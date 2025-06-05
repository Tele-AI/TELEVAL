import logging
import torch
import torchaudio
import re
import ujson
import os
from typing import Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.base import Model
from src.models.src_baichuan.generation import decode_wave_vocoder, GenerationAudioTokens
import warnings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=FutureWarning, module="contextlib")

class BaichuanOmni(Model):
    def __init__(self, path: str, cosy_vocoder_path: str = None, sample_params: Dict[str, Any] = None):
        super().__init__(sample_params)
        logger.info("start load model from {}".format(path))
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval()
        # self.model.config.use_cache = True
        logger.info("successfully load model from {}".format(path))
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, model_max_length=128000)
        self.model.bind_processor(self.tokenizer, training=False, relative_path='/')  # implemented in model_path/modeling_omni.py

        self.vocoder = None
        if cosy_vocoder_path:
            from src.models.src_baichuan.cosy24k_vocoder.cosy24k_vocoder import Cosy24kVocoder
            self.vocoder = Cosy24kVocoder.from_pretrained(cosy_vocoder_path).cuda()
        
        self.sampling_rate = 24000
        self.audio_start_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audio_start_token_id)
        self.audio_end_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audio_end_token_id)

        self.audiogen_start_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audiogen_start_token_id)
        self.audiogen_end_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audiogen_end_token_id)

        self.special_token_partten = re.compile('<\|endoftext\|>|<audiogen_start_baichuan>|<audiogen_end_baichuan>')
        self.system_prompt = "请用【邻家女声】这个声音回答问题。"
        # NOTE (TTTdas): origin system_prompt yields a poor text result when audiogen_flag=False, 
        #               so we use the text_system_prompt from VoiceBench only if audiogen_flag=False
        self.text_system_prompt = "You are a helpful assistant who tries to help answer the user's question."
        self.role_prefix = {
            'system': '<B_SYS>',
            'user': '<C_Q>',
            'assistant': '<C_A>',
            'audiogen': '<audiotext_start_baichuan>'
        }
        config = {
            "default": {
                "do_sample": True,
                "temperature": 0.8,
                "max_new_tokens": 1024,
                "top_k": 20,
                "top_p": 0.85,
                "repetition_penalty": 1.1,
                "pad_token_id": 0
            },  
            "default_text_cache": {
                "do_sample": True,
                "temperature": 0.3,
                "max_new_tokens": 50,
                "top_k": 20,
                "top_p": 0.85,
                "repetition_penalty": 1.05,
                "pad_token_id": 0
            },
            "default_vocoder": {
                "do_sample": True,
                "temperature": 0.5,
                "max_new_tokens": 500,
                "top_k": 5,
                "top_p": 0.85,
                "repetition_penalty": 1.3,
                "pad_token_id": 151699
            },
            "greedy": {
                "do_sample": False,
                "max_new_tokens": 1024,
                "top_k": None,
                "num_beams": 1,
                "temperature": None,
                "top_p": None,
                "pad_token_id": 0
            },
            "greedy_vocoder": {
                "do_sample": False,
                "max_new_tokens": 500,
                "top_k": None,
                "num_beams": 1,
                "temperature": None,
                "top_p": None,
                "pad_token_id": 151699
            }
        }
        # "pad_token_id" from config.json to forbid logging "Setting pad_token_id to eos_token_id for open-end operation"
        self.generation_config = config.get(self.sample_params.get("gen_type", "greedy"), None)
        self.generation_vocoder_config = config["greedy_vocoder"]
        logger.info("generation_config: {}".format(self.generation_config))

    def _preprocess_messages(self, messages, audiogen_flag=False):
        text = ""
        for i, msg in enumerate(messages):
            if audiogen_flag and msg["role"] == "assistant":
                text += self.role_prefix["audiogen"]
            text += self.role_prefix[msg["role"]]
            text += msg["content"]
        if audiogen_flag:
            text += self.role_prefix["audiogen"]
        text += self.role_prefix["assistant"]
        return text
    
    def _wave_concat(self, wave_list, start, overlap=400):
        new_wave_list = []
        cur = start
        for wave in wave_list[start:]:
            if (
                cur - 1 >= 0
                and wave_list[cur - 1].shape[1] > overlap
                and wave.shape[1] > overlap
            ):
                new_wave_list.append(
                    (
                        wave_list[cur - 1][:, -overlap:]
                        * torch.linspace(
                            1.0, 0.0, overlap, device=wave_list[cur - 1].device
                        )[None, :]
                        + wave[:, :overlap]
                        * torch.linspace(
                            0.0, 1.0, overlap, device=wave_list[cur - 1].device
                        )[None, :]
                    )
                )
            new_wave_list.append(wave)
            cur += 1

        return torch.cat(new_wave_list, dim=1)
    
    def _generate_text_step(self, pret, plen, kv_cache_flag, audiogen_flag):
        if audiogen_flag:
            # NOTE (TTTdas): must be 50 as used in s2s_gradio_demo_cosy_multiturn.py, 1024 will cause tensor mismatch in vocoder
            self.generation_config["max_new_tokens"] = 50
        if not kv_cache_flag:
            textret = self.model.generate(
                pret.input_ids.cuda(),
                attention_mask=pret.attention_mask.cuda(), 
                audios = pret.audios.cuda() if pret.audios is not None else None,
                encoder_length=pret.encoder_length.cuda() if pret.encoder_length is not None else None,
                bridge_length=pret.bridge_length.cuda() if pret.bridge_length is not None else None,
                tokenizer=self.tokenizer,
                stop_strings=[self.audiogen_start_token, '<|endoftext|>'] if audiogen_flag else ['<|endoftext|>'],
                return_dict_in_generate=True,
                **self.generation_config
            )
        else:
            textret = self.model.generate(
                pret.sequences,
                attention_mask=torch.ones_like(pret.sequences),
                tokenizer=self.tokenizer,
                past_key_values=(pret.past_key_values),
                stop_strings = [self.audiogen_start_token,',','!','?','，','。','！','？','. '],
                return_dict_in_generate=True,
                **self.generation_config
            )
        newtext = self.tokenizer.decode(textret.sequences[0, plen:])
        return textret, newtext
    
    def _generate_audio_step(self, pret):
        audioret = GenerationAudioTokens.generate(
                    self.model,
                    pret.sequences,
                    attention_mask=torch.ones_like(pret.sequences),
                    past_key_values=(pret.past_key_values if pret.past_key_values is not None else None),
                    return_dict_in_generate=True,
                    **self.generation_vocoder_config
        )
        wave_segment = decode_wave_vocoder(audioret.audios_sequences.clone(), self.vocoder, self.model)
        return audioret, wave_segment

    def _generate(self, message, local_path, audiogen_flag, return_history=False):
        """
        from https://github.com/baichuan-inc/Baichuan-Omni-1.5/blob/main/web_demo/s2s_gradio_demo_cosy_multiturn.py#L268
        """
        if audiogen_flag:
            path, ext = os.path.splitext(local_path)
        pret = self.model.processor([message])
        plen = pret.input_ids.shape[1]
        ret, newtext = self._generate_text_step(pret, plen, False, audiogen_flag)
        result = self.special_token_partten.sub("", newtext)  # show_text & full_text

        full_text = result
        wave_list = []
        if audiogen_flag:
            start = 0
            for i in range(100):
                m = ret.sequences[0, -1].item()
                if m == self.tokenizer.eos_token_id:
                    if ret.sequences.shape[1] - plen > 1:
                        # NOTE (TTTdas): can only use this branch with origin system prompt, 
                        #               which will result in a more stable text output as well as an unstable audio output
                        ret.sequences[0, -1] = (self.model.config.audio_config.audiogen_start_token_id)
                        ret, wave_segment = self._generate_audio_step(ret)
                        wave_list.extend(wave_segment)
                        if return_history:  # use for multiturn generation
                            torchaudio.save(f"{path}_{i}.wav", torch.cat(wave_segment, dim=0).cpu(), self.sampling_rate)
                            full_text += self.audiogen_start_token + ujson.dumps({"path": f"{path}_{i}.wav"}, ensure_ascii=False) + self.audiogen_end_token
                    break
                
                ret.sequences[0, -1] = self.model.config.audio_config.audiogen_start_token_id
                ret, wave_segment = self._generate_audio_step(ret)
                wave_list.extend(wave_segment)
                if return_history:
                    torchaudio.save(f"{path}_{i}.wav", torch.cat(wave_segment, dim=0).cpu(), self.sampling_rate)  # save_local
                    full_text += self.audiogen_start_token + ujson.dumps({"path": f"{path}_{i}.wav"}, ensure_ascii=False) + self.audiogen_end_token

                ret.sequences[0, -1] = self.model.config.audio_config.audiogen_end_token_id
                plen = ret.sequences.shape[1]
                ret, text_segment = self._generate_text_step(ret, plen, True, True)
                new_text = self.special_token_partten.sub("", text_segment)
                full_text += new_text
                result += new_text
            
            if len(wave_list) > start:
                wave_segment = self._wave_concat(wave_list, start, overlap=int(self.sampling_rate * 0.01))
                torchaudio.save(local_path, wave_segment, self.sampling_rate)
            else:
                torchaudio.save(local_path, torch.cat(wave_segment, dim=0).cpu(), self.sampling_rate)

        return result, full_text

    def generate_once(self, audio, **kwargs):
        if kwargs.get("pred_audio"):
            audiogen_flag, local_path = True, kwargs["pred_audio"]
        else:
            audiogen_flag, local_path = False, None
        msgs = [
            {
                "role": "system", 
                "content": self.system_prompt if audiogen_flag else self.text_system_prompt
            },
            {
                "role": "user", 
                "content": self.audio_start_token + ujson.dumps({"path": audio}, ensure_ascii=False) + self.audio_end_token
            }
        ]
        message = self._preprocess_messages(msgs, audiogen_flag)
        pred, _ = self._generate(message, local_path, audiogen_flag)

        return {"pred": pred, "pred_audio": kwargs.get("pred_audio")}

    def generate_multiturn(self, audio, user_history, assistant_history, **kwargs):
        return_history = True
        if kwargs.get("pred_audio"):
            audiogen_flag, local_path = True, kwargs["pred_audio"]
        else:
            audiogen_flag, local_path = False, None
        
        msgs = []
        if len(user_history) == 0:
            msgs.append({"role": "system", "content": self.system_prompt})

        for uh, ah in zip(user_history, assistant_history):
            msgs.append({"role": "user", "content": self.audio_start_token + ujson.dumps({"path": uh}, ensure_ascii=False) + self.audio_end_token})
            msgs.append({"role": "assistant", "content": ah})

        msgs.append({
            "role": "user",
            "content": self.audio_start_token + ujson.dumps({"path": audio}, ensure_ascii=False) + self.audio_end_token
        })
        message = self._preprocess_messages(msgs, audiogen_flag)
        pred, full_text = self._generate(message, local_path, audiogen_flag, return_history=return_history)

        return {"pred": pred, "pred_audio": kwargs.get("pred_audio"), "his": full_text}