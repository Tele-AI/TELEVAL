import logging
import torch
from typing import Dict, Any
import os
import soundfile as sf
import whisper
from transformers import AutoTokenizer, AutoConfig

from src.models.base import Model
from src.models.src_llama_omni2.model.language_model.omni2_speech2s_qwen2 import Omni2Speech2SQwen2ForCausalLM

from src.models.src_llama_omni2.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN
from src.models.src_llama_omni2.cosy2_decoder import SpeechDecoder
from src.models.src_llama_omni2.cosyvoice.utils.file_utils import load_wav
from src.models.src_llama_omni2.cosy2_decoder import process_units

logger = logging.getLogger(__name__)

class LlamaOmni2(Model):
    def __init__(
        self, 
        path: str, 
        encoder_path:str, 
        vocoder_path: str = None, 
        lang: str = "zh",
        sample_params: Dict[str, Any] = None
    ):
        super().__init__(sample_params)
        self.lang = lang
        model_config = AutoConfig.from_pretrained(path)
        model_config.speech_encoder = encoder_path
        model_config.tts_tokenizer = os.path.join(path, "tts_tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        self.model = Omni2Speech2SQwen2ForCausalLM.from_pretrained(path, config=model_config, torch_dtype=torch.bfloat16)
        self.model.cuda()

        self.speech_decoder = SpeechDecoder(
            model_dir=vocoder_path,
            hop_len=None,
            load_jit=False,
            load_trt=False,
            load_onnx=False,
        )
 
        config = {
            "default": {
                "do_sample": False,
                "temperature": 0,
                "max_new_tokens": 256,
                "top_p": None,
                "top_k": None,
                "num_beams": 1
            },
            "greedy": {
                "do_sample": False,
                "max_new_tokens": 1024,
                "top_k": None,
                "num_beams": 1,
                "temperature": None,
                "top_p": None
            }
        }
        
        self.generation_config = config.get(self.sample_params.get("gen_type", "greedy"), None)
        logger.info("generation_config: {}".format(self.generation_config))

    def load_speech(self, path):
        speech = whisper.load_audio(path)
        speech = whisper.pad_or_trim(speech)
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
        return speech

    def process_messages(self, messages):
        assert len(messages) % 2 == 1, "Number of history messages must be odd"
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")[0]
        input_ids[input_ids == self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)] = SPEECH_TOKEN_INDEX
        return input_ids

    def _process_sample(self, audio, multiturn: bool = False, **kwargs):
        messages = []
        speech_list = []

        if multiturn: # multi-turn
            assert "user_history" in kwargs
            assert "assistant_history" in kwargs

            if kwargs["user_history"]:
                for i, turn_speech in enumerate(kwargs["user_history"]):
                    messages.append({
                        "role": "user",
                        "content": DEFAULT_SPEECH_TOKEN,
                    })
                    speech_list.append(self.load_speech(turn_speech))
                    messages.append({
                        "role": "assistant",
                        "content": kwargs["assistant_history"][i],
                    })

        # single-turn \ current-turn
        messages.append({
            "role": "user",
            "content": DEFAULT_SPEECH_TOKEN,
        })
        speech_list.append(self.load_speech(audio))

        input_ids = self.process_messages(messages).unsqueeze(0)
        speech_tensors = torch.nn.utils.rnn.pad_sequence(
            speech_list,
            batch_first=True,
            padding_value=0
        )
        speech_lengths = torch.LongTensor([len(speech) for speech in speech_list])

        return input_ids, speech_tensors, speech_lengths
    
    def _generate_wav(self, output_units, out_path):
        if self.lang == "zh":
            prompt_speech = "./src/models/src_llama_omni2/wav/prompt_zh.wav"
        elif self.lang == "en":
            prompt_speech = "./src/models/src_llama_omni2/wav/prompt_en.wav"
        
        prompt_speech_16k = load_wav(prompt_speech, 16000)
        x = torch.LongTensor(process_units(output_units)).cuda()

        tts_speech = self.speech_decoder.entry(
            x, 
            prompt_speech_16k,
            stream=False,
        )

        sf.write(
            out_path,
            tts_speech.squeeze(dim=0).detach().cpu().numpy(),
            24000,
        )

    def _generate(self, input_ids, speech_tensor, speech_lengths):
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        speech_tensor = speech_tensor.to(dtype=torch.bfloat16, device='cuda', non_blocking=True)
        speech_lengths = speech_lengths.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids, output_units = self.model.generate(
                input_ids,
                speech=speech_tensor,
                speech_lengths=speech_lengths,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                **self.generation_config
            )

        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return output_text, output_units

    def generate_once(self, audio, **kwargs):
        input_ids, speech_tensor, speech_lengths = self._process_sample(audio)

        output_text, output_units = self._generate(input_ids, speech_tensor, speech_lengths)

        if kwargs.get("pred_audio"):
            self._generate_wav(output_units, kwargs["pred_audio"])

        return {"pred": output_text, "pred_audio": kwargs.get("pred_audio")}
    

    def generate_multiturn(self, audio, user_history, assistant_history, **kwargs):
        input_ids, speech_tensor, speech_lengths = self._process_sample(
            audio,
            multiturn=True,
            user_history=user_history,
            assistant_history=assistant_history,
        )

        output_text, output_units = self._generate(input_ids, speech_tensor, speech_lengths)
        
        if kwargs.get("pred_audio"):
            self._generate_wav(output_units, kwargs["pred_audio"])

        return {"pred": output_text, "pred_audio": kwargs.get("pred_audio")}