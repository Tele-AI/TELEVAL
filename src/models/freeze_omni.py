import logging
import torch
import re
from typing import Any, Union, Dict, List, Tuple

import soundfile as sf
from src.models.base import Model
import torchaudio.compliance.kaldi as k
import yaml
import math
from src.models.src_freezeomni.utils import init_encoder_llm, load_checkpoint
from src.models.src_freezeomni.decoder.llm2tts import llm2TTS

logger = logging.getLogger(__name__)

class audioEncoderProcessor:
    def __init__(self, chunk_size = 16):
        self.chunk_size = 16
        self.chunk_overlap = 3
        self.feat_dim = 80
        self.frame_size = 400
        self.frame_shift = 160
        self.frame_overlap = self.frame_size - self.frame_shift
        self.CHUNK = self.frame_shift * self.chunk_size
        self.reset()
    
    def get_chunk_size(self):
        return self.CHUNK
    
    def reset(self):
        self.input_chunk = torch.zeros([1, self.chunk_size + self.chunk_overlap, self.feat_dim])
        self.input_sample = torch.zeros([1, self.CHUNK + self.frame_overlap , 1])
    
    def fbank_shift(self, sample_data):
        # fbank feature shift
        self.input_sample[:, :self.frame_overlap , :] = self.input_sample[:, -self.frame_overlap:, :].clone()
        self.input_sample[:, self.frame_overlap:, :] = sample_data
    
    def chunk_data_shift(self, xs):
        # chunk feature shift
        self.input_chunk[:, :self.chunk_overlap, :] = self.input_chunk[:, -self.chunk_overlap:, :].clone()
        self.input_chunk[:, self.chunk_overlap:, :] = xs.squeeze(0)
    
    def process(self,
                audio: torch.Tensor):
        with torch.no_grad():
            # sample_data = torch.tensor(audio).reshape(1, -1, 1)[:, :, :1] * 32768
            sample_data = audio.reshape(1, -1, 1)[:, :, :1] * 32768
            self.fbank_shift(sample_data)
            # use kaldi api to compute fbank
            xs = k.fbank(waveform = self.input_sample.squeeze(-1), dither=0, 
                         frame_length=25, frame_shift=10, num_mel_bins=self.feat_dim)
            self.chunk_data_shift(xs)
        return self.input_chunk.clone()

class FreezeOmni(Model):
    def __init__(self, path: str, llm_path: str, sample_params = None):
        super().__init__(sample_params)
        logger.info("start load model from {}".format(path))

        with open(path + "/audiollm/train.yaml", "r") as fin:
            configs = yaml.safe_load(fin)
            configs["cmvn_file"] = path + "/audiollm/global_cmvn"
            configs["model_conf"]["llm_path"] = llm_path

        # Init asr model from configs
        self.model = init_encoder_llm(configs)
        load_checkpoint(self.model, path + "/audiollm/final.pt")
        device = torch.device('cuda')
        self.model = self.model.to(device)
        self.model.eval()

        self.audio_processor = audioEncoderProcessor()
        self.tts = llm2TTS(path)

        self.system_prompt = "You are a helpful assistant.{}"
        config = {
            "default": {
                "do_sample": True,
                "temperature": 0.8,
                "max_new_tokens": 128,
                "top_k": 20,
                "top_p": 0.8
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

    def decoder(self, cur_hidden_state, pipeline, cur_text, tts, codec_chunk_size, codec_padding_size, decoder_topk, wav):
        hidden_state_output = torch.cat(cur_hidden_state).squeeze(1)
        cur_text_procced = self.post_process(cur_text)
        embeddings = pipeline.llm_decoder.model.embed_tokens(
                        torch.tensor(pipeline.tokenizer.encode(cur_text_procced)).cuda()
                        )
        for seg in tts.run(embeddings.reshape(-1, 896).unsqueeze(0), decoder_topk,
                            hidden_state_output.reshape(-1, 896).unsqueeze(0), 
                            codec_chunk_size, codec_padding_size):
            wav.append(seg)

    def post_process(self, text):
        """
        Post-processes the input text to standardize various characters and formatting.

        Parameters:
        - text (str): The input text string to be post-processed.

        Actions:
        1. Replaces various Chinese and English punctuation marks with standardized ones.
        2. Removes newline, tab, and other unwanted whitespace characters.
        3. Removes special characters like asterisks, underscores, backticks, and tildes.
        4. Condenses whitespace following periods and colons.
        5. Adjusts the format of numbered lists to use appropriate separators
        6. Ensures the text ends with an appropriate punctuation mark

        Returns:
        - str: The post-processed text string.
        """
        text = text.replace('、', '，')
        text = text.replace('(', ',')
        text = text.replace(')', ',')
        text = text.replace('（', '，')
        text = text.replace('）', '，')

        text = re.sub(r'[\n\r\t]', '', text)
        text = re.sub(r'[*_`~]', '', text)

        text = re.sub(r'(\.|\:)\s+', r'\1', text)
        
        if re.search(r'[\u4e00-\u9fa5]', text):
            text = re.sub(r'(\d+)\.\s*([\u4e00-\u9fa5A-Za-z])', r'\1：\2', text)
        else:
            text = re.sub(r'(\d+)\.\s*([\w])', r'\1:\2', text)
        
        if text and text[-1] not in ["。", "？", "！", ".", "?", "!"]:
            if text[-1] in [",", "，", ";", "；", ":", "：", "、"]:
                text = text[:-1] + "。"
            else:
                text += "。"
        
        return text

    def speech_dialogue(self, 
                        audio: Tuple,
                        role: str=None, 
                        stat: str='sl', 
                        past_key_values=None,
                        last_id=None,
                        past_tokens=None,
                        adapter_cache=None,
                        encoder_cache=None,
                        pe_index=0):
        with torch.no_grad():
            ## input fbank
            feats = audio
            if feats is not None:
                feats = feats.to('cuda')
                feats_lengths = torch.tensor([feats.size(1)]).to('cuda')
            else:
                feats_lengths = None

            extra_inputs = {}
            extra_inputs['top_p'] = self.generation_config["top_p"]
            extra_inputs['top_k'] = self.generation_config["top_k"]
            extra_inputs['temperature'] = self.generation_config["temperature"]
            extra_inputs['do_sample'] = self.generation_config["do_sample"]
            extra_inputs['past_key_values'] = past_key_values
            extra_inputs['stat'] = stat
            extra_inputs['last_id'] = last_id
            extra_inputs['adapter_cache'] = adapter_cache
            extra_inputs['encoder_cache'] = encoder_cache
            extra_inputs['pe_index'] = pe_index
            if role is not None and past_key_values is None:
                # add <|im_end|> in chat_prefix
                extra_inputs['role'] = '<|im_start|>system\n' + role # + '<|im_end|>'

            with torch.autocast(device_type="cuda", 
                       dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32):
                # preprocess system role first 
                if stat == 'pre':
                    past_key_values = self.model.set_system_role(extra_inputs)
                    stat = 'sl'
                else:
                    (last_id, stat, past_key_values, adapter_cache, 
                            encoder_cache, pe_index, hidden_state) = self.model.recognize(
                                feats,
                                feats_lengths,
                                extra_inputs=extra_inputs)
            
            outputs = dict(
                past_key_values=past_key_values,
                stat=stat,
                last_id=last_id,
                adapter_cache=adapter_cache,
                encoder_cache=encoder_cache,
                pe_index=pe_index,
            )

            if stat == 'cs':
                if past_tokens is None:
                    past_tokens = []
                past_tokens.append(last_id[0][0])
                text = self.model.tokenizer.decode(past_tokens, skip_special_tokens=True)
                outputs['hidden_state'] = hidden_state
                outputs['text'] = text
                outputs['past_tokens'] = past_tokens
            
            return outputs

    def _generate(self, input_audio_path, output_audio_path=None, instruction="", past_key_value=None):
        wav, fs = sf.read(input_audio_path)
        wav = torch.tensor(wav)
        
        codec_chunk_size = 40
        codec_padding_size = 10
        decoder_topk = 2
        # Stage0: preprocess
        # set system role, stat will be set to 'sl'
        if past_key_value is None:
            outputs = self.speech_dialogue(None, stat='pre', role=self.system_prompt.format(instruction))
        else:
            outputs = dict(
                past_key_values=past_key_value,
                stat='sl',
                last_id=None,
                adapter_cache=None,
                encoder_cache=None,
                pe_index=0,
            )
        chunk_size = self.audio_processor.get_chunk_size()
        
        # Satge1: start listen
        # stat will be auto set to 'cl' after Stage1
        wav_input = torch.zeros(math.ceil(wav.shape[0] / chunk_size) * chunk_size)
        wav_input[:wav.shape[0]] = wav
        for i in range(0, wav_input.shape[0], chunk_size):
            fbank = self.audio_processor.process(wav_input[i:i + chunk_size])
            outputs = self.speech_dialogue(fbank, **outputs)
            outputs['stat'] = 'cl'
        self.audio_processor.reset()

        outputs['adapter_cache'] = None
        outputs['encoder_cache'] = None
        outputs['pe_index'] = 0
        outputs['stat'] = 'ss'

        # Stage3: start speak
        outputs = self.speech_dialogue(None, **outputs)
        cur_hidden_state = []
        cur_hidden_state.append(outputs['hidden_state'])

        whole_text = ''
        last_text = ''
        cur_text = ''
        wav = []
        # Stage4: contiune speak until stat is set to 'sl'
        # use 'stop' to interrupt generation, stat need to be manually set as 'sl'
        stop = False
        while True:
            if len(outputs['past_tokens']) > self.generation_config["max_new_tokens"]:
                stop = True
            if stop:
                break
            del outputs['text']
            del outputs['hidden_state']
            outputs = self.speech_dialogue(None, **outputs)
            if outputs['stat'] == 'cs':
                whole_text += outputs['text'][len(last_text):]
                cur_hidden_state.append(outputs['hidden_state'])
                # for tts
                if output_audio_path is not None:
                    cur_text += outputs['text'][len(last_text):]
                    suffix_list = ["。", "：", "？", "！", ".", "?","!", "\n"]
                    if outputs['text'][len(last_text):].endswith(tuple(suffix_list)):
                        if len(last_text) > 0 and outputs['text'][len(last_text):].endswith(".") and last_text[-1].isdigit():
                            pass
                        else:
                            if len(cur_hidden_state) > 0:
                                self.decoder(cur_hidden_state, self.model, cur_text, self.tts, 
                                            codec_chunk_size, codec_padding_size, decoder_topk, wav)
                                cur_hidden_state = []
                            cur_text = ""
            
            if outputs['stat'] == 'sl':
                break
            last_text = outputs['text']

        if output_audio_path is not None:
            if len(cur_hidden_state) != 0:
                self.decoder(cur_hidden_state, self.model, cur_text, self.tts, 
                            codec_chunk_size, codec_padding_size, decoder_topk, wav)
            sf.write(output_audio_path, torch.cat(wav, -1).squeeze().float().cpu().numpy(), 24000)
        outputs['stat'] = 'sl'
        outputs['last_id'] = None

        return whole_text, outputs['past_key_values']
    
    def generate_once(self, audio, **kwargs):
        instruction = kwargs.get("instruct", "")
        output_audio_path = kwargs.get("pred_audio", None)
        result, _ = self._generate(audio, output_audio_path)

        return {"pred": result, "pred_audio": output_audio_path}

    def generate_multiturn(self, audio: str, user_history: List[Any], assistant_history: List[Any], **kwargs) -> str:
        output_audio_path = kwargs.get("pred_audio", None)
        past_key_value = kwargs.get("cache", None)
        if len(user_history) > 0 and past_key_value is None:
            raise ValueError(f"Not the first turn, need past_key_value cache for generating!!")

        result, past_key_values = self._generate(audio, output_audio_path, past_key_value=past_key_value)

        return {"pred": result, "cache": past_key_values, "pred_audio": output_audio_path}