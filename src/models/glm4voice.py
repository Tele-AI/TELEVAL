import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "src_glm4")))
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "src_glm4/third_party/Matcha-TTS")))
import warnings
import logging
import re
import uuid
from typing import Dict, Any
import torch
import torchaudio
import json
from threading import Thread
from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor
from queue import Queue

from src.models.base import Model
from src.models.src_glm4.speech_tokenizer.utils import extract_speech_token
from src.models.src_glm4.speech_tokenizer.modeling_whisper import WhisperVQEncoder
from src.models.src_glm4.flow_inference import AudioDecoder
from src.models.src_glm4.audio_process import AudioStreamProcessor

warnings.filterwarnings("ignore", 
    message="`do_sample` is set to `False`.*temperature.*",
    category=UserWarning,
    module="transformers.generation.configuration_utils"
)
logger = logging.getLogger(__name__)

class TokenStreamer:
    def __init__(self, skip_prompt: bool = False, timeout=None):
        self.skip_prompt = skip_prompt
        self.token_queue = Queue()
        self.stop_signal = object()
        self.next_tokens_are_prompt = True
        self.timeout = timeout

    def put(self, value):
        if value.dim() > 1:
            value = value.squeeze(0)
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        for token in value.tolist():
            self.token_queue.put(token)

    def end(self):
        self.token_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.token_queue.get(timeout=self.timeout)
        if value is self.stop_signal:
            raise StopIteration
        return value

class GLM4voice(Model):
    def __init__(self, path: str, speech_tokenizer_path: str, flow_path: str, sample_params: Dict[str, Any] = None):
        super().__init__(sample_params)
        logger.info("start load model from {}".format(path))
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True).cuda().eval()
        self.whisper_model = WhisperVQEncoder.from_pretrained(speech_tokenizer_path).cuda().eval()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(speech_tokenizer_path)
        self.audio_decoder = AudioDecoder(
            config_path=f"{flow_path}/config.yaml",
            flow_ckpt_path=f"{flow_path}/flow.pt",
            hift_ckpt_path=f"{flow_path}/hift.pt",
            device="cuda",
        )

        self.system_prompt = (
            "User will provide you with a speech instruction. Do it step by step. First, "
            "think about the instruction and respond in a interleaved manner, "
            "with 13 text token followed by 26 audio tokens."
        )
        self.system_prompt_text = (
            "User will provide you with a text instruction. Do it step by step. First, "
            "think about the instruction and respond in a interleaved manner, "
            "with 13 text token followed by 26 audio tokens."
        )

        config = {
            "greedy": {
                "do_sample": False,
                "max_new_tokens": 2000,
                "top_k": None,
                "num_beams": 1,
                "temperature": 0,
                "top_p": 1.0
            },
            "default": {
                "max_new_tokens": 2000,
                "temperature": 0.2,
                "top_p": 0.8
            }
        }
        self.generation_config = config.get(self.sample_params.get("gen_type", "greedy"), None)
        logger.info("generation_config: {}".format(self.generation_config))

    @torch.inference_mode()
    def _generate_stream(self, params):
        prompt = params["prompt"]
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        streamer = TokenStreamer(skip_prompt=True)

        thread = Thread(
            target=self.model.generate, 
            kwargs=dict(
                **inputs,
                streamer=streamer,
                **self.generation_config
            )
        )
        thread.start()

        for token_id in streamer:
            yield json.dumps({"token_id": token_id, "error_code": 0}) + "\n"

    def _generate_stream_gate(self, params):
        try:
            yield from self._generate_stream(params)
        except Exception as e:
            print("Caught Unknown Error:", e)
            yield json.dumps({"text": "Server Error", "error_code": 1}) + "\n"

    def _generate(self, audio, input_text=None, previous_input_tokens=None, previous_completion_tokens=None, save_pred_audio=None):
        if audio is None:
            assert input_text is not None
            user_input = input_text
        else:
            audio_tokens = extract_speech_token(self.whisper_model, self.feature_extractor, [audio])[0]
            if len(audio_tokens) == 0:
                raise ValueError("No audio tokens extracted")

            audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
            audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
            user_input = audio_tokens
        
        inputs = ""
        if previous_input_tokens is not None:
            # Gather history
            assert previous_completion_tokens is not None
            inputs = previous_input_tokens + previous_completion_tokens
        
        inputs = inputs.strip()
        if "<|system|>" not in inputs:
            inputs += f"<|system|>\n{self.system_prompt}"
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"

        params = {
            "prompt": inputs
        }
        result = self._generate_stream_gate(params)
        end_token_id = self.tokenizer.convert_tokens_to_ids("<|user|>")  # 151336

        complete_tokens, audio_tokens, text_tokens = [], [], []
        is_finalize = False
        block_size_list = [25, 50, 100, 150, 200]
        block_size_idx = 0
        block_size = block_size_list[block_size_idx]
        audio_offset = self.tokenizer.convert_tokens_to_ids('<|audio_0|>')

        if save_pred_audio:
            tts_speechs, tts_mels = [], []
            prev_mel = None
            is_finalize = False

            end_token_id = self.tokenizer.convert_tokens_to_ids('<|user|>')
            complete_tokens = []
            prompt_speech_feat = torch.zeros(1, 0, 80).to(self.model.device)
            flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(self.model.device)
            this_uuid = str(uuid.uuid4())

        for chunk in result:
            token_id = json.loads(chunk)["token_id"]
            if token_id == end_token_id:
                is_finalize = True
            if len(audio_tokens) >= block_size or (is_finalize and audio_tokens):
                if block_size_idx < len(block_size_list) - 1:
                    block_size_idx += 1
                    block_size = block_size_list[block_size_idx]
                if save_pred_audio:
                    tts_token = torch.tensor(audio_tokens, device=self.model.device).unsqueeze(0)

                    if prev_mel is not None:
                        prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)

                    tts_speech, tts_mel = self.audio_decoder.token2wav(
                        tts_token,
                        uuid=this_uuid,
                        prompt_token=flow_prompt_speech_token.to(self.model.device),
                        prompt_feat=prompt_speech_feat.to(self.model.device),
                        finalize=is_finalize
                    )
                    prev_mel = tts_mel

                    tts_speechs.append(tts_speech.squeeze())
                    tts_mels.append(tts_mel)
                    flow_prompt_speech_token = torch.cat((flow_prompt_speech_token, tts_token), dim=-1)
                audio_tokens = []

            if not is_finalize:
                complete_tokens.append(token_id)
                if token_id >= audio_offset:
                    audio_tokens.append(token_id - audio_offset)
                else:
                    text_tokens.append(token_id)
        
        filtered_tokens = [
            token_id for token_id in complete_tokens 
            if not re.match(r"<\|audio_\d+\|>", self.tokenizer.decode([token_id])) and
            not re.match(r"<\|user\|>", self.tokenizer.decode([token_id]))
        ]
        complete_text = self.tokenizer.decode(filtered_tokens, spaces_between_special_tokens=False)

        if save_pred_audio:
            tts_speech = torch.cat(tts_speechs, dim=-1).cpu()
            with open(save_pred_audio, "wb") as f:
                torchaudio.save(f, tts_speech.unsqueeze(0), 22050, format="wav")
        return inputs, complete_text

    def generate_once(self, audio, **kwargs):
        instruction = kwargs.get("instruct", "")
        save_pred_audio = kwargs.get("pred_audio")
        _, complete_text = self._generate(audio, save_pred_audio=save_pred_audio)

        return {"pred": complete_text, "pred_audio": kwargs.get("pred_audio")}


    def generate_multiturn(self, audio, user_history, assistant_history, **kwargs):
        save_pred_audio = kwargs.get("pred_audio")

        if len(user_history) > 0:
            previous_input_tokens, previous_completion_tokens = kwargs["cache"]
        else:
            previous_input_tokens, previous_completion_tokens = None, None
        
        inputs, complete_text = self._generate(
            audio, 
            save_pred_audio=save_pred_audio,
            previous_input_tokens=previous_input_tokens,
            previous_completion_tokens=previous_completion_tokens
        )

        return {"pred": complete_text, "pred_audio": kwargs.get("pred_audio"), "cache": (inputs, complete_text)}
