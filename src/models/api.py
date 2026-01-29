import logging
import io
import base64
import requests
import threading
import itertools
from typing import Dict
import json
import torchaudio
from src.models.base import Model
from src.utils import retry

import sys
sys.stdout.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)

class GPT4oAudio(Model):
    def __init__(self, llm_name: str, api_keys: Dict, max_workers=None):
        super().__init__(sample_params=None)
        logging.info(f"Using {llm_name} API for judgement...")
        assert len(api_keys) > 0
        self.llm_name = llm_name
        self.api_keys = api_keys

        self.urls = {
            key: (
                f"https://{key}.openai.azure.com/"
                f"openai/deployments/{llm_name}/chat/completions?api-version=2025-01-01-preview"
            )
            for key in api_keys
        }
        self.max_workers = max_workers or len(api_keys)
        self.key_cycle = itertools.cycle(self.api_keys.items())
        self.lock = threading.Lock()

    def get_next_key(self):
        with self.lock:
            key_name, key_value = next(self.key_cycle)
            return key_name, key_value, self.urls[key_name]

    @retry(max_retries=8, sleep_second=3)
    def api_generate(self, messages, api_key, url, modalities):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        input_data = {
            "model": "gpt-4o-audio-preview",
            "modalities": modalities,
            "audio": {
                "voice": "alloy",
                "format": "wav"
            },
            "messages": messages
        }

        response= requests.post(url, headers=headers, data=json.dumps(input_data))
        response.raise_for_status()
        response_data = response.json()
        response = response_data["choices"][0]["message"]

        if "audio" in modalities:
            base64_str = response["audio"]["data"]
            pred = response["audio"]["transcript"].strip()
            assert base64_str is not None
            audio_id = response["audio"]["id"]
        else:
            if "content" not in response:
                logging.info(f"response is unique: {response}")
            pred = response["content"].strip()
            base64_str = None
            audio_id = None
        
        return base64_str, pred, audio_id

    @staticmethod
    def encode_audio(audio):
        with open(audio, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return audio_base64

    def generate_once(self, audio, **kwargs):
        save_pred_audio = kwargs.get("pred_audio", None)

        if audio is None:
            modalities = ["text"]
        else:
            modalities = ["audio", "text"]
        
        audio_base64 = self.encode_audio(audio)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": "wav"
                        }
                    }
                ]
            }
        ]

        key_name, api_key, url = self.get_next_key()
        base64_str, pred, audio_id = self.api_generate(messages, api_key, url, modalities)
        if save_pred_audio:
            audio_bytes = base64.b64decode(base64_str)
            audio_buf = io.BytesIO(audio_bytes)
            waveform, sample_rate = torchaudio.load(audio_buf)
            torchaudio.save(save_pred_audio, waveform, sample_rate=sample_rate)

        return {"pred": pred, "pred_audio": kwargs.get("pred_audio")}
    
    def generate_multiturn(self, audio, user_history, assistant_history, **kwargs):
        messages = []
        for uh, ah in zip(user_history, assistant_history):
            uh_base64 = self.encode_audio(uh)
            messages.append({"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": uh_base64, "format": "wav"}}]})
            messages.append({"role": "assistant", "audio": {"id": ah}})

        save_pred_audio = kwargs.get("pred_audio", None)
        if audio is None:
            modalities = ["text"]
        else:
            modalities = ["audio", "text"]
        
        audio_base64 = self.encode_audio(audio)
        messages.append({"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": audio_base64, "format": "wav"}}]})

        key_name, api_key, url = self.get_next_key()
        base64_str, pred, audio_id = self.api_generate(messages, api_key, url, modalities)

        if save_pred_audio:
            audio_bytes = base64.b64decode(base64_str)
            audio_buf = io.BytesIO(audio_bytes)
            waveform, sample_rate = torchaudio.load(audio_buf)
            torchaudio.save(save_pred_audio, waveform, sample_rate=sample_rate)
        
        return {"pred": pred, "pred_audio": kwargs.get("pred_audio"), "his": audio_id}
    