import logging
import torch
import re
import json
from typing import Dict, Any
from typing import Union
from transformers import (
    AutoTokenizer,
    AutoTokenizer,
    GenerationConfig,
    HfArgumentParser,
)
import numpy as np
import torchaudio
import soundfile as sf
from src.models.base import Model
from src.models.src_speechgpt2.mimo_qwen2_grouped import MIMOLlamaForCausalLM, MIMOModelArguments
from src.models.src_speechgpt2.Codec.models.codec import Generator as SpeechGPT2Tokenizer
from src.models.src_speechgpt2.demo_gradio import MIMOStopper, InputSegment

logger = logging.getLogger(__name__)

def read_wav(audio_path: str, sampling_rate: int):
    wav, raw_sample_rate = torchaudio.load(audio_path)  # (1, T)   tensor
    if raw_sample_rate != sampling_rate:
        wav = torchaudio.functional.resample(
            wav, raw_sample_rate, sampling_rate
        )  # tensor
    return wav

def process_greeting(greeting_source, greeting_line_idx):
    with open(greeting_source, "r") as f:
        for idx, line in enumerate(f):
            if idx == greeting_line_idx:
                greeting = json.loads(line)
                greeting_text = greeting["text"]
                greeting_audio = greeting["audio"]
                break
    return greeting_text, greeting_audio

class SpeechGPT2(Model):
    def __init__(self, path: str, codec_ckpt_path: str, sample_params: Dict[str, Any] = None):
        super().__init__(sample_params)
        codec_config_path = "./src/models/src_speechgpt2/Codec/config/sg2_codec_config.yaml"
        greeting_source = "./src/models/src_speechgpt2/extra/greetings.jsonl"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.sosp_idx = self.tokenizer.convert_tokens_to_ids("<|sosp|>")
        self.eosp_idx = self.tokenizer.convert_tokens_to_ids("<|eosp|>")

        self.empty_token = self.tokenizer.convert_tokens_to_ids("<|empty|>")
        self.end_empty_token = self.tokenizer.convert_tokens_to_ids("<|end_empty|>")
        self.group_size = 3
        self.audio_channels = 3

        parser = HfArgumentParser((MIMOModelArguments,))
        model_args, _ = parser.parse_args_into_dataclasses(
            return_remaining_strings=True
        )
        model_args.model_name_or_path = path

        self.model = MIMOLlamaForCausalLM.from_pretrained(
            path,
            padding_idx=self.tokenizer.pad_token_id,
            sosp_idx=self.sosp_idx,
            eosp_idx=self.eosp_idx,
            args=model_args,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto", 
        )
        self.model.eval()
        self.generator = SpeechGPT2Tokenizer.load_from_checkpoint(
            config_path=codec_config_path, checkpoint_path=codec_ckpt_path
        )
        self.generator = self.generator.to(self.model.device)
        self.generator.eval()
        self.REGEX_HEAD = re.compile(r" ###\n$")

        greeting_text, greeting_audio = process_greeting(greeting_source, 0)
        self.greeting = [
            InputSegment(f"[|SpeechGPT|]: "),
            InputSegment(
                tokenized_text=torch.tensor(greeting_text),
                audio=torch.tensor(greeting_audio).reshape(3, -1),
            ),
            InputSegment(f" ###\n{self.tokenizer.eos_token}"),
        ]
        # self.greeting = None

        config = {
            "default": {
                "do_sample": True,
                "max_new_tokens": 5000,
                "temperature": 0.8,
                "top_p": 0.9,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                ),
            },
            "greedy": {
                "do_sample": False,
                "max_new_tokens": 5000,
                "top_k": None,
                "num_beams": 1,
                "temperature": None,
                "top_p": None,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                ),
            }
        }
        self.generation_config = config.get(self.sample_params.get("gen_type", "greedy"), None)
        logger.info("generation_config: {}".format(self.generation_config))

    def _preprocess(
        self,
        input: Union[None, str] = None,
        add_silence_at_end=True,
        silence_frames=8,
        audio_channels=3,
        group_size=4,
        user_prompt: str = None,
        mode="s2s",
        transcript=None,
        history=None
    ):
        wav = (
            read_wav(input, self.generator.sampling_rate)
            .reshape(1, 1, -1)
            .to(self.model.device)
        )

        tokens = self.generator.inference_tokenize(wav)  # [n_vq, B, t]
        token_flat = (
            tokens.squeeze(1).permute(1, 0).reshape(-1).detach().cpu().numpy()
        )  # [T*n_q]

        silence_tokens = torch.tensor([688, 131, 226])
        token_flat = np.concatenate(
            [token_flat, np.tile(silence_tokens, silence_frames)]
        )
        token_flat = np.concatenate(
            [
                token_flat,
                np.tile(
                    silence_tokens,
                    (
                        group_size * audio_channels
                        - token_flat.shape[0] % (group_size * audio_channels)
                    )
                    // len(silence_tokens),
                ),
            ]
        )
        audio_tokenized = torch.tensor(token_flat)

        prompt = (
            [
                InputSegment(
                    f"You are an helpful assistant. You should answer the user's {'text' if mode[0] == 't' else 'speech'} questions in {'text' if mode[2] == 't' else 'speech'}.\n\n\n",
                ),
                *self.greeting,
            ]
            if not history
            else []
        )
        prompt += [
            InputSegment(f"[|Human|]: "),
            (
                InputSegment("", audio=audio_tokenized, user_prompt=user_prompt)
                if mode[0] == "s"
                else InputSegment(transcript)  # text in
            ),
            InputSegment(f" ###\n[|SpeechGPT|]: "),
        ]

        input_ids = [seg.to_input_id(self.tokenizer, group_size) for seg in prompt]
        input_ids = torch.cat(input_ids, dim=1)

        return input_ids

    def _generate_wav(self,  generated_ids, detokenized_text, text):
        answer = {
            "speech": "",
            "thought": detokenized_text,
            "result": "",
        }
        # Find <|sosp|> and <|eosp|> tokens locations in text channel token sequence
        sosp_idx_locations = (text == self.sosp_idx).nonzero(as_tuple=True)[0]
        eosp_idx_locations = (text == self.eosp_idx).nonzero(as_tuple=True)[0]
        if len(sosp_idx_locations) == 0:
            logging.info("No <|sosp|> token found in the text channel")
        else:
            if len(eosp_idx_locations) == 0:
                eosp_idx_locations = [text.shape[0]]
            sosp_idx_location = sosp_idx_locations[0] * self.group_size
            eosp_idx_location = eosp_idx_locations[0] * self.group_size
            audio_sequence = generated_ids[
                :, sosp_idx_location + self.group_size : eosp_idx_location
            ]

            speech_sequence = audio_sequence[1:].T.flatten()
            assert (speech_sequence < 1024).all()
            answer["result"] = detokenized_text.strip().replace("<|empty|>", ".")

            answer["speech"] = "".join([f"<{i}>" for i in speech_sequence])

            # dump wav
            wav = torch.tensor(0)
            if answer["speech"]:
                tokens = torch.tensor(
                    [int(num) for num in re.findall(r"(\d+)>", answer["speech"])]
                )
                x = (
                    tokens.reshape(-1, 3)
                    .unsqueeze(0)
                    .permute(2, 0, 1)
                    .type(torch.LongTensor)
                    .to(self.model.device)
                )  # [n_vq, B, t]
                wav = self.generator.inference_detokenize(x)
                return (24000, wav.reshape(-1).detach().cpu().numpy())

    def generate_once(self, audio, **kwargs):
        instruction = kwargs.get("instruct", "")
        mode = "s2t"
        if kwargs.get("pred_audio"):
            mode = "s2s"
        if audio is None:
            assert len(kwargs.get("query")) > 0
            mode[0] = "t"
            raise ValueError(f"using text in for speechgpt2")

        with torch.no_grad():
            input_ids = self._preprocess(
                input=audio,
                audio_channels=self.audio_channels,
                group_size=self.group_size,
                mode=mode,
            )

            generation_config = GenerationConfig(**self.generation_config)
            input_ids = input_ids.T.reshape(1, -1)
            prompt_length = input_ids.shape[1] // (self.audio_channels + 1)
            stopping_criteria = [
                MIMOStopper(
                    self.tokenizer.eos_token_id,
                    self.group_size,
                    self.audio_channels,
                    max_tokens=1024 + prompt_length,
                )
            ]
            input_ids = input_ids.to(self.model.device)

            generated_ids = self.model.generate(  
                input_ids,
                generation_config,
                stopping_criteria=stopping_criteria,
            )

            generated_ids = (
                generated_ids.int().cpu().reshape(-1, 4).T[:, prompt_length:]
            )

            text = generated_ids[0, ::self.group_size][:-1]
            detokenized_text = self.tokenizer.decode(text, skip_special_tokens=True)
            response = self.REGEX_HEAD.sub("", detokenized_text)

            if kwargs.get("pred_audio") is not None:
                wav =  self._generate_wav(generated_ids, detokenized_text, text)
                sf.write(kwargs["pred_audio"], wav[1], wav[0])

        return {"pred": response, "pred_audio": kwargs.get("pred_audio")}

    def generate_multiturn(self, audio, user_history, assistant_history, **kwargs):
        mode = "s2t"
        if kwargs.get("pred_audio"):
            mode = "s2s"
        
        if len(user_history) > 0:
            history_id = [kwargs["cache"].to(self.model.device)]
        else:
            history_id = []

        with torch.no_grad():
            input_ids = self._preprocess(
                input=audio,
                audio_channels=self.audio_channels,
                group_size=self.group_size,
                mode=mode,
                history=history_id
            )
            generation_config = GenerationConfig(**self.generation_config)

            input_ids = input_ids.T.reshape(1, -1)
            input_ids = input_ids.to(self.model.device)
            input_ids = torch.cat(history_id + [input_ids], dim=-1)

            prompt_length = input_ids.shape[1] // (self.audio_channels + 1)
            stopping_criteria = [
                MIMOStopper(
                    self.tokenizer.eos_token_id,
                    self.group_size,
                    self.audio_channels,
                    max_tokens=1024 + prompt_length,
                )
            ]
            input_ids = input_ids.to(self.model.device)
            
            generated_ids = self.model.generate(  
                input_ids,
                generation_config,
                stopping_criteria=stopping_criteria,
            )

            his = generated_ids.cpu()

            generated_ids = (
                generated_ids.int().cpu().reshape(-1, 4).T[:, prompt_length:]
            )
            text = generated_ids[0, ::self.group_size][:-1]
            detokenized_text = self.tokenizer.decode(text, skip_special_tokens=True)
            response = self.REGEX_HEAD.sub("", detokenized_text)

            if kwargs.get("pred_audio") is not None:
                wav =  self._generate_wav(generated_ids, detokenized_text, text)
                sf.write(kwargs["pred_audio"], wav[1], wav[0])

        return {"pred": response, "pred_audio": kwargs.get("pred_audio"), "cache": his}
