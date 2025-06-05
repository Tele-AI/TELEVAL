import logging
import torch
import librosa
from typing import Dict, Any
from transformers import AutoModel, AutoTokenizer

from src.models.base import Model


# load omni model default, the default init_vision/init_audio/init_tts is True
# if load vision-only model, please set init_audio=False and init_tts=False
# if load audio-only model, please set init_vision=False


config_logger = logging.getLogger('transformers_modules.MiniCPM-o-2_6.configuration_minicpm')
config_logger.setLevel(logging.WARNING)
config_logger.propagate = False
config_logger.handlers.clear()

modeling_logger = logging.getLogger('transformers_modules.MiniCPM-o-2_6.modeling_minicpmo')
modeling_logger.setLevel(logging.WARNING)
modeling_logger.propagate = False
modeling_logger.handlers.clear()

logger = logging.getLogger(__name__)

class MiniCPMoAudio(Model):
    def __init__(self, path: str, sample_params: Dict[str, Any] = None):
        super().__init__(sample_params)
        model = AutoModel.from_pretrained(
            path,
            trust_remote_code=True,
            attn_implementation="sdpa",  # sdpa or flash_attention_2
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=True,
        )
        # model = AutoModel.from_pretrained(path, trust_remote_code=True,
        #     attn_implementation='sdpa', torch_dtype=torch.bfloat16)  # huggingface
        
        self.model = model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model.init_tts()
        self.model.tts.float()

        config = {
            "default": {
                "do_sample": True,
                "temperature": 0.3,
                "max_new_tokens": 512,
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
    
    def generate_once(self, audio, **kwargs):
        msgs = []
        if kwargs.get("pred_audio"):
            ref_audio, _ = librosa.load("./src/models/src_minicpm/ref_audios/assistant_female_voice.wav", sr=16000, mono=True)
            sys_prompt = self.model.get_sys_prompt(ref_audio=ref_audio, mode="audio_assistant", language="en")
            msgs.append(sys_prompt)
            generate_audio, output_audio_path = True, kwargs["pred_audio"]
        else:
            # if no ref_audio, these can be commented, have influence on the performance 1~2 point reduce
            ref_audio, _ = librosa.load("./src/models/src_minicpm/ref_audios/assistant_female_voice.wav", sr=16000, mono=True)
            sys_prompt = self.model.get_sys_prompt(ref_audio=ref_audio, mode="audio_assistant", language="en")
            msgs.append(sys_prompt)
            
            generate_audio, output_audio_path = False, None

        user_question = {"role": "user", "content": [librosa.load(audio, sr=16000, mono=True)[0]]} # load the user's audio question
        msgs.append(user_question)
        result = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            use_tts_template=True,
            generate_audio=generate_audio,
            output_audio_path=output_audio_path,
            **self.generation_config
        )
        pred = result if isinstance(result, str) else result.text
        # if use tts model, the RESULT struct is OmniOutput(text=pred, spkr_embed=None, audio_wav=tensor[xxx], sampling_rate=24000)
        return {"pred": pred, "pred_audio": kwargs.get("pred_audio")}

    def generate_multiturn(self, audio, user_history, assistant_history, **kwargs):
        msgs = []
        if kwargs.get("pred_audio"):
            ref_audio, _ = librosa.load("./src/minicpm_prompt/assistant_female_voice.wav", sr=16000, mono=True)
            sys_prompt = self.model.get_sys_prompt(ref_audio=ref_audio, mode="audio_assistant", language="en")
            msgs.append(sys_prompt)
            generate_audio, output_audio_path = True, kwargs["pred_audio"]
        else:
            ref_audio, _ = librosa.load("./src/minicpm_prompt/assistant_female_voice.wav", sr=16000, mono=True)
            sys_prompt = self.model.get_sys_prompt(ref_audio=ref_audio, mode="audio_assistant", language="en")
            msgs.append(sys_prompt)
            
            generate_audio, output_audio_path = False, None
    
        for uh, ah in zip(user_history, assistant_history):
            msgs.append({"role": "user", "content": [librosa.load(uh, sr=16000, mono=True)[0]]})
            msgs.append({"role": "assistant", "content": ah})
        user_question = {"role": "user", "content": [librosa.load(audio, sr=16000, mono=True)[0]]}
        msgs.append(user_question)

        result = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            use_tts_template=True,
            generate_audio=generate_audio,
            output_audio_path=output_audio_path,
            **self.generation_config
        )
        pred = result if isinstance(result, str) else result.text
        # if use tts model, the RESULT struct is OmniOutput(text=pred, spkr_embed=None, audio_wav=tensor[xxx], sampling_rate=24000)
        return {"pred": pred, "pred_audio": kwargs.get("pred_audio")}
