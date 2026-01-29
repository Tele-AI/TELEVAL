import logging
from typing import Dict, Any
from src.models.base import Model
from src.models.src_mimo_audio.mimo_audio.mimo_audio import MimoAudio
from src.models.src_mimo_audio.mimo_audio.modeling_mimo_audio import MiMoSampler

logger = logging.getLogger(__name__)

class MiMoAudio(Model):
    def __init__(
        self, 
        model_path: str, 
        mimo_audio_tokenizer_path: str,
        sample_params: Dict[str, Any] = None
    ):
        super().__init__(sample_params)

        self.model = MimoAudio(model_path, mimo_audio_tokenizer_path)

        config = {
            "default": {
                "max_length": 8192,
                "do_sample": True,
                "temperature": 0.6,
                "top_k": 50,
                "top_p": 0.95
            },
            "greedy": {
                "max_length": 8192,
                "do_sample": False,
                "temperature": None,
                "top_k": None,
                "top_p": None
            }
        }
        
        self.generation_config = config.get(self.sample_params.get("gen_type", "greedy"), None)
        self.model.generate_kwargs["max_length"] = self.generation_config.pop("max_length")
        logger.info("generation_config: {}".format(self.generation_config))

        self.model.default_global_sampler = MiMoSampler(**self.generation_config)
        self.model.default_local_sampler = MiMoSampler(**self.generation_config)

        self.think = self.sample_params.get("think", False)
        logger.info("user think: {}, max_length: {}".format(self.think, self.model.generate_kwargs["max_length"]))

    def generate_once(self, audio, **kwargs):
        output_audio_path = kwargs.get("pred_audio", None)
        # For single-turn dialogues, function: spoken_dialogue_sft and spoken_dialogue_sft_multiturn are equivalent.
        if output_audio_path:
            text_channel_output = self.model.spoken_dialogue_sft(
                audio, 
                output_audio_path=output_audio_path, 
                system_prompt=None, 
            )
            text_channel_output = text_channel_output.split("<|eot|>")[0].replace(".....", "")            
        else:
            text_channel_output = self.model.speech2text_dialogue_sft(
                audio, 
                thinking=self.think,
            )
            if self.think:
                text_channel_output = text_channel_output.split("</think>")[-1] if "</think>" in text_channel_output else ""
        return {"pred": text_channel_output, "pred_audio": output_audio_path}
    
    def generate_multiturn(self, audio, user_history, assistant_history, **kwargs):
        output_audio_path = kwargs.get("pred_audio", None)
        message_list = []

        if len(user_history):
            for i, turn_speech in enumerate(user_history):
                message_list.append({"role": "user", "content": turn_speech})
                assistant_reply = assistant_history[i]
                if output_audio_path:
                    message_list.append({"role": "assistant", "content": {"text": assistant_reply[0], "audio": assistant_reply[1]}})
                else:
                    message_list.append({"role": "assistant", "content": assistant_reply[0]})
        message_list.append({"role": "user", "content": audio})

        if output_audio_path:
            text_channel_output = self.model.spoken_dialogue_sft_multiturn(
                message_list,
                output_audio_path=output_audio_path,
                system_prompt=None,
            )
            text_channel_output = text_channel_output.split("<|eot|>")[0].replace(".....", "")
        else:  # speech2text
            text_channel_output = self.model.speech2text_dialogue_sft_multiturn(
                message_list,
                thinking=self.think,
            )
            if self.think:
                text_channel_output = text_channel_output.split("</think>")[-1] if "</think>" in text_channel_output else ""

        return {"pred": text_channel_output, "pred_audio": output_audio_path, "his": [text_channel_output, output_audio_path]}