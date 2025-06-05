import logging
import torch
import json
from typing import Dict, Any
import whisper
import soundfile as sf
from transformers import AutoTokenizer
from src.models.base import Model
from src.models.src_llama_omni.model.language_model.omni_speech_llama import OmniSpeechLlamaForCausalLM
from src.models.src_llama_omni.model.language_model.omni_speech2s_llama import OmniSpeech2SLlamaForCausalLM
from src.models.src_llama_omni.model.builder import load_pretrained_model
from src.models.src_llama_omni.conversation import conv_templates
from src.models.src_llama_omni.utils import disable_torch_init
from src.models.src_llama_omni.datasets.preprocess import tokenizer_speech_token

logger = logging.getLogger(__name__)

def ctc_postprocess(tokens, blank):
    _toks = tokens.squeeze(0).tolist()
    deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
    hyp = [v for v in deduplicated_toks if v != blank]
    hyp = " ".join(list(map(str, hyp)))
    return hyp

class LlamaOmni(Model):
    def __init__(self, path: str, vocoder_path: str = None, sample_params: Dict[str, Any] = None):
        super().__init__(sample_params)
        self.torch_dtype = torch.float16
        logger.info("start load model from {}".format(path))
        self.tokenizer, self.model, context_len = load_pretrained_model(path, model_base=None, s2s=True)
        # will load whisper twice
        logger.info("successfully load model from {}".format(path))
        if vocoder_path:
            from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
            vocoder_cfg = f"{vocoder_path}/config.json"
            with open(vocoder_cfg) as f:
                vocoder_cfg = json.load(f)
            self.vocoder = CodeHiFiGANVocoder(f"{vocoder_path}/g_00500000", vocoder_cfg).cuda()
        else:
            self.vocoder = None

        config = {
            "default": {
                "do_sample": False,
                "temperature": 0,
                "max_new_tokens": 256,
                "top_p": None,
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

    def _process_sample(self, audio, model_config, input_type="mel", mel_size=128, conv_mode="llama_3", instruction=None):
        qs = "<speech>\nPlease directly answer the questions in the user's speech." #item["conversations"][0]["value"]
        if instruction:
            qs += instruction
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        speech = whisper.load_audio(audio)
        if input_type == "raw":
            speech = torch.from_numpy(speech)
            if model_config.speech_normalize:
                speech = torch.nn.functional.layer_norm(speech, speech.shape)
        elif input_type == "mel":
            speech = whisper.pad_or_trim(speech)
            speech = whisper.log_mel_spectrogram(speech, n_mels=mel_size).permute(1, 0)

        input_ids = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt')
        speech_length = torch.LongTensor([speech.shape[0]])

        return input_ids, speech, speech_length

    def _generate_wav(self, output_units, out_path):
        output_units = [(list(map(int, output_units.strip().split())))]
        x = {
            "code": torch.LongTensor(output_units[0]).view(1, -1),
        }
        if self.vocoder:
            from fairseq import utils as fairseq_utils
            x = fairseq_utils.move_to_cuda(x)
            wav = self.vocoder(x, True)

        sf.write(
            out_path,
            wav.detach().cpu().numpy(),
            16000,
        )
    
    def generate_once(self, audio, **kwargs):
        instruction = kwargs.get("instruct", "")
        input_ids, speech_tensor, speech_length = self._process_sample(audio, self.model.config)

        input_ids = input_ids.unsqueeze(0).to(device=self.model.device, non_blocking=True)
        speech_length = speech_length.unsqueeze(0).to(device=self.model.device, non_blocking=True)
        speech_tensor = speech_tensor.unsqueeze(0).to(dtype=self.torch_dtype, device=self.model.device, non_blocking=True)

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                speech=speech_tensor,
                speech_lengths=speech_length,
                use_cache=True,
                pad_token_id=128004,
                streaming_unit_gen=False,
                **self.generation_config
            )
            output_ids, output_units = outputs

        pred = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if kwargs.get("pred_audio"):
            output_units = ctc_postprocess(output_units, blank=self.model.config.unit_vocab_size)
            self._generate_wav(output_units, kwargs["pred_audio"])
        
        return {"pred": pred, "pred_audio": kwargs.get("pred_audio")}

    def generate_multiturn(self, audio, user_history, assistant_history, **kwargs):
        msgs = []
        raise NotImplementedError
        return {"pred": pred, "pred_audio": kwargs.get("pred_audio")}