from typing import Dict
from jiwer import compute_measures
from src.evaluator.base import Evaluator
from src.utils import parallel_batch
from src.evaluator.text_utils import TextProcessor

class ASR(Evaluator):
    """
    Part from https://github.com/BytedanceSpeech/seed-tts-eval/tree/main
    """
    def __init__(self, model: str, max_workers=None):
        if max_workers is not None:
            self.max_workers = max_workers
        from funasr import AutoModel
        self.model = AutoModel(model=model, disable_update=True)
        self.text_processor = TextProcessor(language="zh")
    
    @parallel_batch(default_workers=4)
    def evaluate(self, pred: str, ref: str, pred_info: Dict, **kwargs):
        pred_audio = pred_info["pred_audio"]
        res = self.model.generate(input=pred_audio, batch_size_s=300)
        transcription = res[0]["text"]

        clean_truth, clean_hypo, wer, subs, dele, inse, ref_len = self.compute_wer(hypo=transcription, truth=pred)
        score = {
            "ref_len": ref_len,
            "subs": subs,
            "dele": dele,
            "inse": inse,
            "wer": wer
        }
        return {"key": pred_info["key"], "clean_trans": clean_hypo, "clean_text": clean_truth, "score": score}

    def compute_wer(self, hypo, truth):
        truth = self.text_processor.normalize_and_clean(truth)
        hypo = self.text_processor.normalize_and_clean(hypo)

        truth_chars = " ".join(truth)
        hypo_chars = " ".join(hypo)
        measures = compute_measures(truth_chars, hypo_chars)
        ref_len = len(truth)

        wer = measures["wer"]
        subs = measures["substitutions"]
        dele = measures["deletions"]
        inse = measures["insertions"]

        return truth_chars, hypo_chars, wer, subs, dele, inse, ref_len