from typing import List, Dict
from src.evaluator.base import Evaluator

class Emo2vec(Evaluator):
    def __init__(self, model: str, strict: bool = True):
        from funasr import AutoModel
        self.model = AutoModel(model=model, hub="ms", disable_update=True)
        self.strict = strict

    def evaluate(self, preds, refs, pred_info_list: List[Dict], **kwargs):
        # emo2vec model support batch generate
        pred_audios = [info["pred_audio"] for info in pred_info_list]
        model_outputs = self.model.generate(
            pred_audios, output_dir=None, granularity="utterance", extract_embedding=False
        )

        results = []
        for output, info in zip(model_outputs, pred_info_list):
            label_scores = {
                label.split("/")[-1].lower(): score
                for label, score in zip(output["labels"], output["scores"])
            }
            ref_emotions = [emo.lower() for emo in info["ref_answer_emo"]]

            if self.strict:
                neutral_count = sum(1 for emo in ref_emotions if emo == "neutral")
                if neutral_count <= len(ref_emotions) // 2:
                    # remove "neutral"
                    filtered_ref_emotions = [emo for emo in ref_emotions if emo != "neutral"]
                else:
                    filtered_ref_emotions = ref_emotions
            else:
                filtered_ref_emotions = ref_emotions

            score = max((label_scores.get(emo, 0) for emo in filtered_ref_emotions), default=0)
            results.append({"key": info["key"], "score": score})
        return results
