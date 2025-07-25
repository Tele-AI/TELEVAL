import logging
from typing import Dict, List, Union, Any
from collections import Counter

logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self, rescale="base", power=2):
        rescale_map = {
            "base": lambda x: x,
            "linear": self.linear_rescale,
            "power": lambda x: self.power_rescale(x, power=power)
        }
        logger.info(f"Using rescale type: {rescale}")
        self.rescale_func = rescale_map[rescale]
    
    def _check_scores(self, scores: List[Any]):
        if any(s is None for s in scores if not isinstance(s, dict)):
            raise ValueError("Scores list contains None values, need re-run evaluator.")

    @staticmethod
    def linear_rescale(score):
        return score * 20

    @staticmethod
    def power_rescale(score, power):
        return ((score / 5) ** power) * 100
    
    # @staticmethod
    # def sigmoid_rescale(score, scale=10):
    #     x = (score - 2.5)  # move to 3
    #     return (1 / (1 + np.exp(-scale * x / 5))) * 100

    def statistic(self, scores: List[Any], **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

class AvgInfo(Summarizer):
    def statistic(self, scores: List[Union[float, Dict[str, float]]], **kwargs):
        if isinstance(scores[0], dict):
            keys = scores[0].keys()
            result = {}
            for key in keys:
                values = [float(s[key]) for s in scores if key in s]
                avg = sum(values) / len(values) * 100
                result[key] = "{}: {:.2f}%".format(key, avg)
            return result
        
        # common
        avg = sum(map(float, scores)) / len(scores) * 100
        return {"score": "AVG: {:.2f}%".format(avg)}

class AvgThreshold(Summarizer):
    def __init__(self, rescale, threshold=60, power=2):
        super().__init__(rescale, power)
        self.threshold = threshold
    
    def statistic(self, scores: List[float], **kwargs):
        self._check_scores(scores)
        scores = list(map(lambda x: self.rescale_func(float(x)), scores))
        score_count = Counter(scores)

        avg = sum(scores) / len(scores)
        above_threshold = sum(count for score, count in score_count.items() if score > self.threshold)
        return {"score": "AVG: {:.2f}".format(avg), "above_threshold": "above{}: {}".format(self.threshold, above_threshold)}

class AvgMOS(Summarizer):
    def statistic(self, scores: List[float], **kwargs):
        avg = sum(map(float, scores)) / len(scores)
        return {"score": "DNSMOS: {:.2f}".format(avg)}

class AvgWER(Summarizer):
    def statistic(self, scores: List[Dict], **kwargs):
        """
        score = {"ref_len": ref_len, "subs": subs, "dele": dele, "inse": inse, "wer": wer}
        """
        total_ref_len = 0
        total_subs = 0.0
        total_dele = 0.0
        total_inse = 0.0

        for score in scores:
            total_ref_len += score.get("ref_len", 0)
            total_subs += score.get("subs", 0.0)
            total_dele += score.get("dele", 0.0)
            total_inse += score.get("inse", 0.0)

        if total_ref_len == 0:
            raise ValueError("Not enough ref_len to static")

        avg_wer = (total_subs + total_dele + total_inse) / total_ref_len * 100
        return {"score": "WER: {:.2f}%".format(avg_wer)}