from typing import Dict, Any, Union, List
from src.evaluator.process import SimpleTokenizer, OptionExtractor
from src.config import RefsType
from src.utils import parallel_batch

class Evaluator:
    def evaluate(self, pred, label, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

class ExistMatch(Evaluator):
    """
    referred to https://github.com/DevSinghSachan/emdr2/blob/main/tasks/openqa/dense_retriever/evaluation/qa_validation.py
    """

    def __init__(self, keep_punc=False, max_workers=None):
        self.keep_punc = keep_punc
        if max_workers is not None:
            self.max_workers = max_workers
    
    @parallel_batch(default_workers=4)
    def evaluate(self, pred: str, ref: RefsType, pred_info: Dict, **kwargs):
        # NOTE (TTTdas): If strict sequential matching is required, set keep_punc=False and simply put the ref into a string
        if isinstance(ref, str):
            ref = [ref]
        if not isinstance(ref, List):
            raise ValueError(f"Need List type ref for ExistMatch, but got {type(ref)} instead")
        match = SimpleTokenizer.has_answer(ref, str(pred), uncased=True, keep_punc=self.keep_punc)
        return {"key": pred_info["key"], "pred": pred, "ref": ref, "score": 1 if match else 0}
    

class SingleOptionMatch(Evaluator):
    def __init__(self, max_workers=None, cushion=False):
        self.cushion = cushion
        if max_workers is not None:
            self.max_workers = max_workers
                
    @parallel_batch(default_workers=4)
    def evaluate(self, pred: str, ref: Union[str, List], pred_info: Dict, **kwargs):
        if isinstance(ref, list):
            assert len(ref) == 1
            ref = ref[0]
        match_dict = OptionExtractor.has_answer(ref, str(pred), pred_info.get("query", None), cushion=self.cushion)
        return {"key": pred_info["key"], "pred": pred, "ref": ref, "score": match_dict}