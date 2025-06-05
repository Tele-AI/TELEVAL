from dataclasses import dataclass
from typing import Dict, List, Union, Tuple, Any
from src.dataset import BatchLoader, BatchSaver

"""
request format
single turn: [ { "role": "user", "content": {"audio": audio, "text": text} } ]
multiturn: {"key": xxx, "nrounds": 8, "content": [{"nround": 1, "text": [spkr1_text, spkr2_text], "audio": [spkr1_audio, spkr2_audio]}, {"nround": 2, ...}...]}
"""
TemplateStruct = Union[str, Dict[str, Any], List[Dict[str, Union[str, List[Dict[str, str]]]]]]
RefType = Union[str, List["RefType"], Tuple["RefType", ...]]
RefsType = List[RefType]

@dataclass
class EvalTaskCfg:
    evaluator: str
    summarizer: str

@dataclass
class InferTaskCfg:
    dataset: Union[str, List[str]]
    template: str
    model: str
    eval_task: str
    save_pred_audio: bool = False
    reverse_spkr : bool = False  # for multiturn
    use_model_history: bool = True  # for multiturn

@dataclass
class DatasetRuntimeCtx:
    name: str
    loader: BatchLoader
    saver: BatchSaver
    summary_file: str = None