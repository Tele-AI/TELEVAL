from dataclasses import dataclass
from typing import Dict, List, Union, Tuple, Any
from src.dataset import BatchLoader, BatchSaver

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
    save_latest_only: bool = False  # for multiturn_memory

@dataclass
class DatasetRuntimeCtx:
    name: str
    loader: BatchLoader
    saver: BatchSaver
    summary_file: str = None