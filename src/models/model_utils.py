import torch
import functools
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import infer_auto_device_map
from accelerate.utils import get_balanced_memory
from collections import Counter

logger = logging.getLogger(__name__)

def get_no_split_module_candidates(model):
    class_counter = Counter()

    def count_module_classes(module):
        for child in module.children():
            class_name = child.__class__.__name__
            class_counter[class_name] += 1
            count_module_classes(child)

    count_module_classes(model)

    candidates = {name for name, count in class_counter.items() if count > 1}
    return candidates

def load_model_with_auto_device_map(
    model_name: str,
    max_memory: dict = None,
    no_split_module_classes: list = [],
    dtype=torch.float16,
    return_tokenizer=False
):
    """
    Automatically infer device_map and load a multi-GPU model.

    Args:
        model_name (str): Model name or path.
        max_memory (dict, optional): Max memory per GPU, e.g., {0: "20GiB", 1: "20GiB"}.
            If None, get balance memory.
        no_split_module_classes (list, optional): List of module class names that must not be split.
        dtype (torch.dtype, optional): Model precision (default: float16).
        return_tokenizer (bool, optional): Whether to return tokenizer along with model.

    Returns:
        model or (model, tokenizer)
    """
    # load to CPU first
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True, 
        device_map=None
    )
    candidates = get_no_split_module_candidates(model)
    logger.info(f"Folloing modules can be split: {candidates}")

    illegal = [cls for cls in no_split_module_classes if cls not in candidates]
    if illegal:
        raise ValueError(f"{illegal} not in allowed no_split_module_classes: {candidates}")
    
    if max_memory is None:
        max_memory = get_balanced_memory(model, dtype=dtype)
        # n_gpus = torch.cuda.device_count()
        # if n_gpus == 0:
        #     raise ValueError("No CUDA GPUs detected for max_memory auto-inference.")
        # max_memory = {i: "20GiB" for i in range(n_gpus)}
    
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=no_split_module_classes
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True, 
        device_map=device_map
    )

    if return_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        return model, tokenizer

    return model