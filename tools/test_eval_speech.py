import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import BatchLoader, BatchSaver
from src.registry import registry

dataset_name = "esd"
eval_task = "dnsmos"  # modal_consistency  dnsmos  empathetic_response_audio  dialect_response_audio
eval_task_cfg = registry.get_eval_task(eval_task)
evaluator = registry.get_evaluator(eval_task_cfg.evaluator)
summarizer = registry.get_summarizer(eval_task_cfg.summarizer)

pred_file = f"{dataset_name}.jsonl"
save_file = f"{dataset_name}_{eval_task}.jsonl"

scores = []
all_results = []
data_loader = BatchLoader(pred_file, batch_size=1)
saver = BatchSaver(save_file)

for batch_data in data_loader:
    keys, preds, refs, pred_info_list = [
        list(x) for x in zip(*[
            (
                d["key"],
                d["pred"],
                d["ref"] if isinstance(d["ref"], list) else [d["ref"]],
                {k: d[k] for k in d if k not in ("pred", "ref")}
            )
            for d in batch_data
        ])
    ]
    eval_results = evaluator.evaluate(preds, refs, pred_info_list)
    if len(eval_results) != len(pred_info_list):
        raise ValueError("Lost some results...")
    
    for result, pred_info in zip(eval_results, pred_info_list):
        result.update(pred_info)
        scores.append(result["score"])
        all_results.append(result)

saver.save_all(all_results)
stat = summarizer.statistic(scores)
print(f"stage: {eval_task}, total_score: {stat}")