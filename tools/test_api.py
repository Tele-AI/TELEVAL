import sys
import json
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import BatchLoader, BatchSaver
from src.registry import registry


if __name__ == "__main__":
    static_only = False
    model_name = "freeze_omni"
    evaluator_name = "emo_llm"
    judge_task = "emotion"
    jsonl_files =["esd"]

    summarizer = registry.get_summarizer("AvgThreshold")
    
    for file in jsonl_files:
        input_file = f"res/prediction/{model_name}/{file}.jsonl"
        save_file = f"res/result/{model_name}/{file}_{judge_task}.jsonl"

        print("processing file: ", input_file)
        if static_only:
            scores = []
            with open(save_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    scores.append(int(data["score"]))
            stat = summarizer.statistic(scores)
            print(f"file: {file}, total_score: {stat}")
            raise RuntimeError

        scores = []
        dataloader = BatchLoader(input_file, batch_size=4)
        saver = BatchSaver(save_file)
        evaluator = registry.get_evaluator(evaluator_name)
        
        for idx, batch_data in enumerate(dataloader):
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
                scores.append(result["score"])
                result.update(pred_info)
                saver.save_one(result)

        stat = summarizer.statistic(scores)
        print(f"file: {file}, total_score: {stat}")