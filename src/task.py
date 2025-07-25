import os
import logging
import copy
import json
from tqdm import tqdm
from typing import List, Dict
from src.registry import registry
from src.dataset import BatchLoader, BatchSaver
from src.config import DatasetRuntimeCtx

logger = logging.getLogger(__name__)

class Pipeline:
    @staticmethod
    def create(mode: str, task: str, save_dir: str, **kwargs):
        if mode == "infer":
            return InferenceTask(mode, task, save_dir, **kwargs)
        elif mode == "eval":
            return EvalTask(mode, task, save_dir, **kwargs)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

class BaseTask:
    def __init__(self, mode: str, task: str, save_dir: str, **kwargs):
        self.mode = mode
        self.save_dir = save_dir
        self.kwargs = copy.deepcopy(kwargs)

        self.infer_task_cfg = registry.get_infer_task(task)
        self.model_name = self.kwargs.get("model") or self.infer_task_cfg.model
        
        if self.kwargs.get("save_pred_audio"):
            self.save_pred_audio = eval(self.kwargs.get("save_pred_audio"))
        else:
            self.save_pred_audio = self.infer_task_cfg.save_pred_audio

        if mode == "infer":
            datasets = self.infer_task_cfg.dataset
            if not isinstance(datasets, list):
                datasets = [datasets]
            
            self.template = registry.get_template(self.infer_task_cfg.template)
            self.predictor = registry.get_model(self.model_name)

            self.dataset_infos: List[DatasetRuntimeCtx] = []
            for name in datasets:
                loader = registry.get_dataset(name)
                if kwargs.get("bsz"):  # reset bsz by kwargs, not recommended
                    loader.batch_size = int(kwargs["bsz"])
                saver = BatchSaver(self._get_save_file("prediction", dataset_name=name))
                self.dataset_infos.append(DatasetRuntimeCtx(name, loader, saver))

        elif mode == "eval":
            eval_task_cfg = registry.get_eval_task(self.kwargs.get("eval_task") or self.infer_task_cfg.eval_task)
            datasets = self.infer_task_cfg.dataset
            if not isinstance(datasets, list):
                datasets = [datasets]

            self.evaluator = registry.get_evaluator(eval_task_cfg.evaluator)
            self.summarizer = registry.get_summarizer(eval_task_cfg.summarizer)

            self.dataset_infos: List[DatasetRuntimeCtx] = []
            for name in datasets:
                loader = BatchLoader(
                    self._get_save_file("prediction", dataset_name=name),
                    batch_size=int(self.kwargs.get("bsz") or 1)
                )
                saver = BatchSaver(self._get_save_file("result", dataset_name=name, suffix=self.infer_task_cfg.eval_task))
                summary_file = self._get_save_file("summary", dataset_name=name, suffix=self.infer_task_cfg.eval_task)
                self.dataset_infos.append(DatasetRuntimeCtx(name, loader, saver, summary_file))

    def _get_save_file(self, stage, dataset_name=None, suffix=None):
        dataset_name = dataset_name or self.infer_task_cfg.dataset
        save_path = os.path.join(self.save_dir, stage, self.model_name)
        os.makedirs(save_path, exist_ok=True)

        if suffix is None and self.save_pred_audio:
            save_audio_dir = f"{save_path}/{dataset_name}"
            os.makedirs(save_audio_dir, exist_ok=True)
            self.kwargs["save_audio_dir"] = save_audio_dir

        fname = f"{dataset_name}_{suffix}.jsonl" if suffix else f"{dataset_name}.jsonl"
        return os.path.join(save_path, fname)

    @staticmethod
    def save_summary(summary_file, stat: Dict, save_all=False):
        if summary_file:
            with open(summary_file, "w", encoding="utf-8") as f:
                outputs = stat if save_all else {"score": stat["score"]}
                print(json.dumps(outputs, ensure_ascii=False), file=f)
            logger.info(f"Total score saved to {summary_file}")

    def run(self):
        raise NotImplementedError

class InferenceTask(BaseTask):
    def run(self):
        for dataset_ctx in self.dataset_infos:
            with tqdm(total=len(dataset_ctx.loader), desc="Infer", unit="it") as pbar:  # , disable=True
                for batch_data in dataset_ctx.loader:
                    inputs = [self.template.load(**sample) for sample in batch_data]
                    # inputs: [[{"role": "user", "content": {"audio": audio, "text": text} } ], * batchsize ]
                    # multiturn-input: {"nrounds": "2", "dialogue": [{"role": "A", "round": "1", "content": [...]},{"role": "B", "round": "1", "content": [...]} ,...] }
                    keys = [data.get(dataset_ctx.loader.key_col, i) for i, data in enumerate(batch_data)]
                    pred_args = {
                        "reverse_spkr": self.infer_task_cfg.reverse_spkr,
                        "use_model_history": self.infer_task_cfg.use_model_history,
                        "save_latest_only": self.infer_task_cfg.save_latest_only,
                    }
                    save_audio_dir = self.kwargs.get("save_audio_dir")
                    if save_audio_dir:
                        pred_args["pred_audio"] = [f"{save_audio_dir}/{key}_pred.wav" for key in keys]

                    outputs = self.predictor.inference(inputs, **pred_args)
                    for key, data, output in zip(keys, batch_data, outputs):
                        extra_log = {}  # extra informataion for eval, egs. emotion
                        if dataset_ctx.loader.extra_col:
                            for info in dataset_ctx.loader.extra_col:
                                extra_log[info] = data.get(info)
                        
                        if isinstance(output, dict):
                            # batch single-turn
                            out_log = {
                                "key": key,
                                "query": data.get(dataset_ctx.loader.query_col),
                                "ref": data.get(dataset_ctx.loader.ref_col),
                                **output
                            }
                            out_log.update(extra_log)
                            dataset_ctx.saver.save_one(out_log)

                        elif isinstance(output, list):
                            # single-batch multiturn
                            for round_log in output:
                                out_log = {
                                    "key": key,
                                    **round_log
                                }
                                out_log.update(extra_log)
                                dataset_ctx.saver.save_one(out_log)
                    pbar.update(len(batch_data))

class EvalTask(BaseTask):
    def run(self):
        for dataset_ctx in self.dataset_infos:
            scores = []
            for batch_data in dataset_ctx.loader:
                # sinlge_data: {"key":xxx, "pred":xxx, "ref":xxx, "query":xxx, ...}
                keys, preds, refs, pred_info_list = [
                    list(x) for x in zip(*[
                        (
                            d["key"],
                            d["pred"],
                            d["ref"],
                            {k: d[k] for k in d if k not in ("pred", "ref")}
                        )
                        for d in batch_data
                    ])
                ]

                eval_results = self.evaluator.evaluate(preds, refs, pred_info_list)
                if len(eval_results) != len(pred_info_list):
                    raise ValueError("Lost some results...")
                
                for result, pred_info in zip(eval_results, pred_info_list):
                    if self.kwargs.get("keep_pred_info", True):
                        result.update(pred_info)

                    scores.append(result["score"])
                    dataset_ctx.saver.save_one(result)

            stat = self.summarizer.statistic(scores)
            logger.info(f"total_score: {stat}")
            self.save_summary(dataset_ctx.summary_file, stat, save_all=True)