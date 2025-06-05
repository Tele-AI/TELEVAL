import requests
import json
import logging
import threading
import itertools
from transformers import AutoTokenizer
from src.evaluator.base import Evaluator
from src.evaluator.process import LLMExtractor
from src.registry import registry
from src.prompt.llm_judge import TASK_PROMPT_MAP
from src.utils import retry, parallel_batch

logger = logging.getLogger(__name__)

class LLMScorer(Evaluator):
    def __init__(self, llm_name: str, judge_task: str, api_keys: dict, max_workers=None):
        logging.info(f"Using {llm_name} API for judgement...")
        assert len(api_keys) > 0
        self.prompt_generator = TASK_PROMPT_MAP.get(judge_task)
        if self.prompt_generator is None:
            raise ValueError(f"Unsupported task: {judge_task}")
        self.llm_name = llm_name
        self.api_keys = api_keys

        self.urls = {
            key: (
                f"https://{key}.openai.azure.com/"
                f"openai/deployments/{llm_name}/chat/completions?api-version=2024-02-01"
            )
            for key in api_keys
        }
        self.max_workers = max_workers or len(api_keys)
        self.key_cycle = itertools.cycle(self.api_keys.items())
        self.lock = threading.Lock()

    def get_next_key(self):
        with self.lock:
            key_name, key_value = next(self.key_cycle)
            return key_name, key_value, self.urls[key_name]

    def api_generate(self, temp, api_key, url):
        headers= {
            "Content-Type": "application/json",
            "api-key": api_key
        }
        input_data= {
            "model": self.llm_name,
            "messages": [
                {"role":"user","content": temp}
            ]
        }
        response= requests.post(url, headers=headers, data=json.dumps(input_data))
        response.raise_for_status()
        response_data = response.json()
        return response_data['choices'][0]['message']['content'].strip()

    @retry(max_retries=5, sleep_second=3)
    def get_server_response(self, temp, api_key, url):
        api_response = self.api_generate(temp, api_key, url)
        score, reason = LLMExtractor.extract(
                            api_response, 
                            explain_col="Explanation", 
                            score_col="Score"
                        )
        return score, reason

    @parallel_batch(default_workers=4)
    def evaluate(self, pred, ref, pred_info, **kwargs):
        key_name, api_key, url = self.get_next_key()
        model_input = self.prompt_generator(pred, ref, **pred_info)

        score, reason = self.get_server_response(model_input, api_key, url)
        return {"key": pred_info["key"], "pred": pred, "ref": ref, "score": score, "reason": reason}


class LLMOfflineScorer(Evaluator):
    def __init__(self, llm_name: str, template: str, judge_task: str, generate_params: dict):
        logging.info(f"Using vllm to run {llm_name} offline model for judgement...")
        from vllm import LLM, SamplingParams
        self.template = registry.get_template(template)
        model_path = registry.get_model_cfg(llm_name).get("path")
        if model_path is None:
            raise ValueError(f"{llm_name} model path is required for LLMOfflineScorer")

        self.prompt_generator = TASK_PROMPT_MAP.get(judge_task)

        self.sampling_params = SamplingParams(temperature=generate_params["temperature"], 
                                              top_p=generate_params["top_p"], 
                                              repetition_penalty=generate_params["repetition_penalty"], 
                                              max_tokens=generate_params["max_tokens"])
        self.llm = LLM(model=model_path, tensor_parallel_size=generate_params.get("ngpus", 8))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        """for the latest vllm, no need of SamplingParams"""
        # self.sampling_params = self.llm.get_default_sampling_params()
        # if generate_params.get("max_tokens"):
        #     self.sampling_params.max_tokens = generate_params["max_tokens"]

    def evaluate(self, preds, refs, pred_info_list, **kwargs):
        if not (isinstance(preds, list) and isinstance(refs, list) and isinstance(pred_info_list, list)):
            raise ValueError("Input type must be List")
        if not (len(preds) == len(refs) == len(pred_info_list)):
            raise ValueError("len of pred, ref, pred_info_list must equal")

        message_key_pairs = [
            (
                [
                    {**m, "content": m["content"]["text"]}
                    for m in self.template.load(**dict(text=self.prompt_generator(pred, ref, **pred_info)))
                ],
                pred_info["key"]
            )
            for pred, ref, pred_info in zip(preds, refs, pred_info_list)
        ]
        messages, keys = map(list, zip(*message_key_pairs))

        results = []
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = self.llm.generate(text, self.sampling_params, use_tqdm=False)
        for output, pred, ref, key in zip(outputs, preds, refs, keys):
            generated_text = output.outputs[0].text
            score, reason = LLMExtractor.extract(generated_text, explain_col="Explanation", score_col="Score")
            results.append({
                "key": key, 
                "pred": pred, 
                "ref": ref, 
                "score": score, 
                "reason": reason
            })
        return results