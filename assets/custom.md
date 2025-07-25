## Defining Your Own Configuration
The framework supports the input types required for SLMs inference, including audio input, audio + text instruction input (**NOT RECOMMEND**), and pure text input, as well as pure text input for LLM model inference. For SLMs, we recommend using audio-only input to simulate end-to-end dialogue scenarios.

All configurations can be modified under **```registry/*/*.yaml```**.

### How to Define a Custom `infer_task`

* **```infer_task```** needs to be specified for **both the inference and evaluation stages**.
* **```dataset```, ```template```, ```model```** refer to the custom dataset, data processing template, and model name, respectively. Please refer to their individual documentation for specific definitions.
* **```model```, ```save_pred_audio```, ```eval_task```** can be controlled through global variables in **`main.py`**. These global variables override over the settings in the YAML files.
* **`reverse_spkr`** and **`use_model_history`** are for multi-turn tests and not needed in single-turn tasks.
  * If the data includes two speakers (one for ground-truth), enabling **`reverse_spkr`** swaps their roles. 
  * If **`use_model_history`** is disabled, the model uses the ground-truth text as history instead of its own. (*Note: Some SLMs only support using their own history.*)

```yaml
your_infer_task_name:  # the name of your own infer_task, it should be unique
  class: src.config.InferTaskCfg
  args:
    dataset: your_dataset_name
    template: your_template_name
    model: your_model_name
    save_pred_audio: False
    eval_task: your_eval_task_name
    # reverse_spkr: False  # for multiturn
    # use_model_history: True  # for multiturn
    # save_latest_only: True  # for multiturn
```

### How to Define a Custom `eval_task`

* For the list of supported evaluation tasks, see [Available](task.md#available_eval_task) eval_task.
* The **`eval_task`** only has two parameters: **`evaluator`**, which specifies the evaluator name, and **`summarizer`**, which defines how the scores are processed. See their respective documentation for details.

```yaml
your_eval_task_name: 
  class: src.config.EvalTaskCfg
  args:
    evaluator: your_evaluator_name
    summarizer: your_summarizer_name
```

### How to Define a Custom `dataset`
* The **`BatchLoader`** supports two formats: Parquet files from HuggingFace and local JSONL files.
  * When using HuggingFace, audio is decoded and saved to **`temp_dir`** (auto-deleted after the task). If **`save_query_audio_dir`** is set, audio will be saved there and kept for future reuse.
  * For faster access, it's recommended to convert Parquet to local JSONL and WAV files using **`tools/parquet2jsonl.py`**.

* **`key_col`**, **`ref_col`**, **`query_col`**, and **`extra_col`** control what extra info is saved in ```${save_dir}/prediction/${model}/${infer_task}.jsonl```. These are used for evaluation. Usually, `extra_col` is not needed.
* **`batch_size`** sets how many samples are processed at once. For multi-turn tests, keep it at 1 to avoid OOM errors.

```yaml
dataset-name:
  class: src.dataset.BatchLoader
  args:
    file: path/to/huggingface  # or path/to/*.jsonl
    ref_col: answer  # the reference answer column name in file
    query_col: query  # question col for logger
    batch_size: 1
    # key_col: key  # Prepared for private data, this parameter allows you to set the key_col. The default is "key".
    # extra_col: ["xxx", "xxx"]  # List-type
    # save_query_audio_dir: test_data/audios  # if set, will decode and save test wav when generating from huggingface. This setting is not required for JSONL data.
```

### How to Define a Custom ```model```
<a id="own_model"></a>

* To add a custom model, implement your own model class under **`src/models`**. You only need to define two functions: **`generate_once`** for single-turn inference and **`generate_multiturn`** for multi-turn inference. The return format should be:
  ```text
  return {
    "pred": model's text output,
    "pred_audio": model's audio output path (if save_pred_audio if True),
    "cache": model's kv_cache or generate_id (for multiturn),
    "his": model's history text which is differed from text output (for multiturn, only a few models need)
  }
  ```
  * **`cache`** and **`his`** are optional. If both are missing, the model uses default history (usually the last output). 
    * **`cache`** holds temporary states updated each turn (e.g. kv_cache or token_ids). 
    * **`his`** is for output history. Use this if the model requires a different history format than plain output text. It will be accumulated into **`assistant_history`**.
* If GPU memory is limited, you can use **`load_model_with_auto_device_map`** from **`model_utils.py`** to split the model across multiple GPUs by layers. See the **`split_device`** method in **`kimi_audio.py`** for an example.
* After implementing a custom model, just add a new model config. The parameters in `args` can be customized as needed and should match the `__init__` method of your model class.

```yaml
model_name:
  class: src.models.your_model_class  # your model class defined in model/xxx.py
  args:
    path: path/to/model
    # other_path: other path like tts module if needed
    sample_params:
      gen_type: greedy
```

### How to Define a Custom ```evaluator```

* Implement your evaluator class in **`src/evaluator`**. The returned `**Dict**` must include at least **`key`** and **`score`**. Other fields depend on the **`summarizer`** requirements.
  * Use **`@parallel_batch`** on **`evaluate`** to handle single-batch evaluation. The framework will manage multithreading automatically.
* Add a new evaluator config after implementation. If **`max_workers`** is missing, **`default_workers`** is used.
* LLM API evaluation requires a `key`. See `registry/evaluator/llm.yaml` for examples.

### How to Define a Custom ```template```

* **`template`** is a Jinja data-processing template. Different `role` (e.g., system, instruct, user) needed to be divided. Text data uses the `text` field, and audio data uses the `audio` field.

### How to Define a Custom ```summarizer```
* **`summarizer`** defines how evaluation results are processed and matches the `evaluator`.  
  * For percentage scores, `AvgInfo` is recommended. 
  * For subjective evaluations, `AvgThreshold`  is recommended.  
  * To customize a **`summarizer`**, implement a class with a **`statistic`** method under **`src/summarizer`**.