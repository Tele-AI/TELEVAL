## 自定义参数配置
框架支持 SLM 模型推理所需的音频输入、音频+文本指令输入、纯文本输入，以及 LLM 模型推理所需的纯文本输入。对于 SLM 模型，我们建议只使用音频输入的方法，用以模拟端到端对话场景。

参数配置均在 ```registry/*/*.yaml``` 下调整。

### 自定义infer_task

* ```infer_task```在推理和测评2个阶段，都需要传入。
* ```dataset```, ```template```, ```model```分别为自定义的数据集、数据处理模板和模型名称，具体定义方式详见各自的介绍。
* ```model```, ```save_pred_audio```, ```eval_task```三个参数，可以通过 **```main.py```** 中的全局变量来控制，全局变量优先级高于各yaml文件的配置。
* ```reverse_spkr``` 和 ```use_model_history``` 为多轮测试任务的参数，当多轮测试数据包含两个说话人信息时（其中一个用作ground-truth的评价），开启 ```reverse_spkr``` 会调换两个说话人顺序，关闭 ```use_model_history``` 会使用ground-truth中提供的文本作为模型上一轮的输出历史（**注意：部分模型只支持使用模型自己的历史**），这两个参数单轮测试中无需指定

```yaml
infer_task_name:  # the name of your own infer_task, note that it should be unique
  class: src.config.InferTaskCfg
  args:
    dataset: your_dataset_name
    template: your_dataset_name
    model: your_model_name
    save_pred_audio: False
    eval_task: your_eval_task_name
    # reverse_spkr: False  # for multiturn
    # use_model_history: True  # for multiturn
```

### 自定义eval_task

* 目前框架支持的eval task列表见[可选的eval_task](../README.md#可选的eval_task)
* ```eval_task``` 中只有两个参数，```evaluator``` 为定义的评估器名字，```summarizer``` 为打分结果处理方式，具体定义方式详见各自的介绍。

```yaml
eval_task_name: 
  class: src.config.EvalTaskCfg
  args:
    evaluator: evaluator_name
    summarizer: summarizer_name
```

### 自定义dataset

* 框架的 BatchLoader 支持从huggingface读取parquet文件、读取本地jsonl文件两种格式。
  * 从huggingface读取时，支持将音频解码并保存到temp_dir，当任务结束时会自动删除；如果```save_query_audio_dir``` 参数给定，则会将音频保存到该目录下，任务结束时不会自动删除，便于下次快速调用。
  * 但考虑到网速，建议通过 ```tools/parquet2jsonl.py``` 工具，将parquet文件转为本地jsonl文件和wav，方便多次调用。
* ```key_col```, ```ref_col```, ```query_col```, ```extra_col``` 决定了除模型推理结果以外，保存到```${save_dir}/prediction/${model}/${infer_task}.jsonl``` 里的其他信息，这些信息将用于评测，通常来说不需要 ```extra_col```
* ```batch_size``` 决定了每次推理所用的batch，建议多轮测试时固定为1，以防止OOM

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

### 自定义model

* 如添加自定义模型，需要先在 ```src/models``` 下实现自有模型类，只需实现 ```generate_once``` 和 ```generate_multiturn``` 两个接口函数，用来实现单轮、多轮推理。返回格式为
  ```text
  return {
    "pred": model's text output,
    "pred_audio": model's audio output path (if save_pred_audio if True),
    "cache": model's kv_cache or generate_id (for multiturn),
    "his": model's history text which is differed from text output (for multiturn, only a few models need)
  }
  ```
  * 其中"cache"和"his"不是必须的。如果均不设置，则默认模型使用框架提供的历史信息（通常为模型上一轮输出文本）。"cache"为模型的缓存，可以为kv_cache，也可以为历史token_id，但该值每次都会更新，不会累加，适合多轮返回包含历史的模型；"his"则是模型输出历史，部分模型要求的历史信息和输出文本不同，则使用该参数，会累加到 assistant_history 中。
  * 如果显存受限，可以使用 ```model_utils.py``` 中的 ```load_model_with_auto_device_map``` 方法将模型根据层切分到不同GPU上，可参考 ```kimi_audio.py```中的 ```split_device``` 分支。

* 实现自定义模型后，只需添加新的model config，args中的参数可以根据需要自行添加，和自有模型类中 ```__init__``` 参数一致即可

```yaml
model_name:
  class: src.models.your_model  # your model class
  args:
    path: path/to/model
    sample_params:
      gen_type: greedy
```

### 自定义evaluator

* 如添加自定义评估器，需要先在 ```src/evaluator``` 下实现自有类，返回的 **Dict** 需至少包含key, score两个字段，其他信息则根据 ```summarizer``` 的需要提供
  * 可以使用**装饰器** ```@parallel_batch``` 修饰 ```evaluate``` 函数，```evaluate``` 函数只实现单batch推理，框架会自动执行多线程任务分配。
* 实现评估器方法后，只需添加新的evaluator config，其中如果 ```max_workers``` 不指定，则会使用 ```default_workers```
* **注意**，如果使用LLM API对推理结果进行评价，则需要给入**key**，部分api调用可能需要自行实现。参见```registry/evaluator/llm.yaml```


### 自定义template

* ```template``` 为数据处理Jinjia模板，需要区分不同的 ```role``` （system, instruct, user等），文本数据为text字段，音频数据为audio字段。

### 自定义summarizer

* ```summarizer``` 决定了对评估结果的处理方式，和 ```evaluator``` 相匹配。对于百分制的处理，建议使用 ```AvgInfo``` ，对于主观评估，建议使用```AvgThreshold```，如需自定义 ```summarizer```，需要先在 ```src/summarizer``` 下实现自有类的statistic方法