<p align="left">
        中文</a>&nbsp ｜ &nbsp<a href="README_EN.md">English</a>
</p>
<br>

<div align="center">
<h1>
  TELEVAL
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/datasets/Tele-AI/TELEVAL" target="_blank">HuggingFace Dataset</a>️ • 
📃 <a href="https://arxiv.org/abs/2507.18061" target="_blank">Technical Report</a>
</p>

## 更新
- [Update Jul. 25, 2025] 🔥 技术报告已更新
- [Update Jun. 5, 2025] 测评代码与数据均已开放

## 简介

**TELEVAL** 是一个为语音对话大模型（Spoken-Language Models, SLMs）设计的动态评测基准，针对中文交互场景，划分为三个维度：显性语义（Explicit Semantics）、隐性语义与副语言信息（Paralinguistic & Implicit Semantics）、系统能力（System Abilities）。包含基础知识、方言理解与回应、副语言信息理解与回应等多个任务与测评能力。

- **多维实用性评估 🧠**：覆盖12大任务34个数据集，数据持续扩充中。
- **真实交互测试 🎧**：模结合实际交互需求（如知识问答、拟人陪伴等），构造自然、真实的对话场景，避免任务型指令如“我是个小孩子，我应该...”、“我现在是什么心情？” ，全面考察模型对用户语音的自然对话能力。
- **多语种与多方言数据支持 🌏**：评测数据以中文普通话为主，同时涵盖英文问答与多种中国方言（如粤语、河南话、东北话、上海话、四川话等）。
- **模块化评测框架 🔧**：完整的模型推理与结果评估框架，推理与评估流程解耦，支持使用已有推理结果进行评估，自定义模型、任务与数据集。支持SLM和LLM的推理、评估。

## 环境准备
```bash
python3.10 -m venv televal-env
source televal-env/bin/activate

# Install dependencies for inference & evaluation
pip install -r requirements_all.txt

# evaluation only
pip install -r requirements_eval.txt
```

在```requirements_all.txt```中我们提供了一个综合的环境，满足各个模型的版本依赖。但是```qwen2.5-omni```和```kimi-audio```要求的```transformers```版本较高，因此在执行这两个模型推理时，建议使用
```bash
pip install transformers==4.52.3  # required by qwen2.5-omni
```

## 运行示例

### Stage 0: 数据集准备 (可选)
框架支持从huggingface读取parquet，以及读取本地jsonl文件两种方法。但由于网速的影响，以及部分数据集较大，我们建议在使用前先将数据集下载并保存为 jsonl + wav 的形式，方便反复调用。我们提供了一个 ```parquet2jsonl.py``` 工具可自动执行数据集的下载、格式转换
```bash
# set $save_root_dir to the local directory for saving data
python tools/parquet2jsonl.py
```

如需使用自有数据集，可参考[自定义dataset](assets/custom.md#自定义dataset)中的方式添加自定义数据集进行测试。

### Stage 1: 模型推理 (可选)
下载需要推理的模型，并配置```registry/model/offline.yaml```中相应模型的路径。

任务运行依赖于 ```registry/infer_task``` 中的设置，如果相应```*.yaml```配置文件已修改完成，快速运行可执行
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python main.py --mode "infer" --task "aqa"
```

框架支持全局参数设置，从而避免反复调整配置文件，可执行如下命令
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
infer_task="aqa-llamaqa-zh" # infer tasks defined in registry/infer_task
save_dir="xxx/res"               # prediction and evaluation result saving root dir
save_pred_audio=False        # if True, will save prediction audio
model="freeze_omni"          # model name defined in registry/model
python main.py --mode "infer" --task $infer_task --save_dir $save_dir --save_pred_audio $save_pred_audio --model $model
```

我们提供了一个```run.sh```脚本，可以执行多任务、多模型自动推理。修改```run.sh```中的参数并执行
```bash
bash run.sh
```

### Stage 2: 打分
已有推理结果，可执行如下推理脚本获得在当前eval_task上的得分
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
infer_task="aqa-llamaqa-zh"   # infer tasks defined in registry/infer_task
save_dir="xxx/res"           # prediction and evaluation result saving root dir, sub-dir can be used
save_pred_audio=False         # if True, will save prediction audio
model="freeze_omni"           # model name defined in registry/model
python main.py --mode "eval" --task $infer_task --save_dir $save_dir --model $model
```

同样的，可以使用```run.sh```脚本一站式完成打分。

* 框架也支持自有结果的评测（不执行Stage 1），需确保已有的模型推理结果保存在 ```${save_dir}/prediction/${model}/${infer_task}.jsonl``` ，jsonl文件每一行的json需要至少有```key, pred, ref```字段（也可自行指定修改），之后同样执行推理脚本即可。

### 保存目录结构
模型推理、测评结果自动保存如下
```text
- $save_dir
    ├── prediction
    │   └── $model
    │       └── ${dataset}.jsonl
    ├── result
    │   └── $model
    │       └── ${dataset}_${eval_task}.jsonl
    ├── summary
    │   └── $model
    │       └── ${dataset}_${eval_task}.jsonl
    └── results.csv
```

## 支持的数据集和任务
当前支持34个主数据集（98个子数据集），支持的数据集、任务详见[assets/task.md](assets/task.md)

## 支持的模型
| Model          | Link  |
|:-------------:|:-------:|
| glm-4-voice-9b | [GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice) |
| MiniCPMo2_6-audio | [MiniCPM-o-2.6](https://github.com/OpenBMB/MiniCPM-o) |
| baichuan_omni_1d5 | [Baichuan-Omni-1.5](https://github.com/baichuan-inc/Baichuan-Omni-1.5) |
| llama_omni | [LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni) |
| speechgpt2 | [SpeechGPT-2.0-preview](https://github.com/OpenMOSS/SpeechGPT-2.0-preview) |
| freeze_omni | [Freeze-Omni](https://github.com/VITA-MLLM/Freeze-Omni) |
| qwen2_5_omni | [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) |
| kimi-audio-7b-instruct | [Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio) |

## 数据集信息
数据集信息与对应的测评能力见 [assets/dataset.md](assets/dataset.md#Dataset_Information)

## 开源模型结果
主要的结果如下表所示
| **Model**                | **Basic Knowledge** (%) | **Dialect Comprehension** (%) | **Value Align** (%) | **Chitchat** (%) | **Dialect P&R** (%) | **Emotion P&R** (%) | **Age P&R** (%) | **NSV P&R** (%) | **Scene** (%) | **Acoustic Robustness** (%) | **CER (Speech)** (%) | **DNSMOS (Speech)** ↑ | **Emo (Speech)** (%) |
|:------------------------:|:-------------------:|:-------------------------:|:---------------:|:------------:|:---------------:|:---------------:|:-----------:|:-----------:|:---------:|:-----------------------:|:-------:|:----------:|:--------:|
| GLM-4-Voice              | 31.55               | 13.13                     | 92.55           | 59.50        | 4.57            | 35.55           | 27.81       | 1.89        | 2.28      | 32.88                   | 6.58    | 3.46       | 31.66    |
| MiniCPM-o-2.6            | 36.16               | 16.67                     | 87.60           | 58.29        | 10.98           | 44.03           | 34.56       | 2.08        | 20.37     | 36.18                   | 2.58    | 3.52       | 34.26    |
| Baichuan-Omni-1.5        | 34.84               | 30.68                     | 95.00           | 26.26        | 7.38            | 13.55           | 12.24       | 1.80        | 3.37      | 42.97                   | 7.89    | 3.40       | 24.74    |
| LLaMA-Omni               | 14.63               | 0.00                      | 49.16           | 9.21         | 0.27            | 8.32            | 3.63        | 0.77        | 0.19      | 12.27                   | 8.33    | 3.21       | 37.28    |
| SpeechGPT-2.0-preview    | 9.88                | 4.98                      | 76.41           | 41.22        | 5.17            | 22.59           | 23.63       | 1.52        | 0.52      | 10.70                   | 17.27   | 2.46       | 27.48    |
| Freeze-Omni              | 33.05               | 16.44                     | 87.57           | 30.90        | 5.72            | 20.72           | 13.68       | 1.85        | 17.75     | 30.48                   | 4.88    | 3.49       | 41.05    |
| Qwen2.5-Omni             | 34.77               | 40.54                     | 82.93           | 80.89        | 18.91           | 44.83           | 42.51       | 2.19        | 32.70     | 42.79                   | 1.69    | 3.47       | 52.59    |
| Kimi-Audio               | 37.18               | 25.71                     | 86.67           | 47.95        | 10.18           | 53.17           | 22.77       | 9.19        | 37.11     | 45.30                   | 3.84    | 3.38       | 45.48    |
| GPT4o-Audio (2024-12-17 preview) | 52.93               | 21.15                     | 96.29           | 34.45        | 9.19            | 35.28           | 17.65       | 2.52        | 14.93     | 38.79                   | 1.94    | 3.46       | 24.09    |

* 其中Basic Knowledge、Dialect Comprehension、Dialect P&R为多数据集的加权平均值，Acoustic Robustness为每种声学设置中最差情况的平均值。由于测试的开源模型基本不具备 "无指令条件下方言音频生成"，因此不在此表中展示
* 不同维度的结果见 [assets/result.md](assets/result.md#results)，更多实验结果及分析见 <a href="https://arxiv.org/abs/2507.18061" target="_blank">Technical Report</a>


## 自定义数据集与模型
本框架提供了完整的模型推理、结果评价的流程，支持灵活的任务、数据集、模型定义，只需要修改```registry```下对应配置文件；如需新增模型，则要继承 **```Model```** 类，并实现 **```generate_once```** 与 **```generate_multiturn```** 方法。详见[assets/custom.md](assets/custom.md)


## 致谢与声明
* 本框架中的部分代码引用、修改自 [UltraEval-Audio](https://github.com/OpenBMB/UltraEval-Audio) 和 [OpenCompass](https://github.com/open-compass/opencompass)
* 数据集中```llamaqa-en, triviaqa-en, webq-en```的音频来自[https://huggingface.co/TwinkStart](https://huggingface.co/TwinkStart)，我们对这些数据集进行了人工筛选，去除不适合作为问答测试的数据，并对答案进行了订正，因此总条数会少于源数据集的条数。
* 各SLM的推理实现基于相应开源项目的演示脚本，我们对其进行了结构上的修改，以便无缝集成到TELEVAL框架中。然而，为了确保所有模型都能执行 *greedy_search* 推理，我们调整了一些模型的代码，例如 ```src_freezeomni/audioLLM.py```
