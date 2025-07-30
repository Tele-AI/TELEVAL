<p align="left">
        <a href="README.md">中文</a> &nbsp｜ &nbsp English&nbsp&nbsp
</p>
<br>

<div align="center">
<h1>
  TELEVAL
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/datasets/Tele-AI/TeleSpeech-AudioBench" target="_blank">HuggingFace Dataset</a>️ • 
📃 <a href="https://arxiv.org/abs/2507.18061" target="_blank">Technical Report</a>
</p>

## Updates
- [Update Jul. 25, 2025] 🔥 Technical report updated
- [Update Jun. 5, 2025] Evaluation code and datasets released

## Introduction

**TELEVAL** is a dynamic evaluation benchmark designed for Spoken-Language Models (SLMs), focusing on Chinese interactive scenarios. It covers three main dimensions: **Explicit Semantics**, **Paralinguistic & Implicit Semantics**, and **System Abilities**, with tasks ranging from basic knowledge to dialect understanding and paralinguistic response.

- **Multi-dimensional Evaluation 🧠**: Covers 12 tasks across 34 datasets, with more continuously added.
- **Real-world Interaction Testing 🎧**: Designed around natural, realistic dialogue needs (e.g., knowledge Q&A, human-like companionship), avoiding artificial prompts like “I'm a child, what should I...” or “What mood am I in?”.
- **Multilingual & Dialect-rich Data 🌏**: Primarily based on Mandarin Chinese, with additional coverage of English Q&A and multiple Chinese dialects (e.g., Cantonese, Henan, Northeastern, Shanghainese, Sichuanese).
- **Modular Evaluation Framework 🔧**: A full inference and evaluation framework with a decoupled design. Supports evaluating existing inference results and customizing models, tasks, and datasets. Works for both SLMs and LLMs.

## Environment
```bash
python3.10 -m venv televal-env
source televal-env/bin/activate

# Install dependencies for inference & evaluation
pip install -r requirements_all.txt

# evaluation only
pip install -r requirements_eval.txt
```

We provide a unified environment in `requirements_all.txt` that includes dependencies for all supported models.  
However, `qwen2.5-omni` and `kimi-audio` require a higher version of `transformers`. For these models, we suggest using
```bash
pip install transformers==4.52.3  # required by qwen2.5-omni
```

## Usage

### Stage 0: Dataset Preparation (Optional)

The framework supports loading datasets from HuggingFace (Parquet format) or local JSONL files. Due to network limitations and large dataset sizes, we recommend downloading and converting datasets to `jsonl + wav` format beforehand for repeated use.  

A `parquet2jsonl.py` tool is provided to automate downloading and conversion.
```bash
# set $save_root_dir to the local directory for saving data
python tools/parquet2jsonl.py
```

To use your **own dataset**, refer to [Custom Dataset](assets/custom.md#自定义dataset) for how to add and test custom data.

### Stage 1: Inference (Optional)

Download the model you want to use for inference and set its path in `registry/model/offline.yaml`.

Tasks are configured via YAML files under `registry/infer_task`. Once the corresponding `*.yaml` file is ready, you can quickly run:
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python main.py --mode "infer" --task "aqa"
```

The framework also supports **global argument settings** to avoid repeatedly editing config files. You can run:
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
infer_task="aqa-llamaqa-zh" # infer tasks defined in registry/infer_task
save_dir="xxx/res"               # prediction and evaluation result saving root dir
save_pred_audio=False        # if True, will save prediction audio
model="freeze_omni"          # model name defined in registry/model
python main.py --mode "infer" --task $infer_task --save_dir $save_dir --save_pred_audio $save_pred_audio --model $model
```

We provide a `run.sh` script for **automatic inference across multiple tasks and models**. Modify the parameters in `run.sh` and run:
```bash
bash run.sh
```

### Stage 2: Evaluation
If inference results already exist, you can run the following script to get scores for a given eval_task:
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
infer_task="aqa-llamaqa-zh"   # infer tasks defined in registry/infer_task
save_dir="xxx/res"           # prediction and evaluation result saving root dir, sub-dir can be used
save_pred_audio=False         # if True, will save prediction audio
model="freeze_omni"           # model name defined in registry/model
python main.py --mode "eval" --task $infer_task --save_dir $save_dir --model $model
```

**You can also run scoring directly via ```run.sh``` for a one-stop evaluation.**

The framework also supports evaluating external results (without running Stage 1). Just ensure the model predictions are saved at ```${save_dir}/prediction/${model}/${infer_task}.jsonl```. Each line in the JSONL file must include at least the fields: ```key, pred, ref``` (custom fields are also supported). Then run the same evaluation script as above.

### Directory Structure

Model predictions and evaluation results are automatically saved in the following structure:
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

## Supported Tasks

Currently supports 34 main datasets (98 sub-datasets). For full details, see [assets/task.md](assets/task.md).

## Dataset Information
Dataset details and their corresponding evaluation abilities can be found in [assets/dataset.md](assets/dataset.md#Dataset_Information).

## Supported SLMs
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

## Results of Open-source Models
Key results are shown in the table below:

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

* Basic Knowledge, Dialect Comprehension, and Dialect P&R are weighted averages across multiple datasets. Acoustic Robustness is the average of the worst-case performance under each acoustic condition.  
* Since most open-source models do not support "dialect audio generation without explicit instructions," this part is excluded from the table.
* Results by dimension can be found in [assets/result.md](assets/result.md#results).  
  For more experiments and analysis, see <a href="https://arxiv.org/abs/2507.18061" target="_blank">Technical Report</a>.

## Define Your Own Dataset and Model
This framework provides a complete pipeline for model inference and evaluation. It supports flexible definitions of tasks, datasets, and models by modifying the corresponding config files under `registry`.  

To add a new model, simply inherit the **`Model`** class and implement the **`generate_once`** and **`generate_multiturn`** methods. See [assets/custom.md](assets/custom.md) for more details.

## Acknowledgements

* Parts of the code in this framework are referenced and adapted from [UltraEval-Audio](https://github.com/OpenBMB/UltraEval-Audio) and [OpenCompass](https://github.com/open-compass/opencompass).
* The audio for datasets `llamaqa-en`, `triviaqa-en`, and `webq-en` comes from [https://huggingface.co/TwinkStart](https://huggingface.co/TwinkStart). We manually filtered these datasets to remove unsuitable QA samples and corrected the answers, so the total number of samples is smaller than the original datasets.
* The inference implementations for each SLM are based on demo scripts from their respective open-source projects. We restructured them to integrate seamlessly into the TELEVAL framework. To ensure all models support *greedy_search* inference, we also adjusted some model codes, for example in `src_freezeomni/audioLLM.py`.

## Citation
If you find this project useful, a star ⭐ and citation would be greatly appreciated —— thank you for your support!
```bibtex
@article{li2025televal,
  title={TELEVAL: A Dynamic Benchmark Designed for Spoken Language Models in Chinese Interactive Scenarios},
  author={Zehan Li and Hongjie Chen and Yuxin Zhang and Jing Zhou and Xuening Wang and Hang Lv and Mengjie Du and Yaodong Song and Jie Lian and Jian Kang and Jie Li and Yongxiang Li and Zhongjiang He and Xuelong Li},
  journal={arXiv preprint arXiv:2507.18061},
  year={2025}
}
```