<p align="left">
        中文</a>&nbsp ｜ &nbsp<a href="README_EN.md">English</a>
</p>
<br>

<div align="center">
<h1>
  TeleSpeech-AudioBench
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/datasets/Tele-AI/TeleSpeech-AudioBench" target="_blank">HuggingFace Data</a>️ • 
📃 <a href="https://huggingface.co/datasets/Tele-AI/TeleSpeech-AudioBench" target="_blank">Report(coming soon)</a>
</p>

## 更新
- [Update Jun. 5, 2025] 🔥 测评代码与数据均已开放

## 简介

**TeleSpeech-AudioBench** 旨在探索语音对话大模型（Spoken-Language Models, SLMs）在真实应用中的可行性与实用性，结合实际交互需求（如知识问答、拟人陪伴等），从 7 个关键维度全面衡量模型能力，包括：  
*常识理解、副语言信息感知与回应、拟人程度、声学鲁棒性、音频回应能力、上下文理解及垂类知识掌握*

整体设计以真实应用为导向，强调语言多样性覆盖、交互自然性与评估客观性，主要特点包括：
- **多维实用性评估 🧠**：覆盖 7 大核心维度与多个细分任务，全面检验模型在真实交互中的综合表现。
- **零样本真实交互测试 🎧**：模拟真实使用场景，所有测试均基于 zero-shot 音频输入，无任何文本指令或先验提示，全面考察模型对用户语音的直接响应能力。
- **任务驱动式评估标准 🎯**：不同任务维度对模型输出设定不同要求，例如常识问答允许生成较长回答，而拟人陪伴任务更注重响应的自然度与长度控制。
- **多语种与多方言数据支持 🌏**：评测数据以中文普通话为主，同时涵盖英文问答与多种中国方言（如粤语、河南话、东北话、上海话、四川话等），数据持续扩充中。
- **模块化评测流程 🔧**：推理与评估解耦，支持复用已有测试结果，自定义模型、任务与数据集。
- **可复现与客观性保障 ✅**：所有 SLM 模型统一采用贪心解码策略，评估优先基于客观指标，尽量避免LLM打分引入的主观偏差和随机性，确保实验结果稳定可靠。

## 环境准备
```bash
python3.10 -m venv ctab-env
source ctab-env/bin/activate

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
框架支持从huggingface读取parquet，以及读取本地jsonl文件两种方法。但由于网速的影响，以及部分数据集较大，我们建议在使用前先通过 ```parquet2jsonl.py``` 工具将数据集下载并保存为 jsonl + wav 的形式，方便反复调用。
```bash
# 先修改save_root_dir为要保存到的本地路径
python parquet2jsonl.py
```

如需使用自有数据集，可参考[自定义dataset](assets/custom.md#自定义dataset)中的方式。

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
infer_task="aqa"       # infer tasks defined in registry/infer_task
save_dir="res"         # prediction and evaluation result saving root dir
save_pred_audio=False  # if True, will save prediction audio
model="freeze_omni"    # model name defined in registry/model
python main.py --mode "infer" --task $infer_task --save_dir $save_dir --save_pred_audio $save_pred_audio --model $model
```

对于多模型推理，可以修改并执行```run.sh```脚本
```bash
bash run.sh
```

### Stage 2: 评测
已完成模型的推理，执行如下推理脚本
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
infer_task="aqa"       # infer tasks defined in registry/infer_task
save_dir="res"         # prediction and evaluation result saving root dir
save_pred_audio=False  # if True, will save prediction audio
model="freeze_omni"    # model name defined in registry/model
python main.py --mode "eval" --task $infer_task --save_dir $save_dir --model $model
```

同样的，可以使用```run.sh```脚本一站式完成测评。

框架也支持自有结果的评测（不执行Stage 1），需确保已有的模型推理结果保存在 ```${save_dir}/prediction/${model}/${infer_task}.jsonl``` ，jsonl文件每一行的json需要至少有```key, pred, ref```字段（也可自行指定修改），之后同样执行推理脚本即可。

### 保存目录结构
模型推理、测评结果保存如下
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

## 支持的模型和任务
<a id="支持的模型和任务"></a>

### 测评任务与对应数据集

| ```infer_task```   | ```dataset``` | ```eval_task``` |
|:------------------:|:-------------:|:--------------:|
| aqa-llamaqa-en | ```llamaqa-en```      | ```basic```      |
| aqa-triviaqa-en | ```triviaqa-en```    | ```basic```      |
| aqa-webq-en | ```webq-en```            | ```basic```      |
| aqa-llamaqa-zh | ```llamaqa-zh```      | ```basic```      |
| aqa-triviaqa-zh | ```triviaqa-zh```    | ```basic```      |
| aqa-webq-zh | ```webq-zh```            | ```basic```      |
| aqa-chinesesimpleqa-zh | ```chinesesimpleqa-zh``` | ```basic```      |
| choice-agieval-zh | ```agieval-zh```       | ```choice```        |
| choice-ceval-zh | ```ceval-zh```           | ```choice```        |
| aqa-chinese_quiz-zh | ```chinese_quiz-zh```    | ```basic```        |
| aqa-chinese_quiz-cantonese | ```chinese_quiz-cantonese```    | ```basic```        |
| aqa-chinese_quiz-henan_dialect | ```chinese_quiz-henan_dialect```     | ```basic```        |
| aqa-chinese_quiz-northeastern_mandarin | ```chinese_quiz-northeastern_mandarin``` | ```basic```        |
| aqa-chinese_quiz-shanghainese | ```chinese_quiz-shanghainese``` | ```basic```        |
| aqa-chinese_quiz-sichuanese | ```chinese_quiz-sichuanese```   | ```basic```        |
| aed-audio-instruct | ```aed_combine-zh```     | ```aed_instruct```        |
| emotion-esd | ```esd-zh```     | ```emotion_understand, emotion_response, wer, dnsmos```        |
| acceptance-human-zh | ```human_acceptance-zh```     | ```human_acceptance```        |
| follow-chitchat-cantonese | ```chitchat-cantonese```    | ```dialect_follow, dialect_classify```         |
| follow-chitchat-henan_dialect | ```chitchat-henan_dialect```     | ```dialect_follow, dialect_classify```        |
| follow-chitchat-northeastern_mandarin | ```chitchat-northeastern_mandarin``` | ```dialect_follow, dialect_classify```        |
| follow-chitchat-shanghainese | ```chitchat-shanghainese``` | ```dialect_follow, dialect_classify```        |
| follow-chitchat-sichuanese | ```chitchat-sichuanese```   | ```dialect_follow, dialect_classify```        |
| aqa-livelihood_policy-zh | ```livelihood_policy-zh``` | ```basic``` |
| aqa-livelihood_policy-cantonese | ```livelihood_policy-cantonese```    | ```basic``` |
| aqa-livelihood_policy-henan_dialect | ```livelihood_policy-henan_dialect```     | ```basic``` |
| aqa-livelihood_policy-northeastern_mandarin | ```livelihood_policy-northeastern_mandarin``` | ```basic``` |
| aqa-livelihood_policy-shanghainese | ```livelihood_policy-shanghainese``` | ```basic``` |
| aqa-livelihood_policy-sichuanese | ```livelihood_policy-sichuanese```   | ```basic``` |
| aqa-bubble_noise-zh | ```bubble_noise_{-5dB, 0dB, 5dB, 10dB, 15dB, 20dB}```   | ```basic``` |
| aqa-white_noise-zh | ```white_noise_{-5dB, 0dB, 5dB, 10dB, 15dB, 20dB}```   | ```basic``` |
| aqa-single_background_speaker-zh | ```single_background_speaker_{-5dB, 0dB, 5dB, 10dB, 15dB, 20dB}```   | ```basic``` |
| aqa-multi_background_speakers-zh | ```multi_background_speakers_{-5dB, 0dB, 5dB, 10dB, 15dB, 20dB}```   | ```basic``` |
| aqa-complex_environments-zh | ```complex_environments_{-5dB, 0dB, 5dB, 10dB, 15dB, 20dB}```   | ```basic``` |
| aqa-complex_environments_reverb-zh | ```complex_environments_reverb_{-5dB, 0dB, 5dB, 10dB, 15dB, 20dB}```   | ```basic``` |
| aqa-distortion-zh | ```distortion_rate{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}```   | ```basic``` |
| aqa-lowpass_filtering-zh | ```lowpass_filtering_{100Hz, 200Hz, 300Hz, 400Hz, 500Hz, 600Hz, 700Hz, 800Hz}```  | ```basic``` |
| aqa-packet_loss-zh  | ```packet_loss_rate{10, 20, 30, 40, 50}``` |  ```basic``` |
| aqa-reverberation_RT60-zh | ```reverberation_RT60_{100ms, 500ms, 1000ms, 2000ms, 3000ms}``` |  ```basic``` |
| aqa-different_distance-zh | ```different_distance_{1m, 2m, 3m, 4m, 5m}``` |  ```basic``` |

### 可选的```eval_task```
<a id="可选的eval_task"></a>

| ```eval_task``` | 测评方式  | Metrics | SLM输出模态 |
|:--------------:|:----------:|:----------:|:-------:|
| ```basic```  | 字符串匹配 | ACC | 文本 |
| ```choice``` | 正则匹配 | ACC | 文本 |
| ```emotion_understand``` | LLM打分 | Score | 文本 |
| ```aed_instruct```  | LLM打分 | Score | 文本 |
| ```dialect_follow``` | LLM打分 | Score | 文本 |
| ```human_acceptance``` | LLM打分 | Score | 文本 |
| ```wer```  | 模型输出的文本与音频经ASR后的WER | WER/CER | 音频 |
| ```dnsmos``` | DNSMOS模型打分 |  Score | 音频 |
| ```emotion_response``` | Emo2vec模型基于人工标签打分 | Score | 音频 |
| ```dialect_classify``` | 方言分类模型打分 | Score | 音频 |
* 框架中涉及的匹配算法，使用了较为宽松的匹配策略，但依然可能会有遗漏，无法囊括所有输出情况
* **如需音频的测试，需要在infer_task里将**```save_pred_audio```**设置为True**

### 支持的模型
| ```model```          | Link  |
|:-------------:|:-------:|
| glm-4-voice-9b | [GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice) |
| MiniCPMo2_6-audio | [MiniCPM-o 2.6](https://github.com/OpenBMB/MiniCPM-o) |
| baichuan_omni_1d5 | [Baichuan-Omni-1.5](https://github.com/baichuan-inc/Baichuan-Omni-1.5) |
| llama_omni | [LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni) |
| speechgpt2 | [SpeechGPT-2.0-preview](https://github.com/OpenMOSS/SpeechGPT-2.0-preview) |
| freeze_omni | [Freeze-Omni](https://github.com/VITA-MLLM/Freeze-Omni) |
| qwen2_5_omni | [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) |
| kimi-audio-7b-instruct | [Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio) |

## 数据集信息

| Dataset          | 测评维度     | 测试能力                             | 条数  |
|:--------------------:|:--------------:|:--------------------------------------:|:----:|
| llamaqa-en         | 基础常识     | 英文通用问答 (AQA)                    | 300 |
| triviaqa-en        | 基础常识     | 英文通用问答 (AQA)                    | 837 |
| webq-en            | 基础常识     | 英文通用问答 (AQA)                    | 1938 |
| llamaqa-zh         | 基础常识     | 中文通用问答 (AQA)                     | 300 |
| triviaqa-zh        | 基础常识     | 中文通用问答 (AQA)                    | 837 |
| webq-zh            | 基础常识     | 中文通用问答 (AQA)                    | 1938 |
| chinesesimpleqa-zh | 基础常识     | 中文通用问答, 中国常识 (AQA)            | 2668 |
| agieval-zh         | 基础常识     | 中文单选题  (AQA)                     | 1227 |
| ceval-zh           | 基础常识     | 中文单选题  (AQA)                     | 965 |
| chinese_quiz-zh    | 基础常识     | 中文通用问答, 中国常识 (AQA)           | 827 |
| chinese_quiz-cantonese    | 基础常识     | 方言理解, 中国常识  (AQA)           | 659 |
| chinese_quiz-henan_dialect     | 基础常识     | 方言理解, 中国常识  (AQA)           | 564 |
| chinese_quiz-northeastern_mandarin | 基础常识     | 方言理解, 中国常识   (AQA)          | 615 |
| chinese_quiz-shanghainese | 基础常识     | 方言理解, 中国常识  (AQA)           | 542 |
| chinese_quiz-sichuanese   | 基础常识     | 方言理解, 中国常识  (AQA)           | 674 |
| aed_combine-zh     | 副语言信息     | 音频事件检测                 | 2000 |
| esd-zh     | 副语言信息, 情绪音频回应     | 情绪理解, 情绪回答               | 150 |
| human_acceptance-zh     | 拟人程度     | 回答自然度, 行为对齐              | 300  |
| chitchat-cantonese    | 拟人程度, 方言音频回应     | 方言理解，方言跟随    | 182 |
| chitchat-henan_dialect     | 拟人程度, 方言音频回应     | 方言理解，方言跟随                 | 161 |
| chitchat-northeastern_mandarin | 拟人程度, 方言音频回应     | 方言理解，方言跟随                 | 246 |
| chitchat-shanghainese | 拟人程度, 方言音频回应     | 方言理解，方言跟随                 | 207 |
| chitchat-sichuanese   | 拟人程度, 方言音频回应     | 方言理解，方言跟随                 | 144 |
| noise-zh**            | 声学鲁棒性   | 模型抗噪能力                       | 19500 |
| livelihood_policy-zh  | 垂域知识 (hard) | 中国民生、客服类问答 (AQA) | 1597 |
| livelihood_policy-cantonese    | 垂域知识 (hard)     | 中国民生、客服类方言问答 (AQA) | 804 |
| livelihood_policy-henan_dialect     | 垂域知识 (hard)     | 中国民生、客服类方言问答 (AQA)  | 949 |
| livelihood_policy-northeastern_mandarin | 垂域知识 (hard)     | 中国民生、客服类方言问答 (AQA)  | 908 |
| livelihood_policy-shanghainese | 垂域知识 (hard)     | 中国民生、客服类方言问答 (AQA)  | 810 |
| livelihood_policy-sichuanese   | 垂域知识 (hard)     | 中国民生、客服类方言问答 (AQA)  | 923 |

** ```noize-zh``` 的子数据集构成如下
| Dataset           | 测评维度     | 测试能力                             | 条数  |
|:--------------------:|:--------------:|:--------------------------------------:|:----:|
| bubble_noise_*            | 声学鲁棒性   | 不同信噪比bubble噪声 (AQA)  | 6*300 |
| white_noise_*            | 声学鲁棒性   | 不同信噪比white噪声 (AQA)  | 6*300 |
| single_background_speaker_* | 声学鲁棒性   | 不同信噪比单说话人背景噪声 (AQA)  | 6*300 |
| multi_background_speakers_* | 声学鲁棒性   | 不同信噪比多说话人背景噪声 (AQA)  | 6*300 |
| complex_environments_* | 声学鲁棒性   | 不同信噪比复杂环境场景背景噪声 (AQA)  | 6*300 |
| complex_environments_reverb_* | 声学鲁棒性   | 不同信噪比复杂环境场景(带混响)背景噪声 (AQA)  | 6*300 |
| distortion_rate_* | 声学鲁棒性   | 不同削波失真率 (AQA)  | 6*300 |
| lowpass_filtering_* |  声学鲁棒性   | 不同带宽低通滤波 (AQA)  | 8*300 |
| packet_loss_rate_* |   声学鲁棒性   | 不同丢包率 (AQA)  | 5*300 |
| reverberation_RT60_* |  声学鲁棒性   | 不同混响时间 (AQA)  | 5*300 |
| different_distance_* |  声学鲁棒性   | 说话人不同距离 (AQA)  | 5*300 |


## 开源模型结果

### 1. 基础常识

| Model | llamaqa-en (%) | llamaqa-zh (%) | triviaqa-en (%) | triviaqa-zh (%) | webq-en (%) | webq-zh (%) | chinesesimpleqa-zh (%) | chinese_quiz-zh (%) | agieval-zh (%) | ceval-zh (%)  |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GLM-4-Voice | 67.67  | 53.00  | 34.89  | 27.00  | 37.00  | 34.62  | 14.47  | 47.09  | 34.47  | 41.24   |
| MiniCPM-o 2.6 | 70.67  | 58.33  | 46.95  | 30.59  | 48.50  | 39.42  | 13.68  | 46.25  | 33.90  | 22.28   |
| Baichuan-Omni-1.5 | 69.33  | 58.00  | 34.89  | 29.75  | 42.98  | 39.32  | 15.74  | 51.09  | 27.38  | 28.39   |
| LLaMA-Omni | 70.33  | 22.33  | 31.90  | 6.33  | 34.31  | 6.91  | 0.49  | 0.12  | 0.00  | 0.00   |
| SpeechGPT-2.0-preview | 0.00  | 36.33  | 0.12  | 13.62  | 0.00  | 20.33  | 4.16  | 27.12  | 0.81  | 1.87   |
| Freeze-Omni | 66.00  | 57.67  | 37.87  | 23.78  | 41.95  | 35.60  | 14.48  | 49.76  | 1.39  | 1.24   |
| Qwen2.5-Omni | 69.67  | 58.67  | 43.13  | 29.03  | 44.32  | 35.19  | 13.42  | 56.30  | 20.78  | 24.25   |
| Kimi-Audio | 70.67  | 65.33  | 45.52  | 32.97  | 43.81  | 39.27  | 17.58  | 53.51  | 12.88  | 15.03   |


### 2. 副语言信息、回答自然度

| Model | aed_combine-zh (%) | esd-zh (%) | human_acceptance-zh (%) |
|:-----:|:----------:|:----------:|:----------:|
| GLM-4-Voice    | 2.28 | 40.35 | 50.33 |
| MiniCPM-o 2.6 | 20.37 | 44.03 | 45.51 |
| Baichuan-Omni-1.5 | 3.37 | 15.47 | 41.79 |
| LLaMA-Omni        | 0.19 | 7.57 | 14.93 |
| SpeechGPT-2.0-preview        | 0.52 | 30.80 | 54.85 |
| Freeze-Omni       | 17.75 | 21.12 | 38.47 |
| Qwen2.5-Omni      | 32.7 | 44.77 | 62.52 |
| Kimi-Audio | 37.11 | 52.45 | 53.52 |

### 3. 方言理解
| Model | chinese_quiz-cantonese (%) | chinese_quiz-henan_dialect (%) | chinese_quiz-northeastern_mandarin (%) | chinese_quiz-shanghainese (%) | chinese_quiz-sichuanese (%)  |
|:---:|:---:|:---:|:---:|:---:|:---:|
| GLM-4-Voice | 0.61  | 9.93  | 37.40  | 3.87  | 13.35   |
| MiniCPM-o 2.6 | 15.17  | 10.46  | 35.77  | 1.85  | 17.80   |
| Baichuan-Omni-1.5 | 31.71  | 25.00  | 43.25  | 12.73  | 37.39   |
| LLaMA-Omni | 0.00  | 0.00  | 0.00  | 0.00  | 0.00   |
| SpeechGPT-2.0-preview | 0.30  | 3.37  | 15.77  | 1.29  | 4.01   |
| Freeze-Omni | 1.06  | 13.83  | 38.05  | 2.95  | 24.78   |
| Qwen2.5-Omni | 48.10  | 34.75  | 46.99  | 24.72  | 44.81   |
| Kimi-Audio | 17.91  | 24.65  | 42.76  | 4.24  | 35.91   |

### 4. 方言跟随
| Model | chitchat-cantonese (%) | chitchat-henan_dialect (%) | chitchat-northeastern_mandarin (%) | chitchat-shanghainese (%) | chitchat-sichuanese (%) |
|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|
| GLM-4-Voice    | 1.67 |	2.83 |	12.20 |	0.70 |	2.69 |
| MiniCPM-o 2.6 | 8.42 |	9.44 |	21.27 |	2.67 |	10.33 |
| Baichuan-Omni-1.5 | 6.40 |	7.06 |	11.48 |	2.74 |	8.67 |
| LLaMA-Omni        | 0.73 |	0.12 |	0.28 |	0.04 |	0.17 |
| SpeechGPT-2.0-preview        | 0.70 |	4.40 |	13.11 |	1.08 |	4.00 |
| Freeze-Omni       | 0.70 |	5.81 |	10.94 |	1.29 |	9.42 |
| Qwen2.5-Omni      | 15.56 |	18.29 |	29.06 |	8.75 |	21.08 |
| Kimi-Audio | 8.46 |	11.63 |	16.26 |	1.64 |	12.61 |

### 5. 垂域知识
| Model | livelihood_policy-zh (%) | livelihood_policy-cantonese (%) | livelihood_policy-henan_dialect (%) | livelihood_policy-northeastern_mandarin (%) | livelihood_policy-shanghainese (%) | livelihood_policy-sichuanese (%)  |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GLM-4-Voice | 32.19  | 4.48  | 11.59  | 18.94  | 8.27  | 11.92   |
| MiniCPM-o 2.6 | 30.37  | 16.92  | 13.49  | 20.93  | 11.73  | 16.36   |
| Baichuan-Omni-1.5 | 30.68  | 15.30  | 15.49  | 19.49  | 13.21  | 17.44   |
| LLaMA-Omni | 0.00  | 0.12  | 0.00  | 0.00  | 0.00  | 0.00   |
| SpeechGPT-2.0-preview | 28.49  | 1.99  | 3.69  | 5.84  | 3.33  | 3.90   |
| Freeze-Omni | 33.25  | 5.47  | 9.48  | 15.31  | 6.67  | 15.06   |
| Qwen2.5-Omni | 26.86  | 17.04  | 13.80  | 15.97  | 12.35  | 14.08   |
| Kimi-Audio | 23.98  | 11.69  | 8.64  | 12.78  | 4.44  | 10.29   |

### 6. 声学鲁棒性
节选。详细结果见[Report](#Report)

| Model | bubble_-5dB (%) | white_-5dB (%) | single_bg_spkr_-5dB (%) | multi_bg_spkrs_-5dB (%) | complex_env_-5dB (%) | complex_env_reverb_-5dB (%) | distortion_rate0.6 (%) | lowpass_filter_100Hz (%) | packet_loss_rate50 (%) | reverb_3000ms (%) | distance_5m (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GLM-4-Voice | 18.00  | 19.33  | 15.67  | 8.00  | 37.00  | 20.00  | 49.33  | 47.00  | 47.67  | 50.33  | 49.33  |
| MiniCPM-o 2.6 | 12.00  | 24.00  | 15.67  | 11.67  | 38.33  | 23.33  | 54.33  | 54.67  | 52.67  | 55.33  | 56.00  |
| Baichuan-Omni-1.5 | 28.00  | 39.33  | 24.00  | 20.00  | 49.00  | 30.00  | 56.67  | 54.00  | 55.33  | 58.33  | 58.00  |
| LLaMA-Omni | 5.67  | 5.67  | 5.67  | 2.67  | 13.00  | 6.33  | 19.33  | 19.33  | 17.00  | 19.33  | 21.00  |
| SpeechGPT-2.0-preview | 0.33  | 0.00  | 4.00  | 0.67  | 7.00  | 2.33  | 19.00  | 15.67  | 9.67  | 30.33  | 28.67  |
| Freeze-Omni | 9.00  | 14.00  | 14.67  | 6.33  | 27.33  | 17.00  | 49.33  | 46.33  | 41.33  | 55.33  | 54.67  |
| Qwen2.5-Omni | 27.67  | 39.33  | 25.00  | 17.00  | 44.67  | 30.67  | 55.67  | 58.33  | 54.00  | 59.33  | 59.00  |
| Kimi-Audio  | 24.33   | 40.33   | 30.33   | 16.67   | 45.67   | 27.33   | 62.33   | 63.00   | 61.00   | 64.00   | 63.33   |


### 7. 音频回应能力
| Model | CER ↓ | DNSMOS ↑ | emotion_response ↑ | chitchat-cantonese (%) | chitchat-henan_dialect (%) | chitchat-northeastern_mandarin (%) | chitchat-shanghainese (%) | chitchat-sichuanese (%) |
|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| GLM-4-Voice    | 6.58 |	3.46 |	31.66 | 3.30 |	0.00 |	0.00 |	0.00 |	0.00 |
| MiniCPM-o 2.6 | 2.58 |	3.52 |	34.26 | 0.00 |	0.00 |	0.00 |	0.00 |	0.00 |
| Baichuan-Omni-1.5 | 7.89 |	3.40 |	24.74 | 0.00 |	0.00 |	0.00 |	0.00 |	0.00 |
| LLaMA-Omni        | 8.33 |	3.21 |	37.28 | - | - | - | - | - |
| SpeechGPT-2.0-preview        | 17.27 |	2.46 |	27.48 | 0.00 |	0.00 |	1.22 |	3.86 |	4.17 |
| Freeze-Omni       | 4.88 |	3.49 |	41.05 | 0.00 |	0.00 |	0.00 |	0.00 |	0.00 |
| qwen2_5Qwen2.5-Omni_omni      | 1.69 |	3.47 |	52.59 |	0.00 |	0.00 |	0.00 |	0.00 |	0.00 |
| Kimi-Audio | 3.84 |	3.38 |	45.48 | 0.00 |	0.00 |	0.41 |	0.00 |	0.00 |


## 自定义数据集与模型
框架支持灵活的任务、数据集、模型定义，只需要修改```registry```下对应配置文件；如需新增模型，则要继承<b><code>Model</code></b>类，并实现<b><code>generate_once</code></b>与<b><code>generate_multiturn</code></b>方法。详见[assets/custom.md](assets/custom.md)


## 致谢与声明
* 本框架中的部分代码引用、修改自 [UltraEval-Audio](https://github.com/OpenBMB/UltraEval-Audio) 和 [OpenCompass](https://github.com/open-compass/opencompass)
* 数据集中```llamaqa-en, triviaqa-en, webq-en```的音频来自[https://huggingface.co/TwinkStart](https://huggingface.co/TwinkStart)，我们对这些数据集进行了人工筛选，去除不适合作为问答测试的数据，并对答案进行了订正。
* 各SLM的推理实现基于相应开源项目的演示脚本，我们对其进行了结构上的修改，以便无缝集成到我们的框架中。然而，为了确保所有模型都能执行 *greedy_search* 推理，我们调整了一些模型的代码，例如 ```src_freezeomni/audioLLM.py```