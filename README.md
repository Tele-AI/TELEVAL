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

**TeleSpeech-AudioBench** 聚焦于语音对话大模型（SLM）在真实应用场景中的实用性，从 7 个关键维度全面衡量模型能力，包括：  
*常识理解、副语言信息感知与回应、拟人程度、声学鲁棒性、音频生成能力、上下文理解及垂类知识掌握*

框架设计兼顾多语言支持、灵活扩展与评估可复现性：
- **多维实用性评估 🧠**：覆盖 7 大核心维度与多个细分任务，全面检验模型在真实交互中的综合表现。
- **多语言多方言数据覆盖 🌏**：以中文普通话为主，支持英文问答，以及粤语(cantonese)、河南话(henan_dialect)、东北话(northeastern_mandarin)、上海话(shanghainese)、四川话(sichuanese)等多种中国方言，数据持续扩展中。
- **模块化评测流程 🔧**：推理与评估解耦，支持复用已有测试结果，自定义模型、任务与数据集，兼容多种开发需求。
- **本地LLM友好 ⚡**：支持 VLLM 多卡部署，可将本地大模型作为评估器，无需依赖远程API。
- **可复现与客观性 ✅**：统一采用贪心解码策略，优先选择客观指标进行评估，尽量避免LLM打分造成的偏差与随机性。


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
| ```eval_task``` | 测评方式  | Metric | 模型输出模态 |
|:--------------:|:----------:|:----------:|:-------:|
| ```basic```  | 字符串匹配 | ACC | 文本 |
| ```choice``` | 正则匹配 | ACC | 文本 |
| ```emotion_understand``` | LLM打分 | Score | 文本 |
| ```aed_instruct```  | LLM打分 | Score | 文本 |
| ```dialect_follow``` | LLM打分 | Score | 文本 |
| ```human_acceptance``` | LLM打分 | Score | 文本 |
| ```wer```  | 模型输出的文本与音频ASR后的CER | WER/CER | 音频 |
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
| aed_combine-zh     | 副语言信息     | 音频事件理解                 | 2000 |
| esd-zh     | 副语言信息, 情绪音频生成     | 情绪理解, 情绪回答               | 150 |
| human_acceptance-zh     | 拟人程度     | 回答自然度, 行为对齐              | 300  |
| chitchat-cantonese    | 拟人程度, 方言音频生成     | 方言理解与方言跟随    | 182 |
| chitchat-henan_dialect     | 拟人程度, 方言音频生成     | 方言理解与方言跟随                 | 161 |
| chitchat-northeastern_mandarin | 拟人程度, 方言音频生成     | 方言理解与方言跟随                 | 246 |
| chitchat-shanghainese | 拟人程度, 方言音频生成     | 方言理解与方言跟随                 | 207 |
| chitchat-sichuanese   | 拟人程度, 方言音频生成     | 方言理解与方言跟随                 | 144 |
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

| Model | llamaqa-en (%) | llamaqa-zh (%) | triviaqa-en (%) | triviaqa-zh (%) | webq-en (%) | webq-zh (%) | chinesesimpleqa-zh (%) | chinese_quiz-zh (%) | agieval-zh (%) | ceval-zh (%) |
|:-----:|:----------:|:----------:|:-----------:|:-----------:|:-------:|:-------:|:-----------------:|:-------:|:-------:|:-------:|
| glm-4-voice-9b    | 67.67 | 53.00 | 34.89 |	27.00 |37.00 |	34.62 |	14.47 | 47.09 | **34.47** | **41.24** |
| MiniCPMo2_6-audio | **70.67** |	57.00 |	**46.95** |	30.59 |	**48.50** |	39.42 |	13.68 | 46.25 | 12.80 | 10.36 |
| baichuan_omni_1d5 | 70.33 |	59.00 |	38.59 |	**33.93** |	43.81 |	**40.76** |	**17.65** | 51.21 | 13.69 | 15.96 |
| llama_omni        | 70.33 |	22.33 |	31.90 |	6.33 |	34.31 |	6.91 |	0.49 | 0.12 | 0.00 | 0.00 |
| speechgpt2        | 0.00 |	36.33 |	0.12 |	13.62 |	0.00 |	20.33 |	5.88 | 27.12 | 2.93 | 2.59 |
| freeze_omni       | 66.00 |	57.67 |	37.87 |	23.78 |	41.95 |	35.60 |	14.48 | 49.76 | 1.39 | 1.24 |
| qwen2_5_omni      | 69.67 |	58.67 |	43.13 |	29.03 |	44.32 |	35.19 |	13.42 | **56.30** | 20.78 | 24.25 |
| kimi-audio-7b-instruct | **70.67** |	**65.33** |	45.52 |	32.97 |	43.81 |	39.27 |	17.58 | 53.51 | 12.88 | 15.03 |

### 2. 副语言信息、回答自然度

| Model | aed_combine-zh (%) | esd-zh (%) | human_acceptance-zh (%) |
|:-----:|:----------:|:----------:|:----------:|
| glm-4-voice-9b    | 2.28 | 40.35 | 50.33 |
| MiniCPMo2_6-audio | 20.37 | 44.03 | 45.51 |
| baichuan_omni_1d5 | 3.37 | 15.47 | 41.79 |
| llama_omni        | 0.19 | 7.57 | 14.93 |
| speechgpt2        | 0.52 | 30.80 | 54.85 |
| freeze_omni       | 17.75 | 21.12 | 38.47 |
| qwen2_5_omni      | 32.7 | 44.77 | 62.52 |
| kimi-audio-7b-instruct | 37.11 | 52.45 | 53.52 |

### 3. 方言理解
| Model | chinese_quiz-cantonese (%) | chinese_quiz-henan_dialect (%) | chinese_quiz-northeastern_mandarin (%) | chitchat-shanghainese (%) | chitchat-sichuanese (%) |
|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|
| glm-4-voice-9b    | 0.61 |	9.93 |	37.40 |	3.87 |	13.35 |
| MiniCPMo2_6-audio | 15.17 |	10.46 |	35.77 |	1.85 |	17.80 |
| baichuan_omni_1d5 | 31.56 |	26.42 |	43.74 |	14.58 |	37.54 |
| llama_omni        | 0.00 |	0.00 |	0.00 |	0.00 |	0.00 |
| speechgpt2        | 0.30 |	3.37 |	15.77 |	1.29 |	4.01 |
| freeze_omni       | 1.06 |	13.83 |	38.05 |	2.95 |	24.78 |
| qwen2_5_omni      | 48.10 |	34.75 |	46.99 |	24.72 |	44.81 |
| kimi-audio-7b-instruct | 17.91 |	24.65 |	42.76 |	4.24 |	35.91 |

### 4. 方言跟随
| Model | chitchat-cantonese (%) | chitchat-henan_dialect (%) | chitchat-northeastern_mandarin (%) | chitchat-shanghainese (%) | chitchat-sichuanese (%) |
|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|
| glm-4-voice-9b    | 1.67 |	2.83 |	12.20 |	0.70 |	2.69 |
| MiniCPMo2_6-audio | 8.42 |	9.44 |	21.27 |	2.67 |	10.33 |
| baichuan_omni_1d5 | 6.40 |	7.06 |	11.48 |	2.74 |	8.67 |
| llama_omni        | 0.73 |	0.12 |	0.28 |	0.04 |	0.17 |
| speechgpt2        | 0.70 |	4.40 |	13.11 |	1.08 |	4.00 |
| freeze_omni       | 0.70 |	5.81 |	10.94 |	1.29 |	9.42 |
| qwen2_5_omni      | 15.56 |	18.29 |	29.06 |	8.75 |	21.08 |
| kimi-audio-7b-instruct | 8.46 |	11.63 |	16.26 |	1.64 |	12.61 |

### 5. 垂域知识
| Model | livelihood_policy-zh (%)  | livelihood_policy-cantonese (%) | livelihood_policy-henan_dialect (%) | livelihood_policy-northeastern_mandarin (%) | livelihood_policy-shanghainese (%) | livelihood_policy-sichuanese (%) |
|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| glm-4-voice-9b    | 32.19 |	4.48 |	11.59 |	18.94 |	8.27 |	11.92 |
| MiniCPMo2_6-audio | 32.00 |	15.17 |	10.46 |	35.77 |	1.85 |	17.80 |
| baichuan_omni_1d5 | 37.26 |	17.29 |	17.07 |	17.07 |	12.10 |	15.60 |
| llama_omni        | 0.00 |	0.12 |	0.00 |	0.00 |	0.00 |	0.00 |
| speechgpt2        | 28.49 |	1.99 |	3.69 |	5.84 |	3.33 |	3.90 |
| freeze_omni       | 33.25 |	5.47 |	9.48 |	15.31 |	6.67 |	15.06 |
| qwen2_5_omni      | 26.86 |	17.04 |	13.80 |	15.97 |	12.35 |	14.08 |
| kimi-audio-7b-instruct | 23.98 |	11.69 |	8.64 |	12.78 |	4.44 |	10.29 |

### 6. 声学鲁棒性
节选。详细结果见[Report](#Report)

| Model             | bubble_noise_-5dB (%) | white_noise_-5dB (%) | single_background_speaker_-5dB (%) | multi_background_speakers_-5dB (%) | complex_environments_-5dB (%) | complex_environments_reverb_-5dB (%) | distortion_rate_0.6 (%) | lowpass_filtering_100Hz (%) | packet_loss_rate_50 (%) | reverberation_RT60_3000ms (%) | different_distance_5m (%) |
|:-----------------:|:-----------------:|:----------------:|:------------------------------:|:------------------------------:|:-------------------------:|:--------------------------------:|:-------------------:|:-----------------------:|:-------------------:|:-------------------------:|:---------------------:|
| GLM-4-Voice-9B    | 18.00             | 19.33            | 15.67                          | 8.00                           | 37.00                     | 20.00                            | 49.33               | 47.00                   | 47.67               | 50.33                     | 49.33                 |
| MiniCPM-o 2.6     | 18.33             | 25.33            | 17.33                          | 11.67                          | 40.33                     | 23.33                            | 50.33               | 53.67                   | 50.33               | 53.00                     | 56.00                 |
| baichuan-omni-1.5 | 30.67             | 42.00            | 26.67                          | 22.00                          | 48.33                     | 30.67                            | 59.33               | 58.00                   | 55.33               | 60.33                     | 60.00                 |
| Llama-Omni        | 5.67              | 5.67             | 5.67                           | 2.67                           | 13.00                     | 6.33                             | 19.33               | 19.33                   | 17.00               | 19.33                     | 21.00                 |
| speech-gpt2       | 0.33              | 0.00             | 4.00                           | 0.67                           | 7.00                      | 2.33                             | 19.00               | 15.67                   | 9.67                | 30.33                     | 28.67                 |
| freeze-omni       | 9.00              | 14.00            | 14.67                          | 6.33                           | 27.33                     | 17.00                            | 49.33               | 46.33                   | 41.33               | 55.33                     | 54.67                 |
| qwen2.5-omni      | 27.67             | 39.33            | 25.00                          | 17.00                          | 44.67                     | 30.67                            | 55.67               | 58.33                   | 54.00               | 59.33                     | 59.00                 |
| kimi-audio        | 24.33             | 40.33            | 30.33                          | 16.67                          | 45.67                     | 27.33                            | 62.33               | 63.00                   | 61.00               | 64.00                     | 63.33                 |


### 7. 音频能力
| Model | esd (CER ↓) | esd (DNSMOS ↑) | esd (emotion_response ↑) | chitchat-cantonese (%) | chitchat-henan_dialect (%) | chitchat-northeastern_mandarin (%) | chitchat-shanghainese (%) | chitchat-sichuanese (%) |
|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| glm-4-voice-9b    | 6.58 |	3.46 |	31.66 | 3.30 |	0.00 |	0.00 |	0.00 |	0.00 |
| MiniCPMo2_6-audio | 2.58 |	3.52 |	34.26 | 0.00 |	0.00 |	0.00 |	0.00 |	0.00 |
| baichuan_omni_1d5 | 7.89 |	3.40 |	24.74 | 0.00 |	0.00 |	0.00 |	0.00 |	0.00 |
| llama_omni        | 8.33 |	3.21 |	37.28 | - | - | - | - | - |
| speechgpt2        | 17.27 |	2.46 |	27.48 | 0.00 |	0.00 |	1.22 |	3.86 |	4.17 |
| freeze_omni       | 4.88 |	3.49 |	41.05 | 0.00 |	0.00 |	0.00 |	0.00 |	0.00 |
| qwen2_5_omni      | 1.69 |	3.47 |	52.59 |	0.00 |	0.00 |	0.00 |	0.00 |	0.00 |
| kimi-audio-7b-instruct | 3.84 |	3.38 	45.48 | 0.00 |	0.00 |	0.41 |	0.00 |	0.00 |


## 自定义数据集与模型
框架支持灵活的任务、数据集、模型定义，只需要修改```registry```下对应配置文件；如需新增模型，则要继承<b><code>Model</code></b>类，并实现<b><code>generate_once</code></b>与<b><code>generate_multiturn</code></b>方法。详见[assets/add_model.md](examples/add_model.md)


## 致谢与声明
* 本框架中的部分代码引用、修改自 [UltraEval-Audio](https://github.com/OpenBMB/UltraEval-Audio) 和 [OpenCompass](https://github.com/open-compass/opencompass)
* 各SLM的推理实现基于相应开源项目的演示脚本，我们对其进行了调用上的修改，以便无缝集成到我们的框架中。然而，为了确保所有模型都能执行 *greedy_search* 推理，我们调整了一些模型的代码，例如 ```src_freezeomni/audioLLM.py```