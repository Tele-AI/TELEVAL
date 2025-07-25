## Supported Datasets and Tasks
<a id="Supported_Datasets_Tasks"></a>

### Evaluation Tasks and Corresponding Datasets

| ```infer_task```   | ```dataset``` | ```eval_task``` |
|:------------------:|:-------------:|:--------------:|
| aqa-llamaqa-en | ```llamaqa-en```      | ```basic```      |
| aqa-triviaqa-en | ```triviaqa-en```    | ```basic```      |
| aqa-webq-en | ```webq-en```            | ```basic```      |
| aqa-llamaqa-zh | ```llamaqa-zh```      | ```basic```      |
| aqa-triviaqa-zh | ```triviaqa-zh```    | ```basic```      |
| aqa-webq-zh | ```webq-zh```            | ```basic```      |
| aqa-chinesesimpleqa-zh | ```chinesesimpleqa-zh``` | ```basic```      |
| aqa-chinese_quiz-zh | ```chinese_quiz-zh```    | ```basic```        |
| choice-agieval-zh | ```agieval-zh```       | ```choice```        |
| choice-ceval-zh | ```ceval-zh```           | ```choice```        |
| aqa-chinese_quiz-cantonese | ```chinese_quiz-cantonese```    | ```basic```        |
| aqa-chinese_quiz-henan_dialect | ```chinese_quiz-henan_dialect```     | ```basic```        |
| aqa-chinese_quiz-northeastern_mandarin | ```chinese_quiz-northeastern_mandarin``` | ```basic```        |
| aqa-chinese_quiz-shanghainese | ```chinese_quiz-shanghainese``` | ```basic```        |
| aqa-chinese_quiz-sichuanese | ```chinese_quiz-sichuanese```   | ```basic```        |
| acceptance-human-zh | ```human_accept-zh```     | ```human_acceptance```        |
| chitchat-human-zh | ```human_chitchat-zh```     | ```human_likeness```        |
| multiturn-memory-zh | ```multiturn_memory-zh```     | ```basic```        |
| aed-audio-instruct | ```aed_combine-zh```     | ```aed_instruct```        |
| emotion-esd | ```esd-zh```     | ```emotion_understand, emotion_response```        |
| aqa-para_mix300-zh | ```para_mix300-zh```     | ```basic, para_care```        |
| care-age-zh | ```age-zh```     | ```age_care```        |
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
| aqa-babble_noise-zh | ```babble_noise_{-5dB, 0dB, 5dB, 10dB, 15dB, 20dB}```   | ```basic``` |
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

### Available ```eval_task```
<a id="available_eval_task"></a>

| ```eval_task``` | Evaluation Method  | Metrics | SLM Output Modalities |
|:--------------:|:----------:|:----------:|:-------:|
| ```basic```  | string match | ACC | Text |
| ```choice``` | regex match | ACC | Text |
| ```emotion_understand``` | LLM-as-judge | Score | Text |
| ```aed_instruct```  | LLM-as-judge | Score | Text |
| ```multiturn_fluency```  | LLM-as-judge | Score | Text |
| ```dialect_follow``` | LLM-as-judge | Score | Text |
| ```human_acceptance``` | LLM-as-judge | Score | Text |
| ```human_likeness``` | LLM-as-judge | Score | Text |
| ```para_care``` | LLM-as-judge | Score | Text |
| ```age_care``` | LLM-as-judge | Score | Text |
| ```wer```  | WER of Model Outputs (Text and ASR-transcribed Audio) | WER/CER | Audio |
| ```dnsmos``` | DNSMOS |  Score | Audio |
| ```emotion_response``` | Emo2Vec Scores Based on Human Labels | Score | Audio |
| ```dialect_classify``` | Dialect Classify Model | Score | Audio |
* The matching algorithms used in the framework employ a relatively relaxed matching strategy but may still miss some cases and cannot cover all output scenarios.
* **For audio testing, set** ```save_pred_audio = True``` **in the** ```infer_task``` **configuration.**