## Dataset Information
<a id="Dataset_Information"></a>

| **Dataset**                             | **Samples #** | **Task**                      | **Evaluation Abilities**                                                | **Dimension**                         |
|:-----------------------------------------:|:---------------:|:-------------------------------:|:-------------------------------------------------------------------------:|:---------------------------------------:|
| llamaqa-en                              | 300           | Basic Knowledge               | Commonsence knowledge (EN)                                              | Explicit Semantics                    |
| llamaqa-zh                              | 300           | Basic Knowledge               | Commonsence knowledge (ZH)                                              | Explicit Semantics                    |
| triviaqa-en                             | 837           | Basic Knowledge               | Commonsence knowledge (EN)                                              | Explicit Semantics                    |
| triviaqa-zh                             | 837           | Basic Knowledge               | Commonsence knowledge (ZH)                                              | Explicit Semantics                    |
| webq-en                                 | 1938          | Basic Knowledge               | Commonsence knowledge (EN)                                              | Explicit Semantics                    |
| webq-zh                                 | 1938          | Basic Knowledge               | Commonsence knowledge (ZH)                                              | Explicit Semantics                    |
| chinesesimpleqa-zh                      | 2668          | Basic Knowledge               | Commonsence knowledge (ZH), Chinese cultural and factual knowledge      | Explicit Semantics                    |
| agieval-zh                              | 1227          | Basic Knowledge               | Chinese single-choice, Mixtural task understanding                      | Explicit Semantics                    |
| ceval-zh                                | 965           | Basic Knowledge               | Chinese single-choice, Mixtural task understanding                      | Explicit Semantics                    |
| chinese_quiz-zh                         | 827           | Basic Knowledge               | Commonsence knowledge (zh), Chinese cultural and factual knowledge      | Explicit Semantics                    |
| chinese_quiz-cantonese                  | 659           | Dialect Comprehension         | Dialect understanding, Chinese cultural and factual knowledge (dialect) | Explicit Semantics                    |
| chinese_quiz-henan_dialect              | 564           | Dialect Comprehension         | Dialect understanding, Chinese cultural and factual knowledge (dialect) | Explicit Semantics                    |
| chinese_quiz-northeastern_mandarin      | 615           | Dialect Comprehension         | Dialect understanding, Chinese cultural and factual knowledge (dialect) | Explicit Semantics                    |
| chinese_quiz-shanghainese               | 542           | Dialect Comprehension         | Dialect understanding, Chinese cultural and factual knowledge (dialect) | Explicit Semantics                    |
| chinese_quiz-sichuanese                 | 674           | Dialect Comprehension         | Dialect understanding, Chinese cultural and factual knowledge (dialect) | Explicit Semantics                    |
| multiturn_memory-zh                     | 150           | Context Memory                | Historical information memory                                           | Explicit Semantics                    |
| human_accept-zh                         | 300           | Safety & Values               | Human values                                                            | Explicit Semantics                    |
| human_chitchat-zh                       | 400           | Chitchat                      | Linguistic variation, Informal response                                 | Explicit Semantics                    |
| livelihood_policy-zh                    | 1597          | Domain Knowledge              | Chinese livelihood knowledge                                            | Explicit Semantics                    |
| livelihood_policy-cantonese             | 804           | Domain Knowledge              | Chinese livelihood knowledge (dialect)                                  | Explicit Semantics                    |
| livelihood_policy-henan_dialect         | 949           | Domain Knowledge              | Chinese livelihood knowledge (dialect)                                  | Explicit Semantics                    |
| livelihood_policy-northeastern_mandarin | 908           | Domain Knowledge              | Chinese livelihood knowledge (dialect)                                  | Explicit Semantics                    |
| livelihood_policy-shanghainese          | 810           | Domain Knowledge              | Chinese livelihood knowledge (dialect)                                  | Explicit Semantics                    |
| livelihood_policy-sichuanese            | 923           | Domain Knowledge              | Chinese livelihood knowledge (dialect)                                  | Explicit Semantics                    |
| aed_combine-zh                          | 2000          | Scene                         | Audio event perception                                                  | Paralinguistic and Implicit Semantics |
| esd-zh                                  | 150           | Emotion Perception & Response | Emotion Perception, Emotion Response                                    | Paralinguistic and Implicit Semantics |
| chitchat-cantonese                      | 182           | Dialect Perception & Response | Dialect following ability                                               | Paralinguistic and Implicit Semantics |
| chitchat-henan_dialect                  | 161           | Dialect Perception & Response | Dialect following ability                                               | Paralinguistic and Implicit Semantics |
| chitchat-northeastern_mandarin          | 246           | Dialect Perception & Response | Dialect following ability                                               | Paralinguistic and Implicit Semantics |
| chitchat-shanghainese                   | 207           | Dialect Perception & Response | Dialect following ability                                               | Paralinguistic and Implicit Semantics |
| chitchat-sichuanese                     | 144           | Dialect Perception & Response | Dialect following ability                                               | Paralinguistic and Implicit Semantics |
| para_mix300-zh                          | 300           | NSV Perception & Response     | NSV response, Concern                                                   | Paralinguistic and Implicit Semantics |
| age-zh                                  | 150           | Age Perception & Response     | Age response, Concern                                                   | Paralinguistic and Implicit Semantics |
| noise-zh                                | 19500         | Acoustic Robustness           | Complex acoustic environment                                            | System Abilities                      |



** The sub-dataset composition of ```noize-zh``` is as follows:
| **Dataset**                 | **Samples #** | **Task**            | **Evaluation Abilities** | **Dimension**    |
|:---------------------------:|:-------------:|:-------------------:|:------------------------:|:----------------:|
| babble_*                   | 6*300         | Acoustic Robustness | Babble noise at different SNR            | System Abilities |
| white_*                     | 6*300         | Acoustic Robustness | White noise at different SNR             | System Abilities |
| single_background_speaker_* | 6*300         | Acoustic Robustness | Single background speaker noise at different SNR            | System Abilities |
| multi_background_speakers_* | 6*300         | Acoustic Robustness | Multiple background speaker noise at different SNR            | System Abilities |
| complex_env_*               | 6*300         | Acoustic Robustness | Real-world env noise at different SNR          | System Abilities |
| complex_env_reverb_*        | 6*300         | Acoustic Robustness | Real-world env (reverb) at different SNR     | System Abilities |
| distortion_rate_*           | 6*300         | Acoustic Robustness | Different clipping distortion rates                  | System Abilities |
| lowpass_filter_*            | 8*300         | Acoustic Robustness | Low-pass filtering with different bandwidths                 | System Abilities |
| packet_loss_rate_*          | 5*300         | Acoustic Robustness | Different packet loss rates                    | System Abilities |
| reverb_rt60_*               | 5*300         | Acoustic Robustness | Different reverberation times                   | System Abilities |
