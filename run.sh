#!/bin/bash
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1

max_memory=400
save_dir="res/test"

stage=1
stop_stage=2
eval_bsz=1
save_pred_audio=False

aqa_tasks="aqa-llamaqa-en,aqa-llamaqa-zh,aqa-triviaqa-en,aqa-triviaqa-zh,aqa-webq-en,aqa-webq-zh,aqa-chinesesimpleqa-zh,aqa-chinese_quiz-zh"
choice_tasks="choice-agieval-zh,choice-ceval-zh"
aqa_dialect_tasks="aqa-chinese_quiz-sichuanese,aqa-chinese_quiz-shanghainese,aqa-chinese_quiz-northeastern_mandarin,aqa-chinese_quiz-henan_dialect,aqa-chinese_quiz-cantonese"
chitchat_dialect_tasks="follow-chitchat-sichuanese,follow-chitchat-shanghainese,follow-chitchat-northeastern_mandarin,follow-chitchat-henan_dialect,follow-chitchat-cantonese"

down_tasks="aqa-livelihood_policy-zh,aqa-livelihood_policy-sichuanese,aqa-livelihood_policy-shanghainese,aqa-livelihood_policy-northeastern_mandarin,aqa-livelihood_policy-henan_dialect,aqa-livelihood_policy-cantonese"
noise_tasks="aqa-babble_noise-zh,aqa-white_noise-zh,aqa-distortion-zh,aqa-single_background_speaker-zh,aqa-multi_background_speakers-zh,aqa-lowpass_filtering-zh,aqa-packet_loss-zh,aqa-reverberation_RT60-zh,aqa-complex_environments-zh,aqa-complex_environments_reverb-zh,aqa-different_distance-zh"
multiturn_tasks="multiturn-memory-zh"
para_tasks="aqa-para_mix300-zh"
llm_judge_tasks="emotion-esd,aed-audio-instruct,acceptance-human-zh,chitchat-human-zh,care-age-zh"

declare -A model_tasks
model_tasks=(
    ["MiniCPMo2_6-audio"]="$aqa_tasks,$aqa_dialect_tasks"
    ["baichuan_omni_1d5"]="$aqa_tasks,$aqa_dialect_tasks"
    ["llama_omni"]="$aqa_tasks,$aqa_dialect_tasks"
    ["speechgpt2"]="$aqa_tasks,$aqa_dialect_tasks"
    ["freeze_omni"]="$aqa_tasks,$para_taaqa_dialect_taskssks"
    ["glm-4-voice-9b"]="$aqa_tasks,$aqa_dialect_tasks"
    ["kimi-audio-7b-instruct"]="$aqa_tasks,$aqa_dialect_tasks"
    ["qwen2_5_omni"]="$aqa_tasks,$aqa_dialect_tasks"
)

gpu_list=($(echo $CUDA_VISIBLE_DEVICES | tr ',' ' '))
gpu_counts=${#gpu_list[@]}

get_free_gpu() {
    while true; do
        for gpu in "${gpu_list[@]}"; do
            used_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "NR==$((gpu+1))")
            if [[ "$used_mem" -lt "$max_memory" ]]; then
                echo "$gpu"
                return
            fi
        done
        sleep 30
    done
}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for model in "${!model_tasks[@]}"; do
        IFS=',' read -r -a values <<< "${model_tasks[$model]}"
        for task in "${values[@]}"; do
            gpu=$(get_free_gpu)
            echo "***********************************************"
            echo "processing model: $model using task: $task on GPU: $gpu"
            echo "***********************************************"
            CUDA_VISIBLE_DEVICES=$gpu python main.py \
                --mode "infer" \
                --task $task \
                --save_dir $save_dir \
                --save_pred_audio $save_pred_audio \
                --model $model &
            sleep 40  # Increase sleep time appropriately according to the speed of loading the model
        done
    done
    wait
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    for model in "${!model_tasks[@]}"; do
        IFS=',' read -r -a values <<< "${model_tasks[$model]}"
        for task in "${values[@]}"; do
            python main.py \
                --mode "eval" \
                --save_dir $save_dir \
                --save_pred_audio $save_pred_audio \
                --model $model \
                --bsz $eval_bsz \
                --task $task
        done
    done
    wait
    python tools/save_csv.py --root_dir $save_dir
fi