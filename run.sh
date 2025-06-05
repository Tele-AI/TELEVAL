#!/bin/bash
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

max_memory=400
save_dir="res/test"

stage=1
stop_stage=2
eval_bsz=1
save_pred_audio=False

aqa_tasks="aqa-llamaqa-en,aqa-llamaqa-zh,aqa-triviaqa-en,aqa-triviaqa-zh,aqa-webq-en,aqa-webq-zh,aqa-chinesesimpleqa-zh,aqa-chinese_quiz-zh"
aqa_dialect_tasks="aqa-chinese_quiz-sichuanese,aqa-chinese_quiz-shanghainese,aqa-chinese_quiz-northeastern_mandarin,aqa-chinese_quiz-henan_dialect,aqa-chinese_quiz-cantonese"
chitchat_dialect_tasks="follow-chitchat-sichuanese,follow-chitchat-shanghainese,follow-chitchat-northeastern_mandarin,follow-chitchat-henan_dialect,follow-chitchat-cantonese"

down_tasks="aqa-livelihood_policy-zh,aqa-livelihood_policy-sichuanese,aqa-livelihood_policy-shanghainese,aqa-livelihood_policy-northeastern_mandarin,aqa-livelihood_policy-henan_dialect,aqa-livelihood_policy-cantonese"
noise_tasks="aqa-bubble_noise-zh,aqa-white_noise-zh,aqa-distortion-zh,aqa-single_background_speaker-zh,aqa-multi_background_speakers-zh,aqa-lowpass_filtering-zh,aqa-packet_loss-zh,aqa-reverberation_RT60-zh,aqa-complex_environments-zh,aqa-complex_environments_reverb-zh,aqa-different_distance-zh"
choice_tasks="choice-agieval-zh,choice-ceval-zh"
para_tasks="emotion-esd,aed-audio-instruct"
human_tasks="acceptance-human-zh"

declare -A model_tasks
model_tasks=(
    ["MiniCPMo2_6-audio"]="$aqa_tasks,$para_tasks"
    ["baichuan_omni_1d5"]="$aqa_tasks,$para_tasks"
    ["llama_omni"]="$aqa_tasks,$para_tasks"
    ["speechgpt2"]="$aqa_tasks,$para_tasks"
    ["freeze_omni"]="$aqa_tasks,$para_tasks"
    ["glm-4-voice-9b"]="$aqa_tasks,$para_tasks"
    ["kimi-audio-7b-instruct"]="$aqa_tasks,$para_tasks"
    ["qwen2_5_omni"]="$aqa_tasks,$para_tasks"
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
            sleep 60  # Increase sleep time appropriately according to the speed of loading the model
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