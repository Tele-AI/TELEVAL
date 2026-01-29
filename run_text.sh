#!/bin/bash
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1

max_memory=400
save_dir="res/test"

stage=1
stop_stage=1
eval_bsz=1

text_qa_tasks="text-llamaqa-en,text-llamaqa-zh,text-triviaqa-en,text-triviaqa-zh,text-webq-en,text-webq-zh,text-chinesesimpleqa-zh"
text_dialect_tasks="text-sichuanese,text-shanghainese,text-northeastern_mandarin,text-henan_dialect,text-cantonese"
text_chitchat_dialect_tasks="text-chitchat-sichuanese,text-chitchat-shanghainese,text-chitchat-northeastern_mandarin,text-chitchat-henan_dialect,text-chitchat-cantonese"

text_down_tasks="text-chinese_quiz-zh,text-livelihood_policy-zh"
text_down_dialect_tasks="text-livelihood_policy-sichuanese,text-livelihood_policy-shanghainese,text-livelihood_policy-northeastern_mandarin,text-livelihood_policy-henan_dialect,text-livelihood_policy-cantonese"

text_emo_tasks="text-emo"


declare -A model_tasks
model_tasks=(
    ["qwen3-8b-instruct"]="$text_down_tasks,$text_dialect_tasks,$text_down_tasks"
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
                --save_dir $save_dir \
                --model $model \
                --task $task &
            sleep 40  # Increase sleep time appropriately according to the speed of loading the model
        done
    done
    wait
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    for model in "${!model_tasks[@]}"; do
        IFS=',' read -r -a values <<< "${model_tasks[$model]}"
        # read -a values <<< "${model_tasks[$model]}"
        for task in "${values[@]}"; do
            python main.py \
                --mode "eval" \
                --save_dir $save_dir \
                --model $model \
                --bsz $eval_bsz \
                --task $task
        done
    done
    wait
    python tools/save_csv.py --root_dir $save_dir
fi