#!/bin/bash

# Server GPU Information
CUDA_IDS=0,1,2,3,4,5,6,7
length=${#CUDA_IDS}
N_GPU=$(((length+1)/2))
N_NODES=1

export PYTHONUNBUFFERED=1
export RAY_MEMORY_USAGE_THRESHOLD=0.98

MODEL_PATH=./models/Qwen3-VL-2B-Thinking
TEACHER_MODEL_PATH=./models/Qwen3-VL-4B-Thinking
# MODEL_PATH=./models/Qwen2.5-VL-3B-Instruct
# TEACHER_MODEL_PATH=./models/Qwen2.5-VL-7B-Instruct


TOTAL_EPOCHES=1
ROLLOUT_N=4
MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE=4
GLOBAL_BATCH_SIZE=$(($MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE * $N_GPU * $N_NODES * 4 / $ROLLOUT_N))
VAL_BATCH_SIZE=512
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=8192
# GLOBAL_BATCH_SIZE x ROLLOUT_N / N_GPU / MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE = Updating Policy and Computing Log Prob Iteration
# Checking it is divisible, and if it's true, then showing updating policy iteration number
if [ $((GLOBAL_BATCH_SIZE * ROLLOUT_N % (N_NODES * N_GPU * MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE))) -ne 0 ]; then
    echo "Error: GLOBAL_BATCH_SIZE x ROLLOUT_N must be divisible by N_NODES x N_GPU x MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE"
    exit 1
else
    UPDATING_POLICY_ITER=$((GLOBAL_BATCH_SIZE * ROLLOUT_N / (N_NODES * N_GPU * MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE)))
    echo "Updating Policy and Computing Log Prob Iteration: ${UPDATING_POLICY_ITER}"
fi

# PROEJCT_NAME=Extracting Second name after slash / in Model Path and Teacher Model Path (ex: Qwen3-VL-2B-Thinking-Qwen3-VL-4B-Thinking)
PROEJCT_NAME="$(basename ${MODEL_PATH})-$(basename ${TEACHER_MODEL_PATH})"
EXP_NAME="batch${GLOBAL_BATCH_SIZE}_rollout${ROLLOUT_N}_epoch${TOTAL_EPOCHES}_node${N_NODES}"

CONFIG_FILE="examples/config.yaml"
TRAIN_FILE="PAPOGalaxy/PAPO_ViRL39K_train"
VAL_FILE="PAPOGalaxy/PAPO_MMK12_test"

FORMAT_PROMPT="./examples/format_prompt/math.jinja"
REWARD_FUNCTION="./examples/reward_function/math.py:compute_score"

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 -m verl.trainer.main \
    config=${CONFIG_FILE} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.format_prompt=${FORMAT_PROMPT} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.teacher.model.model_path=${TEACHER_MODEL_PATH} \
    worker.teacher.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=${ROLLOUT_N} \
    worker.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    data.rollout_batch_size=${GLOBAL_BATCH_SIZE} \
    trainer.project_name=${PROEJCT_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.nnodes=${N_NODES} \
    trainer.total_epochs=${TOTAL_EPOCHES} \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    worker.actor.micro_batch_size_per_device_for_update=${MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE} \
    worker.actor.micro_batch_size_per_device_for_experience=${MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE} \
    worker.teacher.micro_batch_size_per_device_for_update=${MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE} \
    worker.teacher.micro_batch_size_per_device_for_experience=${MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=false \
    worker.actor.dynamic_batching=false \
    worker.teacher.dynamic_batching=false \
    trainer.debug=false
