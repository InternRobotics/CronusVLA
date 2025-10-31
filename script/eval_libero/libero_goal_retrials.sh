#!/bin/bash

CUDA_DEVICES=(0 1 2 3)  # can be modified according to demand

# For pre-trained model checkpoints, you can set the same checkpoint N times to test with different random seeds.
CHECKPOINTS=(
    ./checkpoints_3/step-027000-epoch-133-loss=0.0193.pt
    ./checkpoints_3/step-027000-epoch-133-loss=0.0193.pt
    ./checkpoints_3/step-027000-epoch-133-loss=0.0193.pt
    ./checkpoints_3/step-027000-epoch-133-loss=0.0193.pt
)

# CUDA devices number
NUM_CUDA_DEVICES=${#CUDA_DEVICES[@]}
INDEX=0

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CUDA_DEVICE=${CUDA_DEVICES[$((INDEX % NUM_CUDA_DEVICES))]}  # polling to allocate GPU
    SEED=$((42 + CUDA_DEVICE))
    echo "Running on GPU $CUDA_DEVICE with checkpoint $CHECKPOINT"
    
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python experiments/Libero/robot/libero/run_libero_eval.py \
        --model_family cronus \
        --pretrained_checkpoint "$CHECKPOINT" \
        --task_suite_name libero_goal \
        --center_crop True \
        --use_wrist_image False \
        --seed $SEED &
    
    sleep 300
    ((INDEX++))
done

wait