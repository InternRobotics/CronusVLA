#!/bin/bash

CUDA_DEVICES=(0 1 2 3 4 5 6 7)  # can be modified according to demand

# pre-trained model storage directory
CHECKPOINT_DIR=./checkpoints

CHECKPOINTS=($(ls "$CHECKPOINT_DIR"/*.pt | sort))

# CUDA devices number
NUM_CUDA_DEVICES=${#CUDA_DEVICES[@]}
INDEX=0

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CUDA_DEVICE=${CUDA_DEVICES[$((INDEX % NUM_CUDA_DEVICES))]}  # polling to allocate GPU
    echo "Running on GPU $CUDA_DEVICE with checkpoint $CHECKPOINT"
    
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python experiments/Libero/robot/libero/run_libero_eval.py \
        --model_family cronus \
        --pretrained_checkpoint "$CHECKPOINT" \
        --task_suite_name libero_10 \
        --center_crop True \
        --use_wrist_image True &
    
    sleep 2
    ((INDEX++))
done

wait