#!/bin/bash

CUDA_DEVICES=(0 1 2 3 4 5 6 7)  # can be modified according to demand

# pre-trained model storage directory
CHECKPOINT_DIR=./checkpoints

CHECKPOINTS=($(ls "$CHECKPOINT_DIR"/*.pt | sort))

# CUDA devices number
NUM_CUDA_DEVICES=${#CUDA_DEVICES[@]} 

# =============== How many checkpoint do you want to run in each batch ===============
BATCH_SIZE=8    # <-- modify here, 8 checkpoint per batch

# ==========================================================

TOTAL_CHECKPOINTS=${#CHECKPOINTS[@]}
TOTAL_BATCHES=$(( (TOTAL_CHECKPOINTS + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "Total checkpoints: $TOTAL_CHECKPOINTS"
echo "Running in $TOTAL_BATCHES batches (each with up to $BATCH_SIZE checkpoints)."

for ((BATCH_INDEX=0; BATCH_INDEX<TOTAL_BATCHES; BATCH_INDEX++)); do
    echo "================== Running Batch $((BATCH_INDEX+1)) / $TOTAL_BATCHES =================="
    
    START=$((BATCH_INDEX * BATCH_SIZE))
    END=$((START + BATCH_SIZE))
    if ((END > TOTAL_CHECKPOINTS)); then
        END=$TOTAL_CHECKPOINTS
    fi

    INDEX=0
    for ((i=START; i<END; i++)); do
        CHECKPOINT=${CHECKPOINTS[$i]}
        CUDA_DEVICE=${CUDA_DEVICES[$((INDEX % NUM_CUDA_DEVICES))]}  # polling to allocate GPU
        echo "  → Launching on GPU $CUDA_DEVICE : $CHECKPOINT"
        
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python experiments/Libero/robot/libero/run_libero_eval.py \
            --model_family cronus \
            --pretrained_checkpoint "$CHECKPOINT" \
            --task_suite_name libero_goal \
            --center_crop True \
            --use_wrist_image False &
        
        sleep 2
        ((INDEX++))
    done

    wait
    echo "================== Batch $((BATCH_INDEX+1)) finished =================="
done

echo "All batches completed."