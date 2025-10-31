#!/bin/bash

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((RANDOM % 101 + 20000))

# to fix: libcudnn_ops_infer.so.8 with link time referencesymbol 
export LD_LIBRARY_PATH=~/anaconda3/envs/cronusvla/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=~/anaconda3/envs/cronusvla/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_ops_infer.so.8

export hf_token=your_token

# Note: This method cannot execute an entire training process. 
# It can only be used before the backward pass. 
# You can use `from IPython import embed; embed()` before a specific line of code for debugging. 
# Only the forward pass can be executed; the backward pass depends on multi-GPU FSDP (>= 1 GPU) which can not be debugged directly.
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 training/train_debug.py \
  --pretrained_checkpoint path/to/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix bridge_rt_1 \
  --vla.expected_world_size 1 \
  --vla.global_batch_size 8 \
  --vla.per_device_batch_size 8 \
  --vla.learning_rate 2e-5 \
  --data_root_dir path/to/bridge_rt_1 \
  --run_root_dir ./outputs/cronusvla_7B_debug \
  --run_id cronusvla_7B_debug \
  --image_aug True \
  --wandb_project cronusvla \
  --wandb_entity your_name \
  --save_interval 2500 \
  --repeated_diffusion_steps 4 \
  --future_action_window_size 15 \
  --past_action_window_size 6 \
  --action_model_type DiT-B \
  --hf_token hf_token \
  --is_resume False