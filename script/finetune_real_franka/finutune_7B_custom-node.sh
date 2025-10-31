#!/bin/bash

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((RANDOM % 101 + 20000))

# to fix: libcudnn_ops_infer.so.8 with link time referencesymbol 
export LD_LIBRARY_PATH=~/anaconda3/envs/cronusvla/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=~/anaconda3/envs/cronusvla/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_ops_infer.so.8

export hf_token=your_token

torchrun --standalone --nnodes 1 --nproc-per-node 8 training/train.py \
  --pretrained_checkpoint path/to/cronusvla_7B_bridge_rt_1/checkpoints/step-055000-epoch-04-loss=0.0286.pt \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix custom \
  --vla.expected_world_size 8 \
  --vla.global_batch_size 64 \
  --vla.per_device_batch_size 8 \
  --vla.learning_rate 2e-5 \
  --vla.max_steps 40000 \
  --data_root_dir path/to/custom_data \
  --run_root_dir ./outputs/cronusvla_7B_fintune_custom \
  --run_id cronusvla_7B_fintune_custom \
  --image_aug True \
  --wandb_project cronusvla \
  --wandb_entity your_name \
  --save_interval 250 \
  --repeated_diffusion_steps 4 \
  --future_action_window_size 15 \
  --past_action_window_size 6 \
  --action_model_type DiT-B \
  --hf_token hf_token \
  --is_resume False