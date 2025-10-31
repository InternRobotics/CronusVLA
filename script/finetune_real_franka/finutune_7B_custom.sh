#!/bin/bash
#SBATCH --job-name=CronusVLA        # name
#SBATCH -p your_cluster_name
#SBATCH -N 2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=128          # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --output=trash/%x-%j.out           # output file name
#SBATCH -e trash/%x-%j.err

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((RANDOM % 101 + 20000))

# NCCL configuration (must be set correctly on your cluster)
export NCCL_SOCKET_IFNAME=bond0 
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5
export NCCL_TIMEOUT=3600  # longer timeout for stable training

# to fix: libcudnn_ops_infer.so.8 with link time referencesymbol 
export LD_LIBRARY_PATH=~/anaconda3/envs/cronusvla/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=~/anaconda3/envs/cronusvla/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_ops_infer.so.8

export hf_token=your_token

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
 training/train.py \
  --pretrained_checkpoint path/to/cronusvla_modified_ckpt_post-train/step4/checkpoints/reshape_embedding_step4.pt \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix custom \
  --vla.expected_world_size 16 \
  --vla.global_batch_size 256 \
  --vla.per_device_batch_size 16 \
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
  --future_action_window_size 7 \
  --past_action_window_size 3 \
  --action_model_type DiT-B \
  --hf_token hf_token \
  --is_resume False'