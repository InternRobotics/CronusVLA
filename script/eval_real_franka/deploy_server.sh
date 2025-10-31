# for model depoly
cd CronusVLA
conda activate cronusvla
CUDA_VISIBLE_DEVICES=0 python ./experiments/real_franka/super_deploy.py --saved_model_path path/to/checkpoints/finetuned_model.pt --unnorm_key fractal20220817_data --port 10011