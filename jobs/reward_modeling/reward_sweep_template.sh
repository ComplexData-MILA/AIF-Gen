#!/bin/bash
#SBATCH --job-name=aif-gen-reward-long-piecewise-8B
#SBATCH --nodes=1                  # Request 2 nodes
#SBATCH --gpus-per-node=h100:4     # Request 4 H100 GPUs per node
#SBATCH --ntasks-per-node=4        # One task per GPU
#SBATCH --cpus-per-task=6
#SBATCH --mem=0G
#SBATCH --time=1:00:00
#SBATCH --output=out/%x.%j.out     # Include job name + job ID
#SBATCH --error=out/%x.%j.err      # Include job name + job ID 
#SBATCH --mail-type=ALL
#SBATCH --account=aip-rrabba
#SBATCH --mail-user=shahrad_m@icloud.com  # Update with your email
source .env
# Set PyTorch to use more aggressive memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

dataset_name='aifgen-long-piecewise'

accelerate launch --config_file benchmarks/cppo/accelerate_configs/deepspeed_zero2.yaml \
    benchmarks/reward_modeling.py \
    --model_name_or_path Qwen/Qwen3-8B-Base \
    --dataset_name $dataset_name \
    --dataset_index 0 \
    --output_dir "$SCRATCH/Qwen3-8B-REWARD-${dataset_name}" \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --bf16 \
    --learning_rate 5.0e-6 \
    --logging_steps 30 \
    --eval_strategy steps \
    --eval_steps 70 \
    --max_length 2048 \
    --wandb_project $dataset_name \
    --wandb_run_name "Qwen3-8B-REWARD-${dataset_name}-multi-gpu"