#!/bin/bash
#SBATCH --job-name=aif-gen-cppo-short-piecewise
#SBATCH --nodes=1                 # Request 2 nodes
#SBATCH --gpus-per-node=h100:4     # Request 4 H100 GPUs per node
#SBATCH --ntasks-per-node=4        # One task per GPU
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=out/%x.%j.out     # Include job name + job ID
#SBATCH --error=out/%x.%j.err      # Include job name + job ID 
#SBATCH --mail-type=ALL
#SBATCH --account=aip-rrabba
#SBATCH --mail-user=shahrad_m@icloud.com  # Update with your email

source .env

dataset_name='aifgen-short-piecewise'

accelerate launch --config_file benchmarks/cppo/accelerate_configs/deepspeed_zero2.yaml \
    benchmarks/cppo/cppo.py \
    --wandb_project $dataset_name \
    --wandb_run_name "Qwen2-0.5B-CPPO-${dataset_name}-multi-gpu" \
    --dataset_name $dataset_name \
    --sft_model_path Qwen/Qwen2-0.5B-Instruct \
    --value_model_path LifelongAlignment/Qwen2-0.5B-Instruct_${dataset_name}_REWARD_0 \
    --reward_model_path LifelongAlignment/Qwen2-0.5B-Instruct_${dataset_name}_REWARD \
    --learning_rate 5.0e-6 \
    --response_length 256 \
    --num_train_epochs 4 \
    --gradient_checkpointing \
    --per_device_train_batch_size 16 \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 300 \
    --save_steps 300 \
    --bf16 \
    --output_dir "$HOME/Qwen2-0.5B-CPPO-${dataset_name}" \
    --no_remove_unused_columns
