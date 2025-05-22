#!/bin/bash
#SBATCH --job-name=aif-gen-evaluation
#SBATCH --nodes=1                  # Request 2 nodes
#SBATCH --gpus-per-node=h100:4     # Request 4 H100 GPUs per node
#SBATCH --ntasks-per-node=4        # One task per GPU
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=out/%x.%j.out     # Include job name + job ID
#SBATCH --error=out/%x.%j.err      # Include job name + job ID 
#SBATCH --mail-type=ALL
#SBATCH --account=aip-rrabba
#SBATCH --mail-user=shahrad_m@icloud.com  # Update with your email
source .env

dataset_name=${1:-'CPPO-RL'}
dataset_index=${2:-'0'}
checkpoint=${3:-'300'}

#DPO
accelerate launch --config_file benchmarks/dpo/accelerate_configs/deepspeed_zero3.yaml \
    benchmarks/parallel_eval_checkpoints.py \
    --checkpoint_dir "/scratch/s/shahradm/projects/Qwen2-0.5B-DPO-CPPO-RL/dataset-${dataset_index}/checkpoint-${checkpoint}" \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --wandb_run_name "test_eval_Qwen2-0.5B-DPO-rl256-v5-CPPO-${dataset_index}-checkpoint-${checkpoint}" \
    --reward_model_path "LifelongAlignment/Qwen2.5-0.5B-Instruct_CPPO_REWARD" \
    --wandb_project eval_${dataset_name}_post_may_19 \
    --learning_rate 0. \
    --response_length 256 \
    --dataset_name $dataset_name \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --bf16 \
    --output_dir "/scratch/s/shahradm/${dataset_name}/eval_Qwen2-0.5B-DPO-8gpus-rl256-v1-s${dataset_index}" \
    --no_remove_unused_columns


#PPO
accelerate launch --config_file benchmarks/dpo/accelerate_configs/deepspeed_zero3.yaml \
   benchmarks/parallel_eval_checkpoints.py \
   --checkpoint_dir "/home/s/shahradm/Qwen2-0.5B-PPO-CPPO-RL/Qwen2-0.5B-Instruct_CPPO-RL_PPO_${dataset_index}/checkpoint-${checkpoint}" \
   --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
   --wandb_run_name "test_eval_Qwen2-0.5B-PPO-rl256-v5-CPPO-${dataset_index}-checkpoint-${checkpoint}" \
   --reward_model_path "LifelongAlignment/Qwen2.5-0.5B-Instruct_CPPO_REWARD" \
   --wandb_project eval_${dataset_name}_post_may_19 \
   --learning_rate 0. \
   --response_length 256 \
   --dataset_name $dataset_name \
   --per_device_eval_batch_size 32 \
   --per_device_train_batch_size 1 \
   --gradient_accumulation_steps 1 \
   --bf16 \
   --output_dir "/scratch/s/shahradm/${dataset_name}/eval_Qwen2-0.5B-PPO-8gpus-rl256-v1-s${dataset_index}" \
   --no_remove_unused_columns


# CPPO
accelerate launch --config_file benchmarks/dpo/accelerate_configs/deepspeed_zero3.yaml \
   benchmarks/parallel_eval_checkpoints.py \
   --checkpoint_dir "/home/s/shahradm/Qwen2-0.5B-CPPO-CPPO-RL/Qwen2-0.5B-Instruct_CPPO-RL_CPPO_${dataset_index}/checkpoint-${checkpoint}" \
   --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
   --wandb_run_name "test_eval_Qwen2-0.5B-CPPO-rl256-v5-CPPO-${dataset_index}-checkpoint-${checkpoint}" \
   --reward_model_path "LifelongAlignment/Qwen2.5-0.5B-Instruct_CPPO_REWARD" \
   --wandb_project eval_${dataset_name}_post_may_19 \
   --learning_rate 0. \
   --response_length 256 \
   --dataset_name $dataset_name \
   --per_device_eval_batch_size 32 \
   --per_device_train_batch_size 1 \
   --gradient_accumulation_steps 1 \
   --bf16 \
   --output_dir "/scratch/s/shahradm/${dataset_name}/eval_Qwen2-0.5B-CPPO-8gpus-rl256-v1-s${dataset_index}" \
   --no_remove_unused_columns
