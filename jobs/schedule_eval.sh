#!/bin/bash
#SBATCH --job-name=aif-gen-evaluation
#SBATCH --nodes=1                  # Request 2 nodes
#SBATCH --gpus-per-node=h100:4     # Request 4 H100 GPUs per node
#SBATCH --ntasks-per-node=4        # One task per GPU
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=out/%x.%j.out     # Include job name + job ID
#SBATCH --error=out/%x.%j.err      # Include job name + job ID 
#SBATCH --mail-type=ALL
#SBATCH --account=aip-rrabba
#SBATCH --mail-user=shahrad_m@icloud.com  # Update with your email
source .env

dataset_name=${1:-'aifgen-lipschitz'}
dataset_index=${2:-'0'}
checkpoint=${3:-'300'}

#DPO on CPPO dataset - DIFFERENT FILE
# accelerate launch --config_file benchmarks/dpo/accelerate_configs/deepspeed_zero3.yaml \
#     benchmarks/parallel_eval_checkpoints.py \
#     --checkpoint_dir "/scratch/s/shahradm/${dataset_name}/Qwen2-0.5B-DPO-/dataset-${dataset_index}/checkpoint-${checkpoint}" \
#     --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
#     --wandb_run_name "test_eval_Qwen2-0.5B-DPO-rl256-v5-dataset-${dataset_index}-checkpoint-${checkpoint}" \
#     --reward_model_path "/lustre/orion/bif151/scratch/ivan.anokhin/AIF-Gen/${dataset_name}/Qwen2-0.5B-Reward-8gpus/Qwen2-0.5B-Instruct_${dataset_name}_REWARD" \
#     --wandb_project eval_${dataset_name} \
#     --learning_rate 0. \
#     --response_length 256 \
#     --dataset_name $dataset_name \
#     --per_device_eval_batch_size 16 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --bf16 \
#     --output_dir "/lustre/orion/bif151/scratch/ivan.anokhin/AIF-Gen/${dataset_name}/eval_Qwen2-0.5B-DPO-rl256-v5-8gpus-s${dataset_index}" \
#     --no_remove_unused_columns


#PPO - not on CPPO dataset
accelerate launch --config_file benchmarks/dpo/accelerate_configs/deepspeed_zero3.yaml \
   benchmarks/parallel_eval_checkpoints.py \
   --checkpoint_dir "/scratch/s/shahradm/Qwen2-0.5B-PPO-${dataset_name}/Qwen2-0.5B-Instruct_${dataset_name}_PPO_${dataset_index}/checkpoint-${checkpoint}" \
   --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
   --wandb_run_name "test_eval_Qwen2-0.5B-PPO-rl256-v1-dataset-${dataset_index}-checkpoint-${checkpoint}" \
   --reward_model_path "LifelongAlignment/Qwen2-0.5B-Instruct_${dataset_name}_REWARD" \
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

# PPO - on CPPO dataset - DIFFERENT FILE

# CPPO - not on CPPO dataset
accelerate launch --config_file benchmarks/dpo/accelerate_configs/deepspeed_zero3.yaml \
   benchmarks/parallel_eval_checkpoints.py \
   --checkpoint_dir "/home/s/shahradm/links/projects/aip-rrabba/shared/aifgen_experiments/Qwen2-0.5B-CPPO-${dataset_name}/Qwen2-0.5B-Instruct_${dataset_name}_CPPO_${dataset_index}/checkpoint-${checkpoint}" \
   --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
   --wandb_run_name "test_eval_Qwen2-0.5B-CPPO-rl256-v1-dataset-${dataset_index}-checkpoint-${checkpoint}" \
   --reward_model_path "LifelongAlignment/Qwen2-0.5B-Instruct_${dataset_name}_REWARD" \
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

# CPPO - on CPPO dataset - DIFFERENT FILE