#!/bin/bash
#SBATCH --job-name=aif-gen-dpo-ewc-long_piecewise
#SBATCH --nodes=1                 # Request 2 nodes
#SBATCH --gpus-per-node=h100:4     # Request 4 H100 GPUs per node
#SBATCH --ntasks-per-node=4        # One task per GPU
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=out/%x.%j.out     # Include job name + job ID
#SBATCH --error=out/%x.%j.err      # Include job name + job ID
#SBATCH --mail-type=ALL
#SBATCH --account=aip-rrabba
#SBATCH --mail-user=shahrad_m@icloud.com  # Update with your email

source .env

dataset_name='aifgen-long-piecewise'

accelerate launch --config_file benchmarks/dpo/accelerate_configs/deepspeed_zero3.yaml \
    benchmarks/dpo_ewc/dpo_EWC_continual.py \
    --dataset_name benchmarks/continual_data_debug.json  \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --reward_model_path LifelongAlignment/Qwen2-0.5B-Instruct_${dataset_name}_REWARD \
    --learning_rate 1.0e-6 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --gradient_checkpointing \
    --logging_steps 20 \
    --eval_strategy steps \
    --response_length 256 \
    --eval_steps 50000 \
    --save_steps 300 \
    --bf16 \
    --output_dir "/home/s/shahradm/links/projects/aip-rrabba/shared/aifgen_experiments/Qwen2-0.5B-DPO-EWC-${dataset_name}" \
    --no_remove_unused_columns \
    --wandb_project "$dataset_name-post-May-19"   \
    --wandb_run_name "Qwen2-0.5B-DPO-EWC-${dataset_name}-multi-gpu-debug"
