# Adaptation of TRL for Continual Learning

This repository adapts TRL for continual learning. The commands below use a consistent set of parameters that you’ve confirmed to work. You can use any of the entrypoints (uv run, accelerate launch, wandb) with the commands listed below.

### Sync additional dependencies

```sh
uv sync --group benchmarks
```

## Reward Modeling

### Full Training

```sh
uv run benchmarks/reward_modeling.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name preference_axes.json \
    --dataset_index 0 \
    --output_dir Qwen2-0.5B-Reward \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-5 \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 5 \
    --max_length 2048 \
    --wandb_entity aifgen
```

### With LoRA and Push to Hub

```sh
uv run benchmarks/reward_modeling.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_index 0 \
    --dataset_name benchmarks/continual_data_debug.json \
    --output_dir "$SCRATCH/Qwen2-0.5B-Reward-LoRA" \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-4 \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 30 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --push_to_hub True \
    --wandb_project Qwen2-0.5B-Reward-LoRA_test \
    --wandb_run_name dataset_0
```

### Evaluation with LoRA (Local Evaluation)

```sh
uv run benchmarks/continual_eval_checkpoint.py \
    --dataset_name benchmarks/continual_dpo_trainer \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --checkpoint_dir Qwen2-0.5B-DPO-test \
    --learning_rate 0 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 1000 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --bf16 \
    --output_dir Qwen2-0.5B-DPO-test-eval \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
```

### Run a Sweep with wandb

First, create the sweep:

```sh
wandb sweep sweep_configs/ppo_sweep.yaml    # This will output the SWEEP_ID
```

Then, run the agent:

```sh
wandb agent <SWEEP_ID>
```

All these commands ensure that whether you run full training or use LoRA and pushing to the Hub, the parameters remain consistent with your working configuration. Adjust any parameters as needed in your own setup.
