# Adaptation of TRL for Continual Learning with EWC (PPO)

This implementation extends the PPO trainer with Elastic Weight Consolidation (EWC) to mitigate catastrophic forgetting in continual learning scenarios.

### Sync additional dependencies

```sh
uv sync --group benchmarks
```

## Run PPO with EWC

### Lora with EWC

```sh
uv run benchmarks/ppo_ewc/ppo_EWC_continual.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --sft_model_path Qwen/Qwen2-0.5B-Instruct \
    --value_model_path Shahradmz/Qwen2-0.5B-Instruct_continual_data_debug_REWARD_0 \
    --reward_model_path Shahradmz/Qwen2-0.5B-Instruct_continual_data_debug_REWARD \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 10 \
    --gradient_checkpointing \
    --logging_steps 20 \
    --eval_strategy steps \
    --eval_steps 20 \
    --save_steps 20 \
    --bf16 \
    --output_dir "$SCRATCH/Qwen2-0.5B-PPO-test" \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --push_to_hub True \
    --ewc_lambda 20.0 \
```

### Lora with EWC using accelerate

```sh
accelerate launch --config_file benchmarks/ppo/accelerate_configs/deepspeed_zero3.yaml \
    benchmarks/ppo_ewc/ppo_EWC_continual.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --sft_model_path Qwen/Qwen2-0.5B-Instruct \
    --value_model_path Shahradmz/Qwen2-0.5B-Instruct_continual_data_debug_REWARD_0 \
    --reward_model_path Shahradmz/Qwen2-0.5B-Instruct_continual_data_debug_REWARD \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 5 \
    --save_steps 5 \
    --bf16 \
    --output_dir "$SCRATCH"/Qwen2-0.5B-PPO-EWC-test \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --ewc_lambda 20.0 \
    --wandb_project Qwen2-0.5B-PPO_EWC_lora_test
```

### Full training with EWC

```sh
uv run benchmarks/ppo_ewc/ppo_EWC_continual.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --sft_model_path Qwen/Qwen2-0.5B-Instruct \
    --value_model_path Shahradmz/Qwen2-0.5B-Instruct_continual_data_debug_REWARD_0 \
    --reward_model_path Shahradmz/Qwen2-0.5B-Instruct_continual_data_debug_REWARD \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-PPO-EWC \
    --no_remove_unused_columns \
    --ewc_lambda 20.0 \
```

## EWC Parameters

| Parameter                   | Description                                                                                | Default |
| --------------------------- | ------------------------------------------------------------------------------------------ | ------- |
| `ewc_lambda`                | EWC regularization strength. Higher values give stronger regularization.                   | 100.0   |
| `ewc_importance_decay`      | Decay factor for previous task importance (0-1). 0 means only care about most recent task. | 0.5     |
| `fisher_estimation_samples` | Number of samples to use when estimating Fisher information matrix.                        | 200     |

### Run a sweep with wandb

```sh
wandb sweep sweep_configs/ppo_EWC_sweep.yaml    # which returns the SWEEP_ID
```

and

```sh
wandb agent <SWEEP_ID>
```

- All details per task and hyperparameters are going to be loaded in your wandb dashboard.
