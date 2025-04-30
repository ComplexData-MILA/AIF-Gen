# Adaptation of TRL for Continual Learning

### Sync additional dependencies

```sh
uv sync --group benchmarks
```

## Run DPO

### Lora

```sh
uv run benchmarks/dpo/dpo_continual.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
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
    --output_dir "$SCRATCH"/Qwen2-0.5B-DPO-test \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
```

### Lora with accelerate

```sh
accelerate launch --config_file benchmarks/dpo/accelerate_configs/deepspeed_zero3.yaml \
    benchmarks/dpo/dpo_continual.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
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
    --output_dir "$SCRATCH"/Qwen2-0.5B-DPO-test \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --wandb_project Qwen2-0.5B-DPO_lora_test
```

### Full training

```sh
uv run benchmarks/dpo/dpo_continual.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --reward_model_path Shahradmz/Qwen2-0.5B-Instruct_continual_data_debug_REWARD \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns
```

### Run a sweep with wandb

```sh
wandb sweep sweep_configs/dpo_sweep.yaml    # which returns the SWEEP_ID
```

and

```sh
wandb agent <SWEEP_ID>
```

- All details per task and hyperparameters are going to be loaded in your wandb dashboard.
