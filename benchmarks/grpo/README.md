# Adaptation of TRL for Continual Learning

This repository adapts TRL for continual learning. The commands below use a consistent set of parameters that you’ve identified as working. You can use any of the entrypoints (uv run, accelerate launch, wandb) with the following commands.

### Sync additional dependencies

```sh
uv sync --group benchmarks.grpo
```

## Run GRPO

### Using uv run (vanilla Python with PEFT)

```sh
python benchmarks/grpo/grpo_continual.py \
    --dataset_name debug \
    --mock \
    --bf16 \
    --learning_rate 3e-6 \
    --output_dir "$SCRATCH/Qwen2-0.5B-GRPO-test" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 20 \
    --per_device_eval_batch_size 2 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --reward_model_path Shahradmz/Qwen2-0.5B-Instruct_continual_data_debug_REWARD
```
