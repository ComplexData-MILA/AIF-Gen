# Adaptation of TRL for Continual Learning

This repository adapts TRL for continual learning. The commands below use a consistent set of parameters that you’ve identified as working. You can use any of the entrypoints (uv run, accelerate launch, wandb) with the following commands.

### Sync additional dependencies

```sh
uv sync --group benchmarks
```

## Run PPO

### Using uv run (vanilla Python with PEFT)

```sh
uv run benchmarks/ppo/ppo_continual.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --sft_model_path Qwen/Qwen2-0.5B-Instruct \
    --value_model_path Shahradmz/Qwen2-0.5B-Instruct_continual_data_debug_REWARD_0 \
    --reward_model_path Shahradmz/Qwen2-0.5B-Instruct_continual_data_debug_REWARD \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 2 \
    --gradient_accumulation_steps 8 \
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
    --push_to_hub False
```

### Using accelerate launch (with DeepSpeed / multi-GPU)

- Please note that our implementation only works with DeepSpeed and Zero2. The configuration file `benchmarks/ppo/accelerate_configs/deepspeed_zero2.yaml` is provided for this purpose.
  We will add support for other configurations in the future.

```sh
accelerate launch --config_file benchmarks/ppo/accelerate_configs/deepspeed_zero2.yaml \
    benchmarks/ppo/ppo_continual.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --sft_model_path Qwen/Qwen2-0.5B-Instruct \
    --value_model_path Shahradmz/Qwen2-0.5B-Instruct_continual_data_debug_REWARD_0 \
    --reward_model_path Shahradmz/Qwen2-0.5B-Instruct_continual_data_debug_REWARD \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_steps 10 \
    --bf16 \
    --output_dir "$SCRATCH/Qwen2-0.5B-PPO-test" \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --push_to_hub False
```

*Make sure you do not add the dataset index to the reward model name as the script itself iterates over the dataset indices.*

### Full Training (without PEFT push, for local evaluation)

```sh
accelerate launch --config_file benchmarks/ppo/accelerate_configs/deepspeed_zero2.yaml \
     benchmarks/ppo/ppo_continual.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --mock False \
    --sft_model_path Qwen/Qwen2-0.5B-Instruct \
    --value_model_path Shahradmz/Qwen2-0.5B-Instruct_continual_data_debug_REWARD_0 \
    --reward_model_path Shahradmz/Qwen2-0.5B-Instruct_continual_data_debug_REWARD \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --bf16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --logging_steps 20 \
    --eval_strategy steps \
    --eval_steps 20 \
    --output_dir "$SCRATCH/Qwen2-0.5B-PPO" \
    --no_remove_unused_columns \
    --push_to_hub False
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

All details per task and hyperparameters are going to be loaded in your wandb dashboard.

______________________________________________________________________

These commands ensure that all runs (whether via uv run or accelerate launch) use the same consistent set of parameters that you’ve confirmed to work. Adjust any parameters as needed in your configuration before running.

______________________________________________________________________

## Additional Notes
