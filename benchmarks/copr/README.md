# COPR: Continual Learning with Optimal Policy Regularization

This guide explains how to run the `copr.py` script for continual learning using the COPR algorithm.

### Install Dependencies

First, ensure you have the necessary dependencies installed. You can synchronize them using `uv`:

```sh
uv sync --group benchmarks
```

Running COPR
The copr.py script trains a language model using the COPR algorithm on a sequence of tasks. Here are example commands for different training configurations:

## LoRA Training

This configuration uses Low-Rank Adaptation (LoRA) to efficiently train the model.

```sh
uv run benchmarks/copr/copr.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 20 \
    --eval_strategy steps \
    --eval_steps 20 \
    --save_steps 20 \
    --bf16 \
    --output_dir Qwen2-0.5B-COPR-test \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --memory_buffer_size 100 \
    --buffer_ratio 0.1 \
    --use_buffer_ratio True
```

Explanation of Arguments:

```
--dataset_name: The name of the continual learning dataset to use.
--mock: Whether to use a mock dataset for debugging.
--model_name_or_path: The path to the pre-trained language model.
--learning_rate: The learning rate for training.
--num_train_epochs: The number of training epochs.
--per_device_train_batch_size: The batch size per device.
--gradient_accumulation_steps: The number of gradient accumulation steps.
--gradient_checkpointing: Whether to use gradient checkpointing to reduce memory usage.
--logging_steps: The number of steps between logging intervals.
--eval_strategy: The evaluation strategy to use (e.g., steps, epoch, or no).
--eval_steps: The number of steps between evaluation intervals.
--save_steps: The number of steps between saving checkpoints.
--bf16: Whether to use bfloat16 mixed precision training.
--output_dir: The directory to save the training outputs.
--no_remove_unused_columns: Whether to prevent the removal of unused columns in the dataset.
--use_peft: Whether to use Parameter-Efficient Fine-Tuning (PEFT).
--lora_r: LoRA attention dimension.
--lora_alpha: LoRA scaling factor.
--memory_buffer_size: Maximum number of examples to keep in memory buffer.
--buffer_ratio: Percentage of task samples to keep in memory buffer (0.0-1.0). If specified, overrides memory_buffer_size.
--use_buffer_ratio: Whether to use buffer_ratio instead of memory_buffer_size.
```

## LoRA Training with Accelerate

This configuration uses the accelerate library for distributed training, enabling multi-GPU and mixed-precision training.

```sh
accelerate launch --config_file benchmarks/dpo/accelerate_configs/deepspeed_zero3.yaml \
    benchmarks/copr/copr.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_steps 3 \
    --bf16 \
    --output_dir Qwen2-0.5B-COPR-test \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --memory_buffer_size 100 \
    --buffer_ratio 0.3 \
    --use_buffer_ratio True
```

```sh
accelerate launch: Launches the training script using accelerate.
--config_file: Specifies the path to the accelerate configuration file (e.g., deepspeed_zero3.yaml). You'll need to create this file based on your hardware setup. Example configurations can be found in the accelerate documentation.
```

## Full Training

This configuration trains the entire model without using LoRA.

```sh
uv run benchmarks/copr/copr.py \
    --dataset_name debug \
    --mock true \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-COPR \
    --no_remove_unused_columns \
    --memory_buffer_size 100 \
    --buffer_ratio 0.1 \
    --use_buffer_ratio True
```

## Running a Sweep with WandB

To run a hyperparameter sweep using Weights & Biases (WandB):

Create a sweep_config.yaml file that defines the hyperparameters to sweep over. See the WandB documentation for details on the sweep configuration format.

Initiate the sweep:

```sh
wandb sweep sweep_config.yaml
```

Run the sweep agent:

```sh
wandb agent <SWEEP_ID>
```

The sweep results will be logged to your WandB dashboard, where you can view the performance of different hyperparameter configurations.
