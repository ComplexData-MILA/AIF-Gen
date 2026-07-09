# Continual DPO benchmark

## Install benchmark dependencies

```sh
pip install -e .
pip install pytest ruff accelerate==0.34.2 deepspeed==0.16.3 datasets>=3.2.0 numpy pandas wandb transformers>=4.49.0 trl>=0.15.2 peft>=0.14.0
```

If you use the project's preferred workflow, sync the benchmark group instead:

```sh
uv sync --group benchmarks --group dev
```

## What changed in this benchmark

- The continual DPO pipeline now keeps **one trainer/model lifecycle** and swaps task datasets safely instead of reusing a shared `Accelerator` across multiple trainers.
- Reward-model policy evaluation and sampled completion logging are now **explicit task-end operations** via `--eval_policy_metrics` and `--log_completions`.
- Reward models are **not kept on the training path** anymore; they are loaded only for explicit evaluation.
- The default ZeRO-3 config is now a **multi-GPU template**. For a single GPU, either run without DeepSpeed first or use the offload config as a slower fallback.

## Recommended sequence-length knobs

DPO memory still scales with prompt/completion length and dynamic padding. For safer runs, set these explicitly:

- `--max_prompt_length 256`
- `--max_completion_length 256`
- `--max_length 512`

Increase them only after you have a stable baseline.

## Fast baseline (single GPU, no DeepSpeed)

Use this first to measure plain DPO throughput before adding ZeRO/offload complexity.

```sh
python benchmarks/dpo/dpo_continual.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --logging_steps 100 \
    --eval_strategy no \
    --bf16 \
    --max_prompt_length 256 \
    --max_completion_length 256 \
    --max_length 512 \
    --output_dir "$SCRATCH"/Qwen2.5-0.5B-DPO-baseline \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32
```

## Recommended memory-efficient Qwen 3B run (single GPU)

For constrained hardware, prefer **QLoRA** over full fine-tuning.

```sh
python benchmarks/dpo/dpo_continual.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --logging_steps 100 \
    --eval_strategy steps \
    --eval_steps 200 \
    --bf16 \
    --max_prompt_length 256 \
    --max_completion_length 256 \
    --max_length 512 \
    --output_dir "$SCRATCH"/Qwen2.5-3B-DPO-qlora \
    --no_remove_unused_columns \
    --use_peft \
    --load_in_4bit \
    --lora_r 16 \
    --lora_alpha 32
```

Optional explicit task-end reward evaluation/logging:

```sh
    --reward_model_path Shahradmz/Qwen2-0.5B-Instruct_continual_data_debug_REWARD \
    --eval_policy_metrics \
    --log_completions \
    --completion_logging_batches 1
```

Those flags add extra evaluation work on purpose; leave them off for raw throughput measurements.

## Multi-GPU ZeRO-3 (throughput / memory sharding)

`benchmarks/dpo/accelerate_configs/deepspeed_zero3.yaml` is a **template for multi-GPU runs**. Set `num_processes` to the number of GPUs you actually launch.

```sh
accelerate launch --config_file benchmarks/dpo/accelerate_configs/deepspeed_zero3.yaml \
    benchmarks/dpo/dpo_continual.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --logging_steps 100 \
    --eval_strategy steps \
    --eval_steps 200 \
    --bf16 \
    --max_prompt_length 256 \
    --max_completion_length 256 \
    --max_length 512 \
    --output_dir "$SCRATCH"/Qwen2.5-3B-DPO-zero3 \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32
```

Use this config only when you are actually launching multiple processes/GPUs. A one-process ZeRO-3 launch does **not** give multi-GPU parameter sharding.

## Single-GPU DeepSpeed fallback

If plain single-GPU QLoRA still does not fit, the slower fallback is CPU offload:

```sh
accelerate launch --config_file benchmarks/dpo/accelerate_configs/deepspeed_zero3_offload.yaml \
    benchmarks/dpo/dpo_continual.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --logging_steps 100 \
    --eval_strategy no \
    --bf16 \
    --max_prompt_length 256 \
    --max_completion_length 256 \
    --max_length 512 \
    --output_dir "$SCRATCH"/Qwen2.5-3B-DPO-zero3-offload \
    --no_remove_unused_columns \
    --use_peft \
    --load_in_4bit \
    --lora_r 16 \
    --lora_alpha 32
```

This is a fallback for memory pressure, not a fast path.

## Full fine-tuning

Full fine-tuning keeps the policy model trainable and also needs a reference model for standard DPO. On 3B models this is much more memory intensive than LoRA/QLoRA.

```sh
python benchmarks/dpo/dpo_continual.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --logging_steps 100 \
    --eval_strategy steps \
    --eval_steps 200 \
    --bf16 \
    --max_prompt_length 256 \
    --max_completion_length 256 \
    --max_length 512 \
    --output_dir "$SCRATCH"/Qwen2.5-0.5B-DPO-full \
    --no_remove_unused_columns
```

## LoRA vs QLoRA vs full fine-tuning

- **LoRA**: `--use_peft`; keeps the base model in standard precision.
- **QLoRA**: `--use_peft --load_in_4bit`; usually the best single-GPU choice for 3B models.
- **Full fine-tuning**: omit PEFT flags; highest memory use.

With PEFT, TRL can use the base model without a separate train-time reference-model copy, which is typically the lowest-memory DPO setup in this benchmark.

## Important limitations

- Hardware limits still apply. These configs reduce risk; they do **not** guarantee that OOM is impossible.
- Reward-model policy evaluation can still require substantial memory because it temporarily loads an additional model for explicit evaluation.
- If a run is unstable, reduce `--max_prompt_length`, `--max_completion_length`, `--max_length`, and/or increase `--gradient_accumulation_steps` before increasing batch size.

## Run a sweep with wandb

```sh
wandb sweep benchmarks/dpo/sweep_configs/dpo_sweep.yaml
```

Then run the returned sweep ID:

```sh
wandb agent <SWEEP_ID>
```
