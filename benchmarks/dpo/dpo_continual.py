"""Adaptation of the DPO TRL training script for continual learning."""

import os
import time
import warnings
from contextlib import contextmanager
from typing import Any, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import (
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

import wandb as wb

from benchmarks.dataloading import init_continual_dataset
from benchmarks.dpo.continual_dpo_trainer import (
    ContinualDPOArguments,
    ContinualDPOConfig,
    ContinualDPOTrainer,
    StepProfilingCallback,
)


# The code is based on TRL DPO script https://github.com/huggingface/trl/blob/main/trl/scripts/dpo.py

def warn_for_memory_settings(
    training_args: ContinualDPOConfig,
    model_args: ModelConfig,
) -> None:
    if training_args.max_completion_length is None:
        warnings.warn(
            'max_completion_length is unset. Long chosen/rejected responses can still cause large padded DPO batches; '
            'set it explicitly for tighter memory bounds.',
            stacklevel=2,
        )

    if not training_args.gradient_checkpointing:
        warnings.warn(
            'gradient_checkpointing is disabled. For larger DPO runs this will increase activation memory pressure.',
            stacklevel=2,
        )

    if not training_args.bf16 and not training_args.fp16:
        warnings.warn(
            'Neither bf16 nor fp16 is enabled. Full-precision DPO is usually much more memory intensive.',
            stacklevel=2,
        )

    if not model_args.use_peft and training_args.reward_model_path is not None:
        warnings.warn(
            'Full-parameter DPO with an explicit reward-model evaluation path is the highest-memory configuration. '
            'Prefer --use_peft (or QLoRA with --load_in_4bit) for constrained hardware.',
            stacklevel=2,
        )


def get_task_reward_model_path(
    reward_model_root: Optional[str],
    task_index: int,
) -> Optional[str]:
    if reward_model_root is None:
        return None
    return f'{reward_model_root}_{task_index}'


def validate_reward_model_paths(
    reward_model_root: Optional[str],
    num_tasks: int,
) -> None:
    if reward_model_root is None:
        return

    for task_index in range(num_tasks):
        reward_path = get_task_reward_model_path(reward_model_root, task_index)
        try:
            AutoConfig.from_pretrained(reward_path, trust_remote_code=True)
        except Exception as exc:
            if not os.path.exists(reward_path):
                raise ValueError(f'Reward model not found at {reward_path}') from exc
            raise ValueError(
                f'Failed to load reward model at {reward_path}: {exc}'
            ) from exc


def load_reward_model_for_task(
    reward_model_root: Optional[str],
    task_index: int,
    torch_dtype: Optional[torch.dtype],
    trust_remote_code: bool,
) -> Optional[AutoModelForSequenceClassification]:
    reward_path = get_task_reward_model_path(reward_model_root, task_index)
    if reward_path is None:
        return None

    reward_model_kwargs = {
        'num_labels': 1,
        'trust_remote_code': trust_remote_code,
    }
    if torch_dtype is not None:
        reward_model_kwargs['torch_dtype'] = torch_dtype

    return AutoModelForSequenceClassification.from_pretrained(
        reward_path,
        **reward_model_kwargs,
    )


def _build_torch_profiler(
    training_args: ContinualDPOConfig,
) -> Optional[torch.profiler.profile]:
    if not training_args.enable_profiling:
        return None

    active_steps = max(1, training_args.profiling_steps)
    os.makedirs(training_args.profile_output_dir, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=active_steps),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            training_args.profile_output_dir
        ),
        record_shapes=True,
        profile_memory=training_args.profile_memory,
    )


@contextmanager
def _time_phase(label: str, wandb_run: Optional[Any], enabled: bool):
    if not enabled:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f'[{label}] elapsed: {elapsed:.3f}s')
        if wandb_run is not None:
            wb.log({f'profiling/{label}_time_s': float(elapsed)})


def main(
    script_args: ContinualDPOArguments,
    training_args: ContinualDPOConfig,
    model_args: ModelConfig,
) -> None:
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ['auto', None]
        else getattr(torch, model_args.torch_dtype)
    )
    if script_args.wandb_run_name is not None:
        training_args.run_name = script_args.wandb_run_name

    warn_for_memory_settings(training_args, model_args)

    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    else:
        ref_model = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    with _time_phase('data_loading', wb.run, training_args.enable_profiling):
        continual_dataset: list[dict[str, Dataset]] = init_continual_dataset(
            script_args.dataset_name,
            mock=training_args.mock,
            tokenizer=tokenizer,
            tools=training_args.tools,
        )
    output_dir = training_args.output_dir
    eval_enabled = training_args.eval_strategy != 'no'
    explicit_policy_eval = training_args.eval_policy_metrics or training_args.log_completions

    if training_args.eval_policy_metrics and training_args.reward_model_path is None:
        raise ValueError(
            'Cannot use --eval_policy_metrics without --reward_model_path; reward model path must be specified for policy evaluation.'
        )

    if explicit_policy_eval:
        validate_reward_model_paths(
            training_args.reward_model_path,
            len(continual_dataset),
        )

    first_dataset = continual_dataset[0]
    trainer = ContinualDPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=model,
        ref_model=ref_model,
        train_dataset=first_dataset[script_args.dataset_train_split],
        eval_dataset=first_dataset.get(script_args.dataset_test_split),
        peft_config=peft_config,
    )
    profiler = _build_torch_profiler(training_args)
    if training_args.enable_profiling:
        trainer.add_callback(
            StepProfilingCallback(
                profiler=profiler,
                profile_memory=training_args.profile_memory,
            )
        )

    if wb.run is not None:
        wb.log({'dataset/name': script_args.dataset_name})
    if training_args.enable_profiling and torch.cuda.is_available():
        print('CUDA memory summary before continual loop:')
        print(torch.cuda.memory_summary())

    first_task_profiled = False
    for task_index, dataset in enumerate(continual_dataset):
        with _time_phase(
            f'task_{task_index}',
            wb.run,
            training_args.enable_profiling,
        ):
            current_dataset_name = f'dataset-{task_index}'
            training_args.output_dir = f'{output_dir}/{current_dataset_name}'
            trainer.args.output_dir = training_args.output_dir

            if task_index > 0:
                trainer.set_task_datasets(
                    train_dataset=dataset[script_args.dataset_train_split],
                    eval_dataset=dataset.get(script_args.dataset_test_split),
                    dataset_name=current_dataset_name,
                )

            if training_args.enable_profiling and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                print(
                    f'CUDA memory summary at task {task_index} start ({current_dataset_name}):'
                )
                print(torch.cuda.memory_summary())

            print('Training dataset:', current_dataset_name)
            should_profile_first_task = (
                training_args.enable_profiling
                and not first_task_profiled
            )
            if should_profile_first_task:
                profiler.start()
            try:
                trainer.train()
            finally:
                if should_profile_first_task:
                    profiler.stop()
                    first_task_profiled = True
                    print(
                        f'Profiler trace exported to {training_args.profile_output_dir} for task {task_index}.'
                    )

            should_run_task_eval = eval_enabled or explicit_policy_eval
            if should_run_task_eval and trainer.eval_dataset is not None:
                metrics = trainer.evaluate()
                reward_model = None
                try:
                    if explicit_policy_eval:
                        reward_model = load_reward_model_for_task(
                            training_args.reward_model_path,
                            task_index,
                            torch_dtype,
                            model_args.trust_remote_code,
                        )
                        try:
                            metrics.update(trainer.evaluate_policy(reward_model=reward_model))
                            if training_args.log_completions:
                                trainer.generate_completions_table(
                                    reward_model=reward_model,
                                    max_batches=training_args.completion_logging_batches,
                                )
                        except Exception as exc:
                            raise RuntimeError(
                                f'Explicit policy evaluation failed for task {task_index} ({current_dataset_name}).'
                            ) from exc
                finally:
                    if reward_model is not None:
                        del reward_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                metrics['dataset'] = task_index
                trainer.log_metrics(f'eval/dataset/{task_index}', metrics)
                trainer.save_metrics('eval', metrics)
                if wb.run is not None:
                    wb.log({'eval/last': metrics})
                    wb.log({f'task/{current_dataset_name}/last': metrics})

            if training_args.enable_profiling and torch.cuda.is_available():
                peak_allocated_gb = float(torch.cuda.max_memory_allocated() / (1024**3))
                peak_reserved_gb = float(torch.cuda.max_memory_reserved() / (1024**3))
                print(
                    f'Task {task_index} peak CUDA memory (GB): allocated={peak_allocated_gb:.3f}, reserved={peak_reserved_gb:.3f}'
                )
                if wb.run is not None:
                    wb.log(
                        {
                            f'profiling/task_{task_index}_peak_memory_allocated_gb': peak_allocated_gb,
                            f'profiling/task_{task_index}_peak_memory_reserved_gb': peak_reserved_gb,
                        }
                    )

            trainer.save_model(os.path.join(training_args.output_dir, 'last'))
            if training_args.push_to_hub:
                trainer.push_to_hub(
                    dataset_name=f'Continual_DPO_{script_args.dataset_name}_{task_index}'
                )

    if training_args.enable_profiling and torch.cuda.is_available():
        print('CUDA memory summary after continual loop:')
        print(torch.cuda.memory_summary())


if __name__ == '__main__':
    dataclass_types = (ContinualDPOArguments, ContinualDPOConfig, ModelConfig)
    parser = TrlParser(dataclass_types)
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
