# Adaptation of the GRPO TRL training script for continual learning.

import argparse
import os

import torch
import wandb as wb
from continual_grpo_trainer import (
    ContinualGRPOConfig,
    ContinualGRPOTrainer,
    GRPOScriptArguments,
)
from datasets import Dataset
from transformers import (
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

from benchmarks.dataloading import init_continual_dataset


# The code is based on TRL GRPO script https://github.com/huggingface/trl/blob/main/trl/scripts/grpo.py
def main(script_args, training_args, model_args):
    # Determine torch dtype and quantization configs
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ['auto', None]
        else getattr(torch, model_args.torch_dtype)
    )
    if script_args.wandb_run_name is not None:
        training_args.run_name = script_args.wandb_run_name

    quantization_config = get_quantization_config(model_args)

    # Model & Tokenizer Setup
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

    # Load tokenizer and set chat template if needed
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    # Initialize continual dataset
    continual_dataset: list[dict[str, Dataset]] = init_continual_dataset(
        script_args.dataset_name,
        mock=training_args.mock,
        tokenizer=tokenizer,
        tools=None,
    )
    output_dir = training_args.output_dir

    # Validate reward model paths if provided
    if training_args.reward_model_path is not None:
        for i, _ in enumerate(continual_dataset):
            reward_path = os.path.join(training_args.reward_model_path, str(i))
            if not os.path.exists(reward_path):
                raise FileNotFoundError(
                    f'Reward model not found for dataset {i} at {reward_path}'
                )

    # Task Loop
    for i, dataset in enumerate(continual_dataset):
        current_dataset_name: str = f'dataset-{i}'
        training_args.output_dir = f'{output_dir}/dataset-{i}'

        # Reward model
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            f'{training_args.reward_model_path}/{i}', num_labels=1
        )

        # Initialize the GRPO trainer
        trainer = ContinualGRPOTrainer(
            args=training_args,
            processing_class=tokenizer,
            model=model,
            reward_model=reward_model,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split],
            peft_config=peft_config,
        )

        # Train
        trainer.train()

        # Evaluate
        metrics = trainer.evaluate_policy()
        print(f'eval/dataset/{i}')
        metrics['dataset'] = i
        trainer.log_metrics(f'eval/dataset/{i}', metrics)
        trainer.save_metrics(f'eval', metrics)
        wb.log({'eval': {'last': metrics}})  # type: ignore[attr-defined]
        wb.log({f'task/{current_dataset_name}/last': metrics})  # type: ignore[attr-defined]

        # Save and push to hub
        trainer.save_model(training_args.output_dir + f'/dataset-{i}')
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name + f'/dataset-{i}')

        # If using DeepSpeed through Accelerate, tear down the engine after training.
        if hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
            # Remove reference to the DeepSpeed engine to allow proper cleanup.
            del trainer.deepspeed
        # Free cached GPU memory.
        torch.cuda.empty_cache()


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (GRPOScriptArguments, ContinualGRPOConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            'grpo', help='Run the GRPO training script', dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == '__main__':
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
