# Adaptation of the GRPO TRL training script for continual learning.

import argparse
from dataclasses import dataclass, field
from typing import Optional
import os
import wandb

from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config, apply_chat_template

from aif_gen.dataset import DebugContinualDataset, ContinualUltrafeedback2AnthropicDataset

"""
python baselines/trl/grpo_continual.py \
    --dataset_name debug \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir models/minimal/grpo \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --use_peft \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --reward_model_path /home/mila/i/ivan.anokhin/AIF-Gen/Qwen/Qwen2-0.5B-Reward/debug 
"""


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_model_path (`str` or `None`):
            Reward model id of a pretrained model hosted inside a model repo on huggingface.co or local path to a
            directory containing model weights saved using [`~transformers.PreTrainedModel.save_pretrained`].
    """

    reward_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Reward model id of a pretrained model hosted inside a model repo on huggingface.co or "
            "local path to a directory containing model weights saved using `PreTrainedModel.save_pretrained`."
        },
    )

def main(script_args, training_args, model_args):

    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    # Load the dataset
    if script_args.dataset_name == 'debug':
        continual_dataset = DebugContinualDataset()
    elif script_args.dataset_name == 'ultrafeedback2anthropic':
        continual_dataset = ContinualUltrafeedback2AnthropicDataset()
    else:
        raise ValueError(f"Unknown dataset: {script_args.dataset_name}")
    eval_samples = 100

    for i in range(len(continual_dataset.datasets)):
        assert os.path.exists(f"{script_args.reward_model_path}/{i}"), f"Reward model not found for dataset {i}"

    for i, dataset in enumerate(continual_dataset.datasets):
        # Dataset
        dataset = dataset[script_args.dataset_train_split]
        dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
        train_dataset = dataset.select(range(len(dataset) - eval_samples))
        eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))

        # Reward model
        reward_model = AutoModelForSequenceClassification.from_pretrained(f"{script_args.reward_model_path}/{i}", num_labels=1)

        # Initialize the GRPO trainer
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=get_peft_config(model_args),
        )

        # Train and push the model to the Hub
        trainer.train()

        # ToDo: GRPOTrainer doesn't have a evaluate method, so we need to implement it to track the performance at each dataset

        # Save and push to hub
        trainer.save_model(training_args.output_dir + f"/dataset-{i}")
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name + f"/dataset-{i}")



def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (GRPOScriptArguments, GRPOConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("grpo", help="Run the GRPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)