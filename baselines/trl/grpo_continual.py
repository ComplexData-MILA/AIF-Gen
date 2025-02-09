# Adaptation of the GRPO TRL training script for continual learning.

import argparse
from dataclasses import dataclass, field
from typing import Optional
import wandb

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_model_name_or_path (`str` or `None`):
            Reward model id of a pretrained model hosted inside a model repo on huggingface.co or local path to a
            directory containing model weights saved using [`~transformers.PreTrainedModel.save_pretrained`].
    """

    reward_model_name_or_path: Optional[str] = field(
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
    datasets = [load_dataset(
        script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split
    )]

    for i, dataset in enumerate(datasets):

        reward_model = AutoModelForSequenceClassification.from_pretrained(
            script_args.reward_model_name_or_path+f"_dataset{str(i)}", trust_remote_code=model_args.trust_remote_code, num_labels=1
        )

        # Initialize the GRPO trainer
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_model,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            processing_class=tokenizer,
            peft_config=get_peft_config(model_args),
        )

        # Train and push the model to the Hub
        trainer.train()

        # Save and push to hub
        trainer.save_model(training_args.output_dir)
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name)


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