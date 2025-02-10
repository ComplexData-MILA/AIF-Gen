# Adaptation of the PPO TRL training script for continual learning.

import shutil
import wandb
import os

import torch
from accelerate import PartialState
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    apply_chat_template,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from aif_gen.dataset import DebugContinualDataset, ContinualUltrafeedback2AnthropicDataset

"""
python baselines/trl/ppo_continual.py \
    --dataset_name debug \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --total_episodes 20 \
    --use_peft \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --sft_model_path Qwen/Qwen2-0.5B-Instruct \
    --reward_model_path /home/mila/i/ivan.anokhin/AIF-Gen/Qwen/Qwen2-0.5B-Reward/debug \
    --missing_eos_penalty 1.0      
"""


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    value_model_path = "/home/mila/i/ivan.anokhin/AIF-Gen/Qwen/Qwen2-0.5B-Reward/debug/0"
    value_model = AutoModelForSequenceClassification.from_pretrained(
        value_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None

    ################
    # Dataset
    ################
    if script_args.dataset_name == 'debug':
        continual_dataset = DebugContinualDataset()
    elif script_args.dataset_name == 'ultrafeedback2anthropic':
        continual_dataset = ContinualUltrafeedback2AnthropicDataset()
    else:
        raise ValueError(f"Unknown dataset: {script_args.dataset_name}")
    eval_samples = 100

    dataset_text_field = "prompt"
    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    for i in range(len(continual_dataset.datasets)):
        assert os.path.exists(training_args.reward_model_path+f"/{str(i)}"), f"Reward model not found for dataset {i}"

    for i, dataset in enumerate(continual_dataset.datasets):

        # Dataset
        dataset = dataset[script_args.dataset_train_split]
        dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
        train_dataset = dataset.select(range(len(dataset) - eval_samples))
        eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            train_dataset = prepare_dataset(train_dataset, tokenizer)
            eval_dataset = prepare_dataset(eval_dataset, tokenizer)

        # Reward model
        reward_model = AutoModelForSequenceClassification.from_pretrained(training_args.reward_model_path+f"/{str(i)}", num_labels=1)
        # ToDo: Value model is based on the reward model, so we need to think if we want to reinstantiate
        #  the value model with the new reward model when we change dataset or continue to train with the old one

        ################
        # Training
        ################
        trainer = PPOTrainer(
            args=training_args,
            processing_class=tokenizer,
            model=policy,
            ref_model=ref_policy,
            reward_model=reward_model,
            value_model=value_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
        )
        trainer.train()

        # ToDo: PPOTrainer doesn't have an evaluate method, so we need to implement it to track the performance at each dataset

        # Save and push to hub
        trainer.save_model(training_args.output_dir + f"/dataset-{i}")
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name + f"/dataset-{i}")

        trainer.generate_completions()