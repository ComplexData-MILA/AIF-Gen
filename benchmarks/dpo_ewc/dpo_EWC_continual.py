"""Adaptation of the DPO TRL training script for continual learning."""

import os

import torch
import wandb as wb
from continual_dpo_EWC_trainer import (
    ContinualDPOEWCArguments,
    ContinualDPOEWCConfig,
    ContinualDPOEWCTrainer,
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


# The code is based on TRL DPO script https://github.com/huggingface/trl/blob/main/trl/scripts/dpo.py
def main(
    script_args: ContinualDPOEWCArguments,
    training_args: ContinualDPOEWCConfig,
    model_args: ModelConfig,
) -> None:
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
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    else:
        ref_model = None

    # Load tokenizer and set chat template if needed
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    # Distributed training hack
    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # Initialize continual dataset
    continual_dataset: list[dict[str, Dataset]] = init_continual_dataset(
        script_args.dataset_name,
        mock=training_args.mock,
        tokenizer=tokenizer,
        tools=training_args.tools,
    )
    output_dir = training_args.output_dir

    # check if the reward models are present either in the path or in the hub
    if training_args.reward_model_path is not None:
        for i in range(len(continual_dataset)):
            reward_path = training_args.reward_model_path + '_' + str(i)
            # first check the hub if the model is present
            try:
                AutoModelForSequenceClassification.from_pretrained(
                    reward_path, num_labels=1
                )
            except:
                # if not found in the hub, check the local path
                if not os.path.exists(reward_path):
                    raise ValueError(f'Reward model not found at {reward_path}')

    # Task Loop
    for i, dataset in enumerate(continual_dataset):
        current_dataset_name: str = f'dataset-{i}'
        training_args.output_dir = f'{output_dir}/{current_dataset_name}'

        # Load reward model if path provided
        if training_args.reward_model_path is not None:
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                training_args.reward_model_path + f'_{str(i)}', num_labels=1
            )

        trainer = ContinualDPOEWCTrainer(
            args=training_args,
            processing_class=tokenizer,
            model=model,
            ref_model=ref_model,
            reward_model=reward_model
            if training_args.reward_model_path is not None
            else None,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != 'no'
            else None,
            peft_config=peft_config,
        )

        # TODO will throw Invalidate trace cache @ step 10: expected module 11, but got module 19
        # https://github.com/deepspeedai/DeepSpeed/issues/6870
        # Fix with deepspeed fix release
        print('Training dataset:', current_dataset_name)
        trainer.train()

        if training_args.eval_strategy != 'no':
            metrics = trainer.evaluate()
            if i == 0:
                trainer.log({'dataset': {'name': script_args.dataset_name}})
            metrics['dataset'] = i
            # Log evaluation metrics under a hierarchy using slashes for wandb
            print(f'eval/dataset/{i}')
            trainer.log_metrics(f'eval/dataset/{i}', metrics)
            trainer.save_metrics(f'eval', metrics)
            wb.log({'eval': {'last': metrics}})  # type: ignore[attr-defined]
            wb.log({f'task/{current_dataset_name}/last': metrics})  # type: ignore[attr-defined]

        # Save and push to hub
        trainer.save_model(os.path.join(training_args.output_dir, 'last'))
        if training_args.push_to_hub:
            trainer.push_to_hub(
                dataset_name=(
                    'Continual_DPO_' + script_args.dataset_name + '_' + str(i),
                )
            )

        # If using DeepSpeed through Accelerate, tear down the engine after training.
        if hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
            # Remove reference to the DeepSpeed engine to allow proper cleanup.
            del trainer.deepspeed
        # Free cached GPU memory.
        torch.cuda.empty_cache()


if __name__ == '__main__':
    dataclass_types = (ContinualDPOEWCArguments, ContinualDPOEWCConfig, ModelConfig)
    parser = TrlParser(dataclass_types)
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
