"""Evaluating checkpoints obtained from training using the dpo_continual script."""

import glob
import os

import torch
import wandb
from dataloading import init_continual_dataset
from datasets import Dataset
from dpo.continual_dpo_trainer import (
    ContinualDPOArguments,
    ContinualDPOConfig,
    ContinualDPOTrainer,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import (
    DPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


def main(
    script_args: ScriptArguments,
    training_args: DPOConfig,
    model_args: ModelConfig,
) -> None:
    # Determine torch dtype and quantization configs
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ['auto', None]
        else getattr(torch, model_args.torch_dtype)
    )
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

    # Initialize continual dataset
    continual_dataset: list[dict[str, Dataset]] = init_continual_dataset(
        script_args.dataset_name,
        mock=training_args.mock,
        tokenizer=tokenizer,
        tools=getattr(training_args, 'tools', None),
    )
    output_dir = training_args.output_dir

    # Validate reward model paths if provided
    for i, _ in enumerate(continual_dataset):
        reward_path = os.path.join(training_args.reward_model_path, str(i))
        if not os.path.exists(reward_path):
            raise FileNotFoundError(
                f'Reward model not found for dataset {i} at {reward_path}'
            )

    checkpoint_paths = glob.glob(f'{script_args.checkpoint_dir}/*/*')
    checkpoint_paths = sorted([ch for ch in checkpoint_paths if 'checkpoint' in ch])

    # Checkpoint loop
    for checkpoint_path in checkpoint_paths:
        dataset_name = checkpoint_path.split('/')[-2].replace('.', '')
        checkpoint_step = checkpoint_path.split('/')[-1].replace('.', '')
        print(
            f'Evaluating checkpoint: {checkpoint_step} trained on dataset: {dataset_name} on all tasks'
        )
        adapter_name = dataset_name + checkpoint_step
        model.load_adapter(checkpoint_path, adapter_name=adapter_name)
        metrics = {}

        # Task Loop
        for i, dataset in enumerate(continual_dataset):
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                training_args.reward_model_path + f'/{str(i)}', num_labels=1
            )

            training_args.output_dir = f'{output_dir}/dataset-{i}'
            # using ContinualDPOTrainer for all pipelines (PPO, DPO, COPR, ..) only for evaluation
            # ToDo: train_dataset is never used here, pass a dummy dataset to make it clear
            trainer = ContinualDPOTrainer(
                args=training_args,
                processing_class=tokenizer,
                model=model,
                ref_model=ref_model,
                reward_model=reward_model,
                train_dataset=dataset[script_args.dataset_test_split],
                eval_dataset=dataset[script_args.dataset_test_split],
                peft_config=peft_config,
            )

            ev_metrics = trainer.evaluate()
            ev_metrics = {f'dataset-{i}/' + k: v for k, v in ev_metrics.items()}
            metrics.update(ev_metrics)

        wandb.log(metrics)

    print('Evaluation completed for all tasks and checkpoints!')


if __name__ == '__main__':
    dataclass_types = (ContinualDPOArguments, ContinualDPOConfig, ModelConfig)
    parser = TrlParser(dataclass_types)
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
