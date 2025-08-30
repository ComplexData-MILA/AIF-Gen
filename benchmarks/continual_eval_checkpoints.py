import glob
import os
import re

import torch
import wandb as wb
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
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


def main(
    script_args: ContinualDPOArguments,
    training_args: ContinualDPOConfig,
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
        reward_path = training_args.reward_model_path + '_' + str(i)
        if not os.path.exists(reward_path):
            raise FileNotFoundError(
                f'Reward model not found for dataset {i} at {reward_path}'
            )

    checkpoint_paths = glob.glob(f'{script_args.checkpoint_dir}/*/*')

    def extract_indices(path):
        match = re.search(r'dataset-(\d+)/checkpoint-(\d+)', path)
        if match:
            dataset_idx = int(match.group(1))
            checkpoint_idx = int(match.group(2))
            return (dataset_idx, checkpoint_idx)
        else:
            return (float('inf'), float('inf'))  # in case of unexpected format

    checkpoint_paths = [ch for ch in checkpoint_paths if 'checkpoint' in ch]
    checkpoint_paths.sort(key=extract_indices)
    print('checkpoint_paths', checkpoint_paths)

    # Checkpoint loop
    for checkpoint_path in checkpoint_paths:
        dataset_name = checkpoint_path.split('/')[-2].replace('.', '')
        checkpoint_step = checkpoint_path.split('/')[-1].replace('.', '')
        print(
            f'Evaluating checkpoint: {checkpoint_step} trained on dataset: {dataset_name} on all tasks'
        )
        # adapter_name = dataset_name + checkpoint_step
        # model.load_adapter(checkpoint_path, adapter_name=adapter_name)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
        metrics = {}

        # Task Loop
        for i, dataset in enumerate(continual_dataset):
            print('task', i)
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                training_args.reward_model_path + f'_{str(i)}', num_labels=1
            )

            training_args.output_dir = f'{output_dir}/dataset-{i}'
            # using ContinualDPOTrainer for all pipelines (PPO, DPO, COPR, ..) only for evaluation
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
            if training_args.local_rank in (None, -1, 0):
                wb.log({f'task/{dataset_name}/{k}': v for k, v in ev_metrics.items()})

            # If using DeepSpeed through Accelerate, tear down the engine after training.
            if hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
                # Remove reference to the DeepSpeed engine to allow proper cleanup.
                del trainer.deepspeed
            # Free cached GPU memory.
            torch.cuda.empty_cache()

        if training_args.local_rank in (None, -1, 0):
            wb.log(metrics)  # type: ignore[attr-defined]

    print('Evaluation completed for all tasks and checkpoints!')


if __name__ == '__main__':
    dataclass_types = (ContinualDPOArguments, ContinualDPOConfig, ModelConfig)
    parser = TrlParser(dataclass_types)
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
