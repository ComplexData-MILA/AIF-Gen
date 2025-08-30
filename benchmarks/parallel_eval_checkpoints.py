import os

import torch
import wandb as wb
from dataloading import init_continual_dataset
from datasets import Dataset
from dpo.continual_dpo_trainer import (
    ContinualDPOArguments,
    ContinualDPOConfig,
    ContinualDPOTrainer,
)
from safetensors import safe_open
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

    # Checkpoint loop
    checkpoint_path = script_args.checkpoint_dir
    if 'PPO' in checkpoint_path:
        dataset_name = 'dataset-' + checkpoint_path.split('/')[-2].split('_')[-1]
    else:
        dataset_name = checkpoint_path.split('/')[-2].replace('.', '')

    checkpoint_step = checkpoint_path.split('/')[-1].replace('.', '')
    print(
        f'Evaluating checkpoint: {checkpoint_step} trained on dataset: {dataset_name} on all tasks'
    )
    checkpoint_name = dataset_name + '_' + checkpoint_step
    print('checkpoint_name', checkpoint_name)

    if 'PPO' in checkpoint_path:
        # remove the prefix 'policy.' from the keys to load the model; skip the critic and value model
        prefix = 'policy.'
        with safe_open(
            checkpoint_path + '/model.safetensors', framework='pt', device='cpu'
        ) as f:
            clean_sd = {
                k[len(prefix) :] if k.startswith(prefix) else k: f.get_tensor(k)
                for k in f.keys()
                if not (
                    k.startswith('critic_backbone.') or k.startswith('value_model.')
                )
            }

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=model_args.trust_remote_code,
            state_dict=clean_sd,
            **model_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    peft_config = get_peft_config(model_args)

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )

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

        print('evaluating...')
        ev_metrics = trainer.evaluate()
        # ev_metrics = {f'dataset-{i}/' + k: v for k, v in ev_metrics.items()}
        if training_args.local_rank in (None, -1, 0):
            print('ev_metrics', ev_metrics)
            wb.log(ev_metrics)
            wb.log({f'{checkpoint_name}/{k}': v for k, v in ev_metrics.items()})

        # If using DeepSpeed through Accelerate, tear down the engine after training.
        if hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
            # Remove reference to the DeepSpeed engine to allow proper cleanup.
            del trainer.deepspeed
        # Free cached GPU memory.
        torch.cuda.empty_cache()


if __name__ == '__main__':
    dataclass_types = (ContinualDPOArguments, ContinualDPOConfig, ModelConfig)
    parser = TrlParser(dataclass_types)
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
