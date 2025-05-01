"""Adaptation of the CPPO TRL training script for continual learning with task-based logging."""

import os

import torch
from cppo_trainer import CPPOArguments, CPPOConfig, CPPOTrainer
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

import wandb as wb
from benchmarks.dataloading import init_continual_dataset


def main(
    script_args: CPPOArguments,
    training_args: CPPOConfig,
    model_args: ModelConfig,
) -> None:
    # Determine torch dtype and quantization configs
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ['auto', None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    if script_args.wandb_run_name is not None:
        training_args.run_name = script_args.wandb_run_name

    # Model & Tokenizer Setup
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Load main model and (optionally) reference model
    model = str(training_args.sft_model_path)
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    else:
        ref_policy = None

    # Load value model and policy model (main model)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.value_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
    )

    # Load tokenizer and set chat template if needed
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.sft_model_path,
        trust_remote_code=model_args.trust_remote_code,
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

    # Extract clean dataset name for repo naming
    clean_dataset_name = os.path.basename(script_args.dataset_name)
    if '.' in clean_dataset_name:
        clean_dataset_name = clean_dataset_name.split('.')[0]

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
    old_logprobs, old_rewards = None, None

    for i, dataset in enumerate(continual_dataset):
        # Build custom repository name for this task
        custom_repo_name = (
            model.split('/')[-1] + '_' + clean_dataset_name + '_CPPO_' + str(i)
        )
        if training_args.push_to_hub:
            training_args.hub_model_id = custom_repo_name
        training_args.output_dir = os.path.join(output_dir, custom_repo_name)

        # Load reward model based on naming convention (expects suffix with task index)
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            training_args.reward_model_path + '_' + str(i), num_labels=1
        )

        ################
        # Training and Evaluation
        ################
        trainer = CPPOTrainer(
            args=training_args,
            processing_class=tokenizer,
            model=policy,
            ref_model=ref_policy,
            reward_model=reward_model,
            value_model=value_model,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split],
            peft_config=peft_config,
            old_logprobs=old_logprobs,
            old_rewards=old_rewards,
        )
        # Set current task in trainer for task-based logging
        trainer.set_task(f'task_{i}')

        print(f'Training on task: {custom_repo_name}')
        trainer.train()

        if training_args.eval_strategy != 'no':
            # Mark final evaluation for this task so metrics are logged under eval.last as well
            trainer.mark_final_eval(True)
            metrics = trainer.evaluate()
            trainer.mark_final_eval(False)

            # Log dataset and task-specific metrics
            if i == 0:
                trainer.log({'dataset': {'name': script_args.dataset_name}})
            metrics['dataset'] = i
            print(f'Evaluation metrics for dataset {i}: {metrics}')
            trainer.log_metrics(f'eval/dataset/{i}', metrics)
            trainer.save_metrics('eval', metrics)

            # Log metrics to WandB
            wb.log({'eval': {'last': metrics}})
            wb.log({f'task/{custom_repo_name}/last': metrics})

        # Save model checkpoint and optionally push
        if not training_args.push_to_hub:
            trainer.save_model(os.path.join(training_args.output_dir, 'last'))
        else:
            trainer.push_to_hub(
                model_name=custom_repo_name,
                dataset_name='CPPO_' + clean_dataset_name + '_' + str(i),
            )

        ref_policy = None
        old_logprobs, old_rewards = trainer.old_logprobs, trainer.old_rewards

    print('Training completed for all tasks!')


if __name__ == '__main__':
    dataclass_types = (CPPOArguments, CPPOConfig, ModelConfig)
    parser = TrlParser(dataclass_types)
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
