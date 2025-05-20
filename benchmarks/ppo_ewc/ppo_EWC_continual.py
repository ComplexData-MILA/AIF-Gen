"""Adaptation of the PPO TRL training script for continual learning with EWC regularization."""

import os

import torch
import wandb as wb
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
    setup_chat_format,
)

from benchmarks.dataloading import init_continual_dataset
from benchmarks.ppo_ewc.continual_ppo_EWC_trainer import (
    ContinualPPOEWCArguments,
    ContinualPPOEWCConfig,
    ContinualPPOEWCTrainer,
)


def main(
    script_args: ContinualPPOEWCArguments,
    training_args: ContinualPPOEWCConfig,
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

    model = str(training_args.sft_model_path)
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    else:
        ref_policy = None

    # Load tokenizer and set chat template if needed
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.sft_model_path,
        trust_remote_code=model_args.trust_remote_code,
    )

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

    print(f'Training PPO-EWC on {len(continual_dataset)} tasks')
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
        # Load main model and (optionally) reference model
        if i == 0:
            model_path = training_args.sft_model_path
            value_model_path = script_args.value_model_path
        else:
            model_path = os.path.join(training_args.output_dir, 'last')
            value_model_path = os.path.join(
                training_args.output_dir, 'last', 'value_model'
            )
        policy = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
        # EWC-specific: DDPT distributed setup
        if script_args.ignore_bias_buffers:
            policy._ddp_params_and_buffers_to_ignore = [
                name
                for name, buffer in policy.named_buffers()
                if buffer.dtype == torch.bool
            ]

        # Load value model and policy model (main model)
        try:
            value_model = AutoModelForSequenceClassification.from_pretrained(
                value_model_path,
                trust_remote_code=model_args.trust_remote_code,
                num_labels=1,
            )
        except OSError:
            # Maybe it was saved as safetensors?
            value_model = AutoModelForSequenceClassification.from_pretrained(
                value_model_path,
                trust_remote_code=model_args.trust_remote_code,
                num_labels=1,
                from_tf=True,  # or use `subfolder="safetensors"` if you saved a .safetensors file
            )

        # Build custom repository name for this task
        custom_repo_name = (
            model.split('/')[-1] + '_' + clean_dataset_name + '_PPO_EWC_' + str(i)
        )
        if training_args.push_to_hub:
            training_args.hub_model_id = custom_repo_name
        training_args.output_dir = os.path.join(output_dir, custom_repo_name)

        # Load reward model based on naming convention (expects suffix with task index)
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            training_args.reward_model_path + '_' + str(i), num_labels=1
        )

        for idx, _model in enumerate([policy, value_model, reward_model]):
            # Align padding tokens between tokenizer and model
            _model.config.pad_token_id = tokenizer.pad_token_id

            # Use ChatML format if the tokenizer doesn't already have a chat template
            if tokenizer.chat_template is None:
                updated_model, updated_tokenizer = setup_chat_format(_model, tokenizer)
                # Actually store the updated model
                if idx == 0:
                    policy = updated_model
                elif idx == 1:
                    value_model = updated_model
                else:
                    reward_model = updated_model
                tokenizer = updated_tokenizer

        ################
        # Training and Evaluation
        ################
        trainer = ContinualPPOEWCTrainer(
            args=training_args,
            processing_class=tokenizer,
            model=policy,
            ref_model=ref_policy,
            reward_model=reward_model,
            value_model=value_model,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split],
            peft_config=peft_config,
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
                trainer.log({'dataset': {'name': script_args.dataset_name}})  # type: ignore[dict-item]
            metrics['dataset'] = i
            print(f'Evaluation metrics for dataset {i}: {metrics}')
            trainer.log_metrics(f'eval/dataset/{i}', metrics)
            trainer.save_metrics('eval', metrics)

            # Log metrics to WandB
            if training_args.local_rank in (None, -1, 0):
                wb.log({'eval': {'last': metrics}})  # type: ignore[attr-defined]
                wb.log({f'task/{custom_repo_name}/last': metrics})  # type: ignore[attr-defined]

        # Save model checkpoint and optionally push
        last_dir = os.path.join(training_args.output_dir, 'last')
        policy.save_pretrained(last_dir)
        tokenizer.save_pretrained(last_dir)

        value_model_dir = os.path.join(last_dir, 'value_model')
        os.makedirs(value_model_dir, exist_ok=True)
        value_model.save_pretrained(value_model_dir, safe_serialization=False)

        trainer.accelerator.wait_for_everyone()

        if training_args.push_to_hub:
            trainer.push_to_hub(
                model_name=custom_repo_name,
                dataset_name='Continual_PPO_EWC_' + clean_dataset_name + '_' + str(i),
            )

    print('Training completed for all tasks!')


if __name__ == '__main__':
    dataclass_types = (ContinualPPOEWCArguments, ContinualPPOEWCConfig, ModelConfig)
    parser = TrlParser(dataclass_types)
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
