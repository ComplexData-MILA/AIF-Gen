import functools
import inspect
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator, PartialState
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from trl import GRPOConfig, GRPOTrainer, ScriptArguments
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import batch_generation, get_reward


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """Script arguments for the GRPO training script."""

    dataset_name: str = field(
        default='debug',
        metadata={'help': 'The name or path of the continual dataset to use.'},
    )
    wandb_project: Optional[str] = field(
        default='AIFGen-grpo-continual-test',
        metadata={'help': 'Override the default WandB project name.'},
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={'help': 'The WandB entity (team) to use.'},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={'help': 'The WandB run name.'},
    )

    def __post_init__(self) -> None:
        if self.wandb_project:
            os.environ['WANDB_PROJECT'] = self.wandb_project
        if self.wandb_entity:
            os.environ['WANDB_ENTITY'] = self.wandb_entity


@dataclass
class ContinualGRPOConfig(GRPOConfig):
    reward_model_path: Optional[str] = field(
        default='AIF-Gen/Qwen/Qwen2-0.5B-Reward/debug_REWARD',
        metadata={
            'help': 'Reward model id of a pretrained model hosted inside a model repo on huggingface.co or '
            'local path to a directory containing model weights saved using `PreTrainedModel.save_pretrained`.'
        },
    )
    mock: bool = field(
        default=False,
        metadata={'help': 'Whether to use mock dataset.'},
    )
    eval_greedy_policy: bool = field(
        default=False,
        metadata={'help': 'Whether to use greedy policy for evaluation.'},
    )
    dataset_num_proc: int = field(
        default=1,
        metadata={'help': 'Number of processes to use for dataset preprocessing.'},
    )
    response_length: int = field(
        default=53,
        metadata={
            'help': 'Length of the response. Borrowed from PPOCOnfig and used only for evaluation.'
        },
    )


class ContinualGRPOTrainer(GRPOTrainer):
    # Shared accelerator instance across all trainer instances
    shared_accelerator: Optional[Accelerator] = None
    accelerator: Accelerator  # now non-optional after creation

    def __init__(
        self,
        args: Optional[ContinualGRPOConfig] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        reward_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        peft_config: Optional[dict] = None,
    ):
        # catching this here to test our implementation of the configs
        if args is None:
            raise ValueError('`args` cannot be None')

        if ContinualGRPOTrainer.shared_accelerator is None:
            ContinualGRPOTrainer.shared_accelerator = Accelerator(
                gradient_accumulation_steps=args.gradient_accumulation_steps
            )
        self.accelerator = ContinualGRPOTrainer.shared_accelerator

        super().__init__(
            args=args,
            processing_class=processing_class,
            model=model,
            reward_funcs=reward_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
        )

        self.accelerator = (
            ContinualGRPOTrainer.shared_accelerator
        )  # turn the accelerator back to the shared one

        self.eval_policy_dataset = self.preprocess_policy_dataset(eval_dataset)
        # using the same data_collator as in PPO trainer
        data_collator = DataCollatorWithPadding(self.processing_class)
        self.eval_policy_dataloader = DataLoader(
            self.eval_policy_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        # Ensure accelerator is available
        # TODO remove the check once ruff issues are resolved
        # fmt: off
        assert self.accelerator is not None, 'Accelerator must be assigned before prepare()'
        # fmt: on
        self.eval_policy_dataloader = self.accelerator.prepare(
            self.eval_policy_dataloader
        )

    def create_accelerator_and_postprocess(self) -> None:
        # Only initialize a new Accelerator if one does not exist
        if ContinualGRPOTrainer.shared_accelerator is None:
            super().create_accelerator_and_postprocess()
            ContinualGRPOTrainer.shared_accelerator = self.accelerator
        else:
            # Reuse the shared accelerator
            self.accelerator = ContinualGRPOTrainer.shared_accelerator
            self.gather_function = self.accelerator.gather_for_metrics
            if (
                'use_gather_object'
                in inspect.signature(self.gather_function).parameters.keys()
            ):
                self.gather_function = functools.partial(
                    self.gather_function,
                    use_gather_object=self.args.eval_use_gather_object,
                )
            self.is_deepspeed_enabled = (
                getattr(self.accelerator.state, 'deepspeed_plugin', None) is not None
            )
            self.is_fsdp_enabled = (
                getattr(self.accelerator.state, 'fsdp_plugin', None) is not None
            )

    def preprocess_policy_dataset(self, dataset: Dataset) -> Dataset:
        # The code is from TRL PPO script https://github.com/huggingface/trl/blob/main/examples/scripts/ppo/ppo.py
        dataset_text_field = 'prompt'

        def tokenize(element: dict) -> dict[str, list[int]]:
            outputs = self.processing_class(
                element[dataset_text_field],
                padding=False,
            )
            return {'input_ids': outputs['input_ids']}

        def prepare_dataset(ds: Dataset) -> Dataset:
            return ds.map(
                tokenize,
                batched=True,
                remove_columns=ds.column_names,
                num_proc=self.args.dataset_num_proc,
            )

        # Compute only on main process for faster data processing.
        with PartialState().local_main_process_first():
            dataset = prepare_dataset(dataset)
        return dataset

    def evaluate_policy(self) -> dict:
        """Evaluate the policy using the evaluation policy dataloader.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        # The code is heavily based on the training loop of TRL PPOTrainer function https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L677
        mode = self.model.training
        # there is no self.model? TODO
        self.model.eval()
        eval_metrics = defaultdict(list)
        processing_class = self.processing_class
        if self.args.eval_greedy_policy:
            generation_config = GenerationConfig(
                max_new_tokens=self.args.response_length,
                top_k=None,
                do_sample=False,
            )
        else:
            # Using the same hyperpaprams as during PPO training
            generation_config = GenerationConfig(
                max_new_tokens=self.args.response_length,
                temperature=(self.args.temperature + 1e-7),
                top_k=0.0,
                top_p=1.0,
                do_sample=True,
            )

        with torch.no_grad():
            if self.eval_policy_dataloader is not None:
                for batch in self.eval_policy_dataloader:
                    query = batch['input_ids'].to(self.accelerator.device)
                    context_length = query.shape[1]
                    with unwrap_model_for_generation(
                        self.model,
                        self.accelerator,
                        gather_deepspeed3_params=None,
                    ) as unwrapped_model:
                        query_response, _ = batch_generation(
                            unwrapped_model,
                            query,
                            query.shape[0],
                            processing_class.pad_token_id,
                            generation_config,
                        )
                        response = query_response[:, context_length:]
                    postprocessed_response = response
                    postprocessed_query_response = torch.cat(
                        (query, postprocessed_response), 1
                    )
                    _, score, _ = get_reward(
                        # self.reward_model,
                        self.reward_funcs[0],
                        postprocessed_query_response,
                        processing_class.pad_token_id,
                        context_length,
                    )
                    eval_metrics['score'].extend(
                        self.accelerator.gather_for_metrics(score).float().cpu().numpy()
                    )
        self.model.train(mode)
        return {'eval_' + k: float(np.mean(v)) for k, v in eval_metrics.items()}

    def log(
        self, logs: dict[str, Union[float, dict]], start_time: Optional[float] = None
    ) -> None:
        """Log `logs` on the various objects watching training, including stored metrics."""
        train_eval = 'train' if 'loss' in logs else 'eval'
        print(f'Logging {train_eval} metrics...')
        if train_eval == 'eval':
            print('Computing policy metrics...')
            eval_policy_metrics = self.evaluate_policy()
            logs.update(eval_policy_metrics)
        return super().log(logs, start_time)
