import functools
import inspect
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator, PartialState
from datasets import Dataset
from packaging import version
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
)
from transformers.trainer_callback import TrainerCallback
from trl import ScriptArguments
from trl.trainer.ppo_config import PPOConfig
from trl.trainer.ppo_trainer import (
    PPOTrainer,
    batch_generation,
    get_reward,
    unwrap_model_for_generation,
)


@dataclass
class ContinualPPOArguments(ScriptArguments):
    value_model_path: str = field(
        default='AIF-Gen/Qwen/Qwen2-0.5B-Reward/debug_REWARD',
        metadata={'help': 'Path to the value model or a HuggingFace model path.'},
    )
    dataset_name: str = field(
        default='debug',
        metadata={'help': 'The name or path of the continual dataset to use.'},
    )
    wandb_project: Optional[str] = field(
        default='AIFGen-ppo-continual-test',
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
class ContinualPPOConfig(PPOConfig):
    mock: bool = field(
        default=False,
        metadata={'help': 'Whether to use mock dataset.'},
    )
    eval_greedy_policy: bool = field(
        default=False,
        metadata={'help': 'Whether to use greedy policy for evaluation.'},
    )


class ContinualPPOTrainer(PPOTrainer):
    # Shared accelerator instance across all trainer instances
    shared_accelerator: Optional[Accelerator] = None
    accelerator: Accelerator  # now non-optional after creation

    def __init__(
        self,
        args: Optional[PPOConfig] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        reward_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        train_dataset: Optional[Dataset] = None,
        value_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        data_collator: Optional[DataCollator] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        callbacks: Optional[list[TrainerCallback]] = None,
        peft_config: Optional[dict] = None,
    ):
        # catching this here to test our implementation of the configs
        if args is None:
            raise ValueError('`args` cannot be None')

        # Initialize metrics tracking storage
        self._stored_metrics: Dict = defaultdict(lambda: defaultdict(list))
        self.current_task = 'task_0'  # Default task name
        self.is_final_eval = False

        if ContinualPPOTrainer.shared_accelerator is None:
            ContinualPPOTrainer.shared_accelerator = Accelerator(
                gradient_accumulation_steps=args.gradient_accumulation_steps
            )
        self.accelerator = ContinualPPOTrainer.shared_accelerator

        train_dataset = self.preprocess_dataset(
            train_dataset, processing_class, args.dataset_num_proc
        )
        eval_dataset = self.preprocess_dataset(
            eval_dataset, processing_class, args.dataset_num_proc
        )

        super().__init__(
            args,
            processing_class,
            model,
            ref_model,
            reward_model,
            train_dataset,
            value_model,
            data_collator,
            eval_dataset,
            optimizers,
            callbacks,
            peft_config,
        )

        # # No need for anything else as PPO itself is already set up with the reward model
        self.accelerator = (
            ContinualPPOTrainer.shared_accelerator
        )  # turn the accelerator back to the shared one

    def create_accelerator_and_postprocess(self) -> None:
        # Only initialize a new Accelerator if one does not exist
        if ContinualPPOTrainer.shared_accelerator is None:
            super().create_accelerator_and_postprocess()
            ContinualPPOTrainer.shared_accelerator = self.accelerator
        else:
            # Reuse the shared accelerator
            self.accelerator = ContinualPPOTrainer.shared_accelerator
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

    def evaluate(self) -> Dict[str, float]:
        """Custom evaluation method for PPO. Generates completions from the evaluation dataloader,
        computes rewards using the reward model, and returns aggregated metrics.
        """
        mode: bool = self.model.training
        self.model.eval()
        eval_metrics = defaultdict(list)
        processing_class = self.processing_class

        # Configure generation settings (using either greedy or sampling strategy)
        if self.args.eval_greedy_policy:
            generation_config = GenerationConfig(
                max_new_tokens=self.args.response_length,
                top_k=None,
                do_sample=False,
            )
        else:
            generation_config = GenerationConfig(
                max_new_tokens=self.args.response_length,
                temperature=self.args.temperature + 1e-7,
                top_k=0.0,
                top_p=1.0,
                do_sample=True,
            )

        with torch.no_grad():
            for batch in self.eval_dataloader:
                # Move the query tokens to the correct device
                query = batch['input_ids'].to(self.accelerator.device)
                context_length = query.shape[1]

                with unwrap_model_for_generation(
                    self.model, self.accelerator
                ) as unwrapped_model:
                    core_model = unwrapped_model

                    if hasattr(core_model, 'policy'):
                        core_model = core_model.policy
                    elif hasattr(core_model, 'model'):
                        core_model = core_model.model
                    elif hasattr(core_model, 'policy_model'):
                        core_model = core_model.policy_model
                    else:
                        break  # No policy attribute found - will not be able to generate

                    query_response, _ = batch_generation(
                        core_model,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )

                    response = query_response[:, context_length:]

                # Combine the original query with the generated response and calculate the reward
                postprocessed_query_response = torch.cat((query, response), dim=1)
                _, score, _ = get_reward(
                    self.reward_model,
                    postprocessed_query_response,
                    processing_class.pad_token_id,
                    context_length,
                )
                # Gather scores from all processes and append to our metrics
                eval_metrics['score'].extend(
                    self.accelerator.gather_for_metrics(score).float().cpu().numpy()
                )

        self.model.train(mode)
        # Calculate the aggregated metrics
        aggregated_metrics = {
            'eval_' + k: float(np.mean(v)) for k, v in eval_metrics.items()
        }

        # Store metrics in our tracking system
        for key, value in aggregated_metrics.items():
            self.store_metrics({key: value}, train_eval='eval')

        return aggregated_metrics

    # FROM COPR: https://github.com/ComplexData-MILA/AIF-Gen/blob/a8ff4900a4415391c6ddbd003384b5a11b95254c/benchmarks/adapt_copr/copr_trainer.py
    def store_metrics(
        self,
        metrics: Dict,
        train_eval: Optional[str] = None,
        split: Optional[str] = None,
    ) -> None:
        """Override store_metrics to organize metrics by task."""
        if not hasattr(self, '_stored_metrics'):
            self._stored_metric: Dict = defaultdict(lambda: defaultdict(list))

        # Use train_eval if provided, otherwise fall back to split
        effective_split = train_eval if train_eval is not None else split

        # Add task prefix to eval metrics for better organization
        if effective_split == 'eval' and hasattr(self, 'current_task'):
            # Store metrics both in default location and in task-specific location
            for key, value in metrics.items():
                self._stored_metrics[effective_split][key].append(value)
                task_key = f'{self.current_task}/{key}'
                self._stored_metrics['task'][task_key].append(value)

                # For final evaluation, also store in eval.last
                if self.is_final_eval:
                    self._stored_metrics['eval.last'][key].append(value)
        else:
            # For training metrics, keep original behavior but also add to task
            for key, value in metrics.items():
                self._stored_metrics[effective_split or 'train'][key].append(value)

                if hasattr(self, 'current_task'):
                    task_key = f'{self.current_task}/{key}'
                    self._stored_metrics['task'][task_key].append(value)

    def preprocess_dataset(
        self,
        dataset: Dataset,
        processing_class: PreTrainedTokenizerBase,
        dataset_num_proc: int,
    ) -> Dataset:
        # The code is from TRL PPO script https://github.com/huggingface/trl/blob/main/examples/scripts/ppo/ppo.py
        dataset_text_field = 'prompt'

        def tokenize(element: dict) -> dict[str, list[int]]:
            outputs = processing_class(
                element[dataset_text_field],
                padding=False,
            )
            return {'input_ids': outputs['input_ids']}

        def prepare_dataset(ds: Dataset) -> Dataset:
            return ds.map(
                tokenize,
                batched=True,
                remove_columns=ds.column_names,
                num_proc=dataset_num_proc,
            )

        # Compute only on main process for faster data processing.
        with PartialState().local_main_process_first():
            dataset = prepare_dataset(dataset)
        return dataset

    def log(
        self, logs: Dict[str, Union[float, str]], start_time: Optional[float] = None
    ) -> None:
        """Reorganize logs into train, task, and eval.last categories."""
        # store the metrics first because the parent's log method will clear them
        self.store_metrics(logs, 'train') if 'loss' in logs else self.store_metrics(
            logs, 'eval'
        )

        train_eval = 'train' if 'loss' in logs else 'eval'

        # Process stored metrics
        processed_logs = {}

        # Process the train/eval metrics
        for key, metrics in self._stored_metrics.get(train_eval, {}).items():
            if metrics:
                if any(isinstance(m, dict) for m in metrics):
                    # For dictionary metrics, just pass the first one through
                    processed_logs[key] = metrics[0]
                else:
                    # For numerical metrics, compute mean as before
                    processed_logs[key] = (
                        torch.tensor(metrics, dtype=torch.float32).mean().item()
                    )

        # Process task-specific metrics
        for key, metrics in self._stored_metrics.get('task', {}).items():
            if metrics:
                if any(isinstance(m, dict) for m in metrics):
                    # For dictionary metrics, just pass the first one through
                    processed_logs[f'task/{key}'] = metrics[0]
                else:
                    # For numerical metrics, compute mean as before
                    processed_logs[f'task/{key}'] = (
                        torch.tensor(metrics, dtype=torch.float32).mean().item()
                    )

        # Process eval.last metrics (final evaluation results)
        if self.is_final_eval and 'eval.last' in self._stored_metrics:
            for key, metrics in self._stored_metrics['eval.last'].items():
                if metrics:
                    if any(isinstance(m, dict) for m in metrics):
                        processed_logs[f'eval.last/{key}'] = metrics[0]
                    else:
                        processed_logs[f'eval.last/{key}'] = (
                            torch.tensor(metrics, dtype=torch.float32).mean().item()
                        )

        # Clear stored metrics after processing
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # Update the logs with our processed logs
        logs.update(processed_logs)

        # Call the parent's log method
        if version.parse(transformers.__version__) >= version.parse('4.47.0.dev0'):
            return Trainer.log(self, logs, start_time)
        else:  # transformers<=4.46
            return Trainer.log(self, logs)

    def set_task(self, task_name: str) -> 'ContinualPPOTrainer':
        """Set the current task name for better metric organization."""
        self.current_task = task_name
        return self

    def mark_final_eval(self, is_final: bool = True) -> 'ContinualPPOTrainer':
        """Mark that the next evaluation will be the final one for the current task."""
        self.is_final_eval = is_final
        return self

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ) -> None:
        """Save the model, dealing with the case where it's a PEFT model without a policy attribute."""
        # Store the original model
        original_model = self.model

        # For PEFT models (which lack .policy attribute), use the model directly
        if hasattr(self.model, 'base_model'):
            # PEFT model case - don't try to access .policy
            pass  # Keep the model as is
        elif hasattr(self.model, 'policy'):
            # Standard PPO case - use the policy as in the original implementation
            self.model: nn.Module = self.model.policy
        elif hasattr(self.model, 'policy_model'):
            # Standard PPO case - use the policy_model as in the original implementation
            self.model = self.model.policy_model

        # Call the parent class's save_model
        if output_dir is None:
            output_dir = self.args.output_dir

        Trainer.save_model(self, output_dir, _internal_call)

        # Restore the original model
        self.model = original_model
