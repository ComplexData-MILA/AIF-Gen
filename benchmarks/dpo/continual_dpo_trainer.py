import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import PartialState
from accelerate.utils import gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    BaseImageProcessor,
    DataCollator,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl import DPOTrainer, ScriptArguments
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.utils import batch_generation, disable_dropout_in_model, get_reward
from typing_extensions import override

import wandb as wb


@dataclass
class ContinualDPOArguments(ScriptArguments):
    dataset_name: str = field(
        default='debug',
        metadata={'help': 'The name or path of the continual dataset to use.'},
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={
            'help': 'The directory containing the checkpoints to evaluate (used only in eval checkpoints script)'
        },
    )
    wandb_project: Optional[str] = field(
        default='AIFGen-dpo-continual-test',
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
class ContinualDPOConfig(DPOConfig):
    reward_model_path: Optional[str] = field(
        default=None,
        metadata={
            'help': 'The name or path to the reward models folder containing all rewards models for continual learning dataset.'
        },
    )
    mock: bool = field(
        default=False,
        metadata={'help': 'Whether to use mock dataset.'},
    )
    response_length: int = field(
        default=53,
        metadata={
            'help': 'Length of the response. Borrowed from PPOConfig and used only for evaluation.'
        },
    )
    temperature: float = field(
        default=0.7,
        metadata={
            'help': 'Temperature for sampling. Borrowed from PPOConfig and used only for evaluation, taken from OnPolicyConfig config'
        },
    )
    eval_greedy_policy: bool = field(
        default=False,
        metadata={'help': 'Whether to use greedy policy for evaluation.'},
    )
    eval_policy_metrics: bool = field(
        default=False,
        metadata={
            'help': 'Run reward-model policy evaluation only during the explicit task-end evaluation path.'
        },
    )
    log_completions: bool = field(
        default=False,
        metadata={
            'help': 'Log sampled completions only during the explicit task-end evaluation path.'
        },
    )
    completion_logging_batches: int = field(
        default=1,
        metadata={
            'help': 'Number of evaluation batches to sample when --log_completions is enabled.'
        },
    )
    enable_profiling: bool = field(
        default=False,
        metadata={'help': 'Enable profiling hooks for continual DPO runs.'},
    )
    profiling_steps: int = field(
        default=3,
        metadata={
            'help': 'Number of active steps to capture with torch profiler when profiling is enabled.'
        },
    )
    profile_memory: bool = field(
        default=False,
        metadata={'help': 'Enable CUDA memory tracing when profiling is enabled.'},
    )
    profile_output_dir: str = field(
        default='profiles/continual_dpo',
        metadata={'help': 'Output directory for torch profiler TensorBoard traces.'},
    )


class StepProfilingCallback(TrainerCallback):
    """Track per-step timing/memory metrics and optionally advance a torch profiler."""

    def __init__(
        self,
        profiler: Optional[Any] = None,
        profile_memory: bool = False,
    ) -> None:
        """Initialize optional step-level profiler and memory metric tracking.

        Args:
            profiler: Torch profiler instance to advance once per training step.
            profile_memory: Whether to collect CUDA allocated/reserved memory metrics.
        """
        self.profiler = profiler
        self.profile_memory = profile_memory
        self._step_start_time: Optional[float] = None
        self._step_time_total: float = 0.0
        self._steps: int = 0

    @override
    def on_step_begin(self, args, state, control, **kwargs):
        """Capture step start timestamp before each optimizer step."""
        self._step_start_time = time.perf_counter()
        return control

    @override
    def on_step_end(self, args, state, control, **kwargs):
        """Log per-step timing/memory metrics and advance profiler when available."""
        step_start_time = self._step_start_time
        self._step_start_time = None
        if step_start_time is None:
            return control

        elapsed = time.perf_counter() - step_start_time
        self._step_time_total += elapsed
        self._steps += 1

        world_size = max(1, getattr(args, 'world_size', 1))
        global_batch_size = args.per_device_train_batch_size * world_size
        logs: dict[str, float | int] = {
            'step': state.global_step,
            'profiling/step_time_s': float(elapsed),
            'profiling/step_time_avg_s': float(self._step_time_total / self._steps),
            'profiling/samples_per_sec': float(global_batch_size / max(elapsed, 1e-12)),
        }

        if self.profile_memory and torch.cuda.is_available():
            logs['profiling/gpu_memory_allocated_gb'] = float(
                torch.cuda.memory_allocated() / (1024**3)
            )
            logs['profiling/gpu_memory_reserved_gb'] = float(
                torch.cuda.memory_reserved() / (1024**3)
            )

        state.log_history.append(logs)

        if self.profiler is not None:
            self.profiler.step()
        return control


class ContinualDPOTrainer(DPOTrainer):
    @override
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        reward_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[DPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        peft_config: Optional[dict] = None,
    ):
        if args is None:
            raise ValueError('`args` cannot be None')

        eval_policy_dataset = eval_dataset

        super().__init__(
            model,
            ref_model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            processing_class,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
            peft_config,
        )

        self.reward_model = reward_model
        if self.reward_model is not None and not isinstance(self.reward_model, str):
            disable_dropout_in_model(self.reward_model)
            self.reward_model.eval()

        if eval_policy_dataset is not None:
            self.eval_policy_dataset = self.preprocess_policy_dataset(
                eval_policy_dataset
            )
            self.eval_policy_dataloader = self._build_eval_policy_dataloader(
                self.eval_policy_dataset
            )
        else:
            self.eval_policy_dataset = None
            self.eval_policy_dataloader = None

    def _build_eval_policy_dataloader(self, dataset: Dataset) -> DataLoader:
        data_collator = DataCollatorWithPadding(self.processing_class)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=data_collator,
            drop_last=True,
        )
        return self.accelerator.prepare(dataloader)

    def set_task_datasets(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        dataset_name: str = 'task',
    ) -> None:
        self.train_dataset = self._prepare_dataset(
            train_dataset,
            self.processing_class,
            self.args,
            f'{dataset_name}-train',
        )
        self.eval_dataset = None
        self.eval_policy_dataset = None
        self.eval_policy_dataloader = None

        if eval_dataset is not None:
            self.eval_dataset = self._prepare_dataset(
                eval_dataset,
                self.processing_class,
                self.args,
                f'{dataset_name}-eval',
            )
            self.eval_policy_dataset = self.preprocess_policy_dataset(eval_dataset)
            self.eval_policy_dataloader = self._build_eval_policy_dataloader(
                self.eval_policy_dataset
            )

    def set_reward_model(
        self,
        reward_model: Optional[Union[PreTrainedModel, nn.Module, str]],
    ) -> None:
        """Store a reward model for explicit evaluation-only flows.

        This does not move or DeepSpeed-wrap the reward model during training; it only updates
        the optional model used by ``reward_model_context`` when explicit policy evaluation or
        completion logging is requested.
        """
        self.reward_model = reward_model
        if self.reward_model is not None and not isinstance(self.reward_model, str):
            disable_dropout_in_model(self.reward_model)
            self.reward_model.eval()

    def preprocess_policy_dataset(self, dataset: Dataset) -> Dataset:
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

        with PartialState().local_main_process_first():
            dataset = prepare_dataset(dataset)
        return dataset

    @contextmanager
    def reward_model_context(
        self,
        reward_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
    ):
        active_reward_model = reward_model if reward_model is not None else self.reward_model
        if active_reward_model is None:
            yield None
            return

        loaded_in_context = False
        if isinstance(active_reward_model, str):
            active_reward_model = AutoModelForSequenceClassification.from_pretrained(
                active_reward_model,
                num_labels=1,
            )
            loaded_in_context = True

        disable_dropout_in_model(active_reward_model)
        active_reward_model.eval()

        first_parameter = next(active_reward_model.parameters(), None)
        original_device = (
            first_parameter.device if first_parameter is not None else self.accelerator.device
        )
        target_device = self.accelerator.device
        moved_to_accelerator = original_device != target_device
        if moved_to_accelerator:
            active_reward_model = active_reward_model.to(target_device)

        try:
            yield active_reward_model
        finally:
            if moved_to_accelerator:
                active_reward_model = active_reward_model.to(original_device)
            if loaded_in_context:
                del active_reward_model
            if torch.cuda.is_available() and (moved_to_accelerator or loaded_in_context):
                torch.cuda.empty_cache()

    def evaluate_policy(
        self,
        reward_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
    ) -> dict:
        """Evaluate the policy on the evaluation prompts with an optional reward model."""
        if self.eval_policy_dataloader is None:
            return {}

        mode = self.model.training
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
            generation_config = GenerationConfig(
                max_new_tokens=self.args.response_length,
                temperature=(self.args.temperature + 1e-7),
                top_k=0.0,
                top_p=1.0,
                do_sample=True,
            )

        with self.reward_model_context(reward_model) as active_reward_model:
            if active_reward_model is None:
                self.model.train(mode)
                return {}

            with torch.inference_mode():
                for batch in self.eval_policy_dataloader:
                    # `eval_policy_dataloader` is built from `preprocess_policy_dataset`, which stores prompts under
                    # `input_ids` for reward-based policy evaluation only.
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
                    postprocessed_query_response = torch.cat((query, response), 1)
                    _, score, _ = get_reward(
                        active_reward_model,
                        postprocessed_query_response,
                        processing_class.pad_token_id,
                        context_length,
                    )
                    eval_metrics['score'].extend(
                        self.accelerator.gather_for_metrics(score).float().cpu().numpy()
                    )

        self.model.train(mode)
        if not eval_metrics:
            return {}
        return {'eval_' + k: float(np.mean(v)) for k, v in eval_metrics.items()}

    def generate_completions_table(
        self,
        reward_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        max_batches: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """Generate a small sample of completions for explicit task-end inspection."""
        if self.eval_dataset is None:
            return None

        eval_dataloader = self.get_eval_dataloader()
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )
        table = defaultdict(list)
        batches_to_log = (
            max_batches if max_batches is not None else self.args.completion_logging_batches
        )
        if batches_to_log <= 0:
            return None

        mode = self.model.training
        self.model.eval()

        with self.reward_model_context(reward_model) as active_reward_model:
            with torch.inference_mode():
                with unwrap_model_for_generation(
                    self.model,
                    self.accelerator,
                    gather_deepspeed3_params=None,
                ) as unwrapped_model:
                    for batch_index, batch in enumerate(eval_dataloader):
                        if batch_index >= batches_to_log:
                            break

                        # `get_eval_dataloader()` uses the tokenized DPO eval dataset, where prompts are kept under
                        # `prompt_input_ids` together with the chosen/rejected preference targets.
                        query = batch['prompt_input_ids'].to(self.accelerator.device)
                        context_length = query.shape[1]
                        query_response, _ = batch_generation(
                            unwrapped_model,
                            query,
                            query.shape[0],
                            self.processing_class.pad_token_id,
                            generation_config,
                        )
                        response = query_response[:, context_length:]
                        postprocessed_query_response = torch.cat((query, response), 1)

                        if active_reward_model is not None:
                            _, score, _ = get_reward(
                                active_reward_model,
                                postprocessed_query_response,
                                self.processing_class.pad_token_id,
                                context_length,
                            )
                            table['score'].extend(
                                self.accelerator.gather_for_metrics(score)
                                .float()
                                .cpu()
                                .numpy()
                            )

                        queries = gather_object(
                            self.processing_class.batch_decode(
                                query,
                                skip_special_tokens=True,
                            )
                        )
                        responses = gather_object(
                            self.processing_class.batch_decode(
                                response,
                                skip_special_tokens=True,
                            )
                        )
                        table['query'].extend(queries)
                        table['model response'].extend(responses)

        self.model.train(mode)
        if not table:
            return None

        df = pd.DataFrame(table)
        if self.accelerator.is_main_process and wb.run is not None:
            wb.log({'completions': wb.Table(dataframe=df)})
        return df

    def _generate_completions(self) -> None:
        self.generate_completions_table(reward_model=self.reward_model)
