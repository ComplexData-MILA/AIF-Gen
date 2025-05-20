import functools
import inspect
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb as wb
from accelerate import Accelerator, PartialState
from accelerate.utils import gather_object
from datasets import Dataset
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from transformers import (
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
from trl.trainer.utils import (
    batch_generation,
    disable_dropout_in_model,
    get_reward,
    prepare_deepspeed,
)
from typing_extensions import override


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
            'help': 'Length of the response. Borrowed from PPOCOnfig and used only for evaluation.'
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


class ContinualDPOTrainer(DPOTrainer):
    # Shared accelerator instance across all trainer instances
    shared_accelerator: Optional[Accelerator] = None
    accelerator: Accelerator  # now non-optional after creation

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
        # setting a reward model only for evaluation purposes
        # The reward model setting code comes from TRL PPOTrainer https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L284
        self.reward_model = reward_model
        if self.reward_model is not None:
            disable_dropout_in_model(self.reward_model)
        if self.is_deepspeed_enabled:
            if self.reward_model is not None:
                self.reward_model = prepare_deepspeed(
                    self.reward_model,
                    args.per_device_train_batch_size,
                    args.fp16,
                    args.bf16,
                )
        else:
            # Ensure reward_model is a model instance (if given as a str then load it)
            if isinstance(self.reward_model, str):
                from transformers import AutoModelForSequenceClassification

                self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                    self.reward_model
                )
            # Ensure accelerator is ready. It should be set already by DPOTrainer.
            assert self.accelerator is not None, 'Accelerator must be initialized'
            if self.reward_model is not None:
                self.reward_model = self.reward_model.to(self.accelerator.device)

        if eval_policy_dataset is not None:
            self.eval_policy_dataset = self.preprocess_policy_dataset(
                eval_policy_dataset
            )
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

        else:
            self.eval_policy_dataset = None
            self.eval_policy_dataloader = None

    def create_accelerator_and_postprocess(self) -> None:
        # Only initialize a new Accelerator if one does not exist
        if ContinualDPOTrainer.shared_accelerator is None:
            super().create_accelerator_and_postprocess()
            ContinualDPOTrainer.shared_accelerator = self.accelerator
        else:
            # Reuse the shared accelerator
            self.accelerator = ContinualDPOTrainer.shared_accelerator
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
                for idx, batch in enumerate(self.eval_policy_dataloader):
                    print(
                        f'Processing batch {idx} out of {len(self.eval_policy_dataloader)}'
                    )
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
                        self.reward_model,
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
            if self.reward_model is not None:
                print('Computing policy metrics...')
                eval_policy_metrics = self.evaluate_policy()
                logs.update(eval_policy_metrics)

        # TODO: Only generation sample completions every x steps
        do_generate_completions = True
        if do_generate_completions:
            print('Generating completions...')
            self._generate_completions()
            torch.cuda.empty_cache()

        return super().log(logs, start_time)

    def _generate_completions(self) -> None:
        # Config from: https://github.com/huggingface/trl/blob/56e57662053e2d0cc6302dad404820b0c0ec6a91/trl/trainer/ppo_trainer.py#L688
        # generation_config = GenerationConfig(
        #     max_new_tokens=53,
        #     temperature=(0.01 + 1e-7),
        #     top_k=0.0,
        #     top_p=1.0,
        #     do_sample=True,
        # )
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(self.args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        self.model.eval()
        table = defaultdict(list)
        with torch.no_grad():
            with unwrap_model_for_generation(
                self.model,
                self.accelerator,
                gather_deepspeed3_params=None,
            ) as unwrapped_model:
                if self.eval_policy_dataloader is not None:
                    for batch in self.eval_policy_dataloader:
                        query = batch['input_ids']
                        context_length = query.shape[1]
                        query_response, _ = batch_generation(
                            unwrapped_model,
                            query,
                            query.shape[0],
                            self.processing_class.pad_token_id,
                            generation_config,
                        )
                        response = query_response[:, context_length:]
                        postprocessed_response = response
                        postprocessed_query_response = torch.cat(
                            (query, postprocessed_response), 1
                        )
                        _, score, _ = get_reward(
                            self.reward_model,
                            postprocessed_query_response,
                            self.processing_class.pad_token_id,
                            context_length,
                        )

                        queries = gather_object(
                            self.processing_class.batch_decode(
                                query, skip_special_tokens=True
                            )
                        )
                        responses = gather_object(
                            self.processing_class.batch_decode(postprocessed_response)
                        )
                        scores = (
                            self.accelerator.gather_for_metrics(score)
                            .float()
                            .cpu()
                            .numpy()
                        )
                        table['query'].extend(queries)
                        table['model response'].extend(responses)
                        table['score'].extend(scores)
                        break

        self.model.train()
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process or self.accelerator is None:
            print_rich_table(df.iloc[0 : 0 + 5])
            if wb.run is not None:
                wb.log({'completions': wb.Table(dataframe=df)})


def print_rich_table(df: pd.DataFrame) -> Table:
    console = Console(markup=False)
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.print(table)
