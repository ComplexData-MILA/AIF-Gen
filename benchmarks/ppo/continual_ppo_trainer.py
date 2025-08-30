import functools
import gc
import inspect
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator, PartialState
from accelerate.utils import broadcast
from datasets import Dataset
from packaging import version
from torch import Tensor
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
    Trainer,
    TrainerCallback,
    TrainerControl,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import (
    CallbackHandler,
    ExportableState,
    PrinterCallback,
)
from transformers.utils import is_peft_available
from trl import ScriptArguments
from trl.core import masked_mean, masked_whiten
from trl.models import create_reference_model
from trl.trainer.ppo_config import PPOConfig
from trl.trainer.ppo_trainer import (
    PolicyAndValueWrapper,
    PPOTrainer,
    batch_generation,
    get_reward,
    unwrap_model_for_generation,
)
from trl.trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    peft_module_casting_to_bf16,
    prepare_deepspeed,
    selective_log_softmax,
    truncate_response,
)

if is_peft_available():
    from peft import PeftModel, get_peft_model

INVALID_LOGPROB = 1.0


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
    current_task_index: Optional[int] = None
    policy_value_models: Any  # the policy and value model wrapper
    ds_wrapped_models: Any  # TODO work with this after deepspeed is initialized
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
        # Basic setup and validation
        if args is None:
            raise ValueError('`args` cannot be None')
        if ref_model is model:
            raise ValueError(
                '`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the '
                'same as `model`, you must make a copy of it, or `None` if you use peft.'
            )
        if model is None or isinstance(model, str):
            raise ValueError(
                'model must be provided as a torch model (not as a string or None)'
            )
        if value_model is None and train_dataset is None:
            raise ValueError('train_dataset must be provided (got None)')
        if processing_class is None:
            raise ValueError('processing_class must be provided')
        if train_dataset is None:
            raise ValueError('train_dataset must be provided')

        # Initialize task tracking
        self._stored_metrics: Dict = defaultdict(lambda: defaultdict(list))
        self.current_task = (
            f'task_{ContinualPPOTrainer.current_task_index}'
            if ContinualPPOTrainer.current_task_index is not None
            else 'task_0'
        )

        # Set up task index tracking
        is_first_task = False
        if ContinualPPOTrainer.current_task_index is None:
            ContinualPPOTrainer.current_task_index = 0
            is_first_task = True
        else:
            ContinualPPOTrainer.current_task_index += 1
        self.is_final_eval = False

        # Store basic configuration
        self.args = args
        self.processing_class = processing_class
        self.policy_model: Optional[Union[PreTrainedModel, nn.Module]] = model

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)
        self.data_collator = data_collator

        # Setup stop token handling
        if args.stop_token and args.stop_token_id:
            raise ValueError('You cannot set both `stop_token` and `stop_token_id`.')
        elif args.stop_token:
            if args.stop_token == 'eos':
                self.policy_model.generation_config.eos_token_id = (
                    self.stop_token_id
                ) = processing_class.eos_token_id
            else:
                raise ValueError(
                    f"Unknown `stop_token` {args.stop_token}. Allowed values are: `'eos'` and `None` (no stop token)."
                )
        else:
            self.policy_model.generation_config.eos_token_id = self.stop_token_id = (
                args.stop_token_id
            )  # None or int

        if not is_peft_available() and peft_config is not None:
            raise ImportError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_confg, we merge and unload it first
            if isinstance(self.policy_model, PeftModel):
                self.policy_model = self.policy_model.merge_and_unload()

            # get peft model with the given config
            self.policy_model = get_peft_model(self.policy_model, peft_config)
            if args.bf16 and getattr(self.policy_model, 'is_loaded_in_4bit', False):
                peft_module_casting_to_bf16(self.policy_model)

        self.is_peft_model = is_peft_available() and isinstance(
            self.policy_model, PeftModel
        )

        # Set adapter names
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        # Set up reference model - only initialize on first task
        if is_first_task:
            if ref_model:
                self.ref_model = ref_model
            elif self.is_peft_model:
                self.ref_model = None
            else:
                self.ref_model = create_reference_model(self.policy_model)

            ContinualPPOTrainer.class_ref_model = self.ref_model

        else:
            # For subsequent tasks, reuse the reference model
            self.ref_model = ContinualPPOTrainer.class_ref_model

        # Always process new datasets for each task
        self.reward_model = reward_model
        self.train_dataset = self.preprocess_dataset(
            train_dataset, processing_class, args.dataset_num_proc
        )
        self.eval_dataset = self.preprocess_dataset(
            eval_dataset, processing_class, args.dataset_num_proc
        )
        self.train_dataset_len: int = train_dataset.num_rows
        self.value_model = value_model if is_first_task else None
        self.optimizer, self.lr_scheduler = optimizers

        # For transformers >= 4.47
        self.optimizer_cls_and_kwargs = None

        # Calculate batch sizes
        if args.total_episodes is None:
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)

        # Setup accelerator - shared across all tasks
        if ContinualPPOTrainer.shared_accelerator is None:
            accelerator = Accelerator(
                gradient_accumulation_steps=args.gradient_accumulation_steps
            )
            self.accelerator = accelerator
            ContinualPPOTrainer.shared_accelerator = accelerator
        else:
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

        # Configure batch size parameters
        args.world_size = self.accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size,
            args.num_mini_batches,
            '`batch_size` must be a multiple of `num_mini_batches`',
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size,
            args.num_mini_batches,
            '`local_batch_size` must be a multiple of `num_mini_batches`',
        )
        if args.whiten_rewards:
            if args.local_mini_batch_size < 8:
                raise ValueError(
                    f'Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening'
                )

        # Training scheduling
        args.num_total_batches = math.ceil(args.total_episodes / args.batch_size)
        time_tensor = torch.tensor(int(time.time()), device=self.accelerator.device)
        broadcast(time_tensor, 0).item()
        # args.run_name = f'{args.exp_name}__{args.seed}__{time_int}'
        self.local_seed = args.seed + self.accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(
                1, args.num_total_batches // args.num_sample_generations
            )
        self.local_dataloader_batch_size = args.local_batch_size

        # Setup models and optimizers - different handling for first vs subsequent tasks
        if is_first_task:
            # First task: Initialize everything
            for module in [
                self.policy_model,
                self.ref_model,
                self.value_model,
                self.reward_model,
            ]:
                if module is not None:
                    disable_dropout_in_model(module)

            # Create policy and value model wrapper
            self.model = PolicyAndValueWrapper(self.policy_model, self.value_model)
            ContinualPPOTrainer.policy_value_models = self.model
            self.model.config = self.policy_model.config  # needed for pushing to hub
        else:
            # Subsequent tasks: Reuse existing model
            self.model = ContinualPPOTrainer.policy_value_models
            self.model.config = self.policy_model.config  # needed for pushing to hub

        # Always create optimizer and scheduler for each task
        self.create_optimizer_and_scheduler(num_training_steps=args.num_total_batches)

        # Setup trainer callbacks
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(
            self.args.report_to
        )
        self.callbacks = (
            default_callbacks if callbacks is None else default_callbacks + callbacks
        )
        self.callback_handler = CallbackHandler(
            self.callbacks,
            self.model,
            self.processing_class,
            self.optimizer,
            self.lr_scheduler,
        )
        self.add_callback(
            PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK
        )
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb
                for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = (
            getattr(self.accelerator.state, 'deepspeed_plugin', None) is not None
        )
        self.is_fsdp_enabled = (
            getattr(self.accelerator.state, 'fsdp_plugin', None) is not None
        )

        # Create output directories if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Add model tags if supported
        if hasattr(self.model, 'add_model_tags'):
            self.model.add_model_tags(self._tag_names)

        # Setup dataloaders for current task
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,
        )
        torch.manual_seed(args.seed)  # Sync random states

        # Different preparations for first vs subsequent tasks
        if is_first_task:
            self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
                self.model, self.optimizer, self.dataloader
            )
            ContinualPPOTrainer.ds_wrapped_models = self.model
        else:
            # For subsequent tasks, only prepare optimizer and dataloader
            self.optimizer, self.dataloader = self.accelerator.prepare(
                self.optimizer, self.dataloader
            )
            # Reuse the model from the first task
            self.model = ContinualPPOTrainer.ds_wrapped_models

        torch.manual_seed(self.local_seed)  # Reset local seed

        # Prepare eval dataloader
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )
        self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)

        # Handle DeepSpeed setup
        if self.is_deepspeed_enabled:
            # Always prepare reward model for DeepSpeed
            self.reward_model = prepare_deepspeed(
                self.reward_model,
                args.per_device_train_batch_size,
                args.fp16,
                args.bf16,
            )
            print(f'self.ref model is {self.ref_model}')
            # Ref model handling for DeepSpeed
            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError(
                        'No reference model and model is not a Peft model.'
                    )
            elif is_first_task:
                # Only prepare ref_model on first task
                self.ref_model = prepare_deepspeed(
                    self.ref_model,
                    args.per_device_train_batch_size,
                    args.fp16,
                    args.bf16,
                )
                ContinualPPOTrainer.class_ref_model = self.ref_model
            else:
                # Reuse prepared ref_model on subsequent tasks
                self.ref_model = ContinualPPOTrainer.class_ref_model
        else:
            # Non-DeepSpeed path
            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError(
                        'No reference model and model is not a Peft model.'
                    )
            elif is_first_task:
                # Only move ref_model to device on first task
                self.ref_model = self.ref_model.to(self.accelerator.device)  # type: ignore
                ContinualPPOTrainer.class_ref_model = self.ref_model
            else:
                # Reuse ref_model on subsequent tasks
                self.ref_model = ContinualPPOTrainer.class_ref_model

            # Always move reward model to device
            self.reward_model = self.reward_model.to(self.accelerator.device)  # type: ignore

    def train(self) -> None:
        """Override train method to preserve reference model."""
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model

        # Store a backup reference to the ref_model before training
        original_ref_model = self.ref_model
        ref_policy = original_ref_model

        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator() -> DataLoader:
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print('===training policy===')
        start_time = time.time()
        stats_shape = (
            args.num_ppo_epochs,
            args.num_mini_batches,
            args.gradient_accumulation_steps,
        )
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(
                    self.state.max_steps * args.logging_steps
                )
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(
                    self.state.max_steps * args.eval_steps
                )
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(
                    self.state.max_steps * args.save_steps
                )
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data['input_ids'].to(device)
                context_length = queries.shape[1]
                responses: Union[List[Tensor], Tensor] = []
                postprocessed_responses: Union[List[Tensor], Tensor] = []
                logprobs: Union[List[Tensor], Tensor] = []
                ref_logprobs: Union[List[Tensor], Tensor] = []
                scores: Union[List[Tensor], Tensor] = []
                sequence_lengths: Union[List[Tensor], Tensor] = []
                values: Union[List[Tensor], Tensor] = []
                with unwrap_model_for_generation(
                    self.model,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as unwrapped_model:
                    query_responses, logitss = batch_generation(
                        unwrapped_model.policy,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                for i in range(
                    0, queries.shape[0], args.local_rollout_forward_batch_size
                ):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[
                        i : i + args.local_rollout_forward_batch_size
                    ]
                    response = query_response[:, context_length:]
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    logprob = selective_log_softmax(logits, response)
                    del logits
                    torch.cuda.empty_cache()

                    if ref_policy is None:
                        with self.null_ref_context():
                            ref_output = forward(
                                model.policy,
                                query_response,
                                processing_class.pad_token_id,
                            )
                    else:
                        ref_output = forward(
                            ref_policy, query_response, processing_class.pad_token_id
                        )
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax(ref_logits, response)
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if (
                        self.stop_token_id is not None
                    ):  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat(
                        (query, postprocessed_response), 1
                    )
                    sequence_length = (
                        first_true_indices(
                            postprocessed_response == processing_class.pad_token_id
                        )
                        - 1
                    )
                    unwrapped_value_model = accelerator.unwrap_model(model).value_model
                    full_value, _, _ = get_reward(
                        unwrapped_value_model,
                        query_response,
                        processing_class.pad_token_id,
                        context_length,
                    )
                    value = full_value[:, context_length - 1 : -1].squeeze(-1)
                    _, score, _ = get_reward(
                        reward_model,
                        postprocessed_query_response,
                        processing_class.pad_token_id,
                        context_length,
                    )

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                    values.append(value)
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                values = torch.cat(values, 0)
                del (logprob, full_value, value, score)
                torch.cuda.empty_cache()
                gc.collect()

                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                contain_eos_token = torch.any(
                    postprocessed_responses == self.processing_class.eos_token_id,
                    dim=-1,
                )
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(
                    responses.shape[1], device=responses.device
                ).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(
                    ref_logprobs, padding_mask, INVALID_LOGPROB
                )
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # 4. compute rewards
                kl = logprobs - ref_logprobs
                non_score_reward = -args.kl_coef * kl
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(
                    sequence_lengths_p1 < rewards.size(1),
                    sequence_lengths_p1,
                    sequence_lengths,
                )
                rewards[[actual_start, actual_end]] += scores

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(
                        rewards, mask=~padding_mask_p1, shift_mean=False
                    )
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(
                    0, args.local_batch_size, args.local_mini_batch_size
                ):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(
                        0, args.local_mini_batch_size, args.per_device_train_batch_size
                    ):
                        # with accelerator.accumulate(model):
                        micro_batch_end = (
                            micro_batch_start + args.per_device_train_batch_size
                        )
                        micro_batch_inds = mini_batch_inds[
                            micro_batch_start:micro_batch_end
                        ]
                        mb_advantage = advantages[micro_batch_inds]
                        mb_responses = responses[micro_batch_inds]
                        mb_query_responses = query_responses[micro_batch_inds]
                        mb_logprobs = logprobs[micro_batch_inds]
                        mb_return = returns[micro_batch_inds]
                        mb_values = values[micro_batch_inds]

                        output, vpred_temp = forward(
                            model, mb_query_responses, processing_class.pad_token_id
                        )
                        logits = output.logits[:, context_length - 1 : -1]
                        logits /= args.temperature + 1e-7
                        new_logprobs = selective_log_softmax(logits, mb_responses)
                        new_logprobs = torch.masked_fill(
                            new_logprobs,
                            padding_mask[micro_batch_inds],
                            INVALID_LOGPROB,
                        )
                        vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                        vpred = torch.masked_fill(
                            vpred, padding_mask_p1[micro_batch_inds], 0
                        )
                        vpredclipped = torch.clamp(
                            vpred,
                            mb_values - args.cliprange_value,
                            mb_values + args.cliprange_value,
                        )
                        vf_losses1 = torch.square(vpred - mb_return)
                        vf_losses2 = torch.square(vpredclipped - mb_return)
                        vf_loss_max = torch.max(vf_losses1, vf_losses2)
                        vf_loss = 0.5 * masked_mean(
                            vf_loss_max, ~padding_mask_p1[micro_batch_inds]
                        )
                        vf_clipfrac = masked_mean(
                            (vf_losses2 > vf_losses1).float(),
                            ~padding_mask_p1[micro_batch_inds],
                        )
                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        pg_losses = -mb_advantage * ratio
                        pg_losses2 = -mb_advantage * torch.clamp(
                            ratio, 1.0 - args.cliprange, 1.0 + args.cliprange
                        )
                        pg_loss_max = torch.max(pg_losses, pg_losses2)
                        pg_loss = masked_mean(
                            pg_loss_max, ~padding_mask[micro_batch_inds]
                        )
                        loss = pg_loss + args.vf_coef * vf_loss
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
                        with torch.no_grad():
                            pg_clipfrac = masked_mean(
                                (pg_losses2 > pg_losses).float(),
                                ~padding_mask[micro_batch_inds],
                            )
                            prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                            entropy = torch.logsumexp(logits, dim=-1) - torch.sum(
                                prob_dist * logits, dim=-1
                            )
                            approxkl = 0.5 * (logprobs_diff**2).mean()
                            approxkl_stats[
                                ppo_epoch_idx,
                                minibatch_idx,
                                gradient_accumulation_idx,
                            ] = approxkl
                            pg_clipfrac_stats[
                                ppo_epoch_idx,
                                minibatch_idx,
                                gradient_accumulation_idx,
                            ] = pg_clipfrac
                            pg_loss_stats[
                                ppo_epoch_idx,
                                minibatch_idx,
                                gradient_accumulation_idx,
                            ] = pg_loss
                            vf_loss_stats[
                                ppo_epoch_idx,
                                minibatch_idx,
                                gradient_accumulation_idx,
                            ] = vf_loss
                            vf_clipfrac_stats[
                                ppo_epoch_idx,
                                minibatch_idx,
                                gradient_accumulation_idx,
                            ] = vf_clipfrac
                            entropy_stats[
                                ppo_epoch_idx,
                                minibatch_idx,
                                gradient_accumulation_idx,
                            ] = entropy.mean()
                            ratio_stats[
                                ppo_epoch_idx,
                                minibatch_idx,
                                gradient_accumulation_idx,
                            ] = ratio.mean()
                    gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, vpred_temp, logits, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                        mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + scores.mean()
                eps = float(self.state.episode / (time.time() - start_time))
                metrics: Dict[str, Union[float, str]] = {}
                metrics['eps'] = eps
                metrics['objective/kl'] = (
                    self.accelerator.gather_for_metrics(mean_kl).mean().item()
                )
                metrics['objective/entropy'] = (
                    self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                )
                metrics['objective/non_score_reward'] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward)
                    .mean()
                    .item()
                )
                metrics['objective/rlhf_reward'] = (
                    self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                )
                metrics['objective/scores'] = (
                    self.accelerator.gather_for_metrics(scores.mean()).mean().item()
                )
                metrics['policy/approxkl_avg'] = (
                    self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                )
                metrics['policy/clipfrac_avg'] = (
                    self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                )
                metrics['loss/policy_avg'] = (
                    self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                )
                metrics['loss/value_avg'] = (
                    self.accelerator.gather_for_metrics(vf_loss_stats).mean().item()
                )
                metrics['val/clipfrac_avg'] = (
                    self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                )
                metrics['policy/entropy_avg'] = (
                    self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                )
                metrics['val/ratio'] = (
                    self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                )
                metrics['val/ratio_var'] = (
                    self.accelerator.gather_for_metrics(ratio_stats).var().item()
                )
                metrics['val/num_eos_tokens'] = (
                    (responses == processing_class.eos_token_id).sum().item()
                )
                metrics['lr'] = self.lr_scheduler.get_last_lr()[0]
                metrics['episode'] = self.state.episode
                self.state.epoch = (
                    self.state.episode / self.train_dataset_len
                )  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(
                args, self.state, self.control
            )
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(
                    self.args, self.state, self.control
                )

            # Don't delete reference to kl here
            del (
                mean_kl,
                mean_entropy,
                mean_non_score_reward,
                scores,
                metrics,
                non_score_reward,
            )
            torch.cuda.empty_cache()
            gc.collect()

            if (
                args.num_sample_generations > 0
                and (update - 1) % self.sample_generations_freq == 0
            ):
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()

            # Don't delete ref_logprobs here
            del (
                query_responses,
                responses,
                postprocessed_responses,
                logprobs,
                values,
                sequence_lengths,
                contain_eos_token,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
            )
            torch.cuda.empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

        # If ref_model was cleared during training, restore it
        if self.ref_model is None and original_ref_model is not None:
            print('Reference model was cleared during training - restoring')
            self.ref_model = original_ref_model
            ContinualPPOTrainer.class_ref_model = original_ref_model

        # Ensure the class variable is updated
        ContinualPPOTrainer.class_ref_model = self.ref_model
        if self.is_deepspeed_enabled:
            ContinualPPOTrainer.ds_wrapped_models = self.deepspeed
        else:
            ContinualPPOTrainer.ds_wrapped_models = self.model
        ContinualPPOTrainer.policy_value_models = self.model

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
            self.model = self.model.policy
        elif hasattr(self.model, 'policy_model'):
            # Standard PPO case - use the policy_model as in the original implementation
            self.model = self.model.policy_model

        # Call the parent class's save_model
        if output_dir is None:
            output_dir = self.args.output_dir

        Trainer.save_model(self, output_dir, _internal_call)

        # Restore the original model
        self.model = original_model
