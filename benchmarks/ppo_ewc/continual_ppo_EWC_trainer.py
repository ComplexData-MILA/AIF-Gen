import gc
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.trainer_callback import TrainerCallback
from trl.core import masked_mean, masked_whiten
from trl.trainer.ppo_config import PPOConfig

# Import utility functions from your codebase
from benchmarks.ppo.continual_ppo_trainer import (
    ContinualPPOArguments,
    ContinualPPOConfig,
    ContinualPPOTrainer,
    batch_generation,
    first_true_indices,
    forward,
    get_reward,
    selective_log_softmax,
    truncate_response,
    unwrap_model_for_generation,
)

INVALID_LOGPROB = 1.0


@dataclass
class ContinualPPOEWCArguments(ContinualPPOArguments):
    """Arguments for Continual PPO training with EWC regularization."""

    wandb_project: Optional[str] = field(
        default='AIFGen-ppo-EWC-continual-test',
        metadata={'help': 'Override the default WandB project name for EWC runs.'},
    )


@dataclass
class ContinualPPOEWCConfig(ContinualPPOConfig):
    """Configuration for Continual PPO training with EWC regularization."""

    ewc_lambda: float = field(
        default=100.0,
        metadata={
            'help': 'EWC regularization strength. Higher values give stronger regularization.'
        },
    )


class ContinualPPOEWCTrainer(ContinualPPOTrainer):
    # Class-level variables to store Fisher Information and old parameters across tasks
    class_fisher_information: Dict[str, torch.Tensor] = {}
    class_old_params: Dict[str, torch.Tensor] = {}

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
        if args is None:
            raise ValueError('Arguments are required for EWC training')
        # Call parent's init method
        super().__init__(
            args=args,
            processing_class=processing_class,
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            train_dataset=train_dataset,
            value_model=value_model,
            data_collator=data_collator,
            eval_dataset=eval_dataset,
            optimizers=optimizers,
            callbacks=callbacks,
            peft_config=peft_config,
        )
        # Store EWC-specific parameters
        self.ewc_lambda = args.ewc_lambda

        if self.current_task_index == 0:
            # Initialize empty dictionaries for first task
            ContinualPPOEWCTrainer.class_fisher_information = {}
            ContinualPPOEWCTrainer.class_old_params = {}

    def compute_fisher_information(
        self, num_samples: int = 128
    ) -> Dict[str, torch.Tensor]:
        """Compute Fisher Information matrix for the current model parameters."""
        self.model.train()  # Set to training mode to enable gradients
        device = self.accelerator.device
        fisher_info = {}

        # Initialize dictionary for accumulating gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param)

        # Create a sampler to randomly select examples
        sample_size = min(num_samples, len(self.train_dataset))
        sampler = torch.utils.data.RandomSampler(
            self.train_dataset, replacement=False, num_samples=sample_size
        )

        # Create a dataloader for the fisher computation
        fisher_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
        )

        total_processed = 0

        for batch in fisher_loader:
            # Move batch to device
            queries = batch['input_ids'].to(device)
            context_length = queries.shape[1]

            for i in range(
                0, queries.shape[0], self.args.local_rollout_forward_batch_size
            ):
                # Get batch components
                query = queries[i : i + self.args.local_rollout_forward_batch_size]

                # Generate responses without gradients
                with torch.no_grad():
                    with unwrap_model_for_generation(
                        self.model, self.accelerator
                    ) as unwrapped_model:
                        # Run forward pass and get logits
                        output = forward(
                            unwrapped_model.policy,
                            query,
                            self.processing_class.pad_token_id,
                        )

                        logits = output.logits[:, context_length - 1 :]
                        logits = logits / (self.args.temperature + 1e-7)

                        # Sample from the logits to generate responses
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        sampled_tokens = torch.multinomial(
                            probs[:, 0, :], num_samples=1
                        )
                        response = sampled_tokens

                        # For sequence length > 1, continue sampling
                        seq_length = min(self.args.response_length, logits.size(1))
                        for t in range(1, seq_length):
                            if t < logits.size(1):
                                next_token_probs = probs[:, t, :]
                                next_token = torch.multinomial(
                                    next_token_probs, num_samples=1
                                )
                                response = torch.cat([response, next_token], dim=1)

                # Process each example separately to avoid graph conflicts
                for j in range(response.size(0)):
                    # Compute a new forward pass for each example
                    single_query = query[j : j + 1]
                    self.model.zero_grad()

                    # New forward pass to create a fresh computation graph
                    with unwrap_model_for_generation(
                        self.model, self.accelerator
                    ) as unwrapped_model:
                        single_output = forward(
                            unwrapped_model.policy,
                            single_query,
                            self.processing_class.pad_token_id,
                        )

                        single_logits = single_output.logits[:, context_length - 1 :]
                        single_logits = single_logits / (self.args.temperature + 1e-7)

                        # Initialize sample_loss as a tensor, not an integer
                        valid_tokens = []
                        valid_probs = []

                        # Collect all valid tokens and their log probs
                        for t in range(min(response.size(1), single_logits.size(1))):
                            if t < single_logits.size(1):
                                token_idx = response[j, t]
                                token_logits = single_logits[0, t]
                                log_prob = torch.nn.functional.log_softmax(
                                    token_logits, dim=-1
                                )[token_idx]
                                valid_tokens.append(token_idx)
                                valid_probs.append(log_prob)

                        # Only proceed if we have valid tokens
                        if valid_probs:
                            # Sum all log probs to get total loss
                            sample_loss = -torch.sum(
                                torch.stack(valid_probs)
                            )  # Negative log likelihood

                            # Check if loss requires gradients
                            if sample_loss.requires_grad:
                                # Backpropagate with a fresh graph for each example
                                sample_loss.backward()

                                # Accumulate squared gradients in Fisher information
                                for name, param in self.model.named_parameters():
                                    if param.requires_grad and param.grad is not None:
                                        fisher_info[name] += param.grad.detach().pow(2)

                                total_processed += 1

        # Normalize by the number of samples
        if total_processed > 0:
            for name in fisher_info.keys():
                fisher_info[name] /= total_processed
        else:
            print('Warning: No samples were processed for Fisher computation')

        print(f'Computed Fisher information for {total_processed} examples')
        return fisher_info

    def store_current_parameters(self) -> Dict[str, torch.Tensor]:
        """Store the current model parameters.

        Returns:
            Dictionary mapping parameter names to their current values
        """
        old_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                old_params[name] = param.data.clone().detach()
        return old_params

    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute the EWC regularization loss.

        This loss penalizes changes to parameters that were important for previous tasks,
        as determined by their Fisher information matrix.

        Returns:
            EWC regularization loss tensor
        """
        if not ContinualPPOEWCTrainer.class_fisher_information:
            # No previous tasks, so no regularization needed
            return torch.tensor(0.0, device=self.accelerator.device)

        ewc_loss = torch.tensor(0.0, device=self.accelerator.device)

        # Calculate the EWC penalty for each parameter
        for name, param in self.model.named_parameters():
            if (
                name in ContinualPPOEWCTrainer.class_fisher_information
                and param.requires_grad
            ):
                # Get the Fisher information and old parameter values
                fisher = ContinualPPOEWCTrainer.class_fisher_information[name].to(
                    param.device
                )
                old_param = ContinualPPOEWCTrainer.class_old_params[name].to(
                    param.device
                )

                # Calculate squared distance weighted by Fisher information
                delta = param - old_param
                ewc_loss += (fisher * delta.pow(2)).sum()

        # Apply the EWC lambda coefficient and return
        return 0.5 * self.ewc_lambda * ewc_loss

    def train(self) -> None:
        """Override train method to incorporate EWC regularization."""
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

        accelerator.print('===training policy with EWC===')
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
        ewc_loss_stats = torch.zeros(stats_shape, device=device)  # Track EWC loss
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
                        with accelerator.accumulate(model):
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

                            # EWC modification: Add EWC regularization loss
                            ewc_loss = self.compute_ewc_loss()
                            loss = pg_loss + args.vf_coef * vf_loss + ewc_loss

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
                                # Record all statistics, including EWC loss
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
                                # Record EWC loss
                                ewc_loss_stats[
                                    ppo_epoch_idx,
                                    minibatch_idx,
                                    gradient_accumulation_idx,
                                ] = ewc_loss

                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, vpred_temp, logits, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
                        pg_loss, ewc_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
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
                # Add EWC loss to metrics
                metrics['loss/ewc_avg'] = (
                    self.accelerator.gather_for_metrics(ewc_loss_stats).mean().item()
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

            # Continue with rest of training loop

        # After training completes, update the Fisher information and old parameters
        # for the next task
        accelerator.print('Computing Fisher information matrix for the next task...')
        ContinualPPOEWCTrainer.class_fisher_information = (
            self.compute_fisher_information()
        )
        ContinualPPOEWCTrainer.class_old_params = self.store_current_parameters()

        # The rest of train() method (from parent class) remains unchanged
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
            self.class_ref_model = original_ref_model

        # Ensure the class variable is updated
        self.class_ref_model = self.ref_model
        if self.is_deepspeed_enabled:
            self.ds_wrapped_models = self.deepspeed
        else:
            self.ds_wrapped_models = self.model
        self.policy_value_models = self.model

    def update_fisher_and_params(self) -> None:
        """Explicitly update the Fisher information and parameter values.

        This can be called manually at specific points if needed, rather than
        waiting for the automatic update at the end of training.
        """
        self.accelerator.print('Updating Fisher information matrix and parameters...')
        ContinualPPOEWCTrainer.class_fisher_information = (
            self.compute_fisher_information()
        )
        ContinualPPOEWCTrainer.class_old_params = self.store_current_parameters()

    def get_parameter_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Return the most important parameters according to the Fisher information.

        Args:
            top_n: Number of top parameters to return

        Returns:
            Dictionary mapping parameter names to their importance scores
        """
        if not ContinualPPOEWCTrainer.class_fisher_information:
            return {}

        # Calculate importance for each parameter
        importance = {}
        for name, fisher in ContinualPPOEWCTrainer.class_fisher_information.items():
            # Total importance is the sum of all Fisher values
            importance[name] = fisher.sum().item()

        # Sort parameters by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        # Return top N parameters
        return dict(sorted_importance[:top_n])
