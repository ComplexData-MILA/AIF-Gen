from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import deepspeed
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from benchmarks.dpo.continual_dpo_trainer import (
    ContinualDPOArguments,
    ContinualDPOConfig,
    ContinualDPOTrainer,
)


@dataclass
class ContinualDPOEWCArguments(ContinualDPOArguments):
    """Arguments for Continual DPO training with EWC regularization."""

    wandb_project: Optional[str] = field(
        default='AIFGen-dpo-EWC-continual-test',
        metadata={'help': 'Override the default WandB project name.'},
    )


@dataclass
class ContinualDPOEWCConfig(ContinualDPOConfig):
    """Configuration for Continual DPO training with EWC regularization."""

    ewc_lambda: float = field(
        default=100.0,
        metadata={
            'help': 'EWC regularization strength. Higher values give stronger regularization.'
        },
    )


class ContinualDPOEWCTrainer(ContinualDPOTrainer):
    """DPO Trainer enhanced with Elastic Weight Consolidation (EWC) for continual learning.

    EWC keeps a memory of parameter importance from previous tasks and penalizes
    changes to important parameters when learning new tasks.
    """

    # Class-level variables to store Fisher Information and old parameters across tasks
    class_fisher_information: Dict[str, torch.Tensor] = {}
    class_old_params: Dict[str, torch.Tensor] = {}
    current_task_index: int = 0

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        reward_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[ContinualDPOEWCConfig] = None,
        **kwargs: Any,
    ):
        super().__init__(model, ref_model, reward_model, args, **kwargs)

        # Store EWC-specific parameters
        self.ewc_lambda = args.ewc_lambda if args is not None else 100.0

        # Track if we're on the first task
        is_first_task = ContinualDPOEWCTrainer.current_task_index == 0
        if is_first_task:
            # Initialize empty dictionaries for first task
            ContinualDPOEWCTrainer.class_fisher_information = {}
            ContinualDPOEWCTrainer.class_old_params = {}

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        """Compute the DPO loss with additional EWC regularization to prevent
        catastrophic forgetting of previously learned tasks.
        """
        # Regular DPO loss calculation
        regular_loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # Skip EWC loss on first task since there's nothing to preserve yet
        is_first_task = ContinualDPOEWCTrainer.current_task_index == 0
        if is_first_task:
            return (regular_loss, outputs) if return_outputs else regular_loss

        # Calculate EWC regularization loss
        ewc_loss = self.compute_ewc_loss()

        # Combine losses
        total_loss = regular_loss + ewc_loss

        self.log(
            {
                'ewc_loss': ewc_loss.item(),
                'dpo_loss': regular_loss.item(),
                'total_loss': total_loss.item(),
            }
        )

        return (total_loss, outputs) if return_outputs else total_loss

    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute the EWC regularization loss.

        This loss penalizes changes to parameters that were important for previous tasks,
        as determined by their Fisher information matrix.

        Returns:
            EWC regularization loss tensor
        """
        if not ContinualDPOEWCTrainer.class_fisher_information:
            # No previous tasks, so no regularization needed
            return torch.tensor(0.0, device=self.accelerator.device)

        # Calculate the EWC penalty for each parameter
        model = self.accelerator.unwrap_model(self.model)
        ewc_loss = torch.tensor(0.0, device=self.accelerator.device)
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.class_fisher_information:
                continue
            # self.accelerator.print(name, param.shape)
            with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                if self.accelerator.is_main_process:
                    # Get the Fisher information and old parameter values
                    fisher = ContinualDPOEWCTrainer.class_fisher_information[name].to(
                        self.accelerator.device
                    )
                    old_param = ContinualDPOEWCTrainer.class_old_params[name].to(
                        self.accelerator.device
                    )

                    # Calculate squared distance weighted by Fisher information
                    delta = param - old_param
                    ewc_loss = ewc_loss + (fisher * delta.pow(2)).sum()

                    # Apply the EWC lambda coefficient and return
                    ewc_loss = 0.5 * self.ewc_lambda * ewc_loss
                else:
                    # Non-main processes should not compute EWC loss
                    ewc_loss = torch.tensor(0.0, device=self.accelerator.device)

        ewc_loss = self.accelerator.reduce(ewc_loss, 'mean')
        return ewc_loss

    def store_current_parameters(self) -> Dict[str, torch.Tensor]:
        """Store the current model parameters.

        Returns:
            Dictionary mapping parameter names to their current values
        """
        model = self.accelerator.unwrap_model(self.model)
        old_params = {}
        for name, param in model.named_parameters():
            with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                if self.accelerator.is_main_process:
                    if param.requires_grad:
                        old_params[name] = param.data.clone().detach()
        return old_params

    def compute_fisher_information(
        self, num_samples: int = 120
    ) -> Dict[str, torch.Tensor]:
        """Compute Fisher Information matrix for the current model parameters.

        Args:
            num_samples: Number of samples to use for Fisher computation

        Returns:
            Dictionary mapping parameter names to their Fisher information values
        """
        # Get unwrapped model for computing Fisher
        model = self.accelerator.unwrap_model(self.model)

        # Initialize fisher information dictionary
        fisher_info = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param)

        # Create a dataloader for Fisher estimation
        fisher_dataloader = self.get_train_dataloader()

        # Collect samples for Fisher estimation
        sample_count = 0
        for batch in fisher_dataloader:
            if sample_count >= num_samples:
                break

            # Check what keys are available in the batch (for debugging)
            batch_keys = list(batch.keys())

            # Try to determine the batch size from available keys
            batch_size = None
            for key in ['input_ids', 'chosen_input_ids', 'policy_input_ids']:
                if (
                    key in batch
                    and hasattr(batch[key], 'shape')
                    and len(batch[key].shape) > 0
                ):
                    batch_size = batch[key].shape[0]
                    break

            if batch_size is None:
                print(
                    f'Warning: Could not determine batch size. Available keys: {batch_keys}'
                )
                batch_size = 1  # Default fallback

            # Forward pass with gradients
            model.zero_grad()

            try:
                loss, _ = super(ContinualDPOEWCTrainer, self).compute_loss(
                    model, batch, return_outputs=True
                )

                # Check if loss requires gradient
                if not loss.requires_grad:
                    print(
                        "Warning: Loss doesn't require gradients. Adding requires_grad=True"
                    )
                    loss = loss.clone().detach().requires_grad_(True)

                self.accelerator.backward(loss)

                # Accumulate squared gradients as Fisher information estimate
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher_info[name] += param.grad.detach().pow(2)
            except Exception as e:
                print(f'Error during Fisher computation: {e}')
                continue

            # Safely increment sample count
            sample_count += batch_size

        # Normalize by the number of samples
        if sample_count > 0:
            for name in fisher_info.keys():
                fisher_info[name] /= sample_count
        else:
            print('Warning: No samples were processed for Fisher computation')

        print(f'Computed Fisher information for {sample_count} examples')
        return fisher_info

    def train(self) -> Any:
        """Override train method to incorporate EWC regularization."""
        # Regular training
        result = super().train()

        # After training completes, update the Fisher information and old parameters
        # for the next task
        self.accelerator.print(
            'Computing Fisher information matrix for the next task...'
        )

        # Calculate and log EWC loss statistics before computing new Fisher information
        if ContinualDPOEWCTrainer.current_task_index > 0:
            ewc_loss = self.compute_ewc_loss()
            # Log EWC loss details
            self.log(
                {
                    'ewc_stats/total_loss': ewc_loss.item(),
                    'ewc_stats/per_param_avg': ewc_loss.item()
                    / sum(
                        p.numel() for p in self.model.parameters() if p.requires_grad
                    ),
                    'ewc_stats/lambda': self.ewc_lambda,
                    'ewc_stats/task_index': ContinualDPOEWCTrainer.current_task_index,
                }
            )
            self.accelerator.print(
                f'EWC loss for task {ContinualDPOEWCTrainer.current_task_index}: {ewc_loss.item():.4f}'
            )

        # Compute new Fisher information and parameters
        ContinualDPOEWCTrainer.class_fisher_information = (
            self.compute_fisher_information()
        )
        ContinualDPOEWCTrainer.class_old_params = self.store_current_parameters()

        # Increment task index for next time
        ContinualDPOEWCTrainer.current_task_index += 1

        return result
