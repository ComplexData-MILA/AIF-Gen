from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import accelerate
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
    """DPO Trainer enhanced with Elastic Weight Consolidation (EWC) for continual learning."""

    fisher: Dict[str, torch.Tensor] = {}
    old_params: Dict[str, torch.Tensor] = {}

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        reward_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[ContinualDPOEWCConfig] = None,
        **kwargs: Any,
    ):
        self.model_copy = deepcopy(model)  # Copy before deepspeed initialization
        self.ewc_lambda = args.ewc_lambda if args is not None else 100.0
        super().__init__(model, ref_model, reward_model, args, **kwargs)

    def train(self) -> Any:
        result = super().train()

        ContinualDPOEWCTrainer.old_params = {
            name: param.detach().clone()
            for name, param in self.accelerator.unwrap_model(
                self.model
            ).named_parameters()
            if param.requires_grad
        }

        if self.accelerator.is_main_process:
            fisher = self.compute_fisher()
        else:
            # The secondary processes need to have a dictionary initialized
            # on the current device, and with the corrct tensor shapes to enable
            # broadcast to _copy() the tensors.
            fisher = {
                name: torch.zeros_like(param, device=self.accelerator.device)
                for name, param in self.accelerator.unwrap_model(
                    self.model
                ).named_parameters()
                if param.requires_grad
            }

        # TODO: Double check that this broadcast is actually correct
        ContinualDPOEWCTrainer.fisher = accelerate.utils.broadcast(fisher)
        return result

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        dpo_loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        def compute_ewc_loss() -> torch.Tensor:
            ewc_loss = torch.tensor(0.0, device=self.accelerator.device)
            model = self.accelerator.unwrap_model(self.model)
            for name, param in model.named_parameters():
                if name in ContinualDPOEWCTrainer.fisher and param.requires_grad:
                    with (
                        deepspeed.zero.GatheredParameters([param], modifier_rank=None)
                        if hasattr(param, 'ds_id')
                        else nullcontext()
                    ):
                        fisher = ContinualDPOEWCTrainer.fisher[name].to(param.device)
                        old_param = ContinualDPOEWCTrainer.old_params[name].to(
                            param.device
                        )
                        ewc_loss += (fisher * (param - old_param).pow(2)).sum()
            return 0.5 * self.ewc_lambda * ewc_loss

        ewc_loss = compute_ewc_loss()
        total_loss = dpo_loss + ewc_loss
        self.log(
            {
                'ewc_loss': ewc_loss.item(),
                'dpo_loss': dpo_loss.item(),
                'total_loss': total_loss.item(),
            }
        )
        return (total_loss, outputs) if return_outputs else total_loss

    def compute_fisher(
        self, num_samples: int = 120, device: str = 'cuda:0'
    ) -> Dict[str, torch.Tensor]:
        # Computing fisher outside the deepspeed context
        model = deepcopy(self.model_copy)
        model.load_state_dict(
            self.accelerator.unwrap_model(self.model).state_dict(), strict=False
        )
        model = model.to(device)
        model.eval()

        fisher = {
            name: torch.zeros_like(param, device=device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        def guess_batch_size(batch: Any) -> int:
            batch_size = 1
            for key in ['input_ids', 'chosen_input_ids', 'policy_input_ids']:
                if (
                    key in batch
                    and hasattr(batch[key], 'shape')
                    and len(batch[key].shape) > 0
                ):
                    batch_size = batch[key].shape[0]
                    break
            return batch_size

        def move_to_device(batch: Any) -> Any:
            if isinstance(batch, dict):
                return {
                    k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()
                }
            elif isinstance(batch, (list, tuple)):
                return type(batch)(move_to_device(x) for x in batch)
            else:
                return batch.to(device) if hasattr(batch, 'to') else batch

        sample_count = 0
        for batch in self.get_train_dataloader():
            if sample_count >= num_samples:
                break

            sample_count += guess_batch_size(batch)
            batch = move_to_device(batch)

            model.zero_grad(set_to_none=True)
            loss = self.compute_dpo_loss_for_fisher(model, batch)
            loss.backward()

            for name, param in model.named_parameters():
                with (
                    deepspeed.zero.GatheredParameters([param], modifier_rank=None)
                    if hasattr(param, 'ds_id')
                    else nullcontext()
                ):
                    if param.grad is not None:
                        fisher[name] += param.grad.detach().clone().pow(2)

        for name in fisher:
            fisher[name] /= sample_count
        return fisher

    def compute_dpo_loss_for_fisher(
        self,
        model: Union[PreTrainedModel, nn.Module],
        batch: dict[str, Union[torch.Tensor, Any]],
    ) -> torch.Tensor:
        # The following is a patch of super().compute_loss which avoid metrics computation
        # and can be run outside of the accelerate context. It primarily specializes the
        # implementation of get_batch_loss_metrics to skip calls to `self.accelerator.gather_for_metrics`.
        with (
            torch.amp.autocast('cuda')
            if self._peft_has_been_casted_to_bf16
            else nullcontext()
        ):
            model_output = self.concatenated_forward(model, batch)

            if 'ref_chosen_logps' in batch and 'ref_rejected_logps' in batch:
                ref_chosen_logps = batch['ref_chosen_logps']
                ref_rejected_logps = batch['ref_rejected_logps']
            else:
                ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

            losses, _, _ = self.dpo_loss(
                model_output['chosen_logps'],
                model_output['rejected_logps'],
                ref_chosen_logps,
                ref_rejected_logps,
            )
            if self.args.rpo_alpha is not None:
                losses = losses + self.args.rpo_alpha * model_output['nll_loss']
            if self.use_weighting:
                losses = losses * model_output['policy_weights']
            if self.aux_loss_enabled:
                losses = losses + self.aux_loss_coef * model_output['aux_loss']
            return losses.mean().to(self.args.device)
