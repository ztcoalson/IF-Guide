"""
Required set-up for influence function computation.
Prepares models for running EK-FAC and defines the language modeling task
(in our case, cross-entropy training and query loss).
"""
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from kronfluence.task import Task

BATCH_TYPE = Dict[str, torch.Tensor]

class LanguageModelingTask(Task):
    def __init__(
        self, 
        config: dict,
    ) -> None:
        super().__init__()
        self.config = config

    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))
        labels = batch["labels"][..., 1:].contiguous()
        if not sample:
            summed_loss = F.cross_entropy(logits, labels.view(-1), reduction="sum", ignore_index=-100)
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()
                masks = labels.view(-1) == -100
                sampled_labels[masks] = -100
            summed_loss = F.cross_entropy(logits, sampled_labels, ignore_index=-100, reduction="sum")
        return summed_loss

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        shift_labels = batch["labels"][..., 1:].contiguous().view(-1)
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        return F.cross_entropy(logits, shift_labels, ignore_index=-100, reduction="sum")

    def get_influence_tracked_modules(self) -> List[str]:
        total_modules = []
        num_layers = self.config['model']['num_layers']

        if self.config['model']['family'] == "pythia":

            for i in range(num_layers):
                    total_modules.append(f"gpt_neox.layers.{i}.mlp.dense_h_to_4h")
                    total_modules.append(f"gpt_neox.layers.{i}.mlp.dense_4h_to_h")

                    total_modules.append(f"gpt_neox.layers.{i}.attention.query_key_value")
                    total_modules.append(f"gpt_neox.layers.{i}.attention.dense")   

        elif self.config['model']['family'] == "llama-3":

            for i in range(num_layers):
                    total_modules.append(f"model.layers.{i}.mlp.gate_proj")
                    total_modules.append(f"model.layers.{i}.mlp.up_proj")
                    total_modules.append(f"model.layers.{i}.mlp.down_proj")

                    total_modules.append(f"model.layers.{i}.self_attn.q_proj")
                    total_modules.append(f"model.layers.{i}.self_attn.k_proj")
                    total_modules.append(f"model.layers.{i}.self_attn.v_proj")
                    total_modules.append(f"model.layers.{i}.self_attn.o_proj")
        else:
            raise NotImplementedError(
                f"Model family {self.config['model']['family']} not supported for influence tracking."
                " Please add the required modules to the `get_influence_tracked_modules` method in `utils/tasks.py`."
            )

        return total_modules

    def get_attention_mask(self, batch: BATCH_TYPE) -> torch.Tensor:
        return batch["attention_mask"]