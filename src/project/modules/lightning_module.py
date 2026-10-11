"""LitResearchSystem: the imperative shell around a pure nn.Module.

The model is injected via DI — it never knows Lightning exists. This system
owns loss, step metrics, and optimizer configuration only.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn

from project.models.io import ModelInput, ModelOutput


class LitResearchSystem(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float = 1e-4, weight_decay: float = 1e-2):
        super().__init__()
        # Dependency injection: the model is passed in, not instantiated here.
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs: ModelInput) -> ModelOutput:
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs = ModelInput(
            input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask")
        )
        output: ModelOutput = self.model(inputs)
        loss = self.criterion(
            output.logits.view(-1, output.logits.size(-1)),
            batch["labels"].view(-1),
        )
        if output.aux_loss is not None:
            loss = loss + output.aux_loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = ModelInput(
            input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask")
        )
        output: ModelOutput = self.model(inputs)
        loss = self.criterion(
            output.logits.view(-1, output.logits.size(-1)),
            batch["labels"].view(-1),
        )
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Rule-based parameter groups: 2D weights decay, 1D biases/norms do not.
        decay = [p for p in self.model.parameters() if p.requires_grad and p.ndim >= 2]
        no_decay = [p for p in self.model.parameters() if p.requires_grad and p.ndim < 2]
        groups = [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=self.learning_rate)
