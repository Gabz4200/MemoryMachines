"""Objective losses. Kept out of models/ — a backbone computes no loss."""

import torch
import torch.nn as nn


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, targets, label_smoothing=self.smoothing)
