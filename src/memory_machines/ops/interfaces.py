"""Narrow op Protocols — only what implementations need."""

from typing import Protocol

import torch


class ElementwiseOp(Protocol):
    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor: ...
