"""Structural subtyping for swappable components (PEP 544).

Strangers plug in by conforming structurally — no nominal inheritance, no
runtime duck typing with try/except TypeError.
"""

from typing import Protocol, runtime_checkable

import torch

from project.models.io import ModelInput, ModelOutput


@runtime_checkable
class Backbone(Protocol):
    """Any pure nn.Module that speaks the ModelInput -> ModelOutput contract."""

    def __call__(self, inputs: ModelInput) -> ModelOutput: ...


@runtime_checkable
class Op(Protocol):
    """Any primitive op with a device-agnostic forward."""

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor: ...
