"""Structural subtyping (PEP 544) for swappable components across the codebase.

Strangers plug in by conforming structurally — no nominal inheritance, no
runtime duck typing with try/except TypeError. These Protocols are the
single source of truth for the contracts that ``models``, ``blocks``, ``ops``,
``training`` and ``integrations`` rely on; concrete classes do not need to
inherit from them.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn

from memory_machines.models.io import ModelInput, ModelOutput


@runtime_checkable
class HasResetParameters(Protocol):
    def reset_parameters(self) -> None: ...


@runtime_checkable
class TokenMixer(Protocol):
    """Any encoder/decoder token-mixing block operating on (B, L, D) tensors."""

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor: ...


@runtime_checkable
class FeedForward(Protocol):
    """Any feed-forward sub-layer operating on (B, L, D) tensors."""

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class Backbone(Protocol):
    """Any pure nn.Module speaking the ModelInput -> ModelOutput contract."""

    def __call__(self, inputs: ModelInput | torch.Tensor) -> ModelOutput: ...

    def get_input_embeddings(self) -> nn.Module: ...

    def set_input_embeddings(self, value: nn.Module) -> None: ...


@runtime_checkable
class EncoderLayer(Protocol):
    """An encoder layer: token mixer followed by a dense FFN."""

    token_mixer: TokenMixer
    ffn: FeedForward

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class GlobalKVProjection(Protocol):
    """Projects encoder-final states into decoder global K/V (Eq. 1)."""

    def forward(self, encoder_final: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


@runtime_checkable
class Decoder(Protocol):
    """Causal decoder consuming encoder context."""

    def decode_full_context(
        self,
        hidden_states: torch.Tensor,
        encoder_final: torch.Tensor,
        position_ids: torch.Tensor | None = ...,
        *,
        start_pos: int | None = ...,
        seeded_store: bool = ...,
        exit_threshold: float | None = ...,
        num_masked_tokens: int | None = ...,
    ) -> torch.Tensor: ...

    def prefill(
        self,
        hidden_states: torch.Tensor,
        encoder_final: torch.Tensor,
        position_ids: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...

    def reset_parameters(self) -> None: ...


@runtime_checkable
class UpProjection(Protocol):
    """Widens (possibly narrow) decoder states back to hidden size."""

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class LightningSystem(Protocol):
    """Minimal LightningModule-shaped system consumed by the trainer."""

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def configure_optimizers(self) -> Any: ...


@runtime_checkable
class DataModule(Protocol):
    def setup(self, stage: str | None = ...) -> None: ...

    def train_dataloader(self) -> Any: ...

    def val_dataloader(self) -> Any: ...


@runtime_checkable
class Trainer(Protocol):
    def fit(self, model: Any, datamodule: Any = ...) -> None: ...


@runtime_checkable
class OpImpl(Protocol):
    """Any op implementation callable with (x, weight)."""

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor: ...


__all__ = [
    "Backbone",
    "DataModule",
    "Decoder",
    "EncoderLayer",
    "FeedForward",
    "GlobalKVProjection",
    "HasResetParameters",
    "LightningSystem",
    "OpImpl",
    "TokenMixer",
    "Trainer",
    "UpProjection",
]
