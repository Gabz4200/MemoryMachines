"""Typed forward contracts.

Model I/O crosses module boundaries as @dataclasses, never as anonymous
tuples or magic dicts. Adding a field is then a one-line change, not a
call-site scavenger hunt.
"""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ModelInput:
    """Canonical model input. Fields are optional unless the model requires them.

    Extend per domain: images (pixel_values), audio (input_values),
    sequences (input_ids). Keep every field optional so a single
    dataclass serves all model families.
    """

    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    state: Any = None
    inputs_embeds: torch.Tensor | None = None
    pixel_values: torch.Tensor | None = None


@dataclass
class ModelState:
    """Explicit recurrent / cache state. Never stored on the module itself."""

    tensors: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class ModelOutput:
    """Canonical model output; grows instead of breaking unpack callers."""

    logits: torch.Tensor
    new_state: Any = None
    aux_loss: torch.Tensor | None = None
    attn_weights: torch.Tensor | None = None
