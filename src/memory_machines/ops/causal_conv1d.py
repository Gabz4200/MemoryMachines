"""Public op facade: torch.library custom operator for causal_conv1d.

The compiled (Taichi/C++) kernel is NOT written yet — this module reserves
the op namespace, fake shapes, and autograd wiring so the reference path
and the future kernel share one entry point. Any ``"compiled"`` request
currently raises an actionable error; use ``backend="naive"``.
"""

from __future__ import annotations

import torch

from memory_machines.ops.references.causal_conv1d_reference import (
    reference_causal_conv1d_fwd,
    reference_causal_conv1d_update,
    reference_causal_conv1d_update_states,
)
from memory_machines.ops.registry import OpSpec, cpu_loader, register_op, taichi_loader


@torch.library.custom_op("memory_machines::causal_conv1d", mutates_args=())
def causal_conv1d_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    raise NotImplementedError("memory_machines::causal_conv1d has no compiled kernel yet (TODO); use backend='naive'")


@torch.library.register_fake("memory_machines::causal_conv1d")
def _causal_conv1d_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    if weight.dim() != 2:
        raise ValueError(f"weight must be [D, W], got {tuple(weight.shape)}")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(f"x last dim ({x.shape[-1]}) must match weight dim 0 ({weight.shape[0]})")
    if bias is not None and tuple(bias.shape) != (weight.shape[0],):
        raise ValueError(f"bias must be [D], got {tuple(bias.shape)}")
    if residual is not None and residual.shape != x.shape:
        raise ValueError(f"residual shape {tuple(residual.shape)} != x shape {tuple(x.shape)}")
    return torch.empty_like(x)


def causal_conv1d_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # Single implementation of the naive path; the dispatcher delegates here
    # so validation and semantics live in exactly one place (see dispatch.py).
    y = reference_causal_conv1d_fwd(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        initial_state=initial_state,
        activation=activation,
        cu_seqlens=cu_seqlens,
    )
    final_state = None
    if output_final_state:
        final_state = reference_causal_conv1d_update_states(
            x=x,
            state_len=weight.shape[1],
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )
    return y, final_state


register_op(
    OpSpec(
        name="causal_conv1d",
        naive=causal_conv1d_naive,
        torch_op=causal_conv1d_op,
        compiled_cpu=cpu_loader("causal_conv1d"),
        compiled_taichi=taichi_loader("causal_conv1d"),
        description="Depthwise causal Conv1d forward pass.",
    )
)

register_op(
    OpSpec(
        name="causal_conv1d_update",
        naive=reference_causal_conv1d_update,
        compiled_cpu=cpu_loader("causal_conv1d_update"),
        compiled_taichi=taichi_loader("causal_conv1d_update"),
        description="Depthwise causal Conv1d single-token decode step.",
    )
)

__all__ = ["causal_conv1d_naive", "causal_conv1d_op"]
