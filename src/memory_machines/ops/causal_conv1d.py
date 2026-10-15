"""Public op facade: torch.library custom operator for causal_conv1d.

The compiled (Taichi/C++) kernel is NOT written yet — this module reserves
the op namespace, fake shapes, and autograd wiring so the reference path
and the future kernel share one entry point. Any ``"compiled"`` request
currently raises an actionable error; use ``backend="naive"``.
"""

import torch

from memory_machines.ops.references.causal_conv1d_reference import (
    reference_causal_conv1d_fwd,
    reference_causal_conv1d_update,
    reference_causal_conv1d_update_states,
)


@torch.library.custom_op("memory_machines::causal_conv1d", mutates_args=())
def causal_conv1d_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    raise NotImplementedError("memory_machines::causal_conv1d has no compiled kernel yet (TODO); use backend='naive'")


@torch.library.register_fake("memory_machines::causal_conv1d")
def _causal_conv1d_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
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


__all__ = ["causal_conv1d_naive", "causal_conv1d_op"]
