"""Strategy dispatcher.

Backends:
  - "naive": pure PyTorch reference (testing/debugging oracle only)
  - "cpu":   first-class C++ kernel (requires compiled extension)
  - "gpu"/"taichi": Taichi kernel from one source. Which backend runs is chosen
    by ensure_taichi(): CUDA if available, else CPU; set TARCH=vulkan for a
    Vulkan backend (an Intel/AMD iGPU counts). Vulkan and CPU take ordinary CPU
    tensors — PyTorch has no "vulkan" device type, so nothing here needs widening
    to use them.

The naive path is NEVER an implicit fallback: a requested compiled backend
that cannot run is an actionable error, not a silent downgrade.
"""

import torch

from memory_machines.ops.dw_scale import dw_scale as _compiled_op
from memory_machines.ops.references.dw_scale_reference import reference_dw_scale
from memory_machines.ops.references.causal_conv1d_reference import (
    reference_causal_conv1d_fwd,
    reference_causal_conv1d_update,
    reference_causal_conv1d_update_states,
)


def dispatch_dw_scale(x: torch.Tensor, weight: torch.Tensor, backend: str = "naive") -> torch.Tensor:
    if backend == "naive":
        return reference_dw_scale(x, weight)
    if backend in ("cpu", "gpu", "taichi", "compiled"):
        return _compiled_op(x, weight)
    raise ValueError(f"Unknown backend: '{backend}'. Valid: ['naive', 'cpu', 'gpu', 'taichi']")


def dispatch_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    backend: str = "naive",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Causal depthwise conv1d dispatcher.

    Only ``backend="naive"`` (pure-PyTorch reference) exists until the
    compiled kernel lands. A requested compiled backend is an actionable
    error, never a silent downgrade.
    """
    if backend == "naive":
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
    if backend in ("cpu", "gpu", "taichi", "compiled"):
        raise NotImplementedError(f"backend='{backend}' has no causal_conv1d kernel yet (TODO); use backend='naive'")
    raise ValueError(f"Unknown backend: '{backend}'. Valid: ['naive', 'cpu', 'gpu', 'taichi']")


def dispatch_causal_conv1d_update(
    x: torch.Tensor,
    cache: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    activation: str | None = None,
    backend: str = "naive",
) -> tuple[torch.Tensor, torch.Tensor]:
    if backend == "naive":
        return reference_causal_conv1d_update(x=x, cache=cache, weight=weight, bias=bias, residual=residual, activation=activation)
    if backend in ("cpu", "gpu", "taichi", "compiled"):
        raise NotImplementedError(f"backend='{backend}' has no causal_conv1d kernel yet (TODO); use backend='naive'")
    raise ValueError(f"Unknown backend: '{backend}'. Valid: ['naive', 'cpu', 'gpu', 'taichi']")
