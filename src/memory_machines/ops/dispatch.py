"""Strategy dispatcher with an explicit compiled-first fallback chain.

Backends (case-insensitive, whitespace-tolerant):
  - "naive": pure PyTorch reference (testing/debugging oracle only).
  - "auto": try the compiled kernel first, fall back to the reference.
    Order: Taichi kernel -> CPU cpp extension -> pure PyTorch reference.
    A warning is emitted whenever the reference is used because the compiled
    kernel was unavailable.
  - "cpu"/"gpu"/"taichi"/"compiled": compiled kernel entry point. A requested
    compiled backend that cannot run raises an actionable error (no silent
    downgrade).

The ``naive`` path is NEVER an implicit fallback under ``auto``: the reference
is used only after every compiled option has been tried and failed, and each
downgrade surfaces an explicit ``UserWarning``.

The backend logic itself lives in :mod:`memory_machines.ops.registry`; these
functions are thin, typed adapters around ``registry.dispatch`` so every op
shares one implementation.
"""

from __future__ import annotations

import torch

# Import the op modules so they self-register with the op registry when this
# package is imported. They are not referenced by name here beyond that.
from memory_machines.ops import causal_conv1d as _causal_conv1d  # noqa: F401
from memory_machines.ops import attention as _attention  # noqa: F401

from memory_machines.ops.registry import dispatch_op


def dispatch_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Causal depthwise conv1d dispatcher with fallback to the reference.

    Args:
        backend: "naive" uses the reference directly; "auto" (default) tries
            compiled kernels then warns and uses the reference; compiled
            labels try the compiled path only and raise if unavailable.
    """
    return dispatch_op(
        "causal_conv1d",
        backend,
        x,
        weight,
        bias=bias,
        residual=residual,
        initial_state=initial_state,
        output_final_state=output_final_state,
        activation=activation,
        cu_seqlens=cu_seqlens,
    )


def dispatch_causal_conv1d_update(
    x: torch.Tensor,
    cache: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    activation: str | None = None,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-token decode-step dispatcher with fallback to the reference.

    Args:
        backend: "naive" uses the reference directly; "auto" (default) tries
            compiled kernels then warns and uses the reference; compiled
            labels try the compiled path only and raise if unavailable.
    """
    return dispatch_op(
        "causal_conv1d_update",
        backend,
        x,
        cache,
        weight,
        bias=bias,
        residual=residual,
        activation=activation,
    )


def dispatch_universal_pe_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cvpe_attn_bias: torch.Tensor | None = None,
    phi_vq: torch.Tensor | None = None,
    phi_vk: torch.Tensor | None = None,
    spatial_grid: torch.Tensor | None = None,
    ov_spatial_weight: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    bidirectional: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    frvm_chunk_size: int = 128,
    backend: str = "auto",
) -> torch.Tensor:
    """Universal PE attention dispatcher with fallback to the reference.

    Args:
        backend: "naive" uses the reference directly; "auto" (default) tries
            compiled kernels then warns and uses the reference; compiled
            labels try the compiled path only and raise if unavailable.
    """
    return dispatch_op(
        "universal_pe_attention",
        backend,
        query,
        key,
        value,
        cvpe_attn_bias=cvpe_attn_bias,
        phi_vq=phi_vq,
        phi_vk=phi_vk,
        spatial_grid=spatial_grid,
        ov_spatial_weight=ov_spatial_weight,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        bidirectional=bidirectional,
        scale=scale,
        enable_gqa=enable_gqa,
        frvm_chunk_size=frvm_chunk_size,
    )
