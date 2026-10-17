"""Strategy dispatcher.

Backends (case-insensitive, whitespace-tolerant):
  - "naive": pure PyTorch reference (testing/debugging oracle only)
  - "cpu"/"gpu"/"taichi"/"compiled": compiled kernel entry point. These are
    currently aliases for "run the compiled op if available, else raise" —
    physical device placement is chosen inside ``ensure_taichi()``
    (CUDA if available, else CPU; ``TARCH=vulkan`` for Vulkan), not by the
    label. A requested compiled backend that cannot run is an actionable
    error, not a silent downgrade.

The naive path is NEVER an implicit fallback: a requested compiled backend
that cannot run is an actionable error, not a silent downgrade.
"""

import torch

from memory_machines.ops.causal_conv1d import causal_conv1d_naive
from memory_machines.ops.interfaces import ElementwiseOp
from memory_machines.ops.references.causal_conv1d_reference import (
    reference_causal_conv1d_update,
)
from memory_machines.ops.references.dw_scale_reference import reference_dw_scale

# Static protocol wiring (checked by the type checker, not at runtime). The
# compiled facade is imported lazily inside ``dispatch_dw_scale`` so merely
# importing this dispatcher never registers its torch.library ops.
_reference_dw_scale_op: ElementwiseOp = reference_dw_scale

_VALID_BACKENDS = ("naive", "cpu", "gpu", "taichi", "compiled")


def _normalize_backend(backend: str) -> str:
    if not isinstance(backend, str):
        raise TypeError(f"backend must be a str, got {type(backend).__name__}")
    norm = backend.strip().lower()
    if norm not in _VALID_BACKENDS:
        raise ValueError(f"Unknown backend: '{backend}'. Valid: {list(_VALID_BACKENDS)}")
    return norm


def dispatch_dw_scale(x: torch.Tensor, weight: torch.Tensor, backend: str = "naive") -> torch.Tensor:
    backend = _normalize_backend(backend)
    if backend == "naive":
        return reference_dw_scale(x, weight)
    # Lazy import: importing the facade registers torch.library ops, so defer
    # it until a compiled backend is actually requested.
    from memory_machines.ops.dw_scale import dw_scale as compiled_op

    return compiled_op(x, weight)


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
    error, never a silent downgrade. The naive path delegates to
    :func:`causal_conv1d_naive` so semantics live in one place.
    """
    backend = _normalize_backend(backend)
    if backend == "naive":
        return causal_conv1d_naive(
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            output_final_state=output_final_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
        )
    raise NotImplementedError(f"backend='{backend}' has no causal_conv1d kernel yet (TODO); use backend='naive'")


def dispatch_causal_conv1d_update(
    x: torch.Tensor,
    cache: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    activation: str | None = None,
    backend: str = "naive",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-token decode-step dispatcher.

    Returns ``(y, cache)`` where ``cache`` is updated **in place** and
    returned aliased (``y_cache is cache``); callers must not treat the
    return as a fresh tensor.
    """
    backend = _normalize_backend(backend)
    if backend == "naive":
        return reference_causal_conv1d_update(x=x, cache=cache, weight=weight, bias=bias, residual=residual, activation=activation)
    raise NotImplementedError(f"backend='{backend}' has no causal_conv1d kernel yet (TODO); use backend='naive'")
