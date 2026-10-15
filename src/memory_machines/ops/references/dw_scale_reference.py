"""Pure PyTorch reference oracle for dw_scale.

Ground truth for parity tests. Reached ONLY via dispatch(backend="naive") or
explicitly by tests — never as a silent fallback from a compiled backend.

Contract (shared with the compiled path): inputs must share shape, dtype,
and device and be 2D contiguous. Unlike the compiled path this reference
accepts any device but still requires equal shapes (dtypes must match;
no implicit promotion, so parity with the kernel holds exactly).
"""

import torch


def reference_dw_scale(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError(f"dw_scale requires 2D inputs, got {tuple(x.shape)} vs {tuple(weight.shape)}")
    if x.shape != weight.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {weight.shape}")
    if x.dtype != weight.dtype:
        raise ValueError(f"dtype mismatch: {x.dtype} vs {weight.dtype} (require equal dtypes)")
    if x.device != weight.device:
        raise ValueError(f"device mismatch: {x.device} vs {weight.device}")
    return x * weight
