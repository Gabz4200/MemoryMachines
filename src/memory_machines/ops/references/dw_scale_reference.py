"""Pure PyTorch reference oracle for dw_scale.

Ground truth for parity tests. Reached ONLY via dispatch(backend="naive") or
explicitly by tests — never as a silent fallback from a compiled backend.
"""

import torch


def reference_dw_scale(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if x.shape != weight.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {weight.shape}")
    return x * weight
