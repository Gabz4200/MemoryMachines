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


def dispatch_dw_scale(x: torch.Tensor, weight: torch.Tensor, backend: str = "naive") -> torch.Tensor:
    if backend == "naive":
        return reference_dw_scale(x, weight)
    if backend in ("cpu", "gpu", "taichi", "compiled"):
        return _compiled_op(x, weight)
    raise ValueError(f"Unknown backend: '{backend}'. Valid: ['naive', 'cpu', 'gpu', 'taichi']")
