"""Public ops package: re-exports for the stable entry points."""

from memory_machines.ops.causal_conv1d import causal_conv1d_naive, causal_conv1d_op
from memory_machines.ops.dispatch import dispatch_causal_conv1d, dispatch_causal_conv1d_update, dispatch_dw_scale
from memory_machines.ops.dw_scale import dw_scale

__all__ = [
    "causal_conv1d_naive",
    "causal_conv1d_op",
    "dispatch_causal_conv1d",
    "dispatch_causal_conv1d_update",
    "dispatch_dw_scale",
    "dw_scale",
]
