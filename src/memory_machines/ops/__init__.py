"""Public ops package: re-exports for the stable entry points."""

from memory_machines.ops.attention import universal_pe_attention_naive, universal_pe_attention_op
from memory_machines.ops.causal_conv1d import causal_conv1d_naive, causal_conv1d_op
from memory_machines.ops.dispatch import (
    dispatch_causal_conv1d,
    dispatch_causal_conv1d_update,
    dispatch_universal_pe_attention,
)
from memory_machines.ops.registry import (
    OpRegistry,
    OpSpec,
    dispatch_op,
    get_op,
    list_ops,
    register_op,
)

__all__ = [
    "OpRegistry",
    "OpSpec",
    "causal_conv1d_naive",
    "causal_conv1d_op",
    "dispatch_causal_conv1d",
    "dispatch_causal_conv1d_update",
    "dispatch_op",
    "dispatch_universal_pe_attention",
    "get_op",
    "list_ops",
    "register_op",
    "universal_pe_attention_naive",
    "universal_pe_attention_op",
]
