"""Public op facade: universal PE attention torch.library custom op + naive.

The compiled (Taichi / C++) kernel is NOT written yet — this module reserves
the op namespace, fake shapes, and (future) autograd wiring so the reference
path and the future kernel share one entry point. Any ``"compiled"`` request
currently raises an actionable error; use ``backend="naive"``.
"""

from __future__ import annotations

import torch

from memory_machines.ops.references.attention_reference import (
    universal_pe_scaled_dot_product_attention,
)
from memory_machines.ops.registry import OpSpec, cpu_loader, register_op, taichi_loader


@torch.library.custom_op("memory_machines::universal_pe_attention", mutates_args=())
def universal_pe_attention_op(
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
) -> torch.Tensor:
    raise NotImplementedError("memory_machines::universal_pe_attention has no compiled kernel yet (TODO); use backend='naive'")


@torch.library.register_fake("memory_machines::universal_pe_attention")
def _universal_pe_attention_fake(
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
) -> torch.Tensor:
    torch._check(query.dim() >= 3)
    torch._check(key.dim() >= 3)
    torch._check(value.dim() >= 3)
    return query.new_empty(*query.shape[:-1], value.size(-1))


def universal_pe_attention_naive(
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
) -> torch.Tensor:
    return universal_pe_scaled_dot_product_attention(
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


register_op(
    OpSpec(
        name="universal_pe_attention",
        naive=universal_pe_attention_naive,
        torch_op=universal_pe_attention_op,
        compiled_cpu=cpu_loader("universal_pe_attention"),
        compiled_taichi=taichi_loader("universal_pe_attention"),
        description="Universal positional-encoding scaled dot-product attention (CVPE QK + F-RVM OV).",
    )
)

__all__ = [
    "universal_pe_attention_naive",
    "universal_pe_attention_op",
]
