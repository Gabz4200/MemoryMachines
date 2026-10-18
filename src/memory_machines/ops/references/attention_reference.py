"""Pure PyTorch reference oracle for universal PE attention (CVPE + F-RVM).

Ground truth for parity tests against future fused kernels.
Reached ONLY via explicit import - never as a silent fallback.

Mirrors ``torch.nn.functional.scaled_dot_product_attention`` with two
optional PE circuits, passed as precomputed tensors so the reduction stays
modality-agnostic (1D text .. N-D video). All learnable projections and
MLPs live outside, in the parent module:

- QK circuit: ``cvpe_attn_bias`` ``(N,...,Hq,L,S)`` added to logits.
- OV circuit: ``phi_vq`` / ``phi_vk`` zero-bias value projections plus an
optional N-dim ``spatial_grid`` projected by ``ov_spatial_weight``.

When PE args are ``None`` the op collapses to vanilla SDPA. Self- vs
cross-attention falls out of the shapes (``L == S`` shared streams vs
``L != S`` distinct streams) - no flag needed. Causal vs bidirectional is
controlled by ``is_causal`` / ``bidirectional`` (mutually exclusive).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def build_sdpa_bias(L, S, dtype, device, attn_mask=None, is_causal=False, bidirectional=False):
    """Additive ``(L, S)`` SDPA bias: causal tril + user mask.

    ``bidirectional=True`` forces full visibility (exclusive with ``is_causal``).
    Bool ``attn_mask`` uses SDPA semantics (``True`` = participate); float
    masks are added directly. ``is_causal`` with non-square ``(L, S)`` uses
    the upper-left causal bias (lower-triangular ``tril``), matching torch.
    """
    if bidirectional and is_causal:
        raise ValueError("bidirectional=True is incompatible with is_causal=True (pass one or neither)")
    bias = torch.zeros(L, S, dtype=dtype, device=device)
    if is_causal and not bidirectional:
        if attn_mask is not None:
            raise ValueError("is_causal=True requires attn_mask=None (pass one or the other)")
        causal = torch.ones(L, S, dtype=torch.bool, device=device).tril(diagonal=0)
        bias.masked_fill_(causal.logical_not(), float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            bias = bias + torch.where(
                attn_mask.logical_not(),
                torch.tensor(float("-inf"), dtype=dtype, device=device),
                torch.zeros((), dtype=dtype, device=device),
            )
        else:
            bias = bias + attn_mask.to(dtype=dtype)
    return bias


def reference_qk_circuit(query, key, cvpe_attn_bias, attn_bias, scale):
    """Scaled QK logits + SDPA bias + optional CVPE bias (pre-softmax)."""
    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
    logits = query @ key.transpose(-2, -1) * scale_factor
    logits = logits + attn_bias.to(dtype=logits.dtype)
    if cvpe_attn_bias is not None:
        logits = logits + cvpe_attn_bias.to(dtype=logits.dtype)
    return logits


def reference_frvm_gated_reduction(attn_weight, value, phi_vq=None, phi_vk=None, spatial_grid=None, ov_spatial_weight=None, chunk_size=128):
    """Chunked F-RVM gated reduction (or plain ``attn @ value`` fallback).

    ``attn_weight``: ``(..., L, S)`` post-softmax weights.
    ``value``: ``(..., H, S, Ev)`` (already GQA-expanded when needed).
    ``phi_vq``: ``(..., Hq, L, Ev)``; ``phi_vk``: ``(..., H, S, Ev)``.
    Either phi ``None`` disables the OV circuit (vanilla matmul).
    Chunking over ``L`` bounds the ``(C, S, Ev)`` intermediate; the memory
    hazard this solves disappears once a fused kernel lands (TODO).
    """
    if phi_vq is None or phi_vk is None:
        return attn_weight @ value
    if chunk_size <= 0:
        raise ValueError(f"frvm chunk_size ({chunk_size}) must be positive")
    if (spatial_grid is None) != (ov_spatial_weight is None):
        raise ValueError("spatial_grid and ov_spatial_weight must be given together (or both None)")
    L = attn_weight.size(-2)
    outs = []
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        w_chunk = attn_weight[..., start:end, :]
        pvq_chunk = phi_vq[..., start:end, :].unsqueeze(-2)
        pvk_exp = phi_vk.unsqueeze(-3)
        vj_exp = value.unsqueeze(-3)
        delta = pvq_chunk - pvk_exp
        if spatial_grid is not None:
            grid_chunk = spatial_grid[..., start:end, :, :]
            f_spatial = F.linear(grid_chunk, ov_spatial_weight.to(dtype=grid_chunk.dtype))
            # grid has no head dim (B,...,C,S,Ev); delta does (B,...,H,C,S,Ev).
            while f_spatial.dim() < delta.dim():
                f_spatial = f_spatial.unsqueeze(-4)
            delta = delta + f_spatial
        gated = vj_exp * F.silu(vj_exp + delta)
        outs.append((w_chunk.unsqueeze(-1) * gated).sum(dim=-2))
    return torch.cat(outs, dim=-2)


def universal_pe_scaled_dot_product_attention(
    query,
    key,
    value,
    cvpe_attn_bias=None,
    phi_vq=None,
    phi_vk=None,
    spatial_grid=None,
    ov_spatial_weight=None,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    bidirectional=False,
    scale=None,
    enable_gqa=False,
    frvm_chunk_size=128,
):
    """Universal PE SDPA: vanilla SDPA + optional CVPE QK bias + F-RVM OV gate.

    Args:
        query: ``(N,...,Hq,L,E)``; key/value: ``(N,...,H,S,E)/(N,...,H,S,Ev)``.
        cvpe_attn_bias: precomputed QK spatial bias ``(N,...,Hq,L,S)`` or ``None``.
        phi_vq: query-side value projection ``(N,...,Hq,L,Ev)`` or ``None``.
        phi_vk: key-side value projection ``(N,...,H,S,Ev)`` or ``None``.
        spatial_grid: N-dim offsets ``(N,...,L,S,Ndims)`` or ``None``.
        ov_spatial_weight: ``(Ev, Ndims)`` grid projection or ``None``.
        attn_mask: bool (``True`` = participate, SDPA semantics) or float
            additive mask broadcastable to ``(N,...,Hq,L,S)``.
        dropout_p: dropout probability (pass ``0.0`` in eval).
        is_causal: lower-triangular causal mask (requires ``attn_mask=None``).
        bidirectional: full all-to-all visibility, exclusive with ``is_causal``
            (``is_causal=False`` alone is already bidirectional - this flag
            just makes encoder/cross-attn intent explicit).
        scale: pre-softmax scale (default ``1/sqrt(E)``).
        enable_gqa: repeat-interleave K/V (and ``phi_vk``) heads to ``Hq``.
        frvm_chunk_size: query-chunk width bounding the ``(C,S,Ev)`` tensor.

    Returns:
        Attention output ``(N,...,Hq,L,Ev)``.
    """
    L, S = query.size(-2), key.size(-2)
    attn_bias = build_sdpa_bias(L, S, query.dtype, query.device, attn_mask=attn_mask, is_causal=is_causal, bidirectional=bidirectional)
    if enable_gqa:
        nq, nk = query.size(-3), key.size(-3)
        if nq % nk != 0:
            raise ValueError(f"GQA requires Hq ({nq}) divisible by Hkv ({nk})")
        if key.size(-3) != value.size(-3):
            raise ValueError(f"GQA requires Hk ({key.size(-3)}) == Hv ({value.size(-3)})")
        rep = nq // nk
        if rep > 1:
            key = key.repeat_interleave(rep, -3)
            value = value.repeat_interleave(rep, -3)
            if phi_vk is not None and phi_vk.size(-3) != nq:
                phi_vk = phi_vk.repeat_interleave(rep, -3)
    logits = reference_qk_circuit(query, key, cvpe_attn_bias, attn_bias, scale)
    weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = torch.dropout(weights, dropout_p, train=True)
    return reference_frvm_gated_reduction(weights, value, phi_vq, phi_vk, spatial_grid=spatial_grid, ov_spatial_weight=ov_spatial_weight, chunk_size=frvm_chunk_size)


__all__ = ["build_sdpa_bias", "reference_frvm_gated_reduction", "reference_qk_circuit", "universal_pe_scaled_dot_product_attention"]
