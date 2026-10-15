r"""Gated DeltaNet-2 (GDN-2) token mixer: Gated Delta Rule-2 reference.

Implements the NVLabs ``GatedDeltaNet-2`` layer (arXiv:2605.22791) as a pure
PyTorch reference. The recurrence on the matrix state ``S_t in R^{K x V}``
(Eq. 10 of the paper) is::

    S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T
    o_t = S_t^T q_t

with channel-wise erase gate ``b_t in [0, 1]^K`` (``[0, 2]^K`` when
``allow_neg_eigval``), channel-wise write gate ``w_t in [0, 1]^V``, and
channel-wise log-decay ``g_t``. Tying ``b_t = w_t = beta_t * 1`` recovers KDA;
further tying the decay recovers Gated DeltaNet.

Functional API (``torch.nn.functional.scaled_dot_product_attention`` style):

- :func:`gated_delta_rule_2` — dispatcher over the reference paths.
- :func:`gated_delta_rule_2_recurrent_forward` — token-by-token oracle
  (``fused_recurrent`` kernel equivalent, differentiable).
- :func:`gated_delta_rule_2_chunk_forward` — chunkwise WY oracle
  (``chunk`` kernel equivalent: intra / solve / inter math from Appendix A).

Both paths share one math core, so their outputs agree to numerical precision;
gradients flow through plain autograd (no manual backward needed). Compiled
backends (``"compiled"``/``"triton"``/``"ascend"``/``"cuda"``) raise — they
are reserved for the future kernel split (``naive``/``wy_fast``/``chunk_*``).
The short convolutions run through
:class:`~memory_machines.models.blocks.causal_conv1d.ShortConvolution`.

Examples::

    >>> import torch
    >>> from memory_machines.models.blocks import gdn2
    >>> q, k = torch.randn(1, 6, 2, 4), torch.randn(1, 6, 2, 4)
    >>> v, w = torch.randn(1, 6, 2, 4), torch.sigmoid(torch.randn(1, 6, 2, 4))
    >>> g, b = -torch.rand(1, 6, 2, 4), torch.sigmoid(torch.randn(1, 6, 2, 4))
    >>> o, _ = gdn2.gated_delta_rule_2(q, k, v, g, b, w, chunk_size=4)
    >>> o.shape
    torch.Size([1, 6, 2, 4])
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F
from torch import nn

from .causal_conv1d import ShortConvolution

__all__ = [
    "GatedDeltaNet2",
    "apply_beta_guard",
    "gated_delta_rule_2",
    "gated_delta_rule_2_chunk_forward",
    "gated_delta_rule_2_recurrent_forward",
]

_COMPILED_BACKENDS = ("compiled", "triton", "triton_ascend", "ascend", "cuda", "cpu", "taichi", "gpu")
_REFERENCE_BACKENDS = ("naive", "recurrent", "chunk", "wy_fast")
GUARD_MARGIN = 1.0 / 32


def _compute_dtype(*tensors: torch.Tensor) -> torch.dtype:
    if any(t.dtype == torch.float64 for t in tensors):
        return torch.float64
    return torch.float32


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _prepare_qkb(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    scale: float,
    use_qk_l2norm: bool,
    beta_guard: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize/scale q/k and guard b, shared by the recurrent and WY paths."""
    if use_qk_l2norm:
        q, k = _l2norm(q), _l2norm(k)
    if beta_guard:
        b = apply_beta_guard(b, k, g)
    return q * scale, k, b


def apply_beta_guard(
    b: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    margin: float = GUARD_MARGIN,
) -> torch.Tensor:
    r"""apply_beta_guard(b, k, g, margin=1/32) -> Tensor

    Conditional erase-contrast projection (the ``beta_guard`` fused into the
    NVLabs chunk/recurrent kernels). Rows whose key-weighted contrast
    ``r2 = n*nu - a^2`` violates the certified non-expansion budget are shrunk
    about their key-weighted mean; safe rows pass through unchanged. Decay and
    write gates are never modified.

    Args:
        b (Tensor): erase gate ``[..., K]`` in ``[0, 1]`` (or ``[0, 2]``).
        k (Tensor): normalized keys ``[..., K]`` (same shape as ``b``).
        g (Tensor): log-decay ``[..., K]`` (same shape as ``b``).
        margin (float): projection margin. Default: ``1/32``.

    Returns:
        Tensor: guarded erase gate with the same shape as ``b``.

    Examples::

        >>> import torch
        >>> from memory_machines.models.blocks.gdn2 import apply_beta_guard
        >>> b = torch.full((1, 1, 1, 4), 0.5)
        >>> k = torch.nn.functional.normalize(torch.randn(1, 1, 1, 4), dim=-1)
        >>> apply_beta_guard(b, k, torch.zeros(1, 1, 1, 4)).shape
        torch.Size([1, 1, 1, 4])
    """
    k2 = k * k
    n = k2.sum(dim=-1, keepdim=True)
    a = (b * k2).sum(dim=-1, keepdim=True)
    nu = (b * b * k2).sum(dim=-1, keepdim=True)
    r2 = torch.clamp(n * nu - a * a, min=0.0)
    g_max = g.max(dim=-1, keepdim=True).values
    c2 = torch.exp(-2.0 * g_max)
    r2_crit = torch.clamp((c2 - 1.0) * (1.0 - (1.0 - a) ** 2 / c2), min=0.0)
    has_key = n > 1e-20
    unsafe = has_key & (r2 > r2_crit)
    mu = a / torch.clamp(n, min=1e-20)
    eta = torch.sqrt((1.0 - margin) * r2_crit / torch.clamp(r2, min=1e-30)).clamp(0.0, 1.0)
    projected = mu + eta * (b - mu)
    return torch.where(unsafe, projected, b)


def _recurrent_single(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    s0: torch.Tensor,
    scale: float,
    use_qk_l2norm: bool,
    beta_guard: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token loop for one head-group row: all ``[T, ...]`` + ``s0 [K, V]``."""
    t = q.shape[0]
    q, k, b = _prepare_qkb(q, k, g, b, scale, use_qk_l2norm, beta_guard)
    alpha = torch.exp(g)  # [T, K]
    e = b * k  # [T, K]
    z = w * v  # [T, V]
    s = s0
    outs = []
    for i in range(t):
        s_bar = alpha[i].unsqueeze(-1) * s
        r = torch.matmul(s_bar.transpose(0, 1), e[i])  # [V]
        s = s_bar + torch.outer(k[i], z[i] - r)
        outs.append(torch.matmul(s.transpose(0, 1), q[i]))
    o = torch.stack(outs, dim=0) if t else q.new_zeros(0, v.shape[-1])
    return o, s


def _chunk_single(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    s0: torch.Tensor,
    scale: float,
    use_qk_l2norm: bool,
    beta_guard: bool,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunkwise WY forward for one head-group row (paper Appendix A)."""
    t = q.shape[0]
    q, k, b = _prepare_qkb(q, k, g, b, scale, use_qk_l2norm, beta_guard)
    s = s0
    eye_full = torch.eye(chunk_size, dtype=q.dtype, device=q.device)
    outs = []
    for start in range(0, t, chunk_size):
        qc = q[start : start + chunk_size]
        kc = k[start : start + chunk_size]
        vc = v[start : start + chunk_size]
        gc = g[start : start + chunk_size]
        bc = b[start : start + chunk_size]
        wc = w[start : start + chunk_size]
        c = qc.shape[0]
        gamma = torch.exp(torch.cumsum(gc, dim=0))  # [C, K]
        k_bar = kc / gamma
        e_bar = gamma * (bc * kc)
        z = wc * vc
        eye_c = eye_full[:c, :c]
        tmat = torch.tril(e_bar @ k_bar.transpose(0, 1), diagonal=-1)  # [C, C]
        a_mat = torch.linalg.solve_triangular(eye_c + tmat, eye_c, upper=False)  # A = (I + T)^{-1}
        y = a_mat @ e_bar  # [C, K]
        u = a_mat @ z  # [C, V]
        gamma_c = gamma[-1]  # [K]
        k_tail = (gamma_c / gamma) * kc
        q_gamma = gamma * qc
        aqk = torch.tril(q_gamma @ k_bar.transpose(0, 1))
        r = u - y @ s
        outs.append(q_gamma @ s + aqk @ r)
        s = gamma_c.unsqueeze(-1) * s + k_tail.transpose(0, 1) @ r
    o = torch.cat(outs, dim=0) if outs else q.new_zeros(0, v.shape[-1])
    return o, s


def _check_shapes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    initial_state: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    r"""_check_shapes(q, k, v, g, b, w, cu_seqlens, initial_state) -> (B, T, H, HV, K)

    Validate the Gated Delta Rule-2 layouts (``b`` on ``[B, T, HV, K]``, GVA
    divisibility, packed-batch rules) and return the resolved dims.
    """
    if not (q.dim() == k.dim() == g.dim() == b.dim() == 4):
        raise ValueError(f"q/k/g/b must be 4D [B, T, H, K], got {[tuple(t.shape) for t in (q, k, g, b)]}")
    if not (v.dim() == w.dim() == 4):
        raise ValueError(f"v/w must be 4D [B, T, HV, V], got {[tuple(t.shape) for t in (v, w)]}")
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    if tuple(k.shape) != (B, T, H, K) or tuple(g.shape) != (B, T, H, K):
        raise ValueError(f"k/g must match q {tuple(q.shape)}, got {tuple(k.shape)}/{tuple(g.shape)}")
    if tuple(b.shape) != (B, T, HV, K):
        raise ValueError(f"b must be [B, T, HV, K]={(B, T, HV, K)}, got {tuple(b.shape)}")
    if tuple(v.shape) != tuple(w.shape) or v.shape[:2] != q.shape[:2]:
        raise ValueError(f"v/w must share [B, T] with q and match each other, got {tuple(v.shape)}/{tuple(w.shape)}")
    if not (HV >= H and HV % H == 0):
        raise ValueError(f"value heads HV ({HV}) must be a multiple of key heads H ({H})")
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError(f"cu_seqlens packing requires batch 1, got {B}")
        if int(cu_seqlens[-1].item()) != T:
            raise ValueError(f"cu_seqlens must end at T={T}, got {int(cu_seqlens[-1].item())}")
    if initial_state is not None:
        n = len(cu_seqlens) - 1 if cu_seqlens is not None else B
        if tuple(initial_state.shape) != (n, HV, K, V):
            raise ValueError(f"initial_state must be [N, HV, K, V]={(n, HV, K, V)}, got {tuple(initial_state.shape)}")
    return B, T, H, HV, K


def _run_path(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm: bool,
    beta_guard: bool,
    cu_seqlens: torch.Tensor | None,
    chunk_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""_run_path(...) -> (o, state)

    Shared driver for both reference paths: validates shapes, expands key-side
    tensors across value-head groups, splits packed/batched segments, and runs
    ``chunk_size=None`` (recurrent) or ``chunk_size=C`` (WY) per head row.
    """
    B, T, H, HV, K = _check_shapes(q, k, v, g, b, w, cu_seqlens, initial_state)
    V = v.shape[-1]
    if beta_guard and not use_qk_l2norm:
        raise ValueError("beta_guard=True requires use_qk_l2norm=True so the guard sees the normalized key")
    if scale is None:
        scale = K**-0.5
    cdtype = _compute_dtype(q, k, v, g, b, w)
    group = HV // H
    # Repeat key-side tensors across value-head groups (GVA); alias when group == 1.
    if group == 1:
        qr, kr, gr = q.to(cdtype), k.to(cdtype), g.to(cdtype)
    else:
        qr = q.repeat_interleave(group, dim=2).to(cdtype)
        kr = k.repeat_interleave(group, dim=2).to(cdtype)
        gr = g.repeat_interleave(group, dim=2).to(cdtype)
    br = b.to(cdtype)
    vr = v.to(cdtype)
    wr = w.to(cdtype)
    bounds = cu_seqlens.tolist() if cu_seqlens is not None else None
    segs: list[tuple[int, int]] = list(itertools.pairwise(bounds)) if bounds is not None else [(0, T) for _ in range(B)]
    outs = torch.empty(B, T, HV, V, dtype=cdtype, device=q.device)
    finals = []
    for n, (s, e) in enumerate(segs):
        bn = 0 if cu_seqlens is not None else n
        s0 = initial_state[n].to(cdtype) if initial_state is not None else torch.zeros(HV, K, V, dtype=cdtype, device=q.device)
        seg_view = outs[0, s:e] if cu_seqlens is not None else outs[n]
        cur = torch.empty_like(s0)
        for hv in range(HV):
            if chunk_size is not None:
                o_hv, cur_hv = _chunk_single(
                    qr[bn, s:e, hv],
                    kr[bn, s:e, hv],
                    vr[bn, s:e, hv],
                    gr[bn, s:e, hv],
                    br[bn, s:e, hv],
                    wr[bn, s:e, hv],
                    s0[hv],
                    scale,
                    use_qk_l2norm,
                    beta_guard,
                    chunk_size,
                )
            else:
                o_hv, cur_hv = _recurrent_single(
                    qr[bn, s:e, hv],
                    kr[bn, s:e, hv],
                    vr[bn, s:e, hv],
                    gr[bn, s:e, hv],
                    br[bn, s:e, hv],
                    wr[bn, s:e, hv],
                    s0[hv],
                    scale,
                    use_qk_l2norm,
                    beta_guard,
                )
            seg_view[:, hv, :] = o_hv
            cur[hv] = cur_hv
        finals.append(cur)
    final_state = torch.stack(finals, dim=0) if output_final_state else None
    return outs.to(q.dtype), (final_state.to(q.dtype) if final_state is not None else None)


def gated_delta_rule_2_recurrent_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
    beta_guard: bool = False,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, ...) -> (o, state)

    Token-by-token reference oracle for Gated Delta Rule-2 (``fused_recurrent``
    kernel equivalent). Differentiable; doubles as the ground truth for kernel
    parity tests.

    Args:
        q (Tensor): queries ``[B, T, H, K]``.
        k (Tensor): keys ``[B, T, H, K]``.
        v (Tensor): values ``[B, T, HV, V]``.
        g (Tensor): log-decay ``[B, T, H, K]`` (``<= 0`` per channel).
        b (Tensor): erase gate ``[B, T, HV, K]`` in ``[0, 1]`` (``[0, 2]`` with
            negative eigenvalues).
        w (Tensor): write gate ``[B, T, HV, V]`` in ``[0, 1]``.
        scale (float, optional): query scale; default ``1/sqrt(K)``.
        initial_state (Tensor, optional): ``[N, HV, K, V]`` state.
        output_final_state (bool): whether to return the final state.
        use_qk_l2norm (bool): L2-normalize q/k per head. Default: ``False``.
        beta_guard (bool): erase-contrast projection (requires
            ``use_qk_l2norm``). Default: ``False``.
        cu_seqlens (Tensor, optional): ``[N+1]`` packed offsets (``B == 1``).

    Returns:
        Tensor: outputs ``[B, T, HV, V]`` and optional final state.
    """
    return _run_path(q, k, v, g, b, w, scale, initial_state, output_final_state, use_qk_l2norm, beta_guard, cu_seqlens, None)


def gated_delta_rule_2_chunk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
    beta_guard: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, ...) -> (o, state)

    Chunkwise WY reference oracle for Gated Delta Rule-2 (``chunk`` kernel
    equivalent, paper Appendix A). Same result as
    :func:`gated_delta_rule_2_recurrent_forward` via dense intra-chunk solves.

    Args:
        q (Tensor): queries ``[B, T, H, K]``.
        k (Tensor): keys ``[B, T, H, K]``.
        v (Tensor): values ``[B, T, HV, V]``.
        g (Tensor): log-decay ``[B, T, H, K]``.
        b (Tensor): erase gate ``[B, T, HV, K]``.
        w (Tensor): write gate ``[B, T, HV, V]``.
        scale (float, optional): query scale; default ``1/sqrt(K)``.
        initial_state (Tensor, optional): ``[N, HV, K, V]`` state.
        output_final_state (bool): whether to return the final state.
        use_qk_l2norm (bool): L2-normalize q/k per head. Default: ``False``.
        beta_guard (bool): erase-contrast projection. Default: ``False``.
        cu_seqlens (Tensor, optional): ``[N+1]`` packed offsets (``B == 1``).
        chunk_size (int): chunk width. Default: ``64``.

    Returns:
        Tensor: outputs ``[B, T, HV, V]`` and optional final state.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return _run_path(q, k, v, g, b, w, scale, initial_state, output_final_state, use_qk_l2norm, beta_guard, cu_seqlens, chunk_size)


def gated_delta_rule_2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
    beta_guard: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    backend: str = "naive",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""gated_delta_rule_2(q, k, v, g, b, w, ...) -> (o, state)

    SDPA-style entry point for Gated Delta Rule-2. ``backend`` selects the
    reference path; every alias runs pure PyTorch (no kernels yet):

    - ``"naive"``/``"recurrent"`` → :func:`gated_delta_rule_2_recurrent_forward`
    - ``"chunk"``/``"wy_fast"`` → :func:`gated_delta_rule_2_chunk_forward`
    - compiled names (``"compiled"``, ``"triton*"``, ``"ascend"``, ...) raise.

    Args:
        q (Tensor): queries ``[B, T, H, K]``.
        k (Tensor): keys ``[B, T, H, K]``.
        v (Tensor): values ``[B, T, HV, V]``.
        g (Tensor): log-decay ``[B, T, H, K]``.
        b (Tensor): erase gate ``[B, T, HV, K]``.
        w (Tensor): write gate ``[B, T, HV, V]``.
        scale (float, optional): query scale; default ``1/sqrt(K)``.
        initial_state (Tensor, optional): ``[N, HV, K, V]`` state.
        output_final_state (bool): whether to return the final state.
        use_qk_l2norm (bool): L2-normalize q/k per head. Default: ``False``.
        beta_guard (bool): erase-contrast projection. Default: ``False``.
        cu_seqlens (Tensor, optional): ``[N+1]`` packed offsets (``B == 1``).
        chunk_size (int): chunk width for the WY path. Default: ``64``.
        backend (str): reference path selector. Default: ``"naive"``.

    Returns:
        Tensor: outputs ``[B, T, HV, V]`` and optional final state.

    Examples::

        >>> import torch
        >>> from memory_machines.models.blocks.gdn2 import gated_delta_rule_2
        >>> q, k = torch.randn(1, 4, 1, 4), torch.randn(1, 4, 1, 4)
        >>> v = torch.randn(1, 4, 1, 4)
        >>> g, b = -torch.rand(1, 4, 1, 4), torch.sigmoid(torch.randn(1, 4, 1, 4))
        >>> w = torch.sigmoid(torch.randn(1, 4, 1, 4))
        >>> o, _ = gated_delta_rule_2(q, k, v, g, b, w, chunk_size=2)
        >>> o.shape
        torch.Size([1, 4, 1, 4])
    """
    if backend in ("naive", "recurrent"):
        return gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, scale, initial_state, output_final_state, use_qk_l2norm, beta_guard, cu_seqlens)
    if backend in ("chunk", "wy_fast"):
        return gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, scale, initial_state, output_final_state, use_qk_l2norm, beta_guard, cu_seqlens, chunk_size)
    if backend in _COMPILED_BACKENDS:
        raise NotImplementedError(f"backend {backend!r} has no GDN-2 kernel yet (TODO); use backend='naive'")
    raise ValueError(f"Unknown backend: {backend!r}. Valid reference backends: {list(_REFERENCE_BACKENDS)}")


class GatedDeltaNet2(nn.Module):
    r"""GatedDeltaNet2(hidden_size, num_heads, ...)

    GDN-2 token mixer block: projections → short convs → Gated Delta Rule-2 →
    gated RMSNorm → output projection. Mirrors the NVLabs layer parameterization
    (``f_proj`` bottleneck decay, ``A_log``/``dt_bias``, ``b``/``w`` gates,
    ``g_proj`` output gate) with local :class:`ShortConvolution` modules and the
    pure-PyTorch reference oracle (no kernels yet).

    Args:
        hidden_size (int): model width.
        num_heads (int): key/query heads ``H``. Default: ``8``.
        head_dim_k (int): key head dim ``K``. Default: ``32``.
        head_dim_v (int | None): value head dim; default ``head_dim_k`` scaled
            by ``expand_v``.
        expand_v (float): value-dim expansion ratio. Default: ``1.0``.
        num_v_heads (int | None): value heads ``HV``; default ``num_heads``.
            Must equal or be a multiple of ``num_heads`` (GVA).
        use_short_conv (bool): depthwise short convs on q/k/v. Default: ``True``.
        conv_size (int): conv width. Default: ``4``.
        conv_bias (bool): conv bias. Default: ``False``.
        allow_neg_eigval (bool): scale erase gate to ``[0, 2]``. Default: ``False``.
        norm_eps (float): RMSNorm epsilon. Default: ``1e-5``.
        backend (str): reference path (``"naive"``/``"recurrent"``/``"chunk"``/``"wy_fast"``,
            all pure PyTorch; compiled names raise). Default: ``"naive"``.
        chunk_size (int): WY chunk width. Default: ``64``.
        use_qk_l2norm (bool): L2-normalize q/k. Default: ``True``.
        beta_guard (bool): erase-contrast projection. Default: ``False``.

    Examples::

        >>> import torch
        >>> from memory_machines.models.blocks.gdn2 import GatedDeltaNet2
        >>> blk = GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4,
        ...     use_short_conv=False)
        >>> o, _, _ = blk(torch.randn(1, 5, 16))
        >>> o.shape
        torch.Size([1, 5, 16])
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        head_dim_k: int = 32,
        head_dim_v: int | None = None,
        expand_v: float = 1.0,
        num_v_heads: int | None = None,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        norm_eps: float = 1e-5,
        backend: str = "naive",
        chunk_size: int = 64,
        use_qk_l2norm: bool = True,
        beta_guard: bool = False,
    ) -> None:
        super().__init__()
        if backend not in _REFERENCE_BACKENDS:
            if backend in _COMPILED_BACKENDS:
                raise NotImplementedError(f"backend {backend!r} has no GDN-2 kernel yet (TODO); use backend='naive'")
            raise ValueError(f"Unknown backend: {backend!r}. Valid reference backends: {list(_REFERENCE_BACKENDS)}")
        if hidden_size <= 0 or num_heads <= 0 or head_dim_k <= 0:
            raise ValueError("hidden_size/num_heads/head_dim_k must be positive")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        if not (self.num_v_heads >= num_heads and self.num_v_heads % num_heads == 0):
            raise ValueError(f"num_v_heads ({self.num_v_heads}) must be a multiple of num_heads ({num_heads})")
        self.head_k_dim = head_dim_k
        self.head_v_dim = head_dim_v if head_dim_v is not None else int(head_dim_k * expand_v)
        if self.head_v_dim <= 0 or not math.isclose(head_dim_k * expand_v, float(self.head_v_dim), rel_tol=1e-5):
            raise ValueError(f"expand_v={expand_v} is inconsistent with head dims {head_dim_k}->{self.head_v_dim}")
        self.expand_v = expand_v
        self.key_dim = num_heads * head_dim_k
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.allow_neg_eigval = allow_neg_eigval
        self.backend = backend
        self.chunk_size = chunk_size
        self.use_qk_l2norm = use_qk_l2norm
        self.beta_guard = beta_guard

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_short_conv:
            self.q_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, bias=conv_bias)
            self.k_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, bias=conv_bias)
            self.v_conv1d = ShortConvolution(hidden_size=self.value_dim, kernel_size=conv_size, bias=conv_bias)
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        self.b_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.w_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16)))
        dt = torch.exp(torch.rand(self.key_dim, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module) -> None:
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2**-2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True  # type: ignore[assignment]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: dict | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        r"""forward(hidden_states, ...) -> (o, None, cache)

        Args:
            hidden_states (Tensor): input ``[B, T, hidden]``.
            attention_mask (Tensor, optional): ``[B, T]`` 0/1 padding mask.
            past_key_values (dict, optional): cache with ``recurrent_state``
                ``[B, HV, K, V]`` and ``conv_state`` triple.
            use_cache (bool): whether to return the updated cache.
            output_attentions (bool): unused, kept for interface compat.
            cu_seqlens (Tensor, optional): ``[N+1]`` packed offsets (``B == 1``).

        Returns:
            Tensor: outputs ``[B, T, hidden]``, ``None`` (no attention map),
            and the updated cache (or ``None``).
        """
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be [B, T, D], got {tuple(hidden_states.shape)}")
        if attention_mask is not None and cu_seqlens is not None:
            raise ValueError("attention_mask and cu_seqlens cannot be provided at the same time")
        B, _, _ = hidden_states.shape
        cdtype = _compute_dtype(hidden_states)

        last_state = past_key_values.get("recurrent_state") if past_key_values is not None else None
        conv_states = past_key_values.get("conv_state") if past_key_values is not None else None

        if attention_mask is not None:
            return self._forward_masked(hidden_states, attention_mask, last_state, conv_states, use_cache)

        if self.use_short_conv:
            csq, csk, csv = (None, None, None) if conv_states is None else conv_states
            q, csq = self.q_conv1d(x=self.q_proj(hidden_states), cache=csq, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, csk = self.k_conv1d(x=self.k_proj(hidden_states), cache=csk, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, csv = self.v_conv1d(x=self.v_proj(hidden_states), cache=csv, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            new_conv = (csq, csk, csv)
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))
            new_conv = None

        g = (-self.A_log.float().exp().repeat_interleave(self.head_k_dim) * F.softplus(self.f_proj(hidden_states).float() + self.dt_bias)).to(cdtype)
        b = self.b_proj(hidden_states).sigmoid().to(cdtype)
        w = self.w_proj(hidden_states).sigmoid().to(cdtype)
        if self.allow_neg_eigval:
            b = b * 2.0

        def _split(x: torch.Tensor, d: int) -> torch.Tensor:
            return x.view(x.shape[0], x.shape[1], -1, d)

        qh, kh, gh = (_split(t, self.head_k_dim) for t in (q.to(cdtype), k.to(cdtype), g))
        vh = _split(v.to(cdtype), self.head_v_dim)
        bh = _split(b, self.head_k_dim).repeat_interleave(self.num_v_heads // self.num_heads, dim=2)
        wh = _split(w, self.head_v_dim)

        if last_state is not None and last_state.shape[0] != (len(cu_seqlens) - 1 if cu_seqlens is not None else B):
            raise ValueError(f"cached recurrent_state batch {tuple(last_state.shape)} mismatches input batch {B}")

        o, new_state = gated_delta_rule_2(
            qh,
            kh,
            vh,
            gh,
            bh,
            wh,
            initial_state=None if last_state is None else last_state.to(cdtype),
            output_final_state=use_cache,
            use_qk_l2norm=self.use_qk_l2norm,
            beta_guard=self.beta_guard,
            cu_seqlens=cu_seqlens,
            chunk_size=self.chunk_size,
            backend=self.backend,
        )
        gate = self.g_proj(hidden_states)
        gate = gate.view(B, o.shape[1], self.num_v_heads, self.head_v_dim)
        o = self.o_norm(o) * F.silu(gate.to(o.dtype))
        o = self.o_proj(o.reshape(B, o.shape[1], self.value_dim))
        cache = None
        if use_cache:
            cache = {"recurrent_state": new_state, "conv_state": new_conv if self.use_short_conv else None}
        return o, None, cache

    def _forward_masked(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        last_state: torch.Tensor | None,
        conv_states: tuple | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        if last_state is not None or conv_states is not None:
            raise ValueError("attention_mask with cached past_key_values is not supported in the naive reference; pass explicit positions via cu_seqlens or decode without padding")
        B, T, _ = hidden_states.shape
        outs = []
        for i in range(B):
            keep = attention_mask[i].to(torch.bool)
            xi = hidden_states[i : i + 1, keep]
            if xi.shape[1] == 0:
                outs.append(hidden_states[i : i + 1, :0].new_zeros(1, 0, self.hidden_size))
                continue
            oi, _, _ = self.forward(xi, use_cache=False)
            full = oi.new_zeros(1, T, self.hidden_size)
            full[0, keep] = oi[0]
            outs.append(full)
        return torch.cat(outs, dim=0), None, None
