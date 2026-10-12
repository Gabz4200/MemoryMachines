r"""Shared caches, masks, position helpers, and Sinkhorn normalization.

This module holds the small stateless helpers shared by the dense
(:class:`~memory_machines.models.blocks.attention.GroupedQueryAttention`),
windowed, and sparse attention layers: incremental-decode caches, causal and
sliding-window masks, absolute-position resolution, and the Sinkhorn operator
used by :class:`~memory_machines.models.blocks.memory.SinglePassMHC`.
"""

from dataclasses import dataclass

import torch


@dataclass
class CompressedCache:
    r"""CompressedCache(compressed_kv=None, comp_end_pos=None, swa_k=None, swa_v=None, swa_pos=None, total_len=None)

    Incremental-decode cache for
    :class:`~memory_machines.models.blocks.sparse.CompressedSparseAttention`.

    Holds the grow-only global compressed KV store alongside the transient
    sliding-window (SWA) KV slice, so sparse and local branches can resume
    across forward passes.

    Attributes:
        compressed_kv (torch.Tensor, optional): global compressed keys/values of
            shape ``(B, Lc, Hkv, Dh)``. Default: ``None``.
        comp_end_pos (torch.Tensor, optional): inclusive end position of each
            compressed entry, shape ``(Lc,)`` or ``(B, Lc)``. Default: ``None``.
        swa_k (torch.Tensor, optional): cached SWA keys of shape
            ``(B, Hkv, Ws, Dh)``. Default: ``None``.
        swa_v (torch.Tensor, optional): cached SWA values with the same shape
            as ``swa_k``. Default: ``None``.
        swa_pos (torch.Tensor, optional): absolute positions of the cached SWA
            slice, shape ``(B, Ws)``. Default: ``None``.
        total_len (int, optional): total consumed sequence length, used as the
            next ``start_pos``. Default: ``None``.

    Examples::

        >>> cache = CompressedCache(total_len=0)
        >>> cache.total_len
        0
    """

    compressed_kv: torch.Tensor | None = None
    comp_end_pos: torch.Tensor | None = None
    swa_k: torch.Tensor | None = None
    swa_v: torch.Tensor | None = None
    swa_pos: torch.Tensor | None = None
    total_len: int | None = None


@dataclass
class CSA2SharedState:
    r"""CSA2SharedState(compressed_kv=None, comp_end_pos=None, indexer_k=None, topk_indices=None, pool_indices=None, total_len=None)

    Cross-layer shared state for
    :class:`~memory_machines.models.blocks.sparse.CompressedSparseAttention2`.

    A ``full`` layer populates the compressed store, indexer keys, and
    candidate pools; ``reindex``/``reuse`` layers consume them without
    recompressing. SWA KV is deliberately absent: every CSA2 layer
    keeps its own sliding-window cache (CSA2, Fig. 4).

    Attributes:
        compressed_kv (torch.Tensor, optional): shared compressed KV store of
            shape ``(B, Lc, Hkv, Dh)``. Default: ``None``.
        comp_end_pos (torch.Tensor, optional): end position per compressed
            entry, shape ``(Lc,)`` or ``(B, Lc)``. Default: ``None``.
        indexer_k (torch.Tensor, optional): projected indexer keys of shape
            ``(B, Lc, Di)``. Default: ``None``.
        topk_indices (torch.Tensor, optional): selected entry ids per query,
            shape ``(B, Lq, K)``. Default: ``None``.
        pool_indices (torch.Tensor, optional): hierarchical candidate pool per
            query, shape ``(B, Lq, P)``. Default: ``None``.
        total_len (int, optional): total consumed sequence length, set by
            the calling stack (not advanced per layer). Default: ``None``.

    Examples::

        >>> state = CSA2SharedState()
        >>> state.compressed_kv is None
        True
    """

    compressed_kv: torch.Tensor | None = None
    comp_end_pos: torch.Tensor | None = None
    indexer_k: torch.Tensor | None = None
    topk_indices: torch.Tensor | None = None
    pool_indices: torch.Tensor | None = None
    total_len: int | None = None


def _past_total_len(
    past: CompressedCache | CSA2SharedState | None,
) -> int | None:
    r"""_past_total_len(past) -> int | None

    Resolve the consumed sequence length from an incremental cache.

    Prefers the explicit ``total_len`` field, then falls back to the maximum
    stored SWA position plus one.

    Args:
        past (CompressedCache or CSA2SharedState, optional): cache or shared
            state left by a previous forward pass. ``None`` means no history.

    Returns:
        int or None: ``0`` when ``past`` is ``None``, the stored total length
        when available, otherwise ``None`` when the length cannot be inferred.

    Examples::

        >>> _past_total_len(None)
        0
        >>> _past_total_len(CompressedCache(total_len=17))
        17
    """
    if past is None:
        return 0
    if past.total_len is not None:
        return past.total_len
    swa_pos = getattr(past, "swa_pos", None)
    if swa_pos is not None:
        return int(swa_pos.max().item()) + 1
    return None


def _causal_comp_mask(
    q_pos: torch.Tensor,
    comp_end: torch.Tensor,
) -> torch.Tensor:
    r"""_causal_comp_mask(q_pos, comp_end) -> Tensor

    Compute visibility of compressed entries for each query position.

    A compressed entry is visible when its inclusive end position does not lie
    in the future of the query:

    .. math::
        \text{visible}(q, c) = \text{comp\_end}_c \le q

    Args:
        q_pos (Tensor): query positions of shape ``(B, Lq)``.
        comp_end (Tensor): compressed-entry end positions of shape ``(Lc,)``
            or ``(B, Lc)``.

    Returns:
        Tensor: bool mask of shape ``(B, Lq, Lc)`` where ``True`` means the
        entry may be attended.

    Examples::

        >>> import torch
        >>> q = torch.tensor([[3, 7]])
        >>> e = torch.tensor([1, 3, 7])
        >>> _causal_comp_mask(q, e)
        tensor([[[ True,  True, False],
                 [ True,  True,  True]]])
    """
    if comp_end.dim() == 1:
        comp_end = comp_end.unsqueeze(0).expand(q_pos.shape[0], -1)
    return comp_end.unsqueeze(1) <= q_pos.unsqueeze(-1)


def _window_mask(
    q_pos: torch.Tensor,
    k_pos: torch.Tensor,
    window: int,
) -> torch.Tensor:
    r"""_window_mask(q_pos, k_pos, window) -> Tensor

    Compute a causal sliding-window mask from absolute positions.

    Keeps keys whose relative offset falls inside ``[0, window)``:

    .. math::
        \text{keep}(q, k) = 0 \le q - k < \text{window}

    Args:
        q_pos (Tensor): query positions of shape ``(B, Lq)``.
        k_pos (Tensor): key positions of shape ``(B, Lk)``.
        window (int): window width; must be positive.

    Returns:
        Tensor: bool mask of shape ``(B, Lq, Lk)`` where ``True`` means keep.

    Examples::

        >>> import torch
        >>> q = torch.tensor([[2, 3]])
        >>> k = torch.tensor([[0, 1, 2, 3]])
        >>> _window_mask(q, k, 2)
        tensor([[[False,  True,  True, False],
                 [False, False,  True,  True]]])
    """
    rel = q_pos.unsqueeze(-1) - k_pos.unsqueeze(1)
    return (rel >= 0) & (rel < window)


def _update_swa_pos(
    prev_pos: torch.Tensor | None,
    q_pos: torch.Tensor,
    stored_width: int,
) -> torch.Tensor:
    r"""_update_swa_pos(prev_pos, q_pos, stored_width) -> Tensor

    Append current query positions to the SWA position history and truncate.

    Args:
        prev_pos (Tensor, optional): previously stored positions of shape
            ``(B, Wprev)``. ``None`` means no history.
        q_pos (Tensor): current query positions of shape ``(B, Lq)``.
        stored_width (int): number of trailing positions to retain.

    Returns:
        Tensor: updated positions of shape ``(B, min(Wprev + Lq, stored_width))``.

    Examples::

        >>> import torch
        >>> _update_swa_pos(None, torch.tensor([[0, 1, 2]]), 2)
        tensor([[1, 2]])
    """
    if prev_pos is None:
        full = q_pos
    else:
        full = torch.cat([prev_pos.to(q_pos.device), q_pos], dim=1)
    return full[:, -stored_width:]


def _resolve_positions(
    hidden_states: torch.Tensor,
    q_len: int,
    past_len: int,
    position_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""_resolve_positions(hidden_states, q_len, past_len, position_ids) -> (Tensor, Tensor)

    Resolve absolute query and key positions for a forward pass.

    Explicit ``position_ids`` (``(B, L)`` or ``(B,)``-broadcastable 1D) take
    precedence; otherwise positions count up from ``past_len``. Past keys are
    assumed to occupy ``[0, past_len)`` when positions are implicit.

    Args:
        hidden_states (Tensor): current input of shape ``(B, Lq, D)``; used for
            batch size, dtype device, and length checks.
        q_len (int): current query length.
        past_len (int): number of already-consumed tokens.
        position_ids (Tensor, optional): explicit absolute positions, shape
            ``(B, L)`` (last ``q_len`` used) or ``(L,)``. ``None`` selects
            implicit counting.

    Returns:
        Tensor: query positions ``q_pos`` of shape ``(B, Lq)`` and key
        positions ``k_pos`` of shape ``(B, past_len + Lq)``.

    Examples::

        >>> import torch
        >>> x = torch.zeros(1, 2, 4)
        >>> q, k = _resolve_positions(x, 2, 3, None)
        >>> q.tolist(), k.tolist()
        ([[3, 4]], [[0, 1, 2, 3, 4]])
    """
    bsz = hidden_states.shape[0]
    device = hidden_states.device
    if position_ids is not None:
        if position_ids.dim() == 2:
            if position_ids.shape[0] != bsz:
                raise ValueError(f"position_ids batch size ({position_ids.shape[0]}) does not match input batch size ({bsz})")
            q_pos = position_ids[:, -q_len:].to(device)
        else:
            q_pos = position_ids[-q_len:].to(device).unsqueeze(0).expand(bsz, -1)
        past_pos = torch.arange(past_len, device=device, dtype=q_pos.dtype).unsqueeze(0).expand(bsz, -1)
        k_pos = torch.cat([past_pos, q_pos], dim=1) if past_len else q_pos
        return q_pos, k_pos
    q_pos = torch.arange(past_len, past_len + q_len, device=device).unsqueeze(0).expand(bsz, -1)
    k_pos = torch.arange(past_len + q_len, device=device).unsqueeze(0).expand(bsz, -1)
    return q_pos, k_pos


def _check_position_bounds(
    q_pos: torch.Tensor,
    k_pos: torch.Tensor,
    max_position_embeddings: int,
    position_ids: torch.Tensor | None,
) -> None:
    r"""_check_position_bounds(q_pos, k_pos, max_position_embeddings, position_ids) -> None

    Validate explicit absolute positions against the configured maximum.

    Implicit positions are unchecked so decode caches can grow transiently;
    only caller-supplied ``position_ids`` are bounded.

    Args:
        q_pos (Tensor): query positions of shape ``(B, Lq)``.
        k_pos (Tensor): key positions of shape ``(B, Lk)``.
        max_position_embeddings (int): exclusive upper bound for positions.
        position_ids (Tensor, optional): explicit positions that triggered the
            check; ``None`` skips validation.

    Returns:
        None

    Examples::

        >>> import torch
        >>> _check_position_bounds(torch.tensor([[0]]), torch.tensor([[0]]), 8, None) is None
        True
    """
    if position_ids is None:
        return
    if bool((q_pos >= max_position_embeddings).any() | (k_pos >= max_position_embeddings).any()):
        raise ValueError(f"Absolute position id exceeds max_position_embeddings ({max_position_embeddings})")


def _sinkhorn(M: torch.Tensor, iters: int) -> torch.Tensor:
    r"""_sinkhorn(M, iters) -> Tensor

    Project logits onto (approximately) doubly-stochastic matrices.

    Applies alternating row/column normalization to ``exp(M)``:

    .. math::
        P^{(0)} = \exp(M), \quad
        P^{(t+1)} = \mathcal{C}(\mathcal{R}(P^{(t)}))

    where :math:`\mathcal{R}` and :math:`\mathcal{C}` are row and column
    normalizations.

    Args:
        M (Tensor): unnormalized log-weights of shape ``(..., n, n)``. Must be
            finite.
        iters (int): number of row/column normalization iterations.

    Returns:
        Tensor: approximately doubly-stochastic matrix with the same shape as
        ``M``; all-zero rows (underflow) fall back to uniform weights.

    Examples::

        >>> import torch
        >>> P = _sinkhorn(torch.zeros(2, 2), 5)
        >>> P.shape
        torch.Size([2, 2])
    """
    if not bool(torch.isfinite(M).all()):
        raise ValueError("Sinkhorn input must be finite (got NaN/Inf logits)")
    n = M.shape[-1]
    P = torch.exp(M)
    for _ in range(iters):
        P = P / P.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        P = P / P.sum(dim=-2, keepdim=True).clamp_min(1e-12)
    empty_rows = P.sum(dim=-1, keepdim=True) == 0
    if bool(empty_rows.any()):
        P = torch.where(empty_rows, torch.full_like(P, 1.0 / n), P)
    return P
