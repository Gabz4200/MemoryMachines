r"""Shared caches, masks, position helpers, and Sinkhorn normalization."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from .linear_init import init_linear


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""_l2norm(x, eps=1e-6) -> Tensor

    L2-normalize the last dimension of ``x`` (unit-norm rows), with a small
    epsilon for numerical stability. Shared by the GDN-2 and HOPE/TITANS
    fast-weight q/k normalization paths.
    """
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def residual_init_scale(std: float, num_layers: int, enabled: bool = True) -> float | None:
    r"""residual_init_scale(std, num_layers, enabled=True) -> float | None

    Depth-scaled residual initialization scale ``std / sqrt(2 * num_layers)``
    when ``enabled`` is True, otherwise no scaling (returns ``None``).

    Args:
        std (float): base initialization standard deviation.
        num_layers (int): number of layers ``N``.
        enabled (bool, optional): whether to apply scaling.
            Default: ``True``.

    Returns:
        float | None: scaling factor or ``None`` to disable scaling.
    """
    if not enabled:
        return None
    return std / math.sqrt(2 * max(num_layers, 1))


def init_residual_linears(
    std: float,
    num_layers: int,
    enabled: bool,
    *linears: nn.Module,
) -> float | None:
    r"""init_residual_linears(std, num_layers, enabled, *linears) -> scale | None

    Truncated-normal init ``N(0, std^2)`` for every ``linears`` effective
    weight and return the depth-scaled residual scale (or ``None``) for the
    downstream output projection. Accepts dense and factored projections alike,
    so a resident block keeps a single init path. Shared by every block that
    initializes a bounded residual path (SwiGLU, ClassicMLP, LIV double-gated
    base).
    """
    for lin in linears:
        init_linear(lin, std)
    return residual_init_scale(std, num_layers, enabled)


@dataclass
class CompressedCache:
    r"""CompressedCache(compressed_kv=None, comp_end_pos=None, swa_k=None, swa_v=None, swa_pos=None, total_len=None)

    Cache for :class:`~memory_machines.models.blocks.sparse.CompressedSparseAttention`
    containing the compressed keys and values, along with metadata
    needed for updating the cache on the next step.
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

    Shared state for :class:`~memory_machines.models.blocks.sparse.CompressedSparseAttention2`
    to enable cross-layer key/value reuse.
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
        if position_ids.dim() == 3:
            if position_ids.shape[0] != bsz:
                raise ValueError(f"position_ids batch size ({position_ids.shape[0]}) does not match input batch size ({bsz})")
            q_pos = position_ids[:, -q_len:, :].to(device)
            return q_pos, q_pos
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


def _coerce_coords(
    position_ids: torch.Tensor | None,
    q_len: int,
    past_len: int,
    bsz: int,
    device: torch.device,
    dtype: torch.dtype,
    spatial_dims: int,
) -> torch.Tensor:
    r"""_coerce_coords(position_ids, q_len, past_len, bsz, device, dtype, spatial_dims) -> Tensor

    Coerce position IDs into an ``(B, L, N)`` coordinate tensor with zero-padding for unused axes.

    Explicit ``position_ids`` (``(B, L, N)``, ``(B, L)``, or ``(L,)``) are used as-is;
    missing dimensions are zero-padded on the right. When ``None``, positions count up from
    ``past_len``.

    Args:
        position_ids (Tensor, optional): explicit coordinates, shape ``(B, L, N)``, ``(B, L)``,
            or ``(L,)``.
        q_len (int): current query length.
        past_len (int): number of already-consumed tokens.
        bsz (int): batch size.
        device (torch.device): device for output tensor.
        dtype (torch.dtype): dtype for output tensor.
        spatial_dims (int): number of coordinate dimensions.

    Returns:
        Tensor: coordinates of shape ``(B, q_len, spatial_dims)``.

    Examples::

        >>> import torch
        >>> # Implicit counting (text)
        >>> _coerce_coords(None, 3, 0, 1, torch.device('cpu'), torch.float32, 3)
        tensor([[[0., 0., 0.],
                 [1., 0., 0.],
                 [2., 0., 0.]]])
        >>> # Explicit 1D positions
        >>> _coerce_coords(torch.tensor([[0, 1, 2]]), 3, 0, 1, torch.device('cpu'), torch.float32, 3)
        tensor([[[0., 0., 0.],
                 [1., 0., 0.],
                 [2., 0., 0.]]])
        >>> # Explicit 2D positions (images)
        >>> _coerce_coords(torch.tensor([[[0, 0], [0, 1], [1, 0]]]), 3, 0, 1, torch.device('cpu'), torch.float32, 3)
        tensor([[[0., 0., 0.],
                 [0., 1., 0.],
                 [1., 0., 0.]]])
    """
    if position_ids is not None:
        if position_ids.dim() == 3:
            if position_ids.shape[0] != bsz:
                raise ValueError(f"position_ids batch size ({position_ids.shape[0]}) does not match input batch size ({bsz})")
            pos = position_ids[:, -q_len:, :].to(device=device, dtype=dtype)
        elif position_ids.dim() == 2:
            if position_ids.shape[0] != bsz:
                raise ValueError(f"position_ids batch size ({position_ids.shape[0]}) does not match input batch size ({bsz})")
            pos = position_ids[:, -q_len:].to(device=device, dtype=dtype)
        else:
            pos = position_ids[-q_len:].to(device=device, dtype=dtype).unsqueeze(0).expand(bsz, -1)
    else:
        pos = torch.arange(past_len, past_len + q_len, device=device, dtype=dtype).unsqueeze(0).expand(bsz, -1)
    if pos.dim() == 2:
        pos = pos.unsqueeze(-1)
    n = pos.shape[-1]
    if n < spatial_dims:
        pos = torch.nn.functional.pad(pos, (0, spatial_dims - n))
    elif n > spatial_dims:
        pos = pos[..., :spatial_dims]
    return pos


def _multi_axis_distance(
    q_pos: torch.Tensor,
    k_pos: torch.Tensor,
    axis_max: tuple[int, ...] | list[int] | torch.Tensor,
    use_log_transform: bool = True,
    eps: float = 1e-9,
) -> torch.Tensor:
    r"""_multi_axis_distance(q_pos, k_pos, axis_max, use_log_transform=True, eps=1e-9) -> Tensor

    Compute bounded N-dimensional distance vectors in :math:`[0, 1]^N`.

    Args:
        q_pos (Tensor): query coordinates of shape ``(B, Lq, N)``.
        k_pos (Tensor): key coordinates of shape ``(B, Lk, N)``.
        axis_max (tuple[int, ...] | list[int] | torch.Tensor): maximum coordinate
            value per axis (length ``N``).
        use_log_transform (bool, optional): whether to log-warp each axis.
            Default: ``True``.
        eps (float, optional): small constant for numerical stability.
            Default: ``1e-9``.

    Returns:
        Tensor: bounded distance of shape ``(B, Lq, Lk, N)`` where each
            dimension is independently normalized to :math:`[0, 1]`.

    Examples::

        >>> import torch
        >>> q = torch.tensor([[[0.], [1.]]])  # (B=1, Lq=2, N=1)
        >>> k = torch.tensor([[[0.], [1.], [2.]]])  # (B=1, Lk=3, N=1)
        >>> _multi_axis_distance(q, k, (32768,), True)
        tensor([[[[0.0000],
                  [0.0667],
                  [0.1057]],
        <BLANKLINE>
                 [[0.0667],
                  [0.0000],
                  [0.0667]]]])
        >>> # 2D example
        >>> q = torch.tensor([[[0, 0], [0, 1]]])  # (1, 2, 2)
        >>> k = torch.tensor([[[0, 0], [0, 1], [1, 0]]])  # (1, 3, 2)
        >>> _multi_axis_distance(q, k, (10, 10), False)
        tensor([[[[0.0000, 0.0000],
                  [0.0000, 0.1000],
                  [0.1000, 0.0000]],
        <BLANKLINE>
                 [[0.0000, 0.1000],
                  [0.0000, 0.0000],
                  [0.1000, 0.1000]]]])
    """
    if isinstance(axis_max, torch.Tensor):
        am = axis_max.to(device=q_pos.device, dtype=q_pos.dtype)
    else:
        am = torch.as_tensor(axis_max, dtype=q_pos.dtype, device=q_pos.device)
    am = am.view(1, 1, 1, -1)  # (1, 1, 1, N)
    delta = (q_pos.unsqueeze(2) - k_pos.unsqueeze(1)).abs()  # (B, Lq, Lk, N)
    if use_log_transform:
        # Use log1p(delta) / log1p(axis_max + eps)
        num = torch.log1p(delta)
        den = torch.log1p(am + eps)
    else:
        num = delta
        den = am + eps
    norm = num / den
    return torch.clamp(norm, 0.0, 1.0)


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
    # Max-subtract for overflow safety: exp(M) alone overflows to inf -> NaN.
    M = M - M.max(dim=-1, keepdim=True).values
    P = torch.exp(M)
    for _ in range(iters):
        P = P / P.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        P = P / P.sum(dim=-2, keepdim=True).clamp_min(1e-12)
    empty_rows = P.sum(dim=-1, keepdim=True) == 0
    if bool(empty_rows.any()):
        P = torch.where(empty_rows, torch.full_like(P, 1.0 / n), P)
    return P
