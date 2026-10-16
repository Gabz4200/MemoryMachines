r"""Sparse attention building blocks: compressor, indexer, gates, norms, output projection.

These modules implement the pieces composed by
:class:`~memory_machines.models.blocks.sparse.CompressedSparseAttention`:
:meth:`BlockCompressor` builds the ``m:1`` compressed KV store,
:class:`SparseIndexer` retrieves content-relevant entries,
:class:`BranchGate` fuses branches, and :class:`AttentionSink`,
:class:`HeadRMSNorm`, and :class:`GroupedOutputProjection` stabilize and
project the result.
"""

import warnings

import torch
import torch.nn.functional as F
from torch import nn

from .shared import _causal_comp_mask, residual_init_scale


class BlockCompressor(nn.Module):
    r"""BlockCompressor(hidden_size, num_kv_heads, head_dim, compression_ratio=4)

    Compress ``m`` consecutive tokens into one KV entry (``m:1``).

    Each block is a softmax-weighted sum over its ``m`` projected keys/values,
    with learned per-offset, per-head scores plus a ``block_bias``. Partial
    trailing blocks are padded and masked.

    Args:
        hidden_size (int): input feature dimension.
        num_kv_heads (int): number of KV heads ``Hkv``.
        head_dim (int): per-head dimension ``Dh``.
        compression_ratio (int, optional): block size ``m``; must be positive.
            Default: ``4``.

    Examples::

        >>> import torch
        >>> comp = BlockCompressor(hidden_size=16, num_kv_heads=2, head_dim=8)
        >>> kv, ends = comp(torch.randn(1, 8, 16))
        >>> kv.shape, ends.tolist()
        (torch.Size([1, 2, 2, 8]), [3, 7])
    """

    def __init__(
        self,
        hidden_size: int,
        num_kv_heads: int,
        head_dim: int,
        compression_ratio: int = 4,
        init_std: float = 0.006,
    ):
        super().__init__()
        if compression_ratio <= 0:
            raise ValueError(f"compression_ratio ({compression_ratio}) must be positive")
        self.compression_ratio = compression_ratio
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.init_std: float = init_std
        kv_dim = num_kv_heads * head_dim
        self.kv_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.score_proj = nn.Linear(hidden_size, num_kv_heads, bias=False)
        self.block_bias = nn.Parameter(torch.zeros(compression_ratio, num_kv_heads))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        ``kv_proj`` uses truncated normal
        ``N(0, initializer_range^2)`` (DeepSeek-style,
        ``initializer_range = 0.006``). ``score_proj`` and
        ``block_bias`` are zero-initialized so every block
        starts as a uniform average pool and the compressor
        learns selectivity from a neutral, position-independent
        start (cf. Set Transformer PMA, Lee et al., 2019). A
        random score init would give sharp arbitrary softmaxes
        at step 0.
        """
        std = self.init_std
        nn.init.trunc_normal_(self.kv_proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(self.score_proj.weight)
        nn.init.zeros_(self.block_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""forward(hidden_states, start_pos=0) -> (Tensor, Tensor)

        Compress a token sequence into block-level keys/values.

        Args:
            hidden_states (Tensor): input of shape ``(B, L, D)`` with ``L > 0``.
            start_pos (int, optional): absolute position of the first token;
                end positions count from here. Default: ``0``.

        Returns:
            Tensor: compressed entries of shape ``(B, Lc, Hkv, Dh)`` where
            ``Lc = ceil(L / m)``, and end positions ``comp_end`` of shape
            ``(Lc,)``.

        Examples::

            >>> import torch
            >>> comp = BlockCompressor(16, 2, 8, compression_ratio=4)
            >>> kv, ends = comp(torch.randn(2, 6, 16), start_pos=10)
            >>> kv.shape, ends.tolist()
            (torch.Size([2, 2, 2, 8]), [13, 15])
        """
        bsz, seqlen, _ = hidden_states.shape
        if seqlen == 0:
            raise ValueError("BlockCompressor got an empty sequence (seqlen 0)")
        m = self.compression_ratio
        pad = (-seqlen) % m
        if pad:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad))
        total = hidden_states.shape[1]
        lc = total // m
        kv = self.kv_proj(hidden_states).view(bsz, lc, m, self.num_kv_heads, self.head_dim)
        z = self.score_proj(hidden_states).view(bsz, lc, m, self.num_kv_heads) + self.block_bias.unsqueeze(0).unsqueeze(0)
        if pad:
            valid = torch.ones(bsz, lc, m, 1, device=hidden_states.device, dtype=torch.bool)
            valid[:, -1, seqlen % m :, 0] = False
            z = z.masked_fill(~valid, float("-inf"))
        s = F.softmax(z, dim=2)
        comp = (s.unsqueeze(-1) * kv).sum(dim=2)
        comp_end = torch.arange(start_pos + m - 1, start_pos + total, m, device=hidden_states.device)
        # Clamp the trailing partial block to the last real token: without this
        # the final entry points into padding (e.g. L=6,m=4 -> end 7 vs real 5)
        # and stays invisible until q>=7 (over-masking by up to m-1).
        if pad:
            comp_end[-1] = min(int(comp_end[-1].item()), start_pos + seqlen - 1)
        return comp, comp_end


class SparseIndexer(nn.Module):
    r"""SparseIndexer(q_latent_dim, kv_head_dim, num_index_heads=32, index_dim=128)

    Content indexer sharing a low-rank query latent with the main attention path.

    Query latents are lifted to ``(Hidx, Di)`` index queries and per-head
    weights; compressed entries are mean-pooled over heads and projected to
    index keys. Scores are a weighted sum of ReLU similarities over index
    heads, masked to causally visible entries.

    Args:
        q_latent_dim (int): query-latent dimension.
        kv_head_dim (int): compressed-KV head dimension.
        num_index_heads (int, optional): index heads ``Hidx``. Default: ``32``.
        index_dim (int, optional): per-head index dimension ``Di``. Default: ``128``.

    Examples::

        >>> import torch
        >>> idx = SparseIndexer(q_latent_dim=16, kv_head_dim=8)
        >>> q = torch.randn(1, 4, 16)
        >>> kv = torch.randn(1, 3, 2, 8)
        >>> idx.score(q, kv, torch.tensor([[3, 4, 5, 6]]), torch.tensor([1, 3, 5])).shape
        torch.Size([1, 4, 3])
    """

    def __init__(
        self,
        q_latent_dim: int,
        kv_head_dim: int,
        num_index_heads: int = 32,
        index_dim: int = 128,
        init_std: float = 0.006,
    ):
        super().__init__()
        self.num_index_heads = num_index_heads
        self.index_dim = index_dim
        self.init_std: float = init_std
        self.q_up_index = nn.Linear(q_latent_dim, num_index_heads * index_dim, bias=False)
        self.k_index_proj = nn.Linear(kv_head_dim, index_dim, bias=False)
        self.weight_proj = nn.Linear(q_latent_dim, num_index_heads, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        ``q_up_index``/``k_index_proj`` use truncated normal
        ``N(0, initializer_range^2)`` (DeepSeek-style,
        ``initializer_range = 0.006``) for the index query/key
        projections. ``weight_proj`` is zero-initialized so
        retrieval scores start at zero (neutral ranking) instead
        of collapsing onto random entries before the index
        geometry is learned.
        """
        std = self.init_std
        nn.init.trunc_normal_(self.q_up_index.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.trunc_normal_(self.k_index_proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(self.weight_proj.weight)

    def _index_heads(self, query_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""_index_heads(query_latent) -> (Tensor, Tensor)

        Lift query latents to index queries and mixing weights.

        Args:
            query_latent (Tensor): latents of shape ``(B, Lq, Dlat)``.

        Returns:
            Tensor: index queries of shape ``(B, Lq, Hidx, Di)`` and head
            weights of shape ``(B, Lq, Hidx)``.

        Examples::

            >>> import torch
            >>> idx = SparseIndexer(16, 8, num_index_heads=4, index_dim=8)
            >>> qi, w = idx._index_heads(torch.randn(1, 3, 16))
            >>> qi.shape, w.shape
            (torch.Size([1, 3, 4, 8]), torch.Size([1, 3, 4]))
        """
        bsz, lq, _ = query_latent.shape
        qi = self.q_up_index(query_latent).view(bsz, lq, self.num_index_heads, self.index_dim)
        return qi, self.weight_proj(query_latent)

    def score(
        self,
        query_latent: torch.Tensor,
        compressed_kv: torch.Tensor,
        q_pos: torch.Tensor,
        comp_end: torch.Tensor,
    ) -> torch.Tensor:
        r"""score(query_latent, compressed_kv, q_pos, comp_end) -> Tensor

        Score every compressed entry for every query.

        Args:
            query_latent (Tensor): latents of shape ``(B, Lq, Dlat)``.
            compressed_kv (Tensor): store of shape ``(B, Lc, Hkv, Dh)``.
            q_pos (Tensor): query positions of shape ``(B, Lq)``.
            comp_end (Tensor): entry end positions, ``(Lc,)`` or ``(B, Lc)``.

        Returns:
            Tensor: scores of shape ``(B, Lq, Lc)`` with future entries set
            to ``-inf``.

        Examples::

            >>> import torch
            >>> idx = SparseIndexer(16, 8)
            >>> idx.score(torch.randn(1, 2, 16), torch.randn(1, 3, 2, 8),
            ...     torch.tensor([[5, 6]]), torch.tensor([1, 5, 9])).shape
            torch.Size([1, 2, 3])
        """
        qi, w = self._index_heads(query_latent)
        kvec = self.k_index_proj(compressed_kv.mean(dim=2))
        dots = torch.einsum("bqhd,bkd->bqhk", qi, kvec)
        dots = F.relu(dots)
        scores = (w.unsqueeze(-1) * dots).sum(dim=2)
        visible = _causal_comp_mask(q_pos, comp_end)
        return scores.masked_fill(~visible, float("-inf"))

    @staticmethod
    def topk(scores: torch.Tensor, k: int) -> torch.Tensor:
        r"""topk(scores, k) -> Tensor

        Select the ``k`` highest-scoring entries per query.

        Args:
            scores (Tensor): scores of shape ``(B, Lq, Lc)``.
            k (int): entries to keep; clamped to ``Lc``.

        Returns:
            Tensor: indices of shape ``(B, Lq, min(k, Lc))``.

        Examples::

            >>> import torch
            >>> SparseIndexer.topk(torch.tensor([[[1., 5., 2.]]]), 2)
            tensor([[[1, 2]]])
        """
        lc = scores.shape[-1]
        if k > lc:
            warnings.warn(f"SparseIndexer.topk: k ({k}) exceeds the number of entries ({lc}); clamping to {lc}", stacklevel=2)
        k = min(k, lc)
        return scores.topk(k, dim=-1).indices

    def build_pool(
        self,
        scores: torch.Tensor,
        pool_blocks: int,
        block_size: int,
    ) -> torch.Tensor:
        r"""build_pool(scores, pool_blocks, block_size) -> Tensor

        Build a hierarchical candidate pool of whole score blocks.

        Entries are grouped into blocks of ``block_size``; the top
        ``pool_blocks`` blocks (by block-max) are expanded back to entry ids.

        Args:
            scores (Tensor): scores of shape ``(B, Lq, Lc)``.
            pool_blocks (int): number of blocks to keep per query.
            block_size (int): entries per block.

        Returns:
            Tensor: pooled entry ids of shape ``(B, Lq, nb * block_size)``
            clamped to ``[0, Lc - 1]``. Clamping can duplicate the last
            valid id when ``Lc`` is not divisible by ``block_size``; the
            duplicates are harmless (rescore masks them consistently and
            they consume at most ``block_size - 1`` slots) but waste a
            small amount of top-k budget.

        Examples::

            >>> import torch
            >>> idx = SparseIndexer(16, 8)
            >>> idx.build_pool(torch.randn(1, 2, 16), pool_blocks=2, block_size=4).shape
            torch.Size([1, 2, 8])
        """
        bsz, lq, lc = scores.shape
        num_blocks = (lc + block_size - 1) // block_size
        nb = min(pool_blocks, num_blocks)
        # Pad to a whole number of blocks, then select the top-nb by block-max.
        pad = num_blocks * block_size - lc
        if pad:
            scores = F.pad(scores, (0, pad), value=float("-inf"))
        blocked = scores.view(bsz, lq, num_blocks, block_size)
        block_scores = blocked.amax(dim=-1)
        top_blocks = block_scores.topk(nb, dim=-1).indices
        base = (top_blocks * block_size).unsqueeze(-1) + torch.arange(block_size, device=scores.device).view(1, 1, 1, -1)
        pool = base.reshape(bsz, lq, nb * block_size)
        return pool.clamp(max=lc - 1)

    def rescore_pool(
        self,
        query_latent: torch.Tensor,
        indexer_k: torch.Tensor,
        pool_indices: torch.Tensor,
        q_pos: torch.Tensor,
        comp_end: torch.Tensor,
    ) -> torch.Tensor:
        r"""rescore_pool(query_latent, indexer_k, pool_indices, q_pos, comp_end) -> Tensor

        Rescore a prebuilt candidate pool with fresh query latents.

        Used by ``reindex`` layers to refine selection without rescoring the
        full store.

        Args:
            query_latent (Tensor): latents of shape ``(B, Lq, Dlat)``.
            indexer_k (Tensor): cached index keys of shape ``(B, Lc, Di)``.
            pool_indices (Tensor): candidate ids of shape ``(B, Lq, P)``; must
                be non-negative.
            q_pos (Tensor): query positions of shape ``(B, Lq)``.
            comp_end (Tensor): entry end positions, ``(Lc,)`` or ``(B, Lc)``.

        Returns:
            Tensor: pool scores of shape ``(B, Lq, P)`` with invisible or
            out-of-range entries set to ``-inf``.

        Examples::

            >>> import torch
            >>> idx = SparseIndexer(16, 8, num_index_heads=4, index_dim=8)
            >>> idx.rescore_pool(torch.randn(1, 2, 16), torch.randn(1, 6, 8),
            ...     torch.tensor([[[0, 1, 2]]]), torch.tensor([[5, 6]]),
            ...     torch.tensor([1, 3, 5, 7, 9, 11])).shape
            torch.Size([1, 2, 3])
        """
        bsz = query_latent.shape[0]
        if bool((pool_indices < 0).any()):
            raise ValueError("pool_indices must be non-negative (negative ids are never valid compressed entries)")
        lc = indexer_k.shape[1]
        safe_pool = pool_indices.clamp(max=lc - 1)
        qi, w = self._index_heads(query_latent)
        batch_idx = torch.arange(bsz, device=pool_indices.device).view(bsz, 1, 1)
        k_pool = indexer_k[batch_idx, safe_pool]
        dots = F.relu(torch.einsum("bqhd,bqpd->bqhp", qi, k_pool))
        scores = (w.unsqueeze(-1) * dots).sum(dim=2)
        if comp_end.dim() == 1:
            pool_ends = comp_end[safe_pool]
        else:
            pool_ends = comp_end[batch_idx, safe_pool]
        visible = (pool_ends <= q_pos.unsqueeze(-1)) & (pool_indices < lc)
        return scores.masked_fill(~visible, float("-inf"))


class BranchGate(nn.Module):
    r"""BranchGate(hidden_size, num_branches=2)

    Sigmoid fusion gates over sparse branches (dense/selected/window).

    Args:
        hidden_size (int): input feature dimension.
        num_branches (int, optional): number of branches to weight. Default: ``2``.

    Examples::

        >>> import torch
        >>> gate = BranchGate(hidden_size=16, num_branches=3)
        >>> gate(torch.randn(2, 5, 16)).shape
        torch.Size([2, 5, 3])
    """

    def __init__(self, hidden_size: int, num_branches: int = 2):
        super().__init__()
        self.proj = nn.Linear(hidden_size, num_branches)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Zero-initialized so all branches fuse equally (``sigmoid(0) = 0.5``)
        at step 0; the model learns to prefer branches from a balanced
        start instead of a random early commitment (cf. Highway-network
        gating, Srivastava et al., 2015).
        """
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""forward(hidden_states) -> Tensor

        Compute per-token branch weights.

        Args:
            hidden_states (Tensor): input of shape ``(B, L, D)``.

        Returns:
            Tensor: gates of shape ``(B, L, num_branches)`` in ``(0, 1)``.

        Examples::

            >>> import torch
            >>> BranchGate(8)(torch.randn(1, 4, 8)).shape
            torch.Size([1, 4, 2])
        """
        return torch.sigmoid(self.proj(hidden_states))


class HeadRMSNorm(nn.Module):
    r"""HeadRMSNorm(head_dim, eps=1e-6)

    RMS normalization over the head dimension.

    .. math::
        \bar{x} = \frac{x}{\sqrt{\mathrm{mean}(x^2) + \epsilon}} \odot w

    Args:
        head_dim (int): normalized dimension ``Dh``.
        eps (float, optional): numerical stability term. Default: ``1e-6``.

    Examples::

        >>> import torch
        >>> HeadRMSNorm(8)(torch.randn(1, 2, 4, 8)).shape
        torch.Size([1, 2, 4, 8])
    """

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Unit gain so the norm starts as a pure RMS rescale (identity
        up to scale), the standard RMSNorm init (Zhang & Sennrich, 2019).
        """
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x) -> Tensor

        Normalize the last dimension with RMSNorm.

        Args:
            x (Tensor): input of shape ``(..., Dh)``.

        Returns:
            Tensor: normalized output with the same shape as ``x``.

        Examples::

            >>> import torch
            >>> HeadRMSNorm(4)(torch.randn(2, 3, 4)).shape
            torch.Size([2, 3, 4])
        """
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class AttentionSink(nn.Module):
    r"""AttentionSink(num_heads)

    Learnable per-head sink logit absorbing excess attention mass.

    With logits :math:`z` and sink :math:`s_h`, weights are
    :math:`e^{z} / (\sum e^{z} + e^{s_h})`, so the rows sum to less than one
    and fully-masked rows yield zeros instead of NaNs.

    Args:
        num_heads (int): number of attention heads ``H``.

    Examples::

        >>> import torch
        >>> sink = AttentionSink(num_heads=2)
        >>> sink.weights(torch.randn(1, 2, 3, 4)).shape
        torch.Size([1, 2, 3, 4])
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.sink_logit = nn.Parameter(torch.zeros(num_heads))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Zero sink logit (``exp(0) = 1``) so each head starts with a modest
        neutral sink mass of ``1 / (Lk + 1)`` per row; the head then learns
        how much excess mass to absorb (cf. attention-sink behavior, Xiao
        et al., 2023).
        """
        nn.init.zeros_(self.sink_logit)

    def weights(self, logits: torch.Tensor) -> torch.Tensor:
        r"""weights(logits) -> Tensor

        Convert logits to sink-deflated attention weights.

        Args:
            logits (Tensor): raw scores of shape ``(B, H, Lq, Lk)``.

        Returns:
            Tensor: weights with the same shape as ``logits``; rows sum to
            at most one.

        Examples::

            >>> import torch
            >>> round(float(AttentionSink(2).weights(torch.zeros(1, 2, 1, 3)).sum()), 4)
            1.5
        """
        # Upcast to float32: exp() in fp16 overflows at >11 (bf16 at >88) giving
        # inf/inf=NaN; compute in fp32 then cast back.
        l32 = logits.float()
        sink = self.sink_logit.to(torch.float32).view(1, -1, 1, 1)
        denom = torch.exp(l32).sum(dim=-1, keepdim=True) + torch.exp(sink)
        return (torch.exp(l32) / denom).to(logits.dtype)


class GroupedOutputProjection(nn.Module):
    r"""GroupedOutputProjection(num_heads, head_dim, num_groups, group_dim, hidden_size)

    Grouped output projection over attention heads.

    Heads are split into ``num_groups`` contiguous groups; each group is
    projected from ``(H/g) * Dh`` to ``group_dim``, concatenated, and mapped
    back to ``hidden_size``.

    Args:
        num_heads (int): number of heads ``H``; must be divisible by
            ``num_groups``.
        head_dim (int): per-head dimension ``Dh``.
        num_groups (int): number of groups ``g``.
        group_dim (int): per-group output dimension.
        hidden_size (int): final output dimension.

    Examples::

        >>> import torch
        >>> proj = GroupedOutputProjection(4, 8, num_groups=2, group_dim=16, hidden_size=32)
        >>> proj(torch.randn(1, 4, 5, 8)).shape
        torch.Size([1, 5, 32])
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_groups: int,
        group_dim: int,
        hidden_size: int,
        init_std: float = 0.006,
        num_layers: int = 40,
        residual_init_scale: bool = True,
    ):
        super().__init__()
        if num_heads % num_groups != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_groups ({num_groups})")
        self.num_groups = num_groups
        self.group_heads = num_heads // num_groups
        self.init_std: float = init_std
        self.num_layers: int = num_layers
        self.residual_init_scale: bool = residual_init_scale
        self.group_projs = nn.ModuleList([nn.Linear(self.group_heads * head_dim, group_dim, bias=False) for _ in range(num_groups)])
        self.out_proj = nn.Linear(num_groups * group_dim, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Truncated normal ``N(0, init_std^2)`` for the
        per-group projections; the final ``out_proj`` is
        either zero-initialized (identity residual start,
        Fixup/ReZero) or depth-scaled to
        ``init_std / sqrt(2N)`` (Megatron-LM/Qwen residual
        scaling) per ``residual_init_scale``.
        """
        std = self.init_std
        scale = residual_init_scale(std, self.num_layers, self.residual_init_scale)
        for proj in self.group_projs:
            assert isinstance(proj, nn.Linear), "group_projs holds nn.Linear"  # type: ignore[assert-type]
            nn.init.trunc_normal_(proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        if scale is None:
            nn.init.zeros_(self.out_proj.weight)
        else:
            nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=scale, a=-2 * scale, b=2 * scale)

    def forward(self, heads: torch.Tensor) -> torch.Tensor:
        r"""forward(heads) -> Tensor

        Project per-head outputs back to the model dimension.

        Args:
            heads (Tensor): head outputs of shape ``(B, H, Lq, Dh)``.

        Returns:
            Tensor: fused output of shape ``(B, Lq, hidden_size)``.

        Examples::

            >>> import torch
            >>> proj = GroupedOutputProjection(4, 8, 2, 16, 32)
            >>> proj(torch.randn(1, 4, 5, 8)).shape
            torch.Size([1, 5, 32])
        """
        bsz, _, lq, _ = heads.shape
        x = heads.transpose(1, 2)
        groups = x.chunk(self.num_groups, dim=2)
        outs = [proj(g.reshape(bsz, lq, -1)) for proj, g in zip(self.group_projs, groups)]
        return self.out_proj(torch.cat(outs, dim=-1))
