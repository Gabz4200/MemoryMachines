r"""Sparse attention layers: NSA/CSA/HCA and CSA2 Full/Reindex/Reuse.

:class:`CompressedSparseAttention` compresses keys/values ``m:1`` with
:class:`~memory_machines.models.blocks.compression.BlockCompressor`,
retrieves top-``k`` entries with
:class:`~memory_machines.models.blocks.compression.SparseIndexer`, and fuses
sparse + sliding-window branches. :class:`NativeSparseAttention`,
:class:`HeavilyCompressedAttention`, and
:class:`CompressedSparseAttention2` specialize the branch mix and cross-layer
sharing pattern.
"""

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from memory_machines.utils.config_schema import MemoryMachineConfig

from .attention import GroupedQueryAttention, SlidingWindowAttention
from .compression import (
    AttentionSink,
    BlockCompressor,
    BranchGate,
    GroupedOutputProjection,
    HeadRMSNorm,
    SparseIndexer,
)
from .shared import CompressedCache, CSA2SharedState, _past_total_len, _resolve_positions, _update_swa_pos


class CompressedSparseAttention(GroupedQueryAttention):
    r"""CompressedSparseAttention(config, compression_ratio=None, use_indexer=True, use_dense_branch=False, top_k=None, window_size=None)

    NSA/CSA/HCA sparse attention layer.

    The layer drops the parent Q/K/O projections and instead uses a low-rank
    query latent (``q_down``/``q_up_main``), a shared
    :class:`~memory_machines.models.blocks.compression.BlockCompressor`, an
    optional :class:`~memory_machines.models.blocks.compression.SparseIndexer`,
    and an internal :class:`~memory_machines.models.blocks.attention.SlidingWindowAttention`
    branch. Branch outputs (dense and/or top-``k`` selected + window) are fused
    by :class:`~memory_machines.models.blocks.compression.BranchGate` and
    :class:`~memory_machines.models.blocks.compression.GroupedOutputProjection`.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides hidden sizes, head counts, ``csa_*``/``swa_*`` settings,
            and CVPE/FRVM toggles.
        compression_ratio (int, optional): block size ``m``. Default: ``None``
            (falls back to ``config.csa_compression_ratio``).
        use_indexer (bool, optional): whether to enable top-``k`` retrieval.
            Default: ``True``.
        use_dense_branch (bool, optional): whether to attend the full
            compressed store. Default: ``False``.
        top_k (int, optional): entries selected per query. Default: ``None``
            (falls back to ``config.csa_top_k``).
        window_size (int, optional): SWA branch width. Default: ``None``
            (falls back to ``config.swa_window_size``).

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> layer = CompressedSparseAttention(MemoryMachineConfig(hidden_size=32,
        ...     num_attention_heads=4, num_key_value_heads=2, csa_top_k=4))
        >>> out, _, cache = layer(torch.randn(1, 8, 32), use_cache=True)
        >>> out.shape
        torch.Size([1, 8, 32])
    """

    def __init__(
        self,
        config: MemoryMachineConfig,
        compression_ratio: int | None = None,
        use_indexer: bool = True,
        use_dense_branch: bool = False,
        top_k: int | None = None,
        window_size: int | None = None,
    ):
        super().__init__(config)
        del self.q_proj
        del self.k_proj
        del self.o_proj
        self.compression_ratio = compression_ratio if compression_ratio is not None else config.csa_compression_ratio
        self.use_indexer = use_indexer
        self.use_dense_branch = use_dense_branch
        self.top_k = top_k if top_k is not None else config.csa_top_k
        self.window_size = window_size if window_size is not None else config.swa_window_size
        q_latent = config.csa_q_latent_dim if config.csa_q_latent_dim > 0 else config.hidden_size
        self.q_latent_dim = q_latent
        self.q_down = nn.Linear(config.hidden_size, q_latent, bias=False)
        self.q_up_main = nn.Linear(q_latent, self.num_heads * self.head_dim, bias=False)
        self.compressor = BlockCompressor(config.hidden_size, self.num_key_value_heads, self.head_dim, self.compression_ratio)
        self.indexer = SparseIndexer(q_latent, self.head_dim, config.csa_indexer_heads, config.csa_indexer_dim) if use_indexer else None
        n_br = (1 if use_dense_branch or not use_indexer else 0) + (1 if use_indexer else 0) + 1
        self.branch_gate = BranchGate(config.hidden_size, n_br)
        self.q_norm = HeadRMSNorm(self.head_dim, config.rms_norm_eps) if config.csa_use_qkv_norm else None
        self.kv_norm = HeadRMSNorm(self.head_dim, config.rms_norm_eps) if config.csa_use_qkv_norm else None
        self.sink = AttentionSink(self.num_heads) if config.csa_use_sink else None
        self.grouped_out = GroupedOutputProjection(self.num_heads, self.head_dim, config.csa_num_output_groups, config.csa_output_group_dim, config.hidden_size)
        self.swa = SlidingWindowAttention(config, self.window_size)

    def _project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""_project_qkv(hidden_states) -> (Tensor, Tensor, Tensor)

        Disabled parent projection (no Q/K/O projections here).

        Args:
            hidden_states (Tensor): input of shape ``(B, L, hidden)``.

        Returns:
            Tensor: never returns; always raises ``NotImplementedError``. Use
            :meth:`_make_main_queries` instead.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> layer = CompressedSparseAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, csa_top_k=2))
            >>> try:
            ...     layer._project_qkv(__import__("torch").randn(1, 2, 16))
            ... except NotImplementedError:
            ...     print("raised")
            raised
        """
        raise NotImplementedError("CompressedSparseAttention has no parent Q/K/O projections; use _make_main_queries().")

    def _project_heads(self, heads: torch.Tensor, bsz: int, q_len: int) -> torch.Tensor:
        r"""_project_heads(heads, bsz, q_len) -> Tensor

        Disabled parent head projection (uses grouped fusion instead).

        Args:
            heads (Tensor): per-head outputs.
            bsz (int): batch size.
            q_len (int): query length.

        Returns:
            Tensor: never returns; always raises ``NotImplementedError``. Use
            :meth:`_fuse_branches` plus ``grouped_out`` instead.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> layer = CompressedSparseAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, csa_top_k=2))
            >>> try:
            ...     layer._project_heads(torch.randn(1, 2, 2, 8), 1, 2)
            ... except NotImplementedError:
            ...     print("raised")
            raised
        """
        raise NotImplementedError("CompressedSparseAttention has no o_proj; use _fuse_branches() + grouped_out.")

    def _core_attend(
        self,
        q_heads: torch.Tensor,
        kv: torch.Tensor,
        v_query: torch.Tensor,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        r"""_core_attend(q_heads, kv, v_query, q_pos, k_pos, causal=True) -> Tensor

        Score queries against one KV set with optional norm/sink/FRVM.

        Applies optional :class:`~memory_machines.models.blocks.compression.HeadRMSNorm`,
        scaled dot products plus CVPE bias, causal masking, then either sink
        weights or softmax (sanitized), dropout, and plain or FRVM-gated
        reduction.

        Args:
            q_heads (Tensor): queries of shape ``(B, H, Lq, Dh)``.
            kv (Tensor): shared keys/values of shape ``(B, Hkv, Lk, Dh)``.
            v_query (Tensor): per-query reference values ``(B, H, Lq, Dh)``.
            q_pos (Tensor): query positions ``(B, Lq)``.
            k_pos (Tensor): key positions ``(B, Lk)``.
            causal (bool, optional): whether to mask future entries. Default: ``True``.

        Returns:
            Tensor: per-head outputs of shape ``(B, H, Lq, Dh)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> layer = CompressedSparseAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, csa_top_k=2,
            ...     csa_use_qkv_norm=False, csa_use_sink=False, frvm_enabled=False,
            ...     cvpe_enabled=False))
            >>> layer._core_attend(torch.randn(1, 2, 3, 8), torch.randn(1, 1, 5, 8),
            ...     torch.randn(1, 2, 3, 8), torch.tensor([[4, 5, 6]]),
            ...     torch.tensor([[0, 1, 2, 3, 4]])).shape
            torch.Size([1, 2, 3, 8])
        """
        if self.q_norm is not None:
            q_heads = self.q_norm(q_heads)
            kv = self.kv_norm(kv)
        kv_exp = kv.repeat_interleave(self.num_key_value_groups, dim=1)
        logits = torch.matmul(q_heads, kv_exp.transpose(2, 3)) / math.sqrt(self.head_dim)
        if self.cvpe_bias is not None:
            logits = logits + self.cvpe_bias(q_heads, q_pos, k_pos).to(logits.dtype)
        if causal:
            causal_mask = k_pos.unsqueeze(1) <= q_pos.unsqueeze(-1)
            logits = logits.masked_fill(~causal_mask.unsqueeze(1), float("-inf"))
            row_ok = causal_mask.any(dim=-1).view(logits.shape[0], 1, logits.shape[2], 1)
        else:
            row_ok = None
        if self.sink is not None:
            weights = self.sink.weights(logits)
        else:
            weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
            if row_ok is not None:
                weights = torch.where(row_ok, weights, torch.zeros_like(weights))
        weights = self.attention_dropout(weights)
        if self.frvm_gate is not None:
            return self.frvm_gate.gated_reduction(weights, kv_exp, v_query, chunk_size=self.frvm_chunk_size)
        return torch.matmul(weights, kv_exp)

    @staticmethod
    def _selected_positions(full_ends: torch.Tensor, topk: torch.Tensor) -> torch.Tensor:
        r"""_selected_positions(full_ends, topk) -> Tensor

        Gather end positions of the selected compressed entries.

        Args:
            full_ends (Tensor): all end positions, ``(Lc,)`` or ``(B, Lc)``.
            topk (Tensor): selected ids of shape ``(B, Lq, K)``.

        Returns:
            Tensor: selected positions of shape ``(B, Lq, K)``.

        Examples::

            >>> import torch
            >>> CompressedSparseAttention._selected_positions(
            ...     torch.tensor([1, 3, 5]), torch.tensor([[[0, 2]]]))
            tensor([[[1, 5]]])
        """
        if full_ends.dim() == 1:
            return full_ends[topk]
        bsz = topk.shape[0]
        return full_ends[torch.arange(bsz, device=full_ends.device).view(bsz, 1, 1), topk]

    @staticmethod
    def _gather_selected(full_comp: torch.Tensor, topk: torch.Tensor) -> torch.Tensor:
        r"""_gather_selected(full_comp, topk) -> Tensor

        Gather the selected compressed entries.

        Args:
            full_comp (Tensor): full store of shape ``(B, Lc, Hkv, Dh)``.
            topk (Tensor): selected ids of shape ``(B, Lq, K)``.

        Returns:
            Tensor: selected entries of shape ``(B, Lq, K, Hkv, Dh)``.

        Examples::

            >>> import torch
            >>> CompressedSparseAttention._gather_selected(
            ...     torch.randn(1, 5, 2, 4), torch.tensor([[[0, 1]]])).shape
            torch.Size([1, 1, 2, 2, 4])
        """
        bsz = topk.shape[0]
        return full_comp[torch.arange(bsz, device=full_comp.device).view(bsz, 1, 1), topk]

    def _make_main_queries(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""_make_main_queries(hidden_states) -> (Tensor, Tensor, Tensor)

        Build the shared query latent, main queries, and reference values.

        Args:
            hidden_states (Tensor): input of shape ``(B, Lq, hidden)``.

        Returns:
            Tensor: query latent ``cq`` ``(B, Lq, Dlat)``, main queries
            ``(B, H, Lq, Dh)``, and expanded reference values ``(B, H, Lq, Dh)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> layer = CompressedSparseAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, csa_top_k=2))
            >>> cq, q, v = layer._make_main_queries(torch.randn(1, 4, 16))
            >>> cq.shape, q.shape, v.shape
            (torch.Size([1, 4, 16]), torch.Size([1, 2, 4, 8]), torch.Size([1, 2, 4, 8]))
        """
        bsz, q_len, _ = hidden_states.shape
        cq = self.q_down(hidden_states)
        q_main = self.q_up_main(cq).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_q = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).repeat_interleave(self.num_key_value_groups, dim=1)
        return cq, q_main, v_q

    def _fuse_branches(self, branches: list[torch.Tensor], hidden_states: torch.Tensor) -> torch.Tensor:
        r"""_fuse_branches(branches, hidden_states) -> Tensor

        Fuse branch outputs with learned sigmoid gates.

        Args:
            branches (list of Tensor): branch outputs, each ``(B, H, Lq, Dh)``
                or merged ``(B, Lq, hidden)``; count must match the gate heads.
            hidden_states (Tensor): gate inputs of shape ``(B, Lq, hidden)``.

        Returns:
            Tensor: fused heads of shape ``(B, H, Lq, Dh)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> layer = CompressedSparseAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, csa_top_k=2))
            >>> layer._fuse_branches([torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8)],
            ...     torch.randn(1, 3, 16)).shape
            torch.Size([1, 2, 3, 8])
        """
        bsz, q_len, _ = hidden_states.shape
        gates = self.branch_gate(hidden_states).unsqueeze(-1)
        flat = [b.transpose(1, 2).reshape(bsz, q_len, -1) if b.dim() == 4 else b for b in branches]
        stacked = torch.stack(flat, dim=2)
        fused = (gates * stacked).sum(dim=2)
        return fused.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _selected_branch(
        self,
        q_main: torch.Tensor,
        sel: torch.Tensor,
        sel_pos: torch.Tensor,
        v_q: torch.Tensor,
        q_pos: torch.Tensor,
    ) -> torch.Tensor:
        r"""_selected_branch(q_main, sel, sel_pos, v_q, q_pos) -> Tensor

        Attend each query to its own top-``k`` selection (per-query loop).

        Args:
            q_main (Tensor): queries of shape ``(B, H, Lq, Dh)``.
            sel (Tensor): selected entries ``(B, Lq, K, Hkv, Dh)``.
            sel_pos (Tensor): selected positions ``(B, Lq, K)``.
            v_q (Tensor): reference values ``(B, H, Lq, Dh)``.
            q_pos (Tensor): query positions ``(B, Lq)``.

        Returns:
            Tensor: per-head outputs of shape ``(B, H, Lq, Dh)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> layer = CompressedSparseAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, csa_top_k=2,
            ...     frvm_enabled=False, cvpe_enabled=False, csa_use_sink=False,
            ...     csa_use_qkv_norm=False))
            >>> layer._selected_branch(torch.randn(1, 2, 2, 8), torch.randn(1, 2, 2, 1, 8),
            ...     torch.tensor([[[0, 1], [0, 1]]]), torch.randn(1, 2, 2, 8),
            ...     torch.tensor([[3, 4]])).shape
            torch.Size([1, 2, 2, 8])
        """
        # Per-query loop; fuse into one batched gather-matmul for long context.
        outs = []
        for t in range(q_main.shape[2]):
            k_t = sel[:, t, :, :, :].transpose(1, 2)
            outs.append(
                self._core_attend(
                    q_main[:, :, t : t + 1, :],
                    k_t,
                    v_q[:, :, t : t + 1, :],
                    q_pos[:, t : t + 1],
                    sel_pos[:, t, :],
                    causal=True,
                )
            )
        return torch.cat(outs, dim=2)

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past: CompressedCache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, CompressedCache | None]:
        r"""forward(hidden_states, attention_mask=None, position_ids=None, past=None, output_attentions=False, use_cache=False) -> (Tensor, None, CompressedCache | None)

        Run sparse attention with global compressed + window caches.

        Compresses the current chunk, appends it to the cached global store,
        runs dense and/or top-``k`` selected branches plus the SWA branch, and
        fuses them into model-dimension outputs.

        .. note::
            Full attention maps are never materialized; ``output_attentions``
            warns and returns ``None`` for weights.

        Args:
            hidden_states (Tensor): input of shape ``(B, Lq, hidden)``.
            attention_mask (Tensor, optional): mask forwarded to the SWA
                branch. Default: ``None``.
            position_ids (Tensor, optional): absolute positions. Default: ``None``.
            past (CompressedCache, optional): incremental cache. Default: ``None``.
            output_attentions (bool, optional): accepted for API parity; always
                yields ``None`` weights. Default: ``False``.
            use_cache (bool, optional): whether to return the updated cache.
                Default: ``False``.

        Returns:
            Tensor: fused outputs ``(B, Lq, hidden)``, ``None`` weights, and
            the updated :class:`~memory_machines.models.blocks.shared.CompressedCache`
            or ``None``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> layer = CompressedSparseAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, csa_top_k=2))
            >>> out, w, cache = layer(torch.randn(1, 6, 16), use_cache=True)
            >>> out.shape, w, cache.total_len
            (torch.Size([1, 6, 16]), None, 6)
        """
        bsz, q_len, _ = hidden_states.shape
        if output_attentions:
            warnings.warn("CompressedSparseAttention does not materialize full attentions; returning None", UserWarning, stacklevel=2)
        past_len_raw = _past_total_len(past) or 0
        q_pos, _ = _resolve_positions(hidden_states, q_len, past_len_raw, position_ids)
        new_comp, new_ends = self.compressor(hidden_states, start_pos=past_len_raw)
        if past is not None and past.compressed_kv is not None:
            if past.comp_end_pos is None:
                raise ValueError("past.comp_end_pos is None while past.compressed_kv is set (corrupt CompressedCache)")
            full_comp = torch.cat([past.compressed_kv, new_comp], dim=1)
            full_ends = torch.cat([past.comp_end_pos.to(new_ends.device), new_ends], dim=0)
        else:
            full_comp, full_ends = new_comp, new_ends
        cq, q_main, v_q = self._make_main_queries(hidden_states)
        k_pos_full = full_ends.unsqueeze(0).expand(bsz, -1)
        branches: list[torch.Tensor] = []
        if self.use_dense_branch or not self.use_indexer:
            branches.append(self._core_attend(q_main, full_comp.transpose(1, 2), v_q, q_pos, k_pos_full, causal=True))
        if self.indexer is not None:
            scores = self.indexer.score(cq, full_comp, q_pos, full_ends)
            topk = self.indexer.topk(scores, self.top_k)
            sel = self._gather_selected(full_comp, topk)
            branches.append(self._selected_branch(q_main, sel, self._selected_positions(full_ends, topk), v_q, q_pos))
        swa_past = None if past is None else (past.swa_k, past.swa_v)
        swa_past_pos = None if past is None else past.swa_pos
        win_out, _, swa_present = self.swa(hidden_states, attention_mask, position_ids, swa_past, False, use_cache=True, past_len_override=past_len_raw, past_positions=swa_past_pos)
        branches.append(win_out)
        out = self.grouped_out(self._fuse_branches(branches, hidden_states))
        present = None
        if use_cache:
            if swa_present is None:
                raise ValueError("SWA branch returned no cache under use_cache=True")
            w = swa_present[0].shape[2]
            prev_swa_pos = None if past is None else past.swa_pos
            present = CompressedCache(full_comp, full_ends, swa_present[0], swa_present[1], _update_swa_pos(prev_swa_pos, q_pos, w), past_len_raw + q_len)
        return out, None, present


class NativeSparseAttention(CompressedSparseAttention):
    r"""NativeSparseAttention(config, compression_ratio=2)

    NSA variant: dense compressed + top-``k`` selected + window branches.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            base sparse-layer hyperparameters.
        compression_ratio (int, optional): block size ``m``. Default: ``2``.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> layer = NativeSparseAttention(MemoryMachineConfig(hidden_size=16,
        ...     num_attention_heads=2, num_key_value_heads=1, csa_top_k=2))
        >>> layer(torch.randn(1, 6, 16))[0].shape
        torch.Size([1, 6, 16])
    """

    def __init__(self, config: MemoryMachineConfig, compression_ratio: int = 2):
        super().__init__(config, compression_ratio=compression_ratio, use_indexer=True, use_dense_branch=True)


class HeavilyCompressedAttention(CompressedSparseAttention):
    r"""HeavilyCompressedAttention(config, compression_ratio=None)

    HCA variant: dense heavily-compressed + window branches (no indexer).

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``hca_compression_ratio`` when ``compression_ratio`` is unset.
        compression_ratio (int, optional): block size ``m``. Default: ``None``
            (falls back to ``config.hca_compression_ratio``).

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> layer = HeavilyCompressedAttention(MemoryMachineConfig(hidden_size=16,
        ...     num_attention_heads=2, num_key_value_heads=1))
        >>> layer(torch.randn(1, 8, 16))[0].shape
        torch.Size([1, 8, 16])
    """

    def __init__(self, config: MemoryMachineConfig, compression_ratio: int | None = None):
        super().__init__(
            config,
            compression_ratio=compression_ratio if compression_ratio is not None else config.hca_compression_ratio,
            use_indexer=False,
            use_dense_branch=True,
        )


class CompressedSparseAttention2(CompressedSparseAttention):
    r"""CompressedSparseAttention2(config, compression_ratio=None, mode="full", top_k=None)

    CSA2 layer with cross-layer ``full``/``reindex``/``reuse`` sharing.

    - ``full`` compresses, scores the whole store, and publishes
      ``compressed_kv``/``indexer_k``/``topk``/``pool`` to the shared state.
    - ``reindex`` rescues the shared pool with fresh latents (or rescores the
      store) and republishes ``topk``.
    - ``reuse`` reuses the published ``topk`` (broadcasting the last row when
      lengths differ) without rescoring.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides sparse hyperparameters plus ``csa_hier_pool_blocks``,
            ``csa_hier_pool_positions``, and ``csa_hier_block_size``.
        compression_ratio (int, optional): block size ``m``. Default: ``None``
            (falls back to ``config.csa_compression_ratio``).
        mode (str, optional): one of ``"full"``, ``"reindex"``, ``"reuse"``.
            Default: ``"full"``.
        top_k (int, optional): entries per query. Default: ``None`` (falls
            back to ``config.csa_top_k``).

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> from memory_machines.models.blocks.shared import CSA2SharedState
        >>> cfg = MemoryMachineConfig(hidden_size=16, num_attention_heads=2,
        ...     num_key_value_heads=1, csa_top_k=2)
        >>> full = CompressedSparseAttention2(cfg, mode="full")
        >>> out, shared = full.forward_shared(torch.randn(1, 6, 16))
        >>> out.shape, shared.topk_indices.shape
        (torch.Size([1, 6, 16]), torch.Size([1, 6, 2]))
    """

    def __init__(
        self,
        config: MemoryMachineConfig,
        compression_ratio: int | None = None,
        mode: str = "full",
        top_k: int | None = None,
    ):
        super().__init__(config, compression_ratio=compression_ratio, use_indexer=True, use_dense_branch=False, top_k=top_k)
        if mode not in ("full", "reindex", "reuse"):
            raise ValueError(f"mode ({mode}) must be full/reindex/reuse")
        self.mode = mode

    def _hier_pool_blocks(self) -> int:
        r"""_hier_pool_blocks() -> int

        Compute the capped hierarchical pool-block count.

        Args:
            None

        Returns:
            int: ``min(csa_hier_pool_blocks, max(1, csa_hier_pool_positions //
            csa_hier_block_size))``.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> CompressedSparseAttention2(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1))._hier_pool_blocks() > 0
            True
        """
        cfg = self.config
        return min(cfg.csa_hier_pool_blocks, max(1, cfg.csa_hier_pool_positions // cfg.csa_hier_block_size))

    def forward_shared(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        shared: CSA2SharedState | None = None,
        compress_from: torch.Tensor | None = None,
        compress_start_pos: int = 0,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, CSA2SharedState]:
        r"""forward_shared(hidden_states, shared=None, compress_from=None, compress_start_pos=0, position_ids=None, use_cache=False) -> (Tensor, CSA2SharedState)

        Run one CSA2 layer with cross-layer shared state.

        Selected sparse outputs and SWA-window outputs are fused with the
        two-branch gate into model-dimension outputs; the SWA slice of
        ``shared`` advances when ``use_cache`` is set.

        Args:
            hidden_states (Tensor): input of shape ``(B, Lq, hidden)``.
            shared (CSA2SharedState, optional): cross-layer state; created when
                ``None``. Default: ``None``.
            compress_from (Tensor, optional): alternate compression source
                (``full`` mode only, e.g. encoder states). Default: ``None``.
            compress_start_pos (int, optional): start position used with
                ``compress_from``. Default: ``0``.
            position_ids (Tensor, optional): absolute positions. Default: ``None``.
            use_cache (bool, optional): whether to advance SWA/total-len bookkeeping.
                Default: ``False``.

        Returns:
            Tensor: fused outputs ``(B, Lq, hidden)`` and the updated shared
            state.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> cfg = MemoryMachineConfig(hidden_size=16, num_attention_heads=2,
            ...     num_key_value_heads=1, csa_top_k=2)
            >>> full, re = CompressedSparseAttention2(cfg, "full"), CompressedSparseAttention2(cfg, "reindex")
            >>> out, shared = full.forward_shared(torch.randn(1, 6, 16), use_cache=True)
            >>> full2, _ = re.forward_shared(torch.randn(1, 6, 16), shared)
            >>> full2.shape
            torch.Size([1, 6, 16])
        """
        bsz, q_len, _ = hidden_states.shape
        if shared is None:
            shared = CSA2SharedState()
        past_len_raw = _past_total_len(shared) or 0
        q_pos, _ = _resolve_positions(hidden_states, q_len, past_len_raw, position_ids)
        if self.indexer is None:
            raise ValueError("forward_shared in full/reindex mode requires an indexer (use_indexer=False)")
        cq, q_main, v_q = self._make_main_queries(hidden_states)
        if self.mode == "full":
            src = compress_from if compress_from is not None else hidden_states
            start = compress_start_pos if compress_from is not None else past_len_raw
            new_comp, new_ends = self.compressor(src, start_pos=start)
            if compress_from is not None:
                full_comp, full_ends = new_comp, new_ends
            elif shared.compressed_kv is not None:
                if shared.comp_end_pos is None:
                    raise ValueError("shared.comp_end_pos is None while shared.compressed_kv is set (corrupt CSA2SharedState)")
                full_comp = torch.cat([shared.compressed_kv, new_comp], dim=1)
                full_ends = torch.cat([shared.comp_end_pos.to(new_ends.device), new_ends], dim=0)
            else:
                full_comp, full_ends = new_comp, new_ends
            indexer_k = self.indexer.k_index_proj(full_comp.mean(dim=2))
            scores = self.indexer.score(cq, full_comp, q_pos, full_ends)
            topk = self.indexer.topk(scores, self.top_k)
            pool = self.indexer.build_pool(scores, self._hier_pool_blocks(), self.config.csa_hier_block_size)
            shared.compressed_kv, shared.comp_end_pos = full_comp, full_ends
            shared.indexer_k, shared.topk_indices, shared.pool_indices = indexer_k, topk, pool
        elif self.mode == "reindex":
            if shared.compressed_kv is None or shared.indexer_k is None:
                raise ValueError("reindex mode requires a Full layer to run first (empty shared state)")
            full_comp, full_ends = shared.compressed_kv, shared.comp_end_pos
            if full_ends is None:
                raise ValueError("shared.comp_end_pos is None while shared.compressed_kv is set (corrupt CSA2SharedState)")
            if shared.pool_indices is not None and shared.pool_indices.shape[1] == q_len:
                pool_scores = self.indexer.rescore_pool(cq, shared.indexer_k, shared.pool_indices, q_pos, full_ends)
                k = min(self.top_k, pool_scores.shape[-1])
                local = pool_scores.topk(k, dim=-1).indices
                topk = shared.pool_indices.gather(-1, local)
            else:
                scores = self.indexer.score(cq, full_comp, q_pos, full_ends)
                topk = self.indexer.topk(scores, self.top_k)
            shared.topk_indices = topk
        else:
            if shared.compressed_kv is None or shared.topk_indices is None:
                raise ValueError("reuse mode requires a Full/Reindex layer to run first (empty shared state)")
            if shared.comp_end_pos is None:
                raise ValueError("shared.comp_end_pos is None while shared.compressed_kv is set (corrupt CSA2SharedState)")
            full_comp = shared.compressed_kv
            full_ends = shared.comp_end_pos
            topk = shared.topk_indices
            if topk.shape[1] != q_len:
                topk = topk[:, -1:, :].expand(bsz, q_len, -1)
        sel = self._gather_selected(full_comp, topk)
        sparse_out = self._selected_branch(q_main, sel, self._selected_positions(full_ends, topk), v_q, q_pos)
        swa_past = None if shared.swa_k is None else (shared.swa_k, shared.swa_v)
        win_out, _, swa_present = self.swa(hidden_states, None, position_ids, swa_past, False, use_cache=True, past_len_override=past_len_raw, past_positions=shared.swa_pos)
        out = self.grouped_out(self._fuse_branches([sparse_out, win_out], hidden_states))
        if use_cache and swa_present is not None:
            shared.swa_k, shared.swa_v = swa_present
            shared.swa_pos = _update_swa_pos(shared.swa_pos, q_pos, swa_present[0].shape[2])
            shared.total_len = past_len_raw + q_len
        return out, shared
