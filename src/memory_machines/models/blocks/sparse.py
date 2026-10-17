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

import warnings

import torch
import torch.nn.functional as F
from torch import nn

from memory_machines.utils.config_schema import MemoryMachineConfig

from .attention import GroupedQueryAttention
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
    and an internal windowed
    :class:`~memory_machines.models.blocks.attention.GroupedQueryAttention`
    branch (SWA by design). Branch outputs (dense and/or top-``k``
    selected + window) are fused by
    :class:`~memory_machines.models.blocks.compression.BranchGate` and
    :class:`~memory_machines.models.blocks.compression.GroupedOutputProjection`.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides hidden sizes, head counts, ``csa_*``/``window_size*``
            settings, and CVPE/FRVM toggles.
        compression_ratio (int, optional): block size ``m``. Default: ``None``
            (falls back to ``config.csa_compression_ratio``).
        use_indexer (bool, optional): whether to enable top-``k`` retrieval.
            Default: ``True``.
        use_dense_branch (bool, optional): whether to attend the full
            compressed store. Default: ``False``.
        top_k (int, optional): entries selected per query. Default: ``None``
            (falls back to ``config.csa_top_k``).
        window_size (int, optional): SWA branch width, in compressed
            tokens. Default: ``None`` (falls back to
            ``config.window_size_compressed``, i.e. ``256 * m``).

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> layer = CompressedSparseAttention(MemoryMachineConfig(hidden_size=32,
        ...     num_attention_heads=4, num_key_value_heads=2, csa_top_k=4,
        ...     csa_num_output_groups=4))
        >>> out, _, cache = layer(torch.randn(1, 8, 32), use_cache=True)
        >>> out.shape
        torch.Size([1, 8, 32])
    """

    q_norm: HeadRMSNorm | None
    kv_norm: HeadRMSNorm | None
    sink: AttentionSink | None
    grouped_out: GroupedOutputProjection

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
        # CSA drops the parent dense Q/K/O projections (only v_proj is kept as
        # reference values); replace the unused linears with Identity so new
        # checkpoints/optimizers do not carry dead params. NOTE: checkpoints
        # saved before this change contain q_proj/k_proj/o_proj weights —
        # load them with strict=False.
        for _dead in ("q_proj", "k_proj", "o_proj"):
            if hasattr(self, _dead):
                delattr(self, _dead)
            setattr(self, _dead, nn.Identity())
        self.compression_ratio = compression_ratio if compression_ratio is not None else config.csa_compression_ratio
        self.use_indexer = use_indexer
        self.use_dense_branch = use_dense_branch
        self.top_k = top_k if top_k is not None else config.csa_top_k
        self.window_size = window_size if window_size is not None else config.window_size_compressed
        self.init_std: float = config.initializer_range
        self.q_latent_dim = config.csa_q_latent_dim if config.csa_q_latent_dim > 0 else config.hidden_size
        self.q_down = nn.Linear(config.hidden_size, self.q_latent_dim, bias=False)
        self.q_up_main = nn.Linear(self.q_latent_dim, self.num_heads * self.head_dim, bias=False)
        self.compressor = BlockCompressor(
            config.hidden_size,
            self.num_key_value_heads,
            self.head_dim,
            self.compression_ratio,
            init_std=config.initializer_range,
        )
        self.indexer = SparseIndexer(self.q_latent_dim, self.head_dim, config.csa_indexer_heads, config.csa_indexer_dim, init_std=config.initializer_range) if use_indexer else None
        n_br = 2 + (1 if use_dense_branch and use_indexer else 0)
        self.branch_gate = BranchGate(config.hidden_size, n_br)
        self.q_norm = HeadRMSNorm(self.head_dim, config.rms_norm_eps) if config.csa_use_qkv_norm else None
        self.kv_norm = HeadRMSNorm(self.head_dim, config.rms_norm_eps) if config.csa_use_qkv_norm else None
        self.sink = AttentionSink(self.num_heads) if config.csa_use_sink else None
        self.grouped_out = GroupedOutputProjection(
            self.num_heads,
            self.head_dim,
            config.csa_num_output_groups,
            config.csa_output_group_dim,
            config.hidden_size,
            init_std=config.initializer_range,
            num_layers=config.num_layers,
            residual_init_scale=config.residual_init_scale,
        )
        self.swa = GroupedQueryAttention(config, self.token_window_size)
        self.reset_parameters()

    def _project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError("CompressedSparseAttention drops the parent dense Q/K/O projections; use _make_main_queries() instead")

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Truncated normal ``N(0, initializer_range^2)`` for the
        low-rank query path (``q_down``/``q_up_main``) and the
        retained ``v_proj`` reference values; all composed
        submodules (compressor, indexer, branch gate, norms, sink,
        grouped output, window branch) are re-reset to their own
        neutral-start inits so a freshly built sparse layer fuses
        balanced branches from uniform pooling and a zero-start
        (or depth-scaled) residual output.
        """
        std = self.init_std
        nn.init.trunc_normal_(self.q_down.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.trunc_normal_(self.q_up_main.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.trunc_normal_(self.v_proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        self.compressor.reset_parameters()
        if self.indexer is not None:
            self.indexer.reset_parameters()
        self.branch_gate.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.kv_norm is not None:
            self.kv_norm.reset_parameters()
        if self.sink is not None:
            self.sink.reset_parameters()
        self.grouped_out.reset_parameters()
        self.swa.reset_parameters()

    @property
    def token_window_size(self) -> int:
        r"""token_window_size -> int

        SWA branch width in raw tokens.

        The layer window is specified in compressed tokens; the internal
        window branch runs over raw tokens, so it scales by the
        compression ratio (a window of ``w`` compressed entries covers
        ``w * m`` tokens).

        Returns:
            int: ``window_size * compression_ratio`` when the window is
            enabled, else ``window_size`` (<= 0 disables the window).

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> layer = CompressedSparseAttention(
            ...     MemoryMachineConfig(hidden_size=16, num_attention_heads=2,
            ...         num_key_value_heads=1, csa_top_k=2,
            ...         csa_compression_ratio=4, csa_num_output_groups=2),
            ...     window_size=4)
            >>> layer.token_window_size
            16
        """
        if self.window_size <= 0:
            return self.window_size
        return self.window_size * self.compression_ratio

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
            q_pos (Tensor): query positions ``(B, Lq)`` (or ``(B, Lq, N)`` N-dim).
            k_pos (Tensor): key positions ``(B, Lk)`` (or ``(B, Lk, N)`` N-dim).
            causal (bool, optional): whether to mask future entries. Default: ``True``.
            bidirectional (bool, optional): full visibility, exclusive with
                ``causal``. Default: ``False``.

        Returns:
            Tensor: per-head outputs of shape ``(B, H, Lq, Dh)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> layer = CompressedSparseAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, csa_top_k=2,
            ...     csa_use_qkv_norm=False, csa_use_sink=False, frvm_enabled=False,
            ...     cvpe_enabled=False, csa_num_output_groups=2))
            >>> layer._core_attend(torch.randn(1, 2, 3, 8), torch.randn(1, 1, 5, 8),
            ...     torch.randn(1, 2, 3, 8), torch.tensor([[4, 5, 6]]),
            ...     torch.tensor([[0, 1, 2, 3, 4]])).shape
            torch.Size([1, 2, 3, 8])
        """
        q_norm = self.q_norm
        kv_norm = self.kv_norm
        if q_norm is not None and kv_norm is not None:
            q_heads = q_norm(q_heads)
            kv = kv_norm(kv)
        # NOTE: repeat_interleave materializes H copies (general GQA only).
        kv_exp = kv.repeat_interleave(self.num_key_value_groups, dim=1)
        from memory_machines.ops.references.attention_reference import (
            reference_frvm_gated_reduction,
            reference_qk_circuit,
        )

        cvpe_attn_bias, phi_vq, phi_vk, spatial_grid, ov_spatial_weight = self._pe_inputs(
            q_heads, v_query, kv_exp, q_pos, k_pos
        )
        logits = reference_qk_circuit(
            q_heads, kv_exp, cvpe_attn_bias,
            torch.zeros(q_heads.shape[2], kv_exp.shape[2], dtype=q_heads.dtype, device=q_heads.device),
            None,
        )
        q1 = q_pos[..., 0] if q_pos.dim() == 3 else q_pos
        k1 = k_pos[..., 0] if k_pos.dim() == 3 else k_pos
        if causal:
            causal_mask = k1.unsqueeze(1) <= q1.unsqueeze(-1)
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
            per_q = weights.shape[0] * weights.shape[1] * weights.shape[3] * kv_exp.shape[-1]
            eff = min(self.frvm_chunk_size, max(1, 32 * 1024 * 1024 // max(1, per_q)))
            return reference_frvm_gated_reduction(
                weights, kv_exp, phi_vq, phi_vk,
                spatial_grid=spatial_grid, ov_spatial_weight=ov_spatial_weight, chunk_size=eff,
            )
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
            ...     num_attention_heads=2, num_key_value_heads=1, csa_top_k=2, csa_num_output_groups=2))
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
            ...     num_attention_heads=2, num_key_value_heads=1, csa_top_k=2, csa_num_output_groups=2))
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

        Attend each query to its own top-``k`` selection in one batched
        gather-matmul (queries flattened to ``B * Lq`` with a singleton
        key dim, so the per-query path matches :meth:`_core_attend`
        exactly without a Python loop over ``Lq``).

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
            ...     csa_use_qkv_norm=False, csa_num_output_groups=2))
            >>> layer._selected_branch(torch.randn(1, 2, 2, 8), torch.randn(1, 2, 2, 1, 8),
            ...     torch.tensor([[[0, 1], [0, 1]]]), torch.randn(1, 2, 2, 8),
            ...     torch.tensor([[3, 4]])).shape
            torch.Size([1, 2, 2, 8])
        """
        bsz, _, lq, _ = q_main.shape
        flat = bsz * lq
        qf = q_main.transpose(1, 2).reshape(flat, self.num_heads, 1, self.head_dim)
        kf = sel.reshape(flat, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        vqf = v_q.transpose(1, 2).reshape(flat, self.num_heads, 1, self.head_dim)
        qpos = q_pos.reshape(flat, 1)
        kpos = sel_pos.reshape(flat, -1)
        out = self._core_attend(qf, kf, vqf, qpos, kpos, causal=True)
        return out.reshape(bsz, lq, self.num_heads, self.head_dim).transpose(1, 2)

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
        runs dense and/or top-``k`` selected branches plus the window
        branch, and fuses them into model-dimension outputs.

        .. note::
            Full attention maps are never materialized; ``output_attentions``
            warns and returns ``None`` for weights.

        Args:
            hidden_states (Tensor): input of shape ``(B, Lq, hidden)``.
            attention_mask (Tensor, optional): mask forwarded to the window
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
            ...     num_attention_heads=2, num_key_value_heads=1, csa_top_k=2, csa_num_output_groups=2))
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
            full_comp = torch.cat([past.compressed_kv.to(new_comp.device, new_comp.dtype), new_comp], dim=1)
            past_ends = past.comp_end_pos.to(new_ends.device)
            if past_ends.dim() == 1:
                # Broadcast a shared (Lc,) store across the batch.
                full_ends = torch.cat([past_ends, new_ends], dim=0)
                k_pos_full = full_ends.unsqueeze(0).expand(bsz, -1)
            elif past_ends.dim() == 2:
                if past_ends.shape[0] != bsz:
                    raise ValueError(f"past.comp_end_pos batch ({past_ends.shape[0]}) does not match input batch ({bsz})")
                new_ends_b = new_ends.unsqueeze(0).expand(bsz, -1)
                full_ends = torch.cat([past_ends, new_ends_b], dim=-1)
                k_pos_full = full_ends
            else:
                raise ValueError(f"past.comp_end_pos must be (Lc,) or (B, Lc), got {tuple(past_ends.shape)}")
        else:
            full_comp, full_ends = new_comp, new_ends
            k_pos_full = full_ends.unsqueeze(0).expand(bsz, -1) if full_ends.dim() == 1 else full_ends
        cq, q_main, v_q = self._make_main_queries(hidden_states)
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
        ...     num_attention_heads=2, num_key_value_heads=1, csa_top_k=2, csa_num_output_groups=2))
        >>> layer(torch.randn(1, 6, 16))[0].shape
        torch.Size([1, 6, 16])
    """

    def __init__(self, config: MemoryMachineConfig, compression_ratio: int = 2):
        super().__init__(config, compression_ratio=compression_ratio, use_indexer=True, use_dense_branch=True)


class HeavilyCompressedAttention(CompressedSparseAttention):
    r"""HeavilyCompressedAttention(config, compression_ratio=None, window_size=None)

    HCA variant: dense heavily-compressed + window branches (no indexer).

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``hca_compression_ratio`` when ``compression_ratio`` is unset.
        compression_ratio (int, optional): block size ``m``. Default: ``None``
            (falls back to ``config.hca_compression_ratio``).
        window_size (int, optional): SWA branch width, in compressed
            tokens. Default: ``None`` (falls back to
            ``config.window_size_compressed``).

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> layer = HeavilyCompressedAttention(MemoryMachineConfig(hidden_size=16,
        ...     num_attention_heads=2, num_key_value_heads=1, csa_num_output_groups=2))
        >>> layer(torch.randn(1, 8, 16))[0].shape
        torch.Size([1, 8, 16])
    """

    def __init__(self, config: MemoryMachineConfig, compression_ratio: int | None = None, window_size: int | None = None):
        super().__init__(
            config,
            compression_ratio=compression_ratio if compression_ratio is not None else config.hca_compression_ratio,
            use_indexer=False,
            use_dense_branch=True,
            window_size=window_size,
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
        window_size (int, optional): SWA branch width, in compressed tokens.
            Default: ``None`` (falls back to ``config.window_size_compressed``).

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> from memory_machines.models.blocks.shared import CSA2SharedState
        >>> cfg = MemoryMachineConfig(hidden_size=16, num_attention_heads=2,
        ...     num_key_value_heads=1, csa_top_k=2, csa_num_output_groups=2)
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
        window_size: int | None = None,
    ):
        super().__init__(config, compression_ratio=compression_ratio, use_indexer=True, use_dense_branch=False, top_k=top_k, window_size=window_size)
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
            ...     num_attention_heads=2, num_key_value_heads=1, csa_num_output_groups=2))._hier_pool_blocks() > 0
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
        seeded_store: bool = False,
        mode: str | None = None,
    ) -> tuple[torch.Tensor, CSA2SharedState]:
        r"""forward_shared(hidden_states, shared=None, compress_from=None, compress_start_pos=0, position_ids=None, use_cache=False, seeded_store=False, mode=None) -> (Tensor, CSA2SharedState)

        Run one CSA2 layer with cross-layer shared state.

        Selected sparse outputs and SWA-window outputs are fused with the
        two-branch gate into model-dimension outputs. Global KV, indexer
        keys, and Top-K indices are published to ``shared``; SWA KV is
        *not* shared (each layer keeps its own window cache), and
        ``shared.total_len`` is read as the sequence past set by the
        calling stack, never advanced here.

        Args:
            hidden_states (Tensor): input of shape ``(B, Lq, hidden)``.
            shared (CSA2SharedState, optional): cross-layer state; created when
                ``None``. Default: ``None``.
            compress_from (Tensor, optional): alternate compression source
                (``full`` mode only, e.g. encoder states). Default: ``None``.
            compress_start_pos (int, optional): start position used with
                ``compress_from``. Default: ``0``.
            position_ids (Tensor, optional): absolute positions. Default: ``None``.
            use_cache (bool, optional): accepted for API parity; the SWA
                branch runs without a persistent cache because SWA KV is
                layer-local. Default: ``False``.
            seeded_store (bool, optional): when ``True`` (``full`` mode
                only), treat ``shared.compressed_kv``/``comp_end_pos``
                as the already-built global store: no compression of
                ``hidden_states`` and no concatenation. Used by the CED
                decoder, whose global KV is projected from the
                encoder-final states rather than derived from decoder
                hidden states (CSA2, Eq. 1). Default: ``False``.
            mode (str, optional): per-call override of the layer's
                ``full``/``reindex``/``reuse`` mode, used by the looped
                CED decoder to cycle the shared layer's stage without
                mutating module state. Default: ``None``
                (falls back to ``self.mode``).

        Returns:
            Tensor: fused outputs ``(B, Lq, hidden)`` and the updated shared
            state.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> cfg = MemoryMachineConfig(hidden_size=16, num_attention_heads=2,
            ...     num_key_value_heads=1, csa_top_k=2, csa_num_output_groups=2)
            >>> full, re = CompressedSparseAttention2(cfg, mode="full"), CompressedSparseAttention2(cfg, mode="reindex")
            >>> out, shared = full.forward_shared(torch.randn(1, 6, 16), use_cache=True)
            >>> full2, _ = re.forward_shared(torch.randn(1, 6, 16), shared)
            >>> full2.shape
            torch.Size([1, 6, 16])
        """
        bsz, q_len, _ = hidden_states.shape
        if shared is None:
            shared = CSA2SharedState()
        mode = self.mode if mode is None else mode
        if mode not in ("full", "reindex", "reuse"):
            raise ValueError(f"mode ({mode}) must be full/reindex/reuse")
        past_len_raw = shared.total_len or 0
        q_pos, _ = _resolve_positions(hidden_states, q_len, past_len_raw, position_ids)
        if self.indexer is None:
            raise ValueError("forward_shared in full/reindex mode requires an indexer (use_indexer=False)")
        cq, q_main, v_q = self._make_main_queries(hidden_states)
        if mode == "full":
            if seeded_store:
                if shared.compressed_kv is None:
                    raise ValueError("seeded_store=True requires shared.compressed_kv (seed the store before the full layer)")
                if shared.comp_end_pos is None:
                    raise ValueError("seeded_store=True requires shared.comp_end_pos (corrupt CSA2SharedState)")
                full_comp, full_ends = shared.compressed_kv, shared.comp_end_pos
            else:
                src = compress_from if compress_from is not None else hidden_states
                start = compress_start_pos if compress_from is not None else past_len_raw
                new_comp, new_ends = self.compressor(src, start_pos=start)
                if compress_from is not None:
                    full_comp, full_ends = new_comp, new_ends
                elif shared.compressed_kv is not None:
                    if shared.comp_end_pos is None:
                        raise ValueError("shared.comp_end_pos is None while shared.compressed_kv is set (corrupt CSA2SharedState)")
                    past_ends = shared.comp_end_pos.to(new_ends.device)
                    full_comp = torch.cat([shared.compressed_kv.to(new_comp.device, new_comp.dtype), new_comp], dim=1)
                    if past_ends.dim() == 1:
                        full_ends = torch.cat([past_ends, new_ends], dim=0)
                    elif past_ends.dim() == 2:
                        if past_ends.shape[0] != bsz:
                            raise ValueError(f"shared.comp_end_pos batch ({past_ends.shape[0]}) does not match input batch ({bsz})")
                        full_ends = torch.cat([past_ends, new_ends.unsqueeze(0).expand(bsz, -1)], dim=-1)
                    else:
                        raise ValueError(f"shared.comp_end_pos must be (Lc,) or (B, Lc), got {tuple(past_ends.shape)}")
                else:
                    full_comp, full_ends = new_comp, new_ends
            indexer_k = self.indexer.k_index_proj(full_comp.mean(dim=2))
            scores = self.indexer.score(cq, full_comp, q_pos, full_ends)
            topk = self.indexer.topk(scores, self.top_k)
            pool = self.indexer.build_pool(scores, self._hier_pool_blocks(), self.config.csa_hier_block_size)
            shared.compressed_kv, shared.comp_end_pos = full_comp, full_ends
            shared.indexer_k, shared.topk_indices, shared.pool_indices = indexer_k, topk, pool
        elif mode == "reindex":
            if shared.compressed_kv is None or shared.indexer_k is None:
                raise ValueError("reindex mode requires a Full layer to run first (empty shared state)")
            full_comp, full_ends = shared.compressed_kv, shared.comp_end_pos
            if full_ends is None:
                raise ValueError("shared.comp_end_pos is None while shared.compressed_kv is set (corrupt CSA2SharedState)")
            if shared.pool_indices is not None and shared.pool_indices.shape[1] == q_len:
                pool_scores = self.indexer.rescore_pool(cq, shared.indexer_k, shared.pool_indices, q_pos, full_ends)
                k = min(self.top_k, pool_scores.shape[-1])
                # topk on all -inf rows returns arbitrary entries;
                # select only causally visible entries instead.
                finite = pool_scores.isfinite()
                local = pool_scores.topk(k, dim=-1).indices
                vis_count = finite.sum(-1)
                has_vis = vis_count > 0
                last_idx = (finite.shape[-1] - 1) - finite.flip(-1).int().argmax(-1)
                last_idx = last_idx.masked_fill(~has_vis, 0)
                tail = torch.arange(k, device=finite.device).unsqueeze(0).unsqueeze(0)
                local = torch.where(tail < vis_count.unsqueeze(-1), local, last_idx.unsqueeze(-1))
                topk = torch.where(has_vis.unsqueeze(-1), shared.pool_indices.gather(-1, local), shared.pool_indices[:, :, :1])
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
                # Reuse the last query's selection (exact only for single-token decode).
                topk = topk[:, -1:, :].expand(bsz, q_len, -1)
        sel = self._gather_selected(full_comp, topk)
        sparse_out = self._selected_branch(q_main, sel, self._selected_positions(full_ends, topk), v_q, q_pos)
        # Thread explicit positions into the SWA branch so both branches share
        # the same absolute position base.
        win_out, _, _ = self.swa(hidden_states, None, q_pos, None, False, use_cache=False)
        out = self.grouped_out(self._fuse_branches([sparse_out, win_out], hidden_states))
        return out, shared
