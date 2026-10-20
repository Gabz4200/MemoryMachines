r"""Parallel attention ∥ GDN-2 token mixer routed by Single-Pass mHC.

:class:`MHCAttentionGDN2` is a drop-in CED layer that runs a
:class:`~memory_machines.models.blocks.sparse.CompressedSparseAttention2`
attention branch and a
:class:`~memory_machines.models.blocks.gdn2.GatedDeltaNet2Mixer` branch in
parallel, routed by :class:`~memory_machines.models.blocks.memory.SinglePassMHC`.

Following single-pass mHC (DeepSeek-V4.1, §2.4.1 Eq. 6) generalized to two
branches, the ``n`` residual streams ``X_l`` persist between adjacent blocks
and are routed per branch:

.. math::
    X_{l+1} = B_l X_l + C^{a}_l F_a(A^{a}_{l-1} X_l) + C^{g}_l F_g(A^{g}_{l-1} X_l)

where ``F_a`` is attention, ``F_g`` is GDN-2, ``A^{a}``/``A^{g}`` are the two
Sinkhorn/mHC-parameterized read-outs that route the streams each branch needs,
and ``B`` is the shared doubly-stochastic residual mapping. The stack expands
the input into streams once (:func:`to_streams`) and collapses the final
streams with a mean (:func:`from_streams`).
"""

from __future__ import annotations

import torch
from torch import nn

from memory_machines.utils.config_schema import MemoryMachineConfig

from .attention import GroupedQueryAttention
from .fast_weights import SelfReferentialTitanMixer
from .gdn2 import GatedDeltaNet2Mixer
from .memory import SinglePassMHC
from .shared import CSA2SharedState
from .sparse import CompressedSparseAttention, CompressedSparseAttention2, HeavilyCompressedAttention

__all__ = ["MHCAttentionGDN2", "MHCAttentionTitan", "from_streams", "to_streams"]


# variant -> (config window attribute, builder). Each attention variant is
# MultiHead and runs its own sliding window (see MemoryMachineConfig).
def _variant_window_attr(variant: str) -> str:
    attrs = {
        "dense": "window_size",
        "compressed": "window_size_compressed",
        "compressed_sparse": "window_size_sparse",
        "heavily_compressed": "window_size_shared",
    }
    if variant not in attrs:
        raise ValueError(f"ced_attention_variant must be one of {sorted(attrs)}, got {variant!r}")
    return attrs[variant]


def _build_attention(
    variant: str,
    config: MemoryMachineConfig,
    compression_ratio: int | None,
    mode: str,
    top_k: int | None,
    window_size: int,
) -> _SharedStateAdapter | CompressedSparseAttention2:
    if variant == "dense":
        return _SharedStateAdapter(GroupedQueryAttention(config, window_size))
    if variant == "compressed":
        return _SharedStateAdapter(CompressedSparseAttention(config, compression_ratio, window_size=window_size))
    if variant == "compressed_sparse":
        return CompressedSparseAttention2(config, compression_ratio, mode, top_k, window_size=window_size)
    return _SharedStateAdapter(HeavilyCompressedAttention(config, compression_ratio, window_size=window_size))


class _SharedStateAdapter(nn.Module):
    r"""_SharedStateAdapter(inner) -> Module

    Give an attention variant that has no cross-layer store the
    ``forward_shared`` interface CED layers are driven through. ``shared`` is
    passed through unchanged (the variant keeps no encoder-seeded global KV).
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    @property
    def mode(self) -> str | None:
        return getattr(self.inner, "mode", None)

    def reset_parameters(self) -> None:
        reset = getattr(self.inner, "reset_parameters", None)
        if callable(reset):
            reset()

    def forward_shared(
        self,
        hidden_states: torch.Tensor,
        shared: CSA2SharedState | None = None,
        compress_from: torch.Tensor | None = None,
        compress_start_pos: int = 0,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = False,
        seeded_store: bool = False,
    ) -> tuple[torch.Tensor, CSA2SharedState | None]:
        out, _, _ = self.inner(hidden_states, position_ids=position_ids, use_cache=use_cache)
        return out, shared


def to_streams(x: torch.Tensor, num_streams: int) -> torch.Tensor:
    r"""to_streams(x, num_streams) -> Tensor

    Expand a hidden state ``(B, L, d)`` into ``n`` identical residual streams
    ``(B, L, n, d)`` (mHC residual-stream replication).

    **Aliasing contract (read before any in-place op):** the returned tensor is
    ``x.unsqueeze(2).expand(...)`` — every stream is the *same* storage, so the
    streams alias the input. This is safe *only* because recombination
    (:func:`combine_parallel`) writes the residual back through a
    doubly-stochastic (Sinkhorn) mixing matrix ``b``: the write is a convex
    combination of the streams, so it is numerically stable and does not
    double-count. Any in-place op on one stream (a future GDN-2 state update,
    a masked-write, a chunked accumulation) would corrupt **all** streams —
    materialize with ``.contiguous()`` or ``.clone()`` before mutating. The
    streams are collapsed with a mean by :func:`from_streams`, which averages
    the (aliased) storage back to a single stream.

    Examples::

        >>> to_streams(torch.randn(1, 4, 8), 2).shape
        torch.Size([1, 4, 2, 8])
    """
    if x.dim() != 3:
        raise ValueError(f"x must be [B, L, d], got {tuple(x.shape)}")
    return x.unsqueeze(2).expand(x.shape[0], x.shape[1], num_streams, x.shape[2])


def from_streams(streams: torch.Tensor) -> torch.Tensor:
    r"""from_streams(streams) -> Tensor

    Collapse ``n`` streams ``(B, L, n, d)`` to a hidden state ``(B, L, d)`` by
    averaging (the mHC read-out preserves the stream mean).

    Examples::

        >>> from_streams(torch.randn(1, 4, 2, 8)).shape
        torch.Size([1, 4, 8])
    """
    if streams.dim() != 4:
        raise ValueError(f"streams must be [B, L, n, d], got {tuple(streams.shape)}")
    return streams.mean(dim=2)


class MHCAttentionGDN2(nn.Module):
    r"""MHCAttentionGDN2(config, compression_ratio=None, mode="full", top_k=None, variant=None, window_size=None)

    Parallel attention ∥ MultiHead GDN-2 encoder layer routed by Single-Pass mHC.

    Operates on mHC residual streams ``(B, L, n, d)``. Each call computes the
    two-branch coefficients, routes the streams into an attention branch input
    ``A^{a} X`` and a GDN-2 branch input ``A^{g} X`` (single-pass mHC: the
    previous block's ``A``, falling back to the current block's for the first
    block), runs both branches, then recombines with the shared Sinkhorn
    residual mapping and per-branch output scales:

    .. math::
        X_{l+1} = B_l X_l + C^{a}_l F_a(A^{a}_{l-1} X_l) + C^{g}_l F_g(A^{g}_{l-1} X_l)

    ``F_a`` is a MultiHead sliding-window attention variant (see
    ``config.ced_attention_variant``); ``F_g`` is a full
    :class:`~memory_machines.models.blocks.gdn2.GatedDeltaNet2Mixer`. Both
    branches are MultiHead; the attention branch runs its variant's own
    sliding window (``dense`` → ``window_size``, ``compressed`` →
    ``window_size_compressed``, ``compressed_sparse`` → ``window_size_sparse``,
    ``heavily_compressed`` → ``window_size_shared``).

    The GDN-2 branch is the block's *global* mixer: it consumes the whole
    sequence in input order and is never windowed, so the attention branch is
    free to focus its precision on a window. Which reference form performs
    that global mixing is selected by ``config.ced_gdn2_backend``
    (``naive``/``recurrent`` for inference, ``chunk``/``wy_fast`` for a
    training-only speed-up); the form never changes the sequence span.

    This block is an *encoder* layer. The decoder stays plain CSA2 and sees
    GDN-2 only indirectly, through the encoder's global KV (Eq. 1).

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``hidden_size``, head counts, ``mhc_num_streams``,
            ``mhc_sinkhorn_iters``, ``rms_norm_eps``, ``conv_kernel_size``, the
            ``ced_attention_variant``/``window_size*`` SWA configuration, the
            ``ced_gdn2_backend``/``ced_gdn2_chunk_size`` GDN-2 form, and
            the ``csa_*`` sparse-attention hyperparameters.
        compression_ratio (int, optional): attention block size ``m``. Default:
            ``None`` (falls back to ``config.csa_compression_ratio``).
        mode (str, optional): CSA2 sharing mode (``full``/``reindex``/``reuse``,
            ``compressed_sparse`` variant only). Default: ``"full"``.
        top_k (int, optional): entries per query. Default: ``None``.
        variant (str, optional): attention variant; overrides
            ``config.ced_attention_variant``. Default: ``None``.
        window_size (int, optional): attention SWA width, in compressed tokens;
            overrides the variant's config window. Default: ``None``.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> cfg = MemoryMachineConfig(hidden_size=32, num_attention_heads=4,
        ...     num_key_value_heads=2, csa_top_k=4, csa_num_output_groups=4,
        ...     mhc_num_streams=2)
        >>> block = MHCAttentionGDN2(cfg)
        >>> streams = to_streams(torch.randn(1, 8, 32), 2)
        >>> out, shared, a = block.forward_shared(streams)
        >>> out.shape, a.shape
        (torch.Size([1, 8, 2, 32]), torch.Size([1, 8, 2, 2]))
    """

    def __init__(
        self,
        config: MemoryMachineConfig,
        compression_ratio: int | None = None,
        mode: str = "full",
        top_k: int | None = None,
        variant: str | None = None,
        window_size: int | None = None,
    ):
        super().__init__()
        self.num_streams = config.mhc_num_streams
        self.attention_variant = variant if variant is not None else config.ced_attention_variant
        resolved_window = window_size if window_size is not None else getattr(config, _variant_window_attr(self.attention_variant))
        self.attention = _build_attention(self.attention_variant, config, compression_ratio, mode, top_k, resolved_window)
        self.gdn2: GatedDeltaNet2Mixer | SelfReferentialTitanMixer = self._build_branch(config)
        self.mhc = SinglePassMHC(config, parallel_branches=True)
        self.reset_parameters()

    def _build_branch(self, config: MemoryMachineConfig) -> GatedDeltaNet2Mixer | SelfReferentialTitanMixer:
        r"""_build_branch(config) -> GatedDeltaNet2Mixer

        Build the second parallel branch (default: GDN-2). Subclasses
        override to swap in a different memory module; the returned object
        is assigned to :attr:`gdn2` so :meth:`forward_shared` is shared.
        """
        return GatedDeltaNet2Mixer(
            config,
            num_v_heads=config.ced_gdn2_num_v_heads or None,
            expand_v=config.ced_gdn2_expand_v,
            use_short_conv=config.ced_gdn2_use_short_conv,
            conv_bias=config.ced_gdn2_conv_bias,
            liv_conv=config.ced_gdn2_liv_conv,
            backend=config.ced_gdn2_backend,
            chunk_size=config.ced_gdn2_chunk_size,
            use_qk_l2norm=config.ced_gdn2_use_qk_l2norm,
            allow_neg_eigval=config.ced_gdn2_allow_neg_eigval,
            beta_guard=config.ced_gdn2_beta_guard,
            use_resting_state=config.ced_gdn2_use_resting_state,
        )

    @property
    def mode(self) -> str | None:
        r"""mode -> str | None

        CSA2 sharing mode of the attention branch (``full``/``reindex``/
        ``reuse``); ``None`` for attention variants without a sharing mode.
        """
        return self.attention.mode

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Re-applies the attention, GDN-2 and mHC inits so a freshly built
        block starts from a neutral point (balanced branch fusion, uniform
        pooling, zero-start outputs).
        """
        for module in (self.attention, self.gdn2, self.mhc):
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()

    def forward_shared(
        self,
        streams: torch.Tensor,
        shared: CSA2SharedState | None = None,
        prev_a: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = False,
        seeded_store: bool = False,
        compress_from: torch.Tensor | None = None,
        compress_start_pos: int = 0,
    ) -> tuple[torch.Tensor, CSA2SharedState | None, torch.Tensor]:
        r"""forward_shared(streams, shared=None, prev_a=None, position_ids=None, use_cache=False, seeded_store=False, compress_from=None, compress_start_pos=0) -> (Tensor, CSA2SharedState | None, Tensor)

        Run one parallel attention ∥ GDN-2 layer on mHC streams.

        Routes the streams into the two branches with this block's
        coefficients (or the previous block's ``prev_a`` for single-pass
        mHC), runs the attention branch through its
        :meth:`~memory_machines.models.blocks.sparse.CompressedSparseAttention2.forward_shared`
        (threading the shared CSA2 state) and the GDN-2 branch
        independently, then recombines.

        Args:
            streams (Tensor): mHC residual streams ``(B, L, n, d)``.
            shared (CSA2SharedState, optional): cross-layer CSA2 state. Default: ``None``.
            prev_a (Tensor, optional): previous block's routing coefficients
                ``(B, L, 2, n)`` (index 0 attention, 1 GDN-2). ``None`` uses
                this block's own coefficients. Default: ``None``.
            position_ids (Tensor, optional): absolute positions. Default: ``None``.
            use_cache (bool, optional): CSA2 API parity flag. Default: ``False``.
            seeded_store (bool, optional): pre-built global store (decoder). Default: ``False``.
            compress_from (Tensor, optional): alternate compression source. Default: ``None``.
            compress_start_pos (int, optional): start position for ``compress_from``. Default: ``0``.

        Returns:
            Tensor: updated streams ``(B, L, n, d)``, the updated shared state,
            and this block's ``(a1, a2)`` coefficients ``(B, L, 2, n)`` for the
            next block.
        """
        if streams.dim() != 4 or streams.shape[2] != self.num_streams:
            raise ValueError(f"streams must be [B, L, n={self.num_streams}, d], got {tuple(streams.shape)}")
        bsz, lq, _, _ = streams.shape
        if prev_a is not None and (prev_a.dim() != 4 or tuple(prev_a.shape[:3]) != (bsz, lq, 2) or prev_a.shape[3] != self.num_streams):
            raise ValueError(f"prev_a must be [B, L, 2, n]=({bsz}, {lq}, 2, {self.num_streams}), got {tuple(prev_a.shape)}")
        a1, a2, b, c1, c2 = self.mhc.parallel_coefficients(streams)
        pa1 = prev_a[:, :, 0, :].unsqueeze(2) if prev_a is not None else a1
        pa2 = prev_a[:, :, 1, :].unsqueeze(2) if prev_a is not None else a2
        mix1 = self.mhc.mix_input(streams, pa1).squeeze(2)
        mix2 = self.mhc.mix_input(streams, pa2).squeeze(2)
        attn_out, shared = self.attention.forward_shared(
            mix1,
            shared,
            compress_from=compress_from,
            compress_start_pos=compress_start_pos,
            position_ids=position_ids,
            use_cache=use_cache,
            seeded_store=seeded_store,
        )
        gdn_out, _, _ = self.gdn2(mix2)
        out = self.mhc.combine_parallel(streams, b, c1, attn_out, c2, gdn_out)
        next_a = torch.stack((a1.squeeze(2), a2.squeeze(2)), dim=2)
        return out, shared, next_a

    def forward(
        self,
        streams: torch.Tensor,
        shared: CSA2SharedState | None = None,
        prev_a: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""forward(streams, shared=None, prev_a=None, **kwargs) -> Tensor

        Streams-only convenience wrapper around :meth:`forward_shared`.
        """
        out, _, _ = self.forward_shared(streams, shared, prev_a, **kwargs)
        return out


class MHCAttentionTitan(MHCAttentionGDN2):
    r"""MHCAttentionTitan(config, compression_ratio=None, mode="full", top_k=None, variant=None, window_size=None)

    Parallel attention ∥ MultiHead SelfReferentialTitans encoder layer routed by Single-Pass mHC.

    Alternates with :class:`MHCAttentionGDN2` in the CED encoder stack:
    runs a MultiHead sliding-window attention variant in parallel with a
    MultiHead :class:`~memory_machines.models.blocks.fast_weights.SelfReferentialTitanMixer`,
    routed by :class:`~memory_machines.models.blocks.memory.SinglePassMHC`
    over persistent residual streams. Subclass of :class:`MHCAttentionGDN2` so
    existing ``isinstance(layer, MHCAttentionGDN2)`` checks still hold; only
    the second branch is swapped.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides model hyperparameters, attention variant, and Titans configurations.
        compression_ratio (int, optional): attention block size ``m``. Default: ``None``.
        mode (str, optional): CSA2 sharing mode (``full``/``reindex``/``reuse``). Default: ``"full"``.
        top_k (int, optional): entries per query. Default: ``None``.
        variant (str, optional): attention variant override. Default: ``None``.
        window_size (int, optional): attention SWA width override. Default: ``None``.
    """

    def _build_branch(self, config: MemoryMachineConfig) -> GatedDeltaNet2Mixer | SelfReferentialTitanMixer:
        r"""_build_branch(config) -> SelfReferentialTitanMixer

        Build the SelfReferentialTitans branch (the GDN-2 override).
        """
        return SelfReferentialTitanMixer(
            config,
            memory_depth=config.titan_memory_depth,
            fast_expansion=config.titan_fast_expansion,
            objective=config.titan_objective,
            use_short_conv=config.titan_use_short_conv,
            conv_kernel_size=config.titan_conv_kernel_size,
            chunk_size=config.titan_chunk_size if config.titan_chunk_size else None,
            backend=config.titan_backend,
            use_qk_projection=config.titan_use_qk_projection,
        )
