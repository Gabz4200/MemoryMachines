r"""Parallel attention ∥ memory mixer encoder block.

:class:`ParallelAttentionGDN2` is a drop-in CED encoder layer with the
architecture:

.. math::
    h &= \mathrm{PreNorm}(X) \\
    Z &= W_O [\mathrm{Attn}(h) \,\|\, \mathrm{Memory}(h)] \\
    X' &= X + Z \\

The pre-norm is configurable (``config.ced_encoder_norm``, default ``dyT``)
and feeds both branches from the *same* normalized representation. The two
branch outputs, each already in model width ``d`` after its own MultiHead
output projection, are concatenated to ``2d`` and mixed back to ``d`` by one
down-projection ``W_O``. That projection is the only place the two branches
interact: the attention branch keeps its local sliding window and the memory
branch (GDN-2, or Titans on alternating layers) mixes the full sequence. The
CED encoder builds layer 0 with a plain, uncompressed and non-sparse SWA
attention branch (``variant="dense"``) and every later layer with the
configured attention variant.

At step 0 ``W_O`` starts with its attention half at standard scale and its
memory half near zero, so the block begins as a standard attention block and
lets the memory branch grow in without an initial gradient spike.

The dense FFN sub-layer stays a separate :class:`~memory_machines.models.blocks.ffn.DenseFFN`
(after the block, as in the rest of the CED stack), so the full layer is
``X + W_O([Attn(h) || Memory(h)])`` followed by the FFN's own pre-norm
residual.
"""

from __future__ import annotations

import torch
from torch import nn

from memory_machines.utils.config_schema import MemoryMachineConfig

from .attention import GroupedQueryAttention
from .fast_weights import SelfReferentialTitanMixer
from .ffn import build_norm
from .gdn2 import GatedDeltaNet2Mixer
from .linear_init import init_linear_columns
from .linears import FactoredLinear
from .shared import CSA2SharedState
from .sparse import CompressedSparseAttention, CompressedSparseAttention2, HeavilyCompressedAttention

__all__ = ["ParallelAttentionGDN2", "ParallelAttentionTitan"]


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
        self.inner.reset_parameters()

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


class ParallelAttentionGDN2(nn.Module):
    r"""ParallelAttentionGDN2(config, compression_ratio=None, mode="full", top_k=None, variant=None, window_size=None, memory_init_scale=1e-3)

    Parallel attention ∥ MultiHead memory mixer with a fused down-projection.

    Each call normalizes the input once (``config.ced_encoder_norm``, default
    ``dyT``), runs a MultiHead sliding-window attention branch and a MultiHead
    memory branch (GDN-2 here) on that same normalized representation,
    concatenates their ``d``-wide outputs to ``2d``, projects them back to
    ``d`` with :attr:`out_proj`, and adds the input residual:

    .. math::
        h = \mathrm{norm}(X), \quad
        X' = X + W_O [F_a(h) \,\|\, F_g(h)]

    ``F_a`` is the attention variant selected by
    ``config.ced_attention_variant``; ``F_g`` is a full
    :class:`~memory_machines.models.blocks.gdn2.GatedDeltaNet2Mixer`. Both
    branches are MultiHead with their own input and output projections, so the
    down-projection only mixes the two branch outputs.

    The memory branch is the block's *global* mixer: it consumes the whole
    sequence in input order and is never windowed. Which reference form
    performs that global mixing is selected by ``config.ced_gdn2_backend``
    (``naive``/``recurrent`` for inference, ``chunk``/``wy_fast`` for a
    training-only speed-up); the form never changes the sequence span.

    This block is an *encoder* layer. The decoder stays plain CSA2 and sees
    the memory branch only indirectly, through the encoder's global KV (Eq. 1).

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``hidden_size``, head counts, ``ced_encoder_norm``,
            ``rms_norm_eps``, ``conv_kernel_size``, the
            ``ced_attention_variant``/``window_size*`` SWA configuration, the
            ``ced_gdn2_backend``/``ced_gdn2_chunk_size`` GDN-2 form, and the
            ``csa_*`` sparse-attention hyperparameters.
        compression_ratio (int, optional): attention block size ``m``. Default:
            ``None`` (falls back to ``config.csa_compression_ratio``).
        mode (str, optional): CSA2 sharing mode (``full``/``reindex``/``reuse``,
            ``compressed_sparse`` variant only). Default: ``"full"``.
        top_k (int, optional): entries per query. Default: ``None``.
        variant (str, optional): attention variant; overrides
            ``config.ced_attention_variant``. Default: ``None``.
        window_size (int, optional): attention SWA width, in compressed tokens;
            overrides the variant's config window. Default: ``None``.
        memory_init_scale (float, optional): std of the memory half of
            :attr:`out_proj` at init; the attention half keeps standard
            trunc-normal init. Default: ``1e-3``.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> cfg = MemoryMachineConfig(hidden_size=32, num_attention_heads=4,
        ...     num_key_value_heads=2, csa_top_k=4, csa_num_output_groups=4)
        >>> block = ParallelAttentionGDN2(cfg)
        >>> out, shared = block.forward_shared(torch.randn(1, 8, 32))
        >>> out.shape
        torch.Size([1, 8, 32])
    """

    def __init__(
        self,
        config: MemoryMachineConfig,
        compression_ratio: int | None = None,
        mode: str = "full",
        top_k: int | None = None,
        variant: str | None = None,
        window_size: int | None = None,
        memory_init_scale: float = 1e-3,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.initializer_range: float = config.initializer_range
        self.attention_variant = variant if variant is not None else config.ced_attention_variant
        resolved_window = window_size if window_size is not None else getattr(config, _variant_window_attr(self.attention_variant))
        self.norm = build_norm(config.ced_encoder_norm, config.hidden_size, eps=config.rms_norm_eps)
        self.attention = _build_attention(self.attention_variant, config, compression_ratio, mode, top_k, resolved_window)
        self.gdn2: GatedDeltaNet2Mixer | SelfReferentialTitanMixer = self._build_branch(config)
        self.memory_init_scale = memory_init_scale
        self.out_proj: nn.Linear | FactoredLinear = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
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

        Re-applies the pre-norm, attention, and memory-branch inits, then
        re-initializes the fused down-projection: standard trunc-normal for
        the attention half and near-zero for the memory half, so the block
        starts as a standard attention block.
        """
        self.norm.reset_parameters()
        self.attention.reset_parameters()
        self.gdn2.reset_parameters()
        std = self.initializer_range
        out_proj = self.out_proj
        if isinstance(out_proj, FactoredLinear):
            # The memory half starts at ~1e-3 of the attention half, so a single
            # global std would erase the split. Per-column stds keep both halves
            # at their own scale with no truncation loss.
            column_stds = torch.full(
                (out_proj.input_dim,),
                std,
                dtype=out_proj.in_linear.weight.dtype,
                device=out_proj.in_linear.weight.device,
            )
            column_stds[self.hidden_size :] = self.memory_init_scale
            init_linear_columns(out_proj, column_stds)
            return
        weight = out_proj.weight
        attention_half = weight[:, : self.hidden_size]
        memory_half = weight[:, self.hidden_size :]
        nn.init.trunc_normal_(attention_half, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.normal_(memory_half, mean=0.0, std=self.memory_init_scale)

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
        r"""forward_shared(hidden_states, shared=None, compress_from=None, compress_start_pos=0, position_ids=None, use_cache=False, seeded_store=False) -> (Tensor, CSA2SharedState | None)

        Run one parallel attention ∥ memory layer.

        Normalizes the input once, routes the normalized representation to
        both branches, runs the attention branch through its
        :meth:`~memory_machines.models.blocks.sparse.CompressedSparseAttention2.forward_shared`
        (threading the shared CSA2 state) and the memory branch independently,
        concatenates their outputs, and applies the fused down-projection plus
        input residual.

        Args:
            hidden_states (Tensor): input ``(B, L, d)``.
            shared (CSA2SharedState, optional): cross-layer CSA2 state. Default: ``None``.
            compress_from (Tensor, optional): alternate compression source. Default: ``None``.
            compress_start_pos (int, optional): start position for ``compress_from``. Default: ``0``.
            position_ids (Tensor, optional): absolute positions. Default: ``None``.
            use_cache (bool, optional): CSA2 API parity flag. Default: ``False``.
            seeded_store (bool, optional): pre-built global store (decoder). Default: ``False``.

        Returns:
            Tensor: outputs ``(B, L, d)`` and the updated shared state.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> cfg = MemoryMachineConfig(hidden_size=16, num_attention_heads=2,
            ...     num_key_value_heads=1, csa_top_k=2, csa_num_output_groups=2)
            >>> block = ParallelAttentionGDN2(cfg)
            >>> out, _ = block.forward_shared(torch.randn(1, 4, 16))
            >>> out.shape
            torch.Size([1, 4, 16])
        """
        if hidden_states.dim() != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden_states must be [B, L, d={self.hidden_size}], got {tuple(hidden_states.shape)}")
        h = self.norm(hidden_states)
        attn_out, shared = self.attention.forward_shared(
            h,
            shared,
            compress_from=compress_from,
            compress_start_pos=compress_start_pos,
            position_ids=position_ids,
            use_cache=use_cache,
            seeded_store=seeded_store,
        )
        memory_out, _, _ = self.gdn2(h)
        mixed = torch.cat((attn_out, memory_out), dim=-1)
        return hidden_states + self.out_proj(mixed), shared

    def forward(
        self,
        hidden_states: torch.Tensor,
        shared: CSA2SharedState | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""forward(hidden_states, shared=None, **kwargs) -> Tensor

        Convenience wrapper around :meth:`forward_shared` that drops the
        shared state.
        """
        out, _ = self.forward_shared(hidden_states, shared, **kwargs)
        return out


class ParallelAttentionTitan(ParallelAttentionGDN2):
    r"""ParallelAttentionTitan(config, compression_ratio=None, mode="full", top_k=None, variant=None, window_size=None, memory_init_scale=1e-3)

    Parallel attention ∥ MultiHead SelfReferentialTitans encoder layer.

    Alternates with :class:`ParallelAttentionGDN2` in the CED encoder stack:
    runs a MultiHead sliding-window attention variant in parallel with a
    MultiHead :class:`~memory_machines.models.blocks.fast_weights.SelfReferentialTitanMixer`
    on the same pre-normalized input, fused by the shared ``2d -> d``
    down-projection. Subclass of :class:`ParallelAttentionGDN2` so
    ``isinstance(layer, ParallelAttentionGDN2)`` checks hold; only the second
    branch is swapped.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides model hyperparameters, attention variant, and Titans configurations.
        compression_ratio (int, optional): attention block size ``m``. Default: ``None``.
        mode (str, optional): CSA2 sharing mode (``full``/``reindex``/``reuse``). Default: ``"full"``.
        top_k (int, optional): entries per query. Default: ``None``.
        variant (str, optional): attention variant override. Default: ``None``.
        window_size (int, optional): attention SWA width override. Default: ``None``.
        memory_init_scale (float, optional): std of the memory half of the
            fused down-projection at init. Default: ``1e-3``.
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
