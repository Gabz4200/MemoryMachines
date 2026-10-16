r"""Causal encoder-decoder stack over CSA2 / parallel mHC layers.

:class:`CausalEncoderDecoderStack` wires
:class:`~memory_machines.models.blocks.sparse.CompressedSparseAttention2`
encoder/decoder layers with per-decoder global KV projections
(:class:`CEDGlobalKVProjection`) built from the encoder-final hidden
states. The decoder half is a standalone :class:`CausalDecoder`
(composable and usable on its own).

When ``config.ced_attention_gdn2_parallel`` is set (default), every
*encoder* layer becomes a
:class:`~memory_machines.models.blocks.mhc_parallel.MHCAttentionGDN2` block:
a MultiHead sliding-window attention variant
(``config.ced_attention_variant``, default ``compressed_sparse``) runs in
parallel with a MultiHead GDN-2 mixer, routed by Single-Pass mHC over
persistent residual streams. The GDN-2 branch mixes the *whole* encoder
sequence (never a window), so it supplies the global context while the
attention branch keeps its precision local. The decoder is never a GDN-2
block — it stays plain CSA2 and inherits the global context through the
encoder's KV (Eq. 1). Set the flag to ``False`` to recover plain CSA2
encoder layers.

Every encoder layer is followed by a dense (non-MoE) feed-forward sub-layer
(:class:`~memory_machines.models.blocks.ffn.DenseFFN`, kind
``config.ced_encoder_ffn``, default SwiGLU) with a pre-norm residual, and
every decoder layer is followed by its own independently configured FFN
(``config.ced_decoder_ffn``, default SwiGLU).

With ``config.ced_decoder_looping`` the decoder half becomes a Looped
Language Model (arXiv:2510.25741): the repeated ``CSA2 -> DenseFFN`` pattern
collapses into one parameter-shared layer (plus one shared encoder-KV
projection), applied ``ced_decoder_loop_steps`` times, and an
:class:`AdaptiveExitGate` stops the loop early per position once its
cumulative exit probability crosses
``ced_decoder_loop_exit_threshold``.
"""

from dataclasses import replace
from typing import cast

import torch
from torch import nn

from memory_machines.utils.config_schema import MemoryMachineConfig

from .compression import BlockCompressor
from .ffn import DenseFFN
from .mhc_parallel import MHCAttentionGDN2, from_streams, to_streams
from .shared import CSA2SharedState
from .sparse import CompressedSparseAttention2


def decoder_width(config: MemoryMachineConfig) -> int:
    r"""decoder_width(config) -> int

    Resolve the CED decoder width.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``ced_decoder_hidden_size`` (``0`` = symmetric).

    Returns:
        int: ``ced_decoder_hidden_size`` when set, else ``config.hidden_size``.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> decoder_width(MemoryMachineConfig(hidden_size=1024, ced_decoder_hidden_size=512))
        512
        >>> decoder_width(MemoryMachineConfig(hidden_size=1024))
        1024
    """
    return config.ced_decoder_hidden_size or config.hidden_size


def decoder_config(config: MemoryMachineConfig) -> MemoryMachineConfig:
    r"""decoder_config(config) -> MemoryMachineConfig

    Resolve the config the CED decoder half is built with.

    When ``ced_decoder_hidden_size`` is set, ``hidden_size`` is replaced by that
    narrow width so every decoder sub-layer (CSA2, dense FFN, exit gate) is
    sized for it, while the encoder keeps the wide ``hidden_size``.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            base config.

    Returns:
        MemoryMachineConfig: unchanged when symmetric, else ``hidden_size``
        replaced by :func:`decoder_width`.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> decoder_config(MemoryMachineConfig(hidden_size=32, num_attention_heads=4,
        ...     csa_num_output_groups=2,
        ...     ced_decoder_hidden_size=16)).hidden_size
        16
    """
    width = decoder_width(config)
    return config if width == config.hidden_size else replace(config, hidden_size=width)


def _run_stream_layers(
    layers: list[MHCAttentionGDN2],
    hidden_states: torch.Tensor,
    shared: CSA2SharedState | None,
    num_streams: int,
    position_ids: torch.Tensor | None = None,
    seeded_store: bool = False,
    ffns: list[nn.Module] | None = None,
) -> tuple[torch.Tensor, CSA2SharedState | None]:
    r"""_run_stream_layers(layers, hidden_states, shared, num_streams, position_ids=None, seeded_store=False, ffns=None) -> (Tensor, CSA2SharedState | None)

    Run parallel mHC blocks over persistent residual streams.

    Expands ``hidden_states`` into ``num_streams`` mHC streams
    (:func:`~memory_machines.models.blocks.mhc_parallel.to_streams`), threads
    the single-pass routing coefficients ``prev_a`` between adjacent blocks,
    and collapses the final streams with a mean
    (:func:`~memory_machines.models.blocks.mhc_parallel.from_streams`).

    When ``ffns`` is given (one per block), each FFN is applied to the streams
    right after its block.
    """
    streams = to_streams(hidden_states, num_streams)
    if shared is None:
        shared = CSA2SharedState(total_len=0)
    prev_a: torch.Tensor | None = None
    for i, layer in enumerate(layers):
        streams, shared, prev_a = layer.forward_shared(
            streams,
            shared,
            prev_a=prev_a,
            position_ids=position_ids,
            use_cache=True,
            seeded_store=seeded_store,
        )
        if ffns is not None:
            streams = ffns[i](streams)
    return from_streams(streams), shared


class CEDGlobalKVProjection(nn.Module):
    r"""CEDGlobalKVProjection(config, compression_ratio=None)

    Decoder global KV projection from encoder-final hidden states.

    Wraps a :class:`~memory_machines.models.blocks.compression.BlockCompressor`
    so each decoder layer sees the full encoder context as compressed entries.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``hidden_size``, ``num_attention_heads``,
            ``num_key_value_heads``, and ``csa_decoder_compression_ratio``.
            ``csa_num_output_groups`` only needs to satisfy the
            config's head-divisibility invariant (it is not read
            by this projection itself).
        compression_ratio (int, optional): block size ``m``. Default: ``None``
            (falls back to ``config.csa_decoder_compression_ratio``).
        source_size (int, optional): encoder-stream width the projection reads
            (``W_K``/``W_V`` input). Default: ``None``
            (``config.hidden_size``).
        decoder_hidden_size (int, optional): narrow decoder width the KV
            entries are written in; set for asymmetric encoder/decoder
            dimensions, where this projection *is* the cross-attention
            downsample. Default: ``None`` (``config.hidden_size``).

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> proj = CEDGlobalKVProjection(MemoryMachineConfig(hidden_size=32,
        ...     num_attention_heads=4, num_key_value_heads=2, csa_num_output_groups=2))
        >>> kv, ends = proj(torch.randn(1, 8, 32))
        >>> kv.shape
        torch.Size([1, 8, 2, 8])
    """

    def __init__(
        self,
        config: MemoryMachineConfig,
        compression_ratio: int | None = None,
        source_size: int | None = None,
        decoder_hidden_size: int | None = None,
    ):
        super().__init__()
        source = source_size if source_size is not None else config.hidden_size
        hidden = decoder_hidden_size if decoder_hidden_size is not None else config.hidden_size
        if hidden % config.num_attention_heads != 0:
            raise ValueError(f"decoder_hidden_size ({hidden}) must be divisible by num_attention_heads ({config.num_attention_heads})")
        head_dim = hidden // config.num_attention_heads
        self.source_size = source
        self.decoder_hidden_size = hidden
        self.compressor = BlockCompressor(
            source,
            config.num_key_value_heads,
            head_dim,
            compression_ratio if compression_ratio is not None else config.csa_decoder_compression_ratio,
            init_std=config.initializer_range,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Re-applies the wrapped :class:`BlockCompressor` init
        (truncated-normal KV projection, zero-start scores/bias for
        uniform pooling at step 0). This module holds no other
        parameters.
        """
        self.compressor.reset_parameters()

    def forward(self, encoder_final_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""forward(encoder_final_hidden) -> (Tensor, Tensor)

        Compress encoder-final states into decoder-visible global KV.

        Args:
            encoder_final_hidden (Tensor): encoder outputs of shape
                ``(B, Le, D)``.

        Returns:
            Tensor: compressed entries ``(B, Lce, Hkv, Dh)`` and end
            positions ``(Lce,)`` anchored at ``0``.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> proj = CEDGlobalKVProjection(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, csa_num_output_groups=2))
            >>> proj(torch.randn(1, 6, 16))[0].shape
            torch.Size([1, 6, 1, 8])
        """
        return self.compressor(encoder_final_hidden, start_pos=0)


class CausalEncoderDecoderStack(nn.Module):
    r"""CausalEncoderDecoderStack(config, encoder_layers=None, decoder_layers=None)

    Reference causal encoder-decoder (CED) stack over CSA2 layers.

    Encoder layers compress with mode ``full`` (first) then ``reuse``;
    the decoder half is a :class:`CausalDecoder` (CSA2 layers in the
    paper's ``full``/``reindex``/``reuse`` cycle, each seeded with the
    encoder-final global KV, Eq. 1). Decoding supports full-context and
    bounded-replay (last-window) prefill. By default each *encoder* layer
    is a parallel attention ∥ GDN-2 block routed by Single-Pass mHC
    (``config.ced_attention_gdn2_parallel``); the residual streams persist
    between adjacent encoder layers and the routing coefficients are threaded
    single-pass. Each encoder layer is followed by a dense (non-MoE)
    feed-forward sub-layer (:class:`DenseFFN`, kind ``config.ced_encoder_ffn``,
    default SwiGLU); the decoder uses its own independently configured FFN
    (``config.ced_decoder_ffn``). The decoder carries no GDN-2
    branch — it consumes the encoder's global KV, so GDN-2 reaches it only
    indirectly, even under bounded replay. With ``config.ced_decoder_looping``
    the decoder is a parameter-shared looped :class:`CausalDecoder`
    (arXiv:2510.25741, ``ced_decoder_loop_steps``).

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``ced_num_encoder_layers``, ``ced_num_decoder_layers``,
            ``ced_attention_gdn2_parallel``, ``ced_attention_variant``,
            ``ced_encoder_ffn``, ``ced_decoder_ffn``,
            ``ced_gdn2_backend``,
            ``csa_encoder_compression_ratio``,
            ``csa_decoder_compression_ratio``, ``window_size``, and
            ``swa_bounded_replay``.
        encoder_layers (list of nn.Module, optional): explicit encoder layers.
            Default: ``None`` (build from ``config``).
        decoder_layers (list of nn.Module, optional): explicit decoder layers.
            Default: ``None`` (build from ``config``).

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> stack = CausalEncoderDecoderStack(MemoryMachineConfig(hidden_size=32,
        ...     num_attention_heads=4, num_key_value_heads=2, ced_num_encoder_layers=1,
        ...     ced_num_decoder_layers=1, csa_top_k=4, csa_num_output_groups=4))
        >>> enc = torch.randn(1, 8, 32)
        >>> x, shared = stack.encode(enc)
        >>> stack.decode_full_context(torch.randn(1, 4, 32), x).shape
        torch.Size([1, 4, 32])
    """

    def __init__(
        self,
        config: MemoryMachineConfig,
        encoder_layers: list[nn.Module] | None = None,
        decoder_layers: list[nn.Module] | None = None,
    ):
        super().__init__()
        self.config = config
        enc_n = config.ced_num_encoder_layers
        if encoder_layers is not None:
            enc = list(encoder_layers)
        elif config.ced_attention_gdn2_parallel:
            enc = [MHCAttentionGDN2(config, config.csa_encoder_compression_ratio, "full" if i == 0 else "reuse") for i in range(enc_n)]
        else:
            enc = [CompressedSparseAttention2(config, config.csa_encoder_compression_ratio, "full" if i == 0 else "reuse") for i in range(enc_n)]
        self.encoder_layers = nn.ModuleList(enc)
        # Dense encoder FFNs, one per layer (kind and width from the
        # encoder-side config, so the two halves stay independent).
        self.encoder_ffns = nn.ModuleList([DenseFFN(config, config.ced_encoder_ffn, intermediate_size=config.ced_encoder_ffn_intermediate_size or None) for _ in range(len(enc))])
        # The decoder half is a standalone CausalDecoder so the
        # Eq. 1 seeding path lives in exactly one place.
        self.decoder = CausalDecoder(config, layers=decoder_layers)
        self.window_size = config.window_size
        self.reset_parameters()

    @property
    def decoder_layers(self) -> nn.ModuleList:
        r"""decoder_layers() -> nn.ModuleList

        Decoder CSA2 layers (backward-compatible view of
        :attr:`decoder`).

        Returns:
            nn.ModuleList: the decoder's
            :class:`CompressedSparseAttention2` layers.
        """
        return self.decoder.layers

    @property
    def global_projs(self) -> nn.ModuleList:
        r"""global_projs() -> nn.ModuleList

        Per-layer decoder global KV projections (backward-compatible
        view of :attr:`decoder`).

        Returns:
            nn.ModuleList: the decoder's
            :class:`CEDGlobalKVProjection` layers.
        """
        return self.decoder.global_projs

    @property
    def decoder_ffns(self) -> nn.ModuleList:
        r"""decoder_ffns() -> nn.ModuleList

        Per-layer dense decoder feed-forward sub-layers (view of
        :attr:`CausalDecoder.ffns`).

        Returns:
            nn.ModuleList: the decoder's :class:`DenseFFN` layers.
        """
        return self.decoder.ffns

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Re-applies every encoder layer and the decoder init
        so a freshly built (or re-initialized) stack starts from
        balanced branch fusion, uniform pooling, and zero-start
        (or depth-scaled) residual outputs.
        """
        for layer in (*self.encoder_layers, *self.encoder_ffns):
            reset = getattr(layer, "reset_parameters", None)
            if callable(reset):
                reset()
        self.decoder.reset_parameters()

    def encode(self, hidden_states: torch.Tensor, position_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, CSA2SharedState]:
        r"""encode(hidden_states, position_ids=None) -> (Tensor, CSA2SharedState)

        Run the encoder stack and return final states plus shared state.

        Each encoder layer's token mixer is followed by its dense
        feed-forward sub-layer (:class:`~memory_machines.models.blocks.ffn.DenseFFN`,
        ``config.ced_encoder_ffn``).

        Args:
            hidden_states (Tensor): encoder inputs of shape ``(B, Le, D)``.
            position_ids (Tensor, optional): absolute positions, ``(B, Le)``
                or ``(Le,)``. Default: ``None`` (implicit counting).

        Returns:
            Tensor: encoder-final states ``(B, Le, D)`` and the populated
            :class:`~memory_machines.models.blocks.shared.CSA2SharedState`.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> stack = CausalEncoderDecoderStack(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, ced_num_encoder_layers=1,
            ...     ced_num_decoder_layers=1, csa_top_k=4, csa_num_output_groups=2))
            >>> stack.encode(torch.randn(1, 6, 16))[0].shape
            torch.Size([1, 6, 16])
        """
        shared: CSA2SharedState | None = None
        x = hidden_states
        layers = list(self.encoder_layers)
        ffns = list(self.encoder_ffns)
        for layer in layers:
            if not isinstance(layer, (CompressedSparseAttention2, MHCAttentionGDN2)):
                raise TypeError(f"encoder_layers must hold CompressedSparseAttention2 or MHCAttentionGDN2, got {type(layer).__name__}")
        if layers and all(isinstance(layer, MHCAttentionGDN2) for layer in layers):
            x, shared = _run_stream_layers(cast("list[MHCAttentionGDN2]", layers), hidden_states, None, self.config.mhc_num_streams, position_ids, seeded_store=False, ffns=ffns)
        else:
            for i, layer in enumerate(cast("list[CompressedSparseAttention2]", layers)):
                if shared is None:
                    shared = CSA2SharedState(total_len=0)
                x, shared = layer.forward_shared(x, shared, position_ids=position_ids, use_cache=True)
                x = ffns[i](x)
        if shared is None:
            raise ValueError("encode produced no shared state (encoder_layers is empty)")
        return x, shared

    def decode_full_context(
        self,
        hidden_states: torch.Tensor,
        encoder_final: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        *,
        start_pos: int | None = None,
        seeded_store: bool = True,
        exit_threshold: float | None = None,
    ) -> torch.Tensor:
        r"""decode_full_context(hidden_states, encoder_final, position_ids=None, start_pos=None, seeded_store=True, exit_threshold=None) -> Tensor

        Decode with every decoder layer seeded from the encoder-final states.

        Delegates to :class:`CausalDecoder.decode_full_context`:
        the decoder global KV is projected from the encoder-final
        hidden states with each layer's own weights
        (:class:`CEDGlobalKVProjection`, Eq. 1); it is *not*
        derived from the decoder hidden states. The first decoder
        layer builds the shared store; subsequent ``reindex``/
        ``reuse`` layers consume it, so the projection runs once.

        Args:
            hidden_states (Tensor): decoder inputs of shape ``(B, Ld, D)``.
            encoder_final (Tensor): encoder-final states of shape ``(B, Le, D)``.
            position_ids (Tensor, optional): decoder positions. Default: ``None``.
            start_pos (int, optional): position of ``hidden_states[0]``
                when the inputs continue a prior sequence. Default:
                ``None`` (``encoder_final.shape[1]``, i.e. the inputs
                follow the encoded context — pinned by
                :func:`tests.test_cross_layer_positions.test_ced_stack_positions_stay_correct_across_layers`).
            seeded_store (bool, optional): when ``True`` (default), pass the
                projected global KV to every decoder layer as a
                pre-built store (``forward_shared(...,
                seeded_store=True)``) so the ``full`` layer does not
                recompress decoder hidden states. Used by
                :meth:`prefill`'s bounded-replay path, which decodes
                a tail whose own states must *not* enter the global
                store. Default: ``True``.

        Returns:
            Tensor: decoder outputs of shape ``(B, Ld, D)``.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> stack = CausalEncoderDecoderStack(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, ced_num_encoder_layers=1,
            ...     ced_num_decoder_layers=1, csa_top_k=4, csa_num_output_groups=2))
            >>> enc = torch.randn(1, 6, 16)
            >>> stack.decode_full_context(torch.randn(1, 4, 16), enc).shape
            torch.Size([1, 4, 16])
        """
        return self.decoder.decode_full_context(
            hidden_states,
            encoder_final,
            position_ids,
            start_pos=start_pos,
            seeded_store=seeded_store,
            exit_threshold=exit_threshold,
        )

    def prefill(
        self,
        hidden_states: torch.Tensor,
        encoder_final: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""prefill(hidden_states, encoder_final, position_ids=None) -> Tensor

        Prefill the decoder, optionally replaying only the trailing window.

        Delegates to :class:`CausalDecoder.prefill`: when
        ``config.swa_bounded_replay`` is set, only the last
        ``window_size`` decoder tokens (and their positions) are
        decoded; otherwise this is :meth:`decode_full_context`.
        The replayed tail is anchored at its *true* positions
        ``[enc_len - n, enc_len)`` and its own states are kept out
        of the global compressed store (the store is projected from
        the encoder-final states, Eq. 1), so the tail tokens attend
        only to causally visible encoder entries and their own SWA
        window — no future-token leakage.

        Args:
            hidden_states (Tensor): decoder inputs of shape ``(B, Ld, D)``.
            encoder_final (Tensor): encoder-final states of shape ``(B, Le, D)``.
            position_ids (Tensor, optional): decoder positions matching
                ``hidden_states``. Default: ``None``.

        Returns:
            Tensor: decoder outputs for the (possibly truncated) inputs.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> stack = CausalEncoderDecoderStack(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, ced_num_encoder_layers=1,
            ...     ced_num_decoder_layers=1, csa_top_k=4, csa_num_output_groups=2))
            >>> stack.prefill(torch.randn(1, 4, 16), torch.randn(1, 4, 16)).shape
            torch.Size([1, 4, 16])
        """
        return self.decoder.prefill(hidden_states, encoder_final, position_ids)


class AdaptiveExitGate(nn.Module):
    r"""AdaptiveExitGate(config)

    Per-step exit gate of the looped CED decoder (LoopLM, Eq. 3).

    Runs in parallel with the (external) LM head at every loop step and maps
    the decoder hidden state to an instantaneous exit probability
    :math:`\lambda_t = \sigma(W h_t) \in (0, 1)`. The per-step probabilities
    induce the discrete exit distribution

    .. math::
        p_\phi(t \mid x) = \lambda_t S_{t-1}, \quad
        S_t = \prod_{j \le t} (1 - \lambda_j), \quad S_0 = 1,

    with the remaining mass :math:`S_{T-1}` assigned to the final step so the
    distribution sums to one. Inference terminates at the first step whose
    cumulative exit probability crosses a threshold ``q`` (Q-exit).

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``hidden_size`` and ``initializer_range``.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> gate = AdaptiveExitGate(MemoryMachineConfig(hidden_size=16))
        >>> gate(torch.randn(1, 4, 16)).shape
        torch.Size([1, 4])
    """

    def __init__(self, config: MemoryMachineConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.initializer_range: float = config.initializer_range
        self.proj = nn.Linear(config.hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Re-applies the truncated-normal projection init and zeroes the bias,
        so a freshly built gate starts at the neutral exit probability 0.5.
        """
        std = self.initializer_range
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""forward(hidden_states) -> Tensor

        Predict the per-position instantaneous exit probability.

        Args:
            hidden_states (Tensor): decoder states of shape ``(B, L, D)``.

        Returns:
            Tensor: exit probabilities ``(B, L)`` in ``(0, 1)``.
        """
        return torch.sigmoid(self.proj(hidden_states)).squeeze(-1)

    @staticmethod
    def exit_distribution(lambdas: torch.Tensor) -> torch.Tensor:
        r"""exit_distribution(lambdas) -> Tensor

        Build the exit-step distribution of Eq. 3 from per-step exit
        probabilities.

        Args:
            lambdas (Tensor): per-step exit probabilities ``(B, T, L)``.

        Returns:
            Tensor: exit probabilities ``(B, T, L)`` summing to one over the
            step axis; the final step carries the survival mass
            :math:`S_{T-1}`.
        """
        survival = torch.cumprod(1.0 - lambdas, dim=1)
        survival_prev = torch.cat([torch.ones_like(survival[:, :1]), survival[:, :-1]], dim=1)
        unnormalized = lambdas * survival_prev
        return torch.cat([unnormalized[:, :-1], survival_prev[:, -1:]], dim=1)

    @staticmethod
    def cdf(lambdas: torch.Tensor) -> torch.Tensor:
        r"""cdf(lambdas) -> Tensor

        Cumulative exit probability ``CDF(n) = 1 - prod_{j <= n} (1 - lambda_j)``.

        Args:
            lambdas (Tensor): per-step exit probabilities ``(B, T, L)``.

        Returns:
            Tensor: cumulative exit probabilities ``(B, T, L)``; the final
            entry is exactly ``1``.
        """
        return torch.cumsum(AdaptiveExitGate.exit_distribution(lambdas), dim=1)

    @staticmethod
    def exit_steps(lambdas: torch.Tensor, threshold: float) -> torch.Tensor:
        r"""exit_steps(lambdas, threshold) -> Tensor

        Q-exit step: the first step whose cumulative exit probability
        reaches ``threshold``, clamped to the recurrent depth ``T``.

        Args:
            lambdas (Tensor): per-step exit probabilities ``(B, T, L)``.
            threshold (float): exit threshold ``q`` in ``[0, 1]``.

        Returns:
            Tensor: 1-based exit step per position, shape ``(B, L)``.
        """
        cdf = AdaptiveExitGate.cdf(lambdas)
        steps = (cdf < threshold).sum(dim=1).clamp(max=cdf.shape[1] - 1)
        return steps + 1


class VocabularyUpProjection(nn.Module):
    r"""VocabularyUpProjection(config)

    Restore the encoder width before the vocabulary head.

    A narrow decoder (``ced_decoder_hidden_size``) mapped straight to a large
    vocabulary is rank-deficient (the softmax bottleneck), so this layer widens
    the final decoder states back to ``hidden_size`` with a single
    ``W_up`` matmul (512 -> 1024 in the reference configuration):

    .. math::
        h_{\text{final}} = h_{\text{dec}}^{(K)} W_{\text{up}}

    Apply it to :meth:`CausalDecoder` outputs immediately before the
    classification head; the internal recurrent loop stays narrow.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``hidden_size`` (output) and ``ced_decoder_hidden_size``
            (input), plus ``initializer_range``.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> up = VocabularyUpProjection(MemoryMachineConfig(hidden_size=1024,
        ...     ced_decoder_hidden_size=512))
        >>> up(torch.randn(1, 4, 512)).shape
        torch.Size([1, 4, 1024])
    """

    def __init__(self, config: MemoryMachineConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.decoder_hidden_size = decoder_width(config)
        self.initializer_range: float = config.initializer_range
        self.proj = nn.Linear(self.decoder_hidden_size, self.hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Re-applies the truncated-normal init to ``W_up``.
        """
        std = self.initializer_range
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""forward(hidden_states) -> Tensor

        Widen the final decoder states to the encoder width.

        Args:
            hidden_states (Tensor): decoder outputs ``(B, L, D_dec)``.

        Returns:
            Tensor: widened states ``(B, L, hidden_size)``.
        """
        return self.proj(hidden_states)


class CausalDecoder(nn.Module):
    r"""CausalDecoder(config, layers=None, global_projs=None)

    Standalone causal decoder matching the paper's decoder
    (§2.2, Eq. 1): decoder global KV is projected from the
    encoder-final hidden states :math:`H_{L/2}` with
    layer-dependent weights, *not* derived from the decoder's
    own hidden states. Each layer retains its own global Q and
    SWA KV (SWA-capable, per the local modification).

    Wraps :class:`CompressedSparseAttention2` decoder layers in
    the paper's CSA2 mode cycle (``full`` / ``reindex`` /
    ``reuse``) and per-layer :class:`CEDGlobalKVProjection`
    (:math:`C_l = H_{L/2} W^{KV}_l`). Supports full-context
    decoding and SWA Bounded Replay (§2.3.2) prefill. The decoder
    carries no GDN-2 branch: the encoder owns the global GDN-2 mixer
    (``config.ced_attention_gdn2_parallel``), and the decoder inherits that
    global context through the encoder's KV, so bounded replay stays valid.
    Each decoder layer is followed by a dense (non-MoE) feed-forward sub-layer
    (:class:`DenseFFN`, kind ``config.ced_decoder_ffn``, default SwiGLU),
    configured independently of the encoder's FFN.

    With ``config.ced_decoder_looping``, the decoder is built in the Looped
    Language Model style (arXiv:2510.25741): one parameter-shared
    ``CSA2 -> DenseFFN`` layer (and one shared
    :class:`CEDGlobalKVProjection`) is applied ``ced_decoder_loop_steps``
    times. The layer's ``full`` / ``reindex`` / ``reuse`` stage cycles per
    iteration (:meth:`_loop_mode`), so reindexing still happens inside the
    loop, and :attr:`gate` reads the hidden state after each iteration to
    exit positions early (Q-exit).

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``ced_num_decoder_layers``, ``ced_decoder_ffn``,
            ``csa_decoder_compression_ratio``, ``window_size``,
            ``swa_bounded_replay``, ``ced_decoder_looping``,
            ``ced_decoder_loop_steps``, and
            ``ced_decoder_loop_exit_threshold``.
        layers (list of nn.Module, optional): explicit decoder
            layers. Default: ``None`` (build from ``config``).
        global_projs (list of nn.Module, optional): explicit
            per-layer global KV projections. Default: ``None``
            (build from ``config``).
        ffns (list of nn.Module, optional): explicit per-layer dense FFNs;
            must match the number of decoder layers. Default: ``None``
            (build from ``config``).

    Attributes:
        gate (AdaptiveExitGate): exit gate, present only when the decoder is
            looped (``config.ced_decoder_looping``).
        loop_steps (int): recurrent depth ``T_max`` of the looped decoder.
        exit_threshold (float): default Q-exit threshold ``q``.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> dec = CausalDecoder(MemoryMachineConfig(hidden_size=16,
        ...     num_attention_heads=2, num_key_value_heads=1,
        ...     ced_num_decoder_layers=2, csa_top_k=4,
        ...     csa_num_output_groups=2))
        >>> enc = torch.randn(1, 4, 16)
        >>> dec(torch.randn(1, 4, 16), enc).shape
        torch.Size([1, 4, 16])
    """

    def __init__(
        self,
        config: MemoryMachineConfig,
        layers: list[nn.Module] | None = None,
        global_projs: list[nn.Module] | None = None,
        ffns: list[nn.Module] | None = None,
    ):
        super().__init__()
        self.config = config
        # Asymmetric dims (arXiv-style wide-encoder/narrow-decoder): the decoder
        # half is sized by the narrow decoder config, while its global KV still
        # reads the wide encoder stream (the projection doubles as W_K/W_V).
        dec_cfg = decoder_config(config)
        self.decoder_hidden_size = decoder_width(config)
        # LoopLM (arXiv:2510.25741): with looping on, the repeated
        # CSA2 -> DenseFFN pattern collapses into one parameter-shared layer,
        # so the configured decoder depth becomes the loop count instead.
        dec_n = 1 if config.ced_decoder_looping else config.ced_num_decoder_layers
        layers_list = list(layers) if layers is not None else None
        projs_list = list(global_projs) if global_projs is not None else None
        ffns_list = list(ffns) if ffns is not None else None
        if layers_list is not None and projs_list is not None and len(layers_list) != len(projs_list):
            raise ValueError(f"layers ({len(layers_list)}) and global_projs ({len(projs_list)}) must have equal length; each decoder layer pairs with one global KV projection (Eq. 1)")
        if layers_list is not None and projs_list is None and len(layers_list) != dec_n:
            raise ValueError(f"layers ({len(layers_list)}) must have length dec_n={dec_n} when global_projs is defaulted (zip would silently truncate)")
        if projs_list is not None and layers_list is None and len(projs_list) != dec_n:
            raise ValueError(f"global_projs ({len(projs_list)}) must have length dec_n={dec_n} when layers is defaulted (zip would silently truncate)")
        if layers_list is not None:
            built = layers_list
        else:
            # The decoder never carries a GDN-2 branch: the encoder owns the
            # global GDN-2 mixer, and the decoder uses its global KV (Eq. 1).
            built = [
                CompressedSparseAttention2(
                    dec_cfg,
                    dec_cfg.csa_decoder_compression_ratio,
                    "full" if i == 0 else ("reindex" if i % 4 == 1 else "reuse"),
                )
                for i in range(dec_n)
            ]
        if ffns_list is not None and len(ffns_list) != len(built):
            raise ValueError(f"ffns ({len(ffns_list)}) must match the number of decoder layers ({len(built)}); each layer pairs with one dense FFN")
        self.layers = nn.ModuleList(built)
        # Global KV reads the wide encoder stream and is written in the narrow
        # decoder head space: W_K/W_V act as the cross-attention downsample.
        self.global_projs = nn.ModuleList(
            projs_list if projs_list is not None else [CEDGlobalKVProjection(config, source_size=config.hidden_size, decoder_hidden_size=self.decoder_hidden_size) for _ in range(dec_n)]
        )
        # Dense decoder FFNs, one per layer (kind and width from the
        # decoder-side config, independent of the encoder's).
        self.ffns = nn.ModuleList(
            ffns_list if ffns_list is not None else [DenseFFN(dec_cfg, dec_cfg.ced_decoder_ffn, intermediate_size=dec_cfg.ced_decoder_ffn_intermediate_size or None) for _ in range(len(built))]
        )
        self.window_size = config.window_size
        self.loop_steps = config.ced_decoder_loop_steps
        self.exit_threshold = config.ced_decoder_loop_exit_threshold
        if config.ced_decoder_looping:
            self.gate = AdaptiveExitGate(dec_cfg)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Re-applies every decoder layer, global-projection, and dense FFN init
        so a freshly built (or re-initialized) decoder starts from
        balanced branch fusion, uniform pooling, and zero-start
        (or depth-scaled) residual outputs.
        """
        for module in (*self.layers, *self.global_projs, *self.ffns):
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
        gate = getattr(self, "gate", None)
        if gate is not None:
            gate.reset_parameters()

    def decode_full_context(
        self,
        hidden_states: torch.Tensor,
        encoder_final: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        *,
        start_pos: int | None = None,
        seeded_store: bool = True,
        exit_threshold: float | None = None,
    ) -> torch.Tensor:
        r"""decode_full_context(hidden_states, encoder_final, position_ids=None, start_pos=None, seeded_store=True, exit_threshold=None) -> Tensor

        Decode with every decoder layer seeded from the encoder-final states.

        The decoder global KV is projected from the encoder-final
        hidden states with each layer's own weights
        (:class:`CEDGlobalKVProjection`, Eq. 1); it is *not*
        derived from the decoder hidden states. The first layer
        builds the shared store; subsequent ``reindex``/``reuse``
        layers consume it, so the projection runs once. Each layer is
        followed by its dense feed-forward sub-layer
        (:class:`~memory_machines.models.blocks.ffn.DenseFFN`,
        ``config.ced_decoder_ffn``).

        Args:
            hidden_states (Tensor): decoder inputs of shape ``(B, Ld, D)``.
            encoder_final (Tensor): encoder-final states of shape ``(B, Le, D)``.
            position_ids (Tensor, optional): decoder positions. Default: ``None``.
            start_pos (int, optional): position of ``hidden_states[0]``
                when the inputs continue a prior sequence. Default:
                ``None`` (``encoder_final.shape[1]``).
            seeded_store (bool, optional): when ``True`` (default), pass the
                projected global KV to every decoder layer as a
                pre-built store (``forward_shared(...,
                seeded_store=True)``) so the ``full`` layer does not
                recompress decoder hidden states and double-count them
                onto the encoder store. Set ``False`` only to ablate the
                Eq. 1 seeding (decoder self-compression). Default: ``True``.
            exit_threshold (float, optional): Q-exit threshold ``q``
                overriding ``config.ced_decoder_loop_exit_threshold`` for this
                call (looped decoder only). ``1.0`` forces every loop step.
                Default: ``None`` (use the config value).

        Returns:
            Tensor: decoder outputs of shape ``(B, Ld, D)``.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> dec = CausalDecoder(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1,
            ...     ced_num_decoder_layers=1, csa_top_k=4,
            ...     csa_num_output_groups=2))
            >>> dec.decode_full_context(torch.randn(1, 4, 16),
            ...     torch.randn(1, 6, 16)).shape
            torch.Size([1, 4, 16])
        """
        if start_pos is None:
            start_pos = encoder_final.shape[1]
        if self.config.ced_decoder_looping:
            return self._decode_looped(
                hidden_states,
                encoder_final,
                position_ids,
                start_pos=start_pos,
                seeded_store=seeded_store,
                exit_threshold=exit_threshold,
            )
        layers = cast("list[CompressedSparseAttention2]", list(self.layers))
        for layer in layers:
            if not isinstance(layer, CompressedSparseAttention2):
                raise TypeError(f"decoder layers must hold CompressedSparseAttention2 (no GDN-2 branch), got {type(layer).__name__}")
        if not layers:
            return hidden_states
        # Eq. 1: C_l = H_{L/2} W^KV_l — the global store is projected
        # from the encoder-final states once, then shared by every
        # decoder layer (Full/Reindex/Reuse).
        # NOTE: indexer_k is (re)computed inside forward_shared;
        # do not pre-seed it here (dead compute, immediately overwritten).
        comp, ends = self.global_projs[0](encoder_final)
        shared: CSA2SharedState | None = CSA2SharedState(
            compressed_kv=comp,
            comp_end_pos=ends,
            total_len=start_pos,
        )
        x = hidden_states
        for i, layer in enumerate(layers):
            x, shared = layer.forward_shared(
                x,
                shared,
                position_ids=position_ids,
                use_cache=True,
                seeded_store=seeded_store,
            )
            x = self.ffns[i](x)
        return x

    @staticmethod
    def _loop_mode(step: int) -> str:
        r"""_loop_mode(step) -> str

        CSA2 stage for loop iteration ``step`` (0-based).

        Keeps the paper's ``full`` / ``reindex`` / ``reuse`` cycle (a
        ``reindex`` every fourth iteration) while sharing one layer's weights
        across all iterations, so reindexing still happens inside the loop.

        Args:
            step (int): zero-based loop iteration.

        Returns:
            str: ``"full"`` at step 0, ``"reindex"`` when ``step % 4 == 1``,
            else ``"reuse"``.
        """
        return "full" if step == 0 else ("reindex" if step % 4 == 1 else "reuse")

    def _decode_looped(
        self,
        hidden_states: torch.Tensor,
        encoder_final: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        *,
        start_pos: int,
        seeded_store: bool,
        exit_threshold: float | None,
    ) -> torch.Tensor:
        r"""_decode_looped(hidden_states, encoder_final, position_ids=None, *, start_pos, seeded_store, exit_threshold) -> Tensor

        Run the parameter-shared decoder layer for up to ``loop_steps``
        iterations with per-position early exit.

        The encoder-final global KV is projected once (Eq. 1) and shared by
        every iteration; each iteration applies the same CSA2 layer (stage
        from :meth:`_loop_mode`) followed by the same dense FFN, then reads
        the exit gate. A position stops updating at the first iteration whose
        cumulative exit probability reaches ``exit_threshold``; the loop
        stops early once every position has exited.

        Returns:
            Tensor: decoder outputs ``(B, Ld, D)``, each position taken from
            its own exit iteration.
        """
        threshold = self.exit_threshold if exit_threshold is None else exit_threshold
        layer = cast("CompressedSparseAttention2", self.layers[0])
        if not isinstance(layer, CompressedSparseAttention2):
            raise TypeError(f"looped decoder layers must hold CompressedSparseAttention2 (no GDN-2 branch), got {type(layer).__name__}")
        comp, ends = self.global_projs[0](encoder_final)
        shared: CSA2SharedState | None = CSA2SharedState(compressed_kv=comp, comp_end_pos=ends, total_len=start_pos)
        x = hidden_states
        out = hidden_states
        done = torch.zeros(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        survival = torch.ones(hidden_states.shape[:2], device=hidden_states.device)
        cdf = torch.zeros_like(survival)
        for step in range(self.loop_steps):
            x, shared = layer.forward_shared(
                x,
                shared,
                position_ids=position_ids,
                use_cache=True,
                seeded_store=seeded_store,
                mode=self._loop_mode(step),
            )
            x = self.ffns[0](x)
            lam = self.gate(x)
            if step == self.loop_steps - 1:
                newly_done = ~done
            else:
                cdf = cdf + lam * survival
                survival = survival * (1.0 - lam)
                newly_done = (~done) & (cdf >= threshold)
            out = torch.where(newly_done.unsqueeze(-1), x, out)
            done = done | newly_done
            if bool(done.all()):
                break
        return out

    def prefill(
        self,
        hidden_states: torch.Tensor,
        encoder_final: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""prefill(hidden_states, encoder_final, position_ids=None) -> Tensor

        Prefill the decoder, optionally replaying only the trailing window.

        When ``config.swa_bounded_replay`` is set, only the last
        ``window_size`` decoder tokens (and their positions) are
        decoded; otherwise this is :meth:`decode_full_context`.

        The replayed tail is anchored at its *true* positions
        ``[enc_len - n, enc_len)`` and its own states are kept out
        of the global compressed store (the store is projected from
        the encoder-final states, Eq. 1), so the tail tokens attend
        only to causally visible encoder entries and their own SWA
        window — no future-token leakage.

        Args:
            hidden_states (Tensor): decoder inputs of shape ``(B, Ld, D)``.
            encoder_final (Tensor): encoder-final states of shape ``(B, Le, D)``.
            position_ids (Tensor, optional): decoder positions matching
                ``hidden_states``. Default: ``None``.

        Returns:
            Tensor: decoder outputs for the (possibly truncated) inputs.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> dec = CausalDecoder(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1,
            ...     ced_num_decoder_layers=1, csa_top_k=4,
            ...     csa_num_output_groups=2))
            >>> dec.prefill(torch.randn(1, 4, 16),
            ...     torch.randn(1, 4, 16)).shape
            torch.Size([1, 4, 16])
        """
        if not self.config.swa_bounded_replay:
            return self.decode_full_context(hidden_states, encoder_final, position_ids)
        seq_len = hidden_states.shape[1]
        enc_len = encoder_final.shape[1]
        if self.window_size <= 0 or seq_len > enc_len:
            # Nothing to replay (window disabled) or the
            # prompt exceeds the encoded context, so a tail
            # cannot be reconstructed from the encoder states
            # alone: decode the full context.
            return self.decode_full_context(hidden_states, encoder_final, position_ids)
        # SWA Bounded Replay (§2.3.2): replay only the last
        # ``n_win`` tokens, where ``n_win`` is the SWA window
        # size, to reconstruct the layer-local SWA KV states.
        # Those tokens' own states must *not* enter the global
        # compressed store (the store is projected from the
        # encoder-final states, Eq. 1), so decode with a seeded
        # store anchored at the tail's true start position.
        replay_len = min(self.window_size, seq_len)
        tail = hidden_states[:, -replay_len:, :]
        if seq_len != enc_len and position_ids is None:
            raise ValueError(
                f"bounded replay needs explicit position_ids when decoder length ({seq_len}) != encoder length ({enc_len}); "
                "pass position_ids covering the replay window so the tail anchors at its true positions"
            )
        start_pos = enc_len - replay_len if seq_len == enc_len else seq_len - replay_len
        if position_ids is None:
            position_ids = torch.arange(start_pos, enc_len, device=hidden_states.device).unsqueeze(0).expand(hidden_states.shape[0], -1)
        else:
            position_ids = position_ids[:, -replay_len:] if position_ids.dim() == 2 else position_ids[-replay_len:]
            if position_ids.shape[-1] != replay_len:
                raise ValueError(f"position_ids (length {position_ids.shape[-1]}) must cover the replay window (replay_len={replay_len})")
        return self.decode_full_context(
            tail,
            encoder_final,
            position_ids,
            start_pos=start_pos,
            seeded_store=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_final: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""forward(hidden_states, encoder_final, position_ids=None) -> Tensor

        Prefill the decoder (the default decoding path).

        Delegates to :meth:`prefill`: when
        ``config.swa_bounded_replay`` is set and the prompt fits
        the encoded context, only the last ``window_size`` tokens
        are replayed, so the output's sequence dimension is
        ``min(window_size, hidden_states.shape[1])`` (the
        replayed tail), not necessarily the full prompt length.
        See :meth:`prefill` for the exact truncation rules.

        Args:
            hidden_states (Tensor): decoder inputs of shape ``(B, Ld, D)``.
            encoder_final (Tensor): encoder-final states of shape ``(B, Le, D)``.
            position_ids (Tensor, optional): decoder positions. Default: ``None``.

        Returns:
            Tensor: decoder outputs for the (possibly truncated) inputs.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> dec = CausalDecoder(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1,
            ...     ced_num_decoder_layers=1, csa_top_k=4,
            ...     csa_num_output_groups=2))
            >>> dec(torch.randn(1, 4, 16), torch.randn(1, 4, 16)).shape
            torch.Size([1, 4, 16])
        """
        return self.prefill(hidden_states, encoder_final, position_ids)
