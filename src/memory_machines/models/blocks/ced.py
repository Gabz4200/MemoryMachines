r"""Causal encoder-decoder stack over CSA2 layers.

:class:`CausalEncoderDecoderStack` wires
:class:`~memory_machines.models.blocks.sparse.CompressedSparseAttention2`
encoder/decoder layers with per-decoder global KV projections
(:class:`CEDGlobalKVProjection`) built from the encoder-final hidden states.
"""

import torch
from torch import nn

from memory_machines.utils.config_schema import MemoryMachineConfig

from .compression import BlockCompressor
from .shared import CSA2SharedState
from .sparse import CompressedSparseAttention2


class CEDGlobalKVProjection(nn.Module):
    r"""CEDGlobalKVProjection(config, compression_ratio=None)

    Decoder global KV projection from encoder-final hidden states.

    Wraps a :class:`~memory_machines.models.blocks.compression.BlockCompressor`
    so each decoder layer sees the full encoder context as compressed entries.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``hidden_size``, ``num_attention_heads``,
            ``num_key_value_heads``, and ``csa_decoder_compression_ratio``.
        compression_ratio (int, optional): block size ``m``. Default: ``None``
            (falls back to ``config.csa_decoder_compression_ratio``).

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> proj = CEDGlobalKVProjection(MemoryMachineConfig(hidden_size=32,
        ...     num_attention_heads=4, num_key_value_heads=2, csa_num_output_groups=4))
        >>> kv, ends = proj(torch.randn(1, 8, 32))
        >>> kv.shape
        torch.Size([1, 8, 2, 8])
    """

    def __init__(self, config: MemoryMachineConfig, compression_ratio: int | None = None):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        self.compressor = BlockCompressor(
            config.hidden_size,
            config.num_key_value_heads,
            head_dim,
            compression_ratio if compression_ratio is not None else config.csa_decoder_compression_ratio,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Re-applies the wrapped :class:`BlockCompressor` init
        (Xavier-uniform KV projection, zero-start scores/bias for uniform
        pooling at step 0). This module holds no other parameters.
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

    Encoder layers compress with mode ``full`` (first) then ``reuse``; decoder
    layers use ``full``/``reindex``/``reuse`` cycling, each seeded with the
    encoder-final global KV. Decoding supports full-context and bounded-replay
    (last-window) prefill.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``ced_num_encoder_layers``, ``ced_num_decoder_layers``,
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
        dec_n = config.ced_num_decoder_layers
        self.encoder_layers = nn.ModuleList(
            encoder_layers if encoder_layers is not None else [CompressedSparseAttention2(config, config.csa_encoder_compression_ratio, "full" if i == 0 else "reuse") for i in range(enc_n)]
        )
        self.decoder_layers = nn.ModuleList(
            decoder_layers
            if decoder_layers is not None
            else [CompressedSparseAttention2(config, config.csa_decoder_compression_ratio, "full" if i == 0 else ("reindex" if i % 4 == 1 else "reuse")) for i in range(dec_n)]
        )
        self.global_projs = nn.ModuleList([CEDGlobalKVProjection(config) for _ in range(dec_n)])
        self.window_size = config.window_size
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Re-applies every encoder/decoder layer and global-projection init
        so a freshly built (or re-initialized) stack starts from balanced
        branch fusion, uniform pooling, and zero-start residual outputs.
        """
        for layer in (*self.encoder_layers, *self.decoder_layers, *self.global_projs):
            reset = getattr(layer, "reset_parameters", None)
            if callable(reset):
                reset()

    def encode(self, hidden_states: torch.Tensor, position_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, CSA2SharedState]:
        r"""encode(hidden_states, position_ids=None) -> (Tensor, CSA2SharedState)

        Run the encoder stack and return final states plus shared state.

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
        for layer in self.encoder_layers:
            if not isinstance(layer, CompressedSparseAttention2):
                raise TypeError(f"encoder_layers must hold CompressedSparseAttention2, got {type(layer).__name__}")
            if shared is None:
                shared = CSA2SharedState(total_len=0)
            x, shared = layer.forward_shared(x, shared, position_ids=position_ids, use_cache=True)
        if shared is None:
            raise ValueError("encode produced no shared state (encoder_layers is empty)")
        return x, shared

    def decode_full_context(
        self,
        hidden_states: torch.Tensor,
        encoder_final: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""decode_full_context(hidden_states, encoder_final, position_ids=None) -> Tensor

        Decode with every decoder layer seeded from the encoder-final states.

        Args:
            hidden_states (Tensor): decoder inputs of shape ``(B, Ld, D)``.
            encoder_final (Tensor): encoder-final states of shape ``(B, Le, D)``.
            position_ids (Tensor, optional): decoder positions. Default: ``None``.

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
        shared: CSA2SharedState | None = None
        x = hidden_states
        for layer, proj in zip(self.decoder_layers, self.global_projs):
            if not isinstance(layer, CompressedSparseAttention2):
                raise TypeError(f"decoder_layers must hold CompressedSparseAttention2, got {type(layer).__name__}")
            comp, ends = proj(encoder_final)
            if shared is None:
                shared = CSA2SharedState(compressed_kv=comp, comp_end_pos=ends, total_len=encoder_final.shape[1])
                if layer.indexer is not None:
                    shared.indexer_k = layer.indexer.k_index_proj(comp.mean(dim=2))
            x, shared = layer.forward_shared(x, shared, position_ids=position_ids, use_cache=True)
        return x

    def prefill(
        self,
        hidden_states: torch.Tensor,
        encoder_final: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""prefill(hidden_states, encoder_final, position_ids=None) -> Tensor

        Prefill the decoder, optionally replaying only the trailing window.

        When ``config.swa_bounded_replay`` is set, only the last
        ``window_size`` decoder tokens (and their positions) are decoded;
        otherwise this is :meth:`decode_full_context`.

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
            >>> stack.prefill(torch.randn(1, 4, 16), torch.randn(1, 6, 16)).shape
            torch.Size([1, 4, 16])
        """
        if not self.config.swa_bounded_replay:
            return self.decode_full_context(hidden_states, encoder_final, position_ids)
        tail = hidden_states[:, -self.window_size :, :]
        if position_ids is not None:
            position_ids = position_ids[:, -self.window_size :] if position_ids.dim() == 2 else position_ids[-self.window_size :]
        return self.decode_full_context(tail, encoder_final, position_ids)
