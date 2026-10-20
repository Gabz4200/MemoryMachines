"""Backbone models: pure nn.Module composition roots speaking ModelInput -> ModelOutput.

``CheapLLM`` is the registered causal encoder-decoder language model: it
*owns* the CED encoder (built in ``__init__``), composes the
:class:`~memory_machines.models.blocks.ced.CausalDecoder` as its decoder half,
and projects the (possibly narrow) decoder states back to vocabulary space with
:class:`~memory_machines.models.blocks.ced.VocabularyUpProjection` before the
LM head. Everything is driven by
:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`.
"""

from typing import cast

import torch
from torch import nn

from memory_machines.models.blocks import (
    GroupedQueryAttention,
    SwiGLU,
)
from memory_machines.models.blocks.blt import (
    VocabularyDecoder,
    VocabularyEncoder,
    compute_patch_ids,
)
from memory_machines.models.blocks.ced import (
    CausalDecoder,
    MaskedTokenEmbedding,
    VocabularyUpProjection,
    _build_encoder,
    _run_encoder_layers,
    _validate_encoder_layers,
)
from memory_machines.models.blocks.shared import CSA2SharedState
from memory_machines.models.io import ModelInput, ModelOutput
from memory_machines.models.registry import register_model
from memory_machines.utils.config_schema import MemoryMachineConfig


class _GQAWrapper(nn.Module):
    """GroupedQueryAttention returning hidden states only (drops weights/cache)."""

    def __init__(self, attn: GroupedQueryAttention):
        super().__init__()
        self.attn = attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _, _ = self.attn(x)
        return out


@register_model("default")
@register_model("example")
class ExampleBackbone(nn.Module):
    """Minimal flat composition: embedding + GQA/SwiGLU blocks, a norm, a head.

    Owns a single embedding and a single LM head and returns vocabulary
    logits (``ModelOutput(logits=...)``); the HF adapter in
    ``integrations/transformers`` reuses ``self.head`` as ``lm_head`` so the
    head tensor is registered exactly once.

    Padding is *not* supported: ``ModelInput.attention_mask`` is accepted for
    API parity but ignored, so padded tokens attend to each other. Pass a
    properly-shaped bool/additive mask to :class:`GroupedQueryAttention`
    directly when padding matters.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        vocab_size: int = 1000,
        backend: str = "naive",
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be positive")
        if num_layers <= 0:
            raise ValueError(f"num_layers ({num_layers}) must be positive")
        if num_heads <= 0:
            raise ValueError(f"num_heads ({num_heads}) must be positive")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size ({vocab_size}) must be positive")
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        # ``backend`` is accepted for API symmetry with the CED stack but is
        # not consumed: this flat GQA/SwiGLU backbone has no backend switch.
        self.backend = backend
        cfg = MemoryMachineConfig(
            hidden_size=embed_dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            # Unused by this backbone, but validated at boot: keep 1 head/group.
            csa_num_output_groups=num_heads,
        )
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": _GQAWrapper(GroupedQueryAttention(cfg)),
                        "ff": SwiGLU(cfg),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed = cast(nn.Embedding, value)

    def forward(self, inputs: ModelInput) -> ModelOutput:
        x = inputs.inputs_embeds
        if x is None:
            if inputs.input_ids is None:
                raise ValueError("ModelInput requires input_ids or inputs_embeds")
            x = self.embed(inputs.input_ids)
        for block in self.blocks:
            pair = cast(nn.ModuleDict, block)
            x = pair["ff"](pair["attn"](x))
        x = self.norm(x)
        return ModelOutput(logits=self.head(x), new_state=None)


@register_model("cheapllm")
@register_model("CheapLLM")
class CheapLLM(nn.Module):
    """CheapLLM(config, encoder_layers=None, decoder_layers=None, global_projs=None, decoder_ffns=None, vocab_size=None)

    Causal encoder-decoder LLM that owns the encoder and composes the decoder.

    The encoder half (token mixers + dense FFNs) is built in ``__init__`` via
    :func:`~memory_machines.models.blocks.ced._build_encoder`. The decoder half
    is a :class:`~memory_machines.models.blocks.ced.CausalDecoder`, so the
    Eq. 1 global-KV seeding path stays in one place. The (possibly narrow)
    decoder output is widened back to ``hidden_size`` by
    :class:`~memory_machines.models.blocks.ced.VocabularyUpProjection` and
    fed to an LM head to produce logits.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            drives every sub-module (encoder, decoder, up-projection, head).
        encoder_layers (list of nn.Module, optional): explicit encoder layers.
            Default: ``None`` (build from ``config``).
        decoder_layers (list of nn.Module, optional): explicit decoder layers.
            Default: ``None`` (build from ``config``).
        global_projs (list of nn.Module, optional): explicit decoder global KV
            projections. Default: ``None`` (build from ``config``).
        decoder_ffns (list of nn.Module, optional): explicit decoder FFNs.
            Default: ``None`` (build from ``config``).
        vocab_size (int, optional): output vocabulary size. Default: ``None``
            (falls back to ``config.vocab_size``).

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> from memory_machines.models.io import ModelInput
        >>> cfg = MemoryMachineConfig(hidden_size=16, vocab_size=32,
        ...     num_attention_heads=2, num_key_value_heads=1,
        ...     ced_num_encoder_layers=1, ced_num_decoder_layers=1,
        ...     csa_top_k=4, csa_num_output_groups=2)
        >>> CheapLLM(cfg)(ModelInput(input_ids=torch.randint(0, 32, (1, 6)))).logits.shape
        torch.Size([1, 6, 32])
    """

    def __init__(
        self,
        config: MemoryMachineConfig,
        encoder_layers: list[nn.Module] | None = None,
        decoder_layers: list[nn.Module] | None = None,
        global_projs: list[nn.Module] | None = None,
        decoder_ffns: list[nn.Module] | None = None,
        vocab_size: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size if vocab_size is not None else config.vocab_size
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size ({self.vocab_size}) must be positive")
        if config.hidden_size <= 0:
            raise ValueError(f"hidden_size ({config.hidden_size}) must be positive")
        # Encoder: owned by this model (token mixers + dense FFNs). Validate
        # the caller-supplied layers once at construction so a wrong-typed
        # entry surfaces here, not inside ``encode``.
        self.encoder_layers, self.encoder_ffns = _build_encoder(config, encoder_layers)
        _validate_encoder_layers(self.encoder_layers)
        # Decoder: the CausalDecoder half (Eq. 1 global-KV seeding).
        self.decoder = CausalDecoder(
            config,
            layers=decoder_layers,
            global_projs=global_projs,
            ffns=decoder_ffns,
        )
        # Learnable masked-token embedding for Perceiver-AR generation
        # (variable count of new tokens appended at the end of input).
        self.masked_token_embed = MaskedTokenEmbedding(
            config.ced_masked_token_count,
            config.hidden_size,
        )
        # Widen (possibly narrow) decoder states back to hidden_size for the head.
        self.up = VocabularyUpProjection(config)
        self.embed = nn.Embedding(self.vocab_size, config.hidden_size)
        self.head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.head.weight = self.embed.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """reset_parameters() -> None

        Re-applies every encoder layer, encoder FFN, the decoder, and the
        up-projection init. The embedding uses PyTorch's default init; the
        head is re-initialized only when it is not weight-tied to the embedding.
        """
        # Every registered component here (CSA2/parallel-hybrid, DenseFFN, CEDGlobalKVProjection,
        # CausalDecoder, VocabularyUpProjection) defines ``reset_parameters``;
        # the ``getattr`` guard is defensive only and should never skip a
        # module in practice. A missing ``reset_parameters`` is a contract
        # violation, not a reason to stay silent.
        for layer in (*self.encoder_layers, *self.encoder_ffns):
            reset = getattr(layer, "reset_parameters", None)
            if callable(reset):
                reset()
        self.decoder.reset_parameters()
        self.up.reset_parameters()
        if not self.config.tie_word_embeddings:
            std = self.config.initializer_range
            nn.init.trunc_normal_(self.head.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed = cast(nn.Embedding, value)

    @property
    def decoder_layers(self) -> nn.ModuleList:
        """Backward-compatible view of :attr:`decoder`."""
        return self.decoder.layers

    @property
    def global_projs(self) -> nn.ModuleList:
        """Per-layer decoder global KV projections (view of :attr:`decoder`)."""
        return self.decoder.global_projs

    @property
    def decoder_ffns(self) -> nn.ModuleList:
        """Per-layer dense decoder feed-forward sub-layers (view of :attr:`decoder`)."""
        return self.decoder.ffns

    def encode(self, hidden_states: torch.Tensor, position_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, CSA2SharedState]:
        """encode(hidden_states, position_ids=None) -> (Tensor, CSA2SharedState)

        Run the encoder stack and return final states plus shared state.

        Each encoder layer's token mixer is followed by its dense feed-forward
        sub-layer.

        Args:
            hidden_states (Tensor): encoder inputs of shape ``(B, Le, D)``.
            position_ids (Tensor, optional): absolute positions, ``(B, Le)``
                or ``(Le,)``. Default: ``None`` (implicit counting).

        Returns:
            Tuple[Tensor, CSA2SharedState]: encoder-final states ``(B, Le, D)``
            and the populated :class:`~memory_machines.models.blocks.shared.CSA2SharedState`.
        """
        layers = list(self.encoder_layers)
        ffns = list(self.encoder_ffns)
        return _run_encoder_layers(layers, ffns, hidden_states, position_ids)

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
        """Decode with every decoder layer seeded from the encoder-final states.

        Delegates to :meth:`CausalDecoder.decode_full_context`: the decoder
        global KV is projected from the encoder-final hidden states with each
        layer's own weights (:class:`CEDGlobalKVProjection`, Eq. 1).
        """
        return self.decoder.decode_full_context(hidden_states, encoder_final, position_ids, start_pos=start_pos, seeded_store=seeded_store, exit_threshold=exit_threshold)

    def prefill(self, hidden_states: torch.Tensor, encoder_final: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Prefill decode via bounded replay (last window).

        Delegates to :meth:`CausalDecoder.prefill`: truncates ``hidden_states``
        to the last window before decoding, conditioned on the encoder context.
        """
        return self.decoder.prefill(hidden_states, encoder_final, position_ids)

    def forward(self, inputs: ModelInput | torch.Tensor) -> ModelOutput:
        """forward(ModelInput | Tensor) -> ModelOutput

        Standard causal LM forward: embed the input, encode it, decode it
        full-context, widen through the vocabulary up-projection, and apply
        the LM head. ``inputs.attention_mask`` is accepted for API parity
        (the CED decoder is causal and ignores padding) but not consumed.

        A bare ``torch.Tensor`` is accepted as a shorthand for
        ``ModelInput(input_ids=tensor)`` so the HF adapter and ad-hoc
        scripts can call ``model(ids)`` directly; ``ModelInput`` is still
        the canonical contract and is required for anything beyond
        ``input_ids`` (e.g. ``num_masked_tokens`` for Perceiver-AR).
        """
        if isinstance(inputs, torch.Tensor):
            inputs = ModelInput(input_ids=inputs)
        x = inputs.inputs_embeds
        if x is None:
            if inputs.input_ids is None:
                raise ValueError("CheapLLM requires input_ids or inputs_embeds")
            x = self.embed(inputs.input_ids)
        # Perceiver-AR masked-token generation: append learnable masked embeddings.
        L = inputs.num_masked_tokens
        if L is not None and L > 0 and self.config.ced_masked_query:
            # Append masked-token embeddings at the end.
            mask_indices = torch.arange(L, device=x.device)
            masked_embeds = self.masked_token_embed(mask_indices)  # (L, D)
            masked_embeds = masked_embeds.unsqueeze(0).expand(x.shape[0], -1, -1)  # (B, L, D)
            x = torch.cat([x, masked_embeds], dim=1)
        encoder_final, _ = self.encode(x)
        # Decode: only masked tokens produce queries; full context provides K/V.
        if L is not None and L > 0 and self.config.ced_masked_query:
            N = x.shape[1]
            # Query stream = masked positions (last L); KV stream = full sequence.
            # Pass position_ids explicitly so the masked queries anchor at
            # [N-L, N) by contract, not by accident of the fallback arithmetic
            # inside CausalDecoder.decode_full_context (which derives positions
            # from start_pos when position_ids is None).
            dec_pos = torch.arange(N - L, N, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
            dec_out = self.decoder.decode_full_context(
                x[:, N - L :, :],
                encoder_final,
                position_ids=dec_pos,
                num_masked_tokens=L,
                start_pos=N - L,
            )
            dec_out = dec_out[:, -L:, :]
        else:
            dec_out = self.decoder.decode_full_context(x, encoder_final)
        out = self.up(dec_out)
        logits = self.head(out)
        return ModelOutput(logits=logits, new_state=None)


@register_model("blt")
@register_model("ByteLatentTransformer")
class BLTBackbone(nn.Module):
    """BLTBackbone(config, encoder_layers=None, decoder_layers=None, global_projs=None, decoder_ffns=None, vocab_size=None)

    Byte Latent Transformer: local byte models around the CED latent stack.

    :class:`~memory_machines.models.blocks.blt.VocabularyEncoder` replaces the
    token embedding: it turns byte ids into patch representations (hash n-gram
    byte embeddings, causal byte transformer, byte-to-patch cross-attention).
    The CED encoder/decoder then play the role of the paper's Latent
    Transformer and operate on patches instead of tokens, including the
    Perceiver-AR masked-patch path. :class:`~memory_machines.models.blocks.blt.VocabularyDecoder`
    replaces the LM head: it maps latent patch representations back to
    next-byte logits through patch-to-byte cross-attention and a causal byte
    transformer.

    The strided patching fallback uses ``config.blt_patch_size``; pass
    ``ModelInput.patch_ids`` to supply an entropy or space patching produced
    elsewhere. In full (unmasked) mode byte ``i`` attends to the latent of
    patch ``p(i) - 1`` (the learned begin-of-bytes patch when ``p(i) == 0``),
    so its next-byte target is never inside the attended patch and logits at
    ``i`` predict byte ``i + 1``. With ``num_masked_tokens=L`` (requires
    ``ced_masked_query=True``) the CED decoder produces ``L`` masked patch
    states and the vocabulary decoder expands each into
    ``config.blt_max_patch_bytes`` byte slots conditioned on that patch's
    latent decoder output.

    ``ModelInput.attention_mask`` is accepted for API parity but not consumed
    (the byte transformer is causal and has no padding mask), so batches must
    contain packed byte sequences.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            drives the byte models and the latent CED stack.
        encoder_layers (list of nn.Module, optional): explicit latent encoder
            layers. Default: ``None`` (build from ``config``).
        decoder_layers (list of nn.Module, optional): explicit latent decoder
            layers. Default: ``None`` (build from ``config``).
        global_projs (list of nn.Module, optional): explicit decoder global KV
            projections. Default: ``None`` (build from ``config``).
        decoder_ffns (list of nn.Module, optional): explicit decoder FFNs.
            Default: ``None`` (build from ``config``).
        vocab_size (int, optional): byte vocabulary size. Default: ``None``
            (``config.blt_vocab_size`` or ``config.vocab_size``).

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> from memory_machines.models.io import ModelInput
        >>> cfg = MemoryMachineConfig(hidden_size=16, vocab_size=32,
        ...     num_attention_heads=2, num_key_value_heads=1,
        ...     ced_num_encoder_layers=1, ced_num_decoder_layers=1,
        ...     csa_top_k=2, csa_num_output_groups=2,
        ...     blt_local_hidden_size=16, blt_local_layers=1,
        ...     blt_local_window_size=8)
        >>> BLTBackbone(cfg)(ModelInput(input_ids=torch.randint(0, 32, (1, 6)))).logits.shape
        torch.Size([1, 6, 32])
    """

    def __init__(
        self,
        config: MemoryMachineConfig,
        encoder_layers: list[nn.Module] | None = None,
        decoder_layers: list[nn.Module] | None = None,
        global_projs: list[nn.Module] | None = None,
        decoder_ffns: list[nn.Module] | None = None,
        vocab_size: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size if vocab_size is not None else (config.blt_vocab_size or config.vocab_size)
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size ({self.vocab_size}) must be positive")
        if config.hidden_size <= 0:
            raise ValueError(f"hidden_size ({config.hidden_size}) must be positive")
        self.vocab_encoder = VocabularyEncoder(config, vocab_size=self.vocab_size)
        self.vocab_decoder = VocabularyDecoder(config, vocab_size=self.vocab_size)
        if config.blt_tie_word_embeddings:
            self.vocab_decoder.output.weight = self.vocab_encoder.byte_embeddings.weight
        # The latent transformer is the existing CED stack, now over patches.
        self.encoder_layers, self.encoder_ffns = _build_encoder(config, encoder_layers)
        _validate_encoder_layers(self.encoder_layers)
        self.decoder = CausalDecoder(
            config,
            layers=decoder_layers,
            global_projs=global_projs,
            ffns=decoder_ffns,
        )
        self.masked_token_embed = MaskedTokenEmbedding(
            config.ced_masked_token_count,
            config.hidden_size,
        )
        self.up = VocabularyUpProjection(config)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-apply byte-model, latent-stack, and vocabulary-head initialization."""
        self.vocab_encoder.reset_parameters()
        self.vocab_decoder.reset_parameters()
        for layer in (*self.encoder_layers, *self.encoder_ffns):
            reset = getattr(layer, "reset_parameters", None)
            if callable(reset):
                reset()
        self.decoder.reset_parameters()
        self.up.reset_parameters()
        if self.config.blt_tie_word_embeddings:
            # reset_parameters re-initialized both tensors; keep them tied.
            self.vocab_decoder.output.weight = self.vocab_encoder.byte_embeddings.weight

    def get_input_embeddings(self) -> nn.Module:
        return self.vocab_encoder.byte_embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.vocab_encoder.byte_embeddings = cast(nn.Embedding, value)

    @property
    def embed(self) -> nn.Embedding:
        """The byte embedding table (input-embedding API parity)."""
        return self.vocab_encoder.byte_embeddings

    @property
    def decoder_layers(self) -> nn.ModuleList:
        """Backward-compatible view of :attr:`decoder`."""
        return self.decoder.layers

    @property
    def global_projs(self) -> nn.ModuleList:
        """Per-layer decoder global KV projections (view of :attr:`decoder`)."""
        return self.decoder.global_projs

    @property
    def decoder_ffns(self) -> nn.ModuleList:
        """Per-layer dense decoder feed-forward sub-layers (view of :attr:`decoder`)."""
        return self.decoder.ffns

    def encode(self, hidden_states: torch.Tensor, position_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, CSA2SharedState]:
        """encode(hidden_states, position_ids=None) -> (Tensor, CSA2SharedState)

        Run the latent encoder stack over patch representations.
        """
        return _run_encoder_layers(list(self.encoder_layers), list(self.encoder_ffns), hidden_states, position_ids)

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
        """Decode latent patch states seeded by the encoder-final states."""
        return self.decoder.decode_full_context(
            hidden_states,
            encoder_final,
            position_ids,
            start_pos=start_pos,
            seeded_store=seeded_store,
            exit_threshold=exit_threshold,
        )

    def prefill(self, hidden_states: torch.Tensor, encoder_final: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Prefill decode via bounded replay (last window)."""
        return self.decoder.prefill(hidden_states, encoder_final, position_ids)

    def forward(self, inputs: ModelInput | torch.Tensor) -> ModelOutput:
        """forward(ModelInput | Tensor) -> ModelOutput

        Run bytes through the local encoder, the latent CED stack, and the
        local decoder.

        Full mode returns next-byte logits ``(B, N, vocab_size)``; position
        ``i`` predicts byte ``i + 1`` (shift the labels by one, as for
        :class:`CheapLLM`). Masked-patch mode
        (``num_masked_tokens=L``) returns ``(B, L * blt_max_patch_bytes,
        vocab_size)`` logits for the future patches.
        """
        if isinstance(inputs, torch.Tensor):
            inputs = ModelInput(input_ids=inputs)
        if inputs.input_ids is None and inputs.inputs_embeds is None:
            raise ValueError("BLTBackbone requires input_ids or inputs_embeds")
        patch_ids = inputs.patch_ids
        if patch_ids is None:
            if inputs.input_ids is None:
                raise ValueError("patch_ids is required when inputs_embeds is given (the strided fallback needs byte ids)")
            patch_ids = compute_patch_ids(inputs.input_ids, self.config.blt_patch_size)
        patch_reps, byte_hidden = self.vocab_encoder(inputs.input_ids, patch_ids, inputs.inputs_embeds)
        num_patches = patch_reps.shape[1]
        x = patch_reps
        L = inputs.num_masked_tokens
        if L is not None and L > 0 and not self.config.ced_masked_query:
            raise ValueError("num_masked_tokens requires ced_masked_query=True")
        if L is not None and L < 0:
            raise ValueError(f"num_masked_tokens ({L}) must be non-negative")
        if L is not None and L > self.config.ced_masked_token_count:
            raise ValueError(f"num_masked_tokens ({L}) exceeds ced_masked_token_count ({self.config.ced_masked_token_count})")
        masked = L is not None and L > 0
        if masked:
            assert L is not None
            mask_indices = torch.arange(L, device=x.device)
            masked_embeds = self.masked_token_embed(mask_indices).unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([x, masked_embeds], dim=1)
        encoder_final, _ = self.encode(x)
        if masked:
            assert L is not None
            total = x.shape[1]
            dec_pos = torch.arange(total - L, total, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
            dec_out = self.decoder.decode_full_context(
                x[:, total - L :, :],
                encoder_final,
                position_ids=dec_pos,
                num_masked_tokens=L,
                start_pos=total - L,
            )
            dec_out = dec_out[:, -L:, :]
            masked_reps = self.up(dec_out)
            # Masked slots condition on their own patch's latent decoder output
            # (the prefix is never attended).
            patch_reps = torch.cat([encoder_final[:, :num_patches], masked_reps], dim=1)
            logits = self.vocab_decoder.decode_masked_patches(masked_reps, patch_reps, first_patch_index=num_patches)
        else:
            dec_out = self.decoder.decode_full_context(x, encoder_final)
            # Patch p of byte i holds bytes up to i, so query i reads latent p - 1
            # (the begin patch for p == 0). Even then logits[i] predicts byte i + 1
            # only after that byte's own patch; see VocabularyDecoder.forward.
            logits = self.vocab_decoder(byte_hidden, self.up(dec_out), patch_ids)
        return ModelOutput(logits=logits, new_state=None)
