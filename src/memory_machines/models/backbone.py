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
    CompressedSparseAttention2,
    GroupedQueryAttention,
    MHCAttentionGDN2,
    SwiGLU,
)
from memory_machines.models.blocks.ced import (
    CausalDecoder,
    VocabularyUpProjection,
    _build_encoder,
    _run_stream_layers,
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

    Returns hidden states (no vocabulary head) so the HF adapter in
    ``integrations.transformers`` owns the single LM head.
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
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
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
        # HF passes a generic nn.Module; the attribute is an nn.Embedding.
        self.embed = cast(nn.Embedding, value)

    def forward(self, inputs: ModelInput) -> ModelOutput:
        x = inputs.inputs_embeds
        if x is None:
            if inputs.input_ids is None:
                raise ValueError("ModelInput requires input_ids or inputs_embeds")
            x = self.embed(inputs.input_ids)
        for block in self.blocks:
            # Cast keeps the type checker honest on ModuleDict indexing.
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
        # Encoder: owned by this model (token mixers + dense FFNs).
        self.encoder_layers, self.encoder_ffns = _build_encoder(config, encoder_layers)
        # Decoder: the CausalDecoder half (Eq. 1 global-KV seeding).
        self.decoder = CausalDecoder(config, layers=decoder_layers, global_projs=global_projs, ffns=decoder_ffns)
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
        for layer in (*self.encoder_layers, *self.encoder_ffns):
            reset = getattr(layer, "reset_parameters", None)
            if callable(reset):
                reset()
        self.decoder.reset_parameters()
        self.up.reset_parameters()
        if not self.config.tie_word_embeddings:
            std = self.config.initializer_range
            nn.init.trunc_normal_(self.head.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)

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
        shared: CSA2SharedState | None = None
        x = hidden_states
        layers = list(self.encoder_layers)
        ffns = list(self.encoder_ffns)
        for layer in layers:
            if not isinstance(layer, (CompressedSparseAttention2, MHCAttentionGDN2)):
                raise TypeError(f"encoder_layers must hold CompressedSparseAttention2 or MHCAttentionGDN2, got {type(layer).__name__}")
        if layers and any(isinstance(layer, MHCAttentionGDN2) for layer in layers):
            x, shared = _run_stream_layers(cast("list[MHCAttentionGDN2]", layers), hidden_states, None, self.config.mhc_num_streams, position_ids, seeded_store=False, ffns=ffns)
        else:
            for i, layer in enumerate(cast("list[CompressedSparseAttention2]", layers)):
                if shared is None:
                    shared = CSA2SharedState(total_len=0)
                x, shared = layer.forward_shared(x, shared, position_ids=position_ids, use_cache=True)
                x = ffns[i](x, step=i)
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

    def forward(self, inputs: ModelInput) -> ModelOutput:
        """forward(ModelInput) -> ModelOutput

        Standard causal LM forward: embed the input, encode it, decode it
        full-context, widen through the vocabulary up-projection, and apply
        the LM head.
        """
        x = inputs.inputs_embeds
        if x is None:
            if inputs.input_ids is None:
                raise ValueError("CheapLLM requires input_ids or inputs_embeds")
            x = self.embed(inputs.input_ids)
        encoder_final, _ = self.encode(x)
        dec_out = self.decoder.decode_full_context(x, encoder_final)
        out = self.up(dec_out)
        logits = self.head(out)
        return ModelOutput(logits=logits, new_state=None)
