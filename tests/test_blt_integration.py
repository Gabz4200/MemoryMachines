"""BLTBackbone integration: registry, patch-lag contract, config and parity.

``tests/test_blt.py`` covers the local byte models in isolation; this module
pins the composition-level contract of the registered ``BLTBackbone``: which
patch ids reach :meth:`VocabularyDecoder.forward`, how ``vocab_size`` resolves,
what gradient paths are live end to end, and that ``CheapLLM`` is unaffected.
"""

from typing import Any

import pytest
import torch

from memory_machines.models.backbone import BLTBackbone, CheapLLM
from memory_machines.models.blocks.blt import (
    VocabularyDecoder,
    VocabularyEncoder,
    compute_patch_ids,
)
from memory_machines.models.io import ModelInput
from memory_machines.models.registry import get_model
from memory_machines.utils.config_schema import MemoryMachineConfig

VOCAB = 32
LOCAL = 16
N = 8


def _cfg(**kw: Any) -> MemoryMachineConfig:
    base: dict[str, Any] = {
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 32,
        "ced_num_encoder_layers": 1,
        "ced_num_decoder_layers": 1,
        "csa_top_k": 2,
        "csa_num_output_groups": 2,
        "csa_output_group_dim": 8,
        "csa_indexer_heads": 2,
        "csa_indexer_dim": 8,
        "csa_hier_pool_blocks": 2,
        "csa_hier_pool_positions": 8,
        "csa_hier_block_size": 2,
        "vocab_size": VOCAB,
        "window_size": 8,
        "blt_local_hidden_size": LOCAL,
        "blt_local_layers": 1,
        "blt_local_window_size": 8,
        "blt_local_ffn_intermediate_size": 32,
        "blt_ngram_hash_size": 64,
        "blt_max_patch_bytes": 4,
        "blt_patch_size": 2,
    }
    base.update(kw)
    return MemoryMachineConfig(**base)


def _ids(n: int = N, batch: int = 1) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, VOCAB, (batch, n))


def _record_decoder_patch_ids(model: BLTBackbone, monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    """Wrap ``vocab_decoder.forward`` and collect the patch ids it receives."""
    calls: list[torch.Tensor] = []
    original = model.vocab_decoder.forward

    def recorder(byte_states: torch.Tensor, patch_reps: torch.Tensor, patch_ids: torch.Tensor) -> torch.Tensor:
        calls.append(patch_ids)
        return original(byte_states, patch_reps, patch_ids)

    monkeypatch.setattr(model.vocab_decoder, "forward", recorder)
    return calls


# ---------------------------------------------------------------- registry --
@pytest.mark.parametrize("name", ["blt", "ByteLatentTransformer"])
def test_registry_resolves_blt_aliases(name: str) -> None:
    model = get_model(name, config=_cfg())
    assert isinstance(model, BLTBackbone)


def test_registry_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError, match="unknown model"):
        get_model("definitely-not-a-model", config=_cfg())


# ------------------------------------------------------------ full forward --
def test_full_mode_shapes_for_tensor_and_model_input() -> None:
    model = BLTBackbone(_cfg()).eval()
    ids = _ids(batch=2)
    with torch.no_grad():
        from_tensor = model(ids).logits
        from_input = model(ModelInput(input_ids=ids)).logits
    assert tuple(from_tensor.shape) == (2, N, VOCAB)
    assert tuple(from_input.shape) == (2, N, VOCAB)
    assert torch.equal(from_tensor, from_input)


# ---------------------------------------------------------- patch-lag pins --
def test_explicit_patch_ids_are_forwarded_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    model = BLTBackbone(_cfg()).eval()
    ids = _ids(n=6)
    patch_ids = torch.tensor([[0, 0, 0, 1, 1, 2]])
    calls = _record_decoder_patch_ids(model, monkeypatch)
    with torch.no_grad():
        out = model(ModelInput(input_ids=ids, patch_ids=patch_ids))
    assert tuple(out.logits.shape) == (1, 6, VOCAB)
    assert len(calls) == 1
    assert torch.equal(calls[0], patch_ids)


def test_strided_fallback_is_forwarded_as_computed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    model = BLTBackbone(cfg).eval()
    ids = _ids(n=7)
    calls = _record_decoder_patch_ids(model, monkeypatch)
    with torch.no_grad():
        model(ModelInput(input_ids=ids))
    expected = compute_patch_ids(ids, cfg.blt_patch_size)
    assert len(calls) == 1
    assert torch.equal(calls[0], expected)
    assert calls[0].tolist() == [[0, 0, 1, 1, 2, 2, 3]]


# ----------------------------------------------------------- inputs_embeds --
def test_inputs_embeds_without_patch_ids_raises() -> None:
    model = BLTBackbone(_cfg()).eval()
    ids = _ids(n=6)
    with torch.no_grad():
        embeds = model.vocab_encoder.byte_embeddings(ids)
    with pytest.raises(ValueError, match="patch_ids"):
        model(ModelInput(inputs_embeds=embeds))


def test_inputs_embeds_matches_input_ids_without_ngrams() -> None:
    cfg = _cfg(blt_use_ngram_embeddings=False)
    model = BLTBackbone(cfg).eval()
    ids = _ids(n=6, batch=2)
    patch_ids = compute_patch_ids(ids, cfg.blt_patch_size)
    with torch.no_grad():
        embeds = model.vocab_encoder.byte_embeddings(ids)
        by_ids = model(ModelInput(input_ids=ids, patch_ids=patch_ids)).logits
        by_embeds = model(ModelInput(inputs_embeds=embeds, patch_ids=patch_ids)).logits
    torch.testing.assert_close(by_embeds, by_ids, atol=1e-6, rtol=1e-6)


def test_inputs_embeds_with_patch_ids_shapes() -> None:
    model = BLTBackbone(_cfg()).eval()
    ids = _ids(n=6)
    patch_ids = compute_patch_ids(ids, 2)
    with torch.no_grad():
        embeds = model.vocab_encoder.byte_embeddings(ids)
        out = model(ModelInput(inputs_embeds=embeds, patch_ids=patch_ids))
    assert tuple(out.logits.shape) == (1, 6, VOCAB)
    assert torch.isfinite(out.logits).all()


# ----------------------------------------------------------- attention mask --
def test_attention_mask_is_ignored() -> None:
    model = BLTBackbone(_cfg()).eval()
    ids = _ids(n=6)
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        without = model(ModelInput(input_ids=ids)).logits
        with_mask = model(ModelInput(input_ids=ids, attention_mask=mask)).logits
    assert torch.equal(with_mask, without)


# ------------------------------------------------------------- masked mode --
def test_masked_mode_shape() -> None:
    cfg = _cfg(ced_masked_query=True, ced_masked_token_count=4, blt_max_patch_bytes=4)
    model = BLTBackbone(cfg).eval()
    with torch.no_grad():
        out = model(ModelInput(input_ids=_ids(), num_masked_tokens=2))
    assert tuple(out.logits.shape) == (1, 2 * cfg.blt_max_patch_bytes, VOCAB)


@pytest.mark.parametrize("max_patch_bytes", [2, 6], ids=["two", "six"])
def test_masked_mode_length_tracks_max_patch_bytes(max_patch_bytes: int) -> None:
    cfg = _cfg(
        ced_masked_query=True,
        ced_masked_token_count=4,
        blt_max_patch_bytes=max_patch_bytes,
    )
    model = BLTBackbone(cfg).eval()
    with torch.no_grad():
        out = model(ModelInput(input_ids=_ids(), num_masked_tokens=3))
    assert out.logits.shape[1] == 3 * max_patch_bytes


def test_masked_mode_feeds_latent_outputs_to_local_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(ced_masked_query=True, ced_masked_token_count=4)
    model = BLTBackbone(cfg).eval()
    captured: dict[str, torch.Tensor] = {}
    first_indices: list[int] = []
    original = model.vocab_decoder.decode_masked_patches

    def spy(masked_states: torch.Tensor, patch_reps: torch.Tensor, first_patch_index: int) -> torch.Tensor:
        captured["masked_states"] = masked_states
        captured["patch_reps"] = patch_reps
        first_indices.append(first_patch_index)
        return original(masked_states, patch_reps, first_patch_index)

    monkeypatch.setattr(model.vocab_decoder, "decode_masked_patches", spy)
    with torch.no_grad():
        model(ModelInput(input_ids=_ids(), num_masked_tokens=2))
    first = first_indices[0]
    # Each masked byte slot conditions on its own patch's latent decoder output
    # (the global output o_j), not on the masked-token encoder state.
    assert first == N // cfg.blt_patch_size
    assert torch.equal(captured["patch_reps"][:, first:], captured["masked_states"])


@pytest.mark.parametrize("masked", [None, 0], ids=["none", "zero"])
def test_zero_masked_tokens_is_full_mode(masked: int | None) -> None:
    cfg = _cfg(ced_masked_query=True, ced_masked_token_count=4)
    model = BLTBackbone(cfg).eval()
    with torch.no_grad():
        out = model(ModelInput(input_ids=_ids(), num_masked_tokens=masked))
    assert tuple(out.logits.shape) == (1, N, VOCAB)


def test_masked_tokens_without_masked_query_raises() -> None:
    cfg = _cfg(ced_masked_query=False)
    model = BLTBackbone(cfg).eval()
    with pytest.raises(ValueError, match="ced_masked_query"):
        model(ModelInput(input_ids=_ids(), num_masked_tokens=2))


@pytest.mark.parametrize("masked", [1, 2, 3, 5], ids=["one", "table", "above-table", "well-above-table"])
def test_masked_patch_count_is_free_per_forward(masked: int) -> None:
    cfg = _cfg(ced_masked_query=True, ced_masked_token_count=2, blt_max_patch_bytes=3)
    model = BLTBackbone(cfg).eval()
    out = model(ModelInput(input_ids=_ids(), num_masked_tokens=masked))
    assert tuple(out.logits.shape) == (1, masked * 3, VOCAB)


def test_negative_masked_tokens_raises() -> None:
    cfg = _cfg(ced_masked_query=True, ced_masked_token_count=4)
    model = BLTBackbone(cfg).eval()
    with pytest.raises(ValueError, match="non-negative"):
        model(ModelInput(input_ids=_ids(), num_masked_tokens=-1))


# --------------------------------------------------------------- gradients --
def test_gradients_flow_end_to_end() -> None:
    model = BLTBackbone(_cfg())
    out = model(ModelInput(input_ids=_ids(n=6)))
    out.logits.float().pow(2).sum().backward()

    named = {
        "byte_embeddings": model.vocab_encoder.byte_embeddings.weight,
        "vocab_output": model.vocab_decoder.output.weight,
        "up": model.up.proj.weight,
        "encoder_layer": next(iter(model.encoder_layers[0].parameters())),
    }
    for label, param in named.items():
        assert param.grad is not None, label
        assert torch.isfinite(param.grad).all(), label

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        assert not torch.isnan(param.grad).any(), name


# --------------------------------------------------------------- causality --
def test_causality_last_byte_only_changes_last_row() -> None:
    model = BLTBackbone(_cfg()).eval()
    ids = _ids()
    changed = ids.clone()
    changed[:, -1] = (changed[:, -1] + 1) % VOCAB
    with torch.no_grad():
        base = model(ids).logits
        new = model(changed).logits
    assert torch.equal(base[:, :-1], new[:, :-1])
    assert not torch.equal(base[:, -1], new[:, -1])


def test_causality_middle_byte_leaves_earlier_rows_unchanged() -> None:
    model = BLTBackbone(_cfg()).eval()
    ids = _ids()
    changed = ids.clone()
    changed[:, 2] = (changed[:, 2] + 1) % VOCAB
    with torch.no_grad():
        base = model(ids).logits
        new = model(changed).logits
    assert torch.equal(base[:, :2], new[:, :2])
    assert not torch.equal(base, new)


# -------------------------------------------------------- config validation --
@pytest.mark.parametrize(
    "kwarg,value",
    [
        ("blt_vocab_size", -1),
        ("blt_patch_size", 0),
        ("blt_local_hidden_size", -1),
        ("blt_local_hidden_size", 1),
        ("blt_local_layers", 0),
        ("blt_local_ffn", "nope"),
        ("blt_local_ffn_intermediate_size", -1),
        ("blt_local_norm", "nope"),
        ("blt_ngram_sizes", (1,)),
        ("blt_ngram_hash_size", 1),
        ("blt_cross_attn_heads", -1),
        ("blt_cross_attn_heads", 3),
        ("blt_max_patch_bytes", 0),
        ("blt_local_max_position_embeddings", -1),
    ],
    ids=[
        "vocab_size-negative",
        "patch_size-zero",
        "local_hidden-negative",
        "local_hidden-not-divisible",
        "local_layers-zero",
        "local_ffn-unknown",
        "local_ffn_intermediate-negative",
        "local_norm-unknown",
        "ngram_sizes-too-small",
        "ngram_hash_size-too-small",
        "cross_attn_heads-negative",
        "cross_attn_heads-not-divisor",
        "max_patch_bytes-zero",
        "local_max_position_embeddings-negative",
    ],
)
def test_config_rejects_invalid_blt_values(kwarg: str, value: object) -> None:
    with pytest.raises(ValueError):
        _cfg(**{kwarg: value})


@pytest.mark.parametrize(
    "kwarg,value",
    [
        ("blt_vocab_size", 0),
        ("blt_local_hidden_size", 0),
        ("blt_local_ffn_intermediate_size", 0),
        ("blt_local_max_position_embeddings", 0),
        ("blt_cross_attn_heads", 0),
    ],
    ids=[
        "vocab_size-zero",
        "local_hidden-zero",
        "local_ffn_intermediate-zero",
        "local_max_position_embeddings-zero",
        "cross_attn_heads-zero",
    ],
)
def test_config_accepts_boundary_blt_values(kwarg: str, value: int) -> None:
    cfg = _cfg(**{kwarg: value})
    assert getattr(cfg, kwarg) == value


# ------------------------------------------------------------ vocab sizing --
def test_vocab_size_falls_back_to_config() -> None:
    model = BLTBackbone(_cfg())
    assert model.vocab_size == VOCAB
    assert model.vocab_decoder.output.out_features == VOCAB


def test_blt_vocab_size_overrides_config() -> None:
    model = BLTBackbone(_cfg(blt_vocab_size=24))
    assert model.vocab_size == 24
    assert model.vocab_decoder.output.out_features == 24
    assert model.vocab_encoder.byte_embeddings.num_embeddings == 24


def test_explicit_vocab_size_overrides_both() -> None:
    model = BLTBackbone(_cfg(blt_vocab_size=24), vocab_size=20)
    assert model.vocab_size == 20
    assert model.vocab_decoder.output.out_features == 20


@pytest.mark.parametrize("vocab_size", [0, -1], ids=["zero", "negative"])
def test_explicit_non_positive_vocab_size_raises(vocab_size: int) -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        BLTBackbone(_cfg(), vocab_size=vocab_size)


# ----------------------------------------------------------- embeddings API --
def test_embedding_api_agrees_with_byte_embeddings() -> None:
    model = BLTBackbone(_cfg())
    byte_embeddings = model.vocab_encoder.byte_embeddings
    assert model.get_input_embeddings() is byte_embeddings
    assert model.embed is byte_embeddings

    replacement = torch.nn.Embedding(VOCAB, LOCAL)
    model.set_input_embeddings(replacement)
    assert model.get_input_embeddings() is replacement
    assert model.embed is replacement
    assert model.vocab_encoder.byte_embeddings is replacement


def test_word_embeddings_untied_by_default() -> None:
    model = BLTBackbone(_cfg())
    assert not model.config.blt_tie_word_embeddings
    assert model.vocab_decoder.output.weight is not model.vocab_encoder.byte_embeddings.weight
    model.reset_parameters()
    assert model.vocab_decoder.output.weight is not model.vocab_encoder.byte_embeddings.weight


def test_tied_word_embeddings_survive_reset_parameters() -> None:
    model = BLTBackbone(_cfg(blt_tie_word_embeddings=True))
    assert model.vocab_decoder.output.weight is model.vocab_encoder.byte_embeddings.weight
    model.reset_parameters()
    assert model.vocab_decoder.output.weight is model.vocab_encoder.byte_embeddings.weight


# ----------------------------------------------------------------- CheapLLM --
def test_cheapllm_ignores_patch_ids_and_keeps_shape() -> None:
    model = CheapLLM(_cfg()).eval()
    ids = _ids(n=6)
    patch_ids = torch.tensor([[0, 0, 1, 1, 2, 2]])
    with torch.no_grad():
        without = model(ModelInput(input_ids=ids)).logits
        with_patch_ids = model(ModelInput(input_ids=ids, patch_ids=patch_ids)).logits
    assert tuple(without.shape) == (1, 6, VOCAB)
    assert torch.equal(with_patch_ids, without)
    assert torch.isfinite(without).all()


# ----------------------------------------------------- training/export wiring --
def test_lightning_training_step_consumes_blt_logits() -> None:
    from memory_machines.modules.lightning_module import LitResearchSystem

    model = BLTBackbone(_cfg())
    system = LitResearchSystem(model)
    ids = _ids()
    loss = system.training_step({"input_ids": ids, "labels": _ids()}, 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    grad = model.vocab_decoder.output.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_public_exports_include_blt_symbols() -> None:
    from memory_machines import models
    from memory_machines.models import blocks

    assert models.BLTBackbone is BLTBackbone
    assert blocks.VocabularyEncoder is VocabularyEncoder
    assert blocks.VocabularyDecoder is VocabularyDecoder
    assert blocks.compute_patch_ids is compute_patch_ids
