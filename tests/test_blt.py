"""Byte Latent Transformer local models and patch-level backbone.

Covers the paper's local encoder (byte transformer + cross-attention pooling
into patch queries) and local decoder (byte queries + cross-attention to patch
representations + byte transformer), plus the patch-level BLT backbone that
uses the CED stack as the latent transformer.
"""

from typing import Any

import pytest
import torch

from memory_machines.models.backbone import BLTBackbone
from memory_machines.models.blocks.blt import (
    VocabularyDecoder,
    VocabularyEncoder,
    compute_patch_ids,
)
from memory_machines.models.io import ModelInput
from memory_machines.utils.config_schema import MemoryMachineConfig

VOCAB = 32
LOCAL = 16


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


def _patch_ids(n: int, patch_size: int = 2) -> torch.Tensor:
    return compute_patch_ids(torch.zeros(1, n, dtype=torch.long), patch_size)


# ---------------------------------------------------------------- patching --
def test_compute_patch_ids_strided() -> None:
    ids = torch.zeros(1, 7, dtype=torch.long)
    assert compute_patch_ids(ids, 2).tolist() == [[0, 0, 1, 1, 2, 2, 3]]


def test_compute_patch_ids_rejects_bad_patch_size() -> None:
    with pytest.raises(ValueError, match="patch_size"):
        compute_patch_ids(torch.zeros(1, 4, dtype=torch.long), 0)


# --------------------------------------------------------- local encoder ----
def test_vocabulary_encoder_shapes() -> None:
    cfg = _cfg()
    enc = VocabularyEncoder(cfg).eval()
    byte_ids = torch.randint(0, VOCAB, (2, 8))
    patch_ids = compute_patch_ids(byte_ids, 2)
    patch_reps, byte_hidden = enc(byte_ids, patch_ids)
    assert tuple(patch_reps.shape) == (2, 4, cfg.hidden_size)
    assert tuple(byte_hidden.shape) == (2, 8, LOCAL)


def test_vocabulary_encoder_is_causal_across_patches() -> None:
    cfg = _cfg()
    enc = VocabularyEncoder(cfg).eval()
    byte_ids = torch.randint(0, VOCAB, (1, 8))
    patch_ids = compute_patch_ids(byte_ids, 2)
    # Perturb a byte in the last patch only.
    changed = byte_ids.clone()
    changed[:, -1] = (changed[:, -1] + 1) % VOCAB
    with torch.no_grad():
        base_reps, base_bytes = enc(byte_ids, patch_ids)
        new_reps, new_bytes = enc(changed, patch_ids)
    torch.testing.assert_close(base_reps[:, :-1], new_reps[:, :-1])
    torch.testing.assert_close(base_bytes[:, :-1], new_bytes[:, :-1])
    assert (base_reps[:, -1] - new_reps[:, -1]).abs().max().item() > 0


def test_vocabulary_encoder_rejects_non_contiguous_patch_ids() -> None:
    cfg = _cfg()
    enc = VocabularyEncoder(cfg).eval()
    byte_ids = torch.randint(0, VOCAB, (1, 6))
    bad = torch.tensor([[0, 0, 2, 2, 3, 3]])
    with pytest.raises(ValueError, match="patch_ids"):
        enc(byte_ids, bad)


def test_vocabulary_encoder_ngram_tables_are_per_size() -> None:
    cfg = _cfg(blt_ngram_sizes=(2, 3), blt_ngram_hash_size=32)
    enc = VocabularyEncoder(cfg)
    assert len(enc.ngram_embeddings) == 2
    assert enc.ngram_embeddings[0].num_embeddings == 32


# --------------------------------------------------------- local decoder ----
def test_vocabulary_decoder_shapes() -> None:
    cfg = _cfg()
    dec = VocabularyDecoder(cfg, vocab_size=VOCAB).eval()
    patch_ids = _patch_ids(6).expand(2, -1)
    logits = dec(torch.randn(2, 6, LOCAL), torch.randn(2, 3, cfg.hidden_size), patch_ids)
    assert tuple(logits.shape) == (2, 6, VOCAB)


def test_vocabulary_decoder_reads_only_its_own_patch() -> None:
    cfg = _cfg()
    dec = VocabularyDecoder(cfg, vocab_size=VOCAB).eval()
    patch_ids = torch.tensor([[0, 0, 1, 1, 2, 2]])
    byte_states = torch.randn(1, 6, LOCAL)
    patch_reps = torch.randn(1, 3, cfg.hidden_size)
    with torch.no_grad():
        base = dec(byte_states, patch_reps, patch_ids)
        perturbed = patch_reps.clone()
        perturbed[:, 1] += 1.0
        new = dec(byte_states, perturbed, patch_ids)
    # decoder patch id 2 selects KV entry 2 = latent patch 1 (begin-patch offset)
    own = patch_ids[0] == 2
    assert (base[0, own] - new[0, own]).abs().max().item() > 0
    assert torch.equal(base[0, ~own], new[0, ~own])


def test_vocabulary_decoder_masked_patch_mode_shapes() -> None:
    cfg = _cfg(blt_max_patch_bytes=3)
    dec = VocabularyDecoder(cfg, vocab_size=VOCAB).eval()
    masked = torch.randn(2, 2, cfg.hidden_size)
    patch_reps = torch.randn(2, 5, cfg.hidden_size)
    logits = dec.decode_masked_patches(masked, patch_reps, first_patch_index=3)
    assert tuple(logits.shape) == (2, 2 * 3, VOCAB)


# -------------------------------------------------------------- backbone ----
def test_blt_backbone_forward_shapes() -> None:
    cfg = _cfg()
    model = BLTBackbone(cfg).eval()
    ids = torch.randint(0, VOCAB, (2, 8))
    out = model(ModelInput(input_ids=ids))
    assert tuple(out.logits.shape) == (2, 8, VOCAB)


def test_blt_backbone_accepts_explicit_patch_ids() -> None:
    cfg = _cfg()
    model = BLTBackbone(cfg).eval()
    ids = torch.randint(0, VOCAB, (1, 6))
    patch_ids = torch.tensor([[0, 0, 0, 1, 1, 2]])
    out = model(ModelInput(input_ids=ids, patch_ids=patch_ids))
    assert tuple(out.logits.shape) == (1, 6, VOCAB)


def test_blt_backbone_masked_patch_mode_shapes() -> None:
    cfg = _cfg(ced_masked_query=True, ced_masked_token_count=4, blt_max_patch_bytes=3)
    model = BLTBackbone(cfg).eval()
    ids = torch.randint(0, VOCAB, (1, 8))
    out = model(ModelInput(input_ids=ids, num_masked_tokens=2))
    assert tuple(out.logits.shape) == (1, 2 * 3, VOCAB)


def test_blt_masked_decoder_queries_and_kv_are_encoder_states() -> None:
    """Masked patch queries/KV come from the latent encoder, not raw patch reps."""
    cfg = _cfg(ced_masked_query=True, ced_masked_token_count=4, blt_max_patch_bytes=3)
    torch.manual_seed(0)
    model = BLTBackbone(cfg).eval()
    L = 2
    captured: dict[str, torch.Tensor] = {}
    real = model.decoder.decode_full_context

    def spy(hidden_states, encoder_final, position_ids=None, **kw):
        captured["queries"] = hidden_states.detach().clone()
        captured["kv"] = kw["kv_states"].detach().clone()
        captured["encoder"] = encoder_final.detach().clone()
        return real(hidden_states, encoder_final, position_ids, **kw)

    model.decoder.decode_full_context = spy
    try:
        with torch.no_grad():
            model(ModelInput(input_ids=torch.randint(0, VOCAB, (1, 8)), num_masked_tokens=L))
    finally:
        del model.decoder.decode_full_context
    assert torch.equal(captured["queries"], captured["encoder"][:, -L:, :])
    assert torch.equal(captured["kv"], captured["encoder"])


def test_blt_forwards_the_kv_stream_to_the_decoder() -> None:
    model = BLTBackbone(_cfg()).eval()
    h, enc, kv = torch.randn(1, 2, 16), torch.randn(1, 6, 16), torch.randn(1, 6, 16)
    kv_pos = torch.arange(6).unsqueeze(0)
    via_model = model.decode_full_context(h, enc, kv_states=kv, kv_position_ids=kv_pos)
    via_decoder = model.decoder.decode_full_context(h, enc, kv_states=kv, kv_position_ids=kv_pos)
    torch.testing.assert_close(via_model, via_decoder)


def test_blt_backbone_is_causal() -> None:
    cfg = _cfg()
    model = BLTBackbone(cfg).eval()
    ids = torch.randint(0, VOCAB, (1, 8))
    changed = ids.clone()
    changed[:, -1] = (changed[:, -1] + 1) % VOCAB
    with torch.no_grad():
        base = model(ids).logits
        new = model(changed).logits
    torch.testing.assert_close(base[:, :-1], new[:, :-1])
    assert (base[:, -1] - new[:, -1]).abs().max().item() > 0


def test_blt_backbone_bare_tensor_matches_modelinput() -> None:
    cfg = _cfg()
    model = BLTBackbone(cfg).eval()
    ids = torch.randint(0, VOCAB, (1, 6))
    with torch.no_grad():
        a = model(ids).logits
        b = model(ModelInput(input_ids=ids)).logits
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_blt_backbone_reset_parameters_is_finite() -> None:
    cfg = _cfg()
    model = BLTBackbone(cfg)
    model.reset_parameters()
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), name


def test_blt_backbone_exposes_byte_embedding_as_input_embeddings() -> None:
    cfg = _cfg()
    model = BLTBackbone(cfg)
    assert model.get_input_embeddings() is model.vocab_encoder.byte_embeddings
    new_emb = torch.nn.Embedding(VOCAB, LOCAL)
    model.set_input_embeddings(new_emb)
    assert model.get_input_embeddings() is new_emb


def test_blt_backbone_invalid_vocab_size_raises() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        BLTBackbone(_cfg(vocab_size=0))


# ---------------------------------------------------------------- config ----
@pytest.mark.parametrize(
    "kwarg,value",
    [
        ("blt_patch_size", 0),
        ("blt_local_layers", 0),
        ("blt_max_patch_bytes", 0),
        ("blt_ngram_hash_size", 1),
        ("blt_local_ffn", "nope"),
        ("blt_local_norm", "nope"),
    ],
)
def test_blt_config_rejects_invalid_values(kwarg: str, value: object) -> None:
    with pytest.raises(ValueError):
        _cfg(**{kwarg: value})


def test_blt_backbone_inputs_embeds_matches_input_ids_without_ngrams() -> None:
    cfg = _cfg(blt_use_ngram_embeddings=False)
    model = BLTBackbone(cfg).eval()
    ids = torch.randint(0, VOCAB, (2, 6))
    patch_ids = compute_patch_ids(ids, cfg.blt_patch_size)
    with torch.no_grad():
        embeds = model.vocab_encoder.byte_embeddings(ids)
        a = model(ModelInput(input_ids=ids, patch_ids=patch_ids)).logits
        b = model(ModelInput(inputs_embeds=embeds, patch_ids=patch_ids)).logits
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_blt_backbone_ties_byte_embeddings_to_output_head() -> None:
    cfg = _cfg(blt_tie_word_embeddings=True)
    model = BLTBackbone(cfg)
    assert model.vocab_decoder.output.weight is model.vocab_encoder.byte_embeddings.weight
    model.reset_parameters()
    assert model.vocab_decoder.output.weight is model.vocab_encoder.byte_embeddings.weight


def test_blt_backbone_local_width_defaults_to_hidden_size() -> None:
    cfg = _cfg(blt_local_hidden_size=0)
    model = BLTBackbone(cfg)
    assert model.vocab_encoder.local_hidden_size == cfg.hidden_size
    assert model.vocab_decoder.local_hidden_size == cfg.hidden_size
