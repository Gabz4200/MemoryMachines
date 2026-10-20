"""BLT local byte models: encoder, decoder, strided patching, and validation.

Covers the ``[begin_patch] + patch_reps`` decoder key contract and the
masked-patch decode path through the public entry points only.
"""

from collections.abc import Callable
from typing import Any

import pytest
import torch

from memory_machines.models.blocks.blt import (
    VocabularyDecoder,
    VocabularyEncoder,
    compute_patch_ids,
)
from memory_machines.utils.config_schema import MemoryMachineConfig

VOCAB = 32
LOCAL = 16
HIDDEN = 16


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


def _patch_ids(num_bytes: int, batch: int = 1, patch_size: int = 2) -> torch.Tensor:
    return compute_patch_ids(torch.zeros(batch, num_bytes, dtype=torch.long), patch_size)


@pytest.mark.parametrize(
    "num_bytes,patch_size,expected",
    [
        (7, 2, [0, 0, 1, 1, 2, 2, 3]),
        (5, 2, [0, 0, 1, 1, 2]),
        (6, 3, [0, 0, 0, 1, 1, 1]),
        (4, 1, [0, 1, 2, 3]),
    ],
    ids=["non_divisible_7_by_2", "non_divisible_5_by_2", "divisible_6_by_3", "patch_size_1"],
)
def test_compute_patch_ids_strided(num_bytes: int, patch_size: int, expected: list[int]) -> None:
    ids = compute_patch_ids(torch.zeros(1, num_bytes, dtype=torch.long), patch_size)
    assert ids.tolist() == [expected]
    assert ids.dtype == torch.int64
    assert tuple(ids.shape) == (1, num_bytes)


def test_compute_patch_ids_broadcasts_over_batch() -> None:
    ids = compute_patch_ids(torch.zeros(3, 5, dtype=torch.long), 2)
    assert ids.tolist() == [[0, 0, 1, 1, 2]] * 3


@pytest.mark.parametrize("patch_size", [0, -1, -7], ids=["zero", "negative", "large_negative"])
def test_compute_patch_ids_rejects_non_positive_patch_size(patch_size: int) -> None:
    with pytest.raises(ValueError, match="patch_size"):
        compute_patch_ids(torch.zeros(1, 4, dtype=torch.long), patch_size)


@pytest.mark.parametrize(
    "cfg_kwargs,expected_local",
    [
        ({}, LOCAL),
        ({"blt_use_ngram_embeddings": False}, LOCAL),
        ({"blt_local_hidden_size": 0}, HIDDEN),
    ],
    ids=["default", "ngrams_disabled", "local_width_defaults_to_hidden_size"],
)
def test_encoder_shapes(cfg_kwargs: dict[str, object], expected_local: int) -> None:
    cfg = _cfg(**cfg_kwargs)
    enc = VocabularyEncoder(cfg).eval()
    byte_ids = torch.randint(0, VOCAB, (2, 8))
    patch_reps, byte_states = enc(byte_ids, compute_patch_ids(byte_ids, 2))
    assert tuple(patch_reps.shape) == (2, 4, cfg.hidden_size)
    assert tuple(byte_states.shape) == (2, 8, expected_local)


def test_encoder_accepts_precomputed_embeddings() -> None:
    # Without n-grams the pre-embedded path must match the byte-id path exactly.
    cfg = _cfg(blt_use_ngram_embeddings=False)
    enc = VocabularyEncoder(cfg).eval()
    byte_ids = torch.randint(0, VOCAB, (2, 8))
    patch_ids = compute_patch_ids(byte_ids, 2)
    with torch.no_grad():
        by_ids, _ = enc(byte_ids, patch_ids)
        by_embeds, _ = enc(patch_ids=patch_ids, byte_embeds=enc.byte_embeddings(byte_ids))
    assert tuple(by_embeds.shape) == (2, 4, cfg.hidden_size)
    torch.testing.assert_close(by_ids, by_embeds, atol=0, rtol=0)


def test_encoder_strided_fallback_when_patch_ids_missing() -> None:
    cfg = _cfg()
    enc = VocabularyEncoder(cfg).eval()
    with torch.no_grad():
        patch_reps, byte_states = enc(torch.randint(0, VOCAB, (1, 7)))
    assert tuple(patch_reps.shape) == (1, 4, cfg.hidden_size)
    assert tuple(byte_states.shape) == (1, 7, LOCAL)


@pytest.mark.parametrize(
    "cfg_kwargs,ctor_vocab,expected",
    [
        ({}, None, VOCAB),
        ({"blt_vocab_size": 48}, None, 48),
        ({"blt_vocab_size": 48}, 20, 20),
        ({}, 40, 40),
    ],
    ids=["config_vocab_size", "blt_vocab_size_override", "ctor_beats_blt_vocab_size", "ctor_vocab_size"],
)
def test_encoder_vocab_size_resolution(cfg_kwargs: dict[str, object], ctor_vocab: int | None, expected: int) -> None:
    enc = VocabularyEncoder(_cfg(**cfg_kwargs), vocab_size=ctor_vocab)
    assert enc.vocab_size == expected
    assert enc.byte_embeddings.num_embeddings == expected


@pytest.mark.parametrize(
    "cfg_kwargs,expected_tables",
    [
        ({}, 6),
        ({"blt_ngram_sizes": (2, 3)}, 2),
        ({"blt_ngram_sizes": (2, 3, 4, 5)}, 4),
        ({"blt_use_ngram_embeddings": False}, 0),
        ({"blt_use_ngram_embeddings": False, "blt_ngram_sizes": (2, 3, 4)}, 0),
    ],
    ids=["default_sizes_3_to_8", "two_sizes", "four_sizes", "disabled", "disabled_with_sizes"],
)
def test_encoder_ngram_tables(cfg_kwargs: dict[str, object], expected_tables: int) -> None:
    cfg = _cfg(**cfg_kwargs)
    enc = VocabularyEncoder(cfg)
    assert len(enc.ngram_embeddings) == expected_tables
    for table in enc.ngram_embeddings:
        assert table.num_embeddings == cfg.blt_ngram_hash_size
        assert table.embedding_dim == LOCAL


def test_encoder_ngram_warmup_positions_stay_finite() -> None:
    # A 3-gram has no full window at bytes 0-1; those positions must still encode.
    cfg = _cfg(blt_ngram_sizes=(3,), blt_ngram_hash_size=16)
    enc = VocabularyEncoder(cfg).eval()
    with torch.no_grad():
        patch_reps, byte_states = enc(torch.randint(0, VOCAB, (2, 6)))
    assert torch.isfinite(byte_states).all()
    assert torch.isfinite(patch_reps).all()


_ENCODER_BAD_CASES = [
    pytest.param(lambda enc: enc(), "byte", id="no_inputs"),
    pytest.param(lambda enc: enc(torch.zeros(6, dtype=torch.long)), "byte_ids", id="byte_ids_1d"),
    pytest.param(lambda enc: enc(patch_ids=_patch_ids(6), byte_embeds=torch.randn(1, 6, LOCAL // 2)), "byte_embeds", id="byte_embeds_wrong_width"),
    pytest.param(lambda enc: enc(byte_embeds=torch.randn(1, 6, LOCAL)), "patch_ids", id="byte_embeds_without_patch_ids"),
    pytest.param(lambda enc: enc(torch.zeros(1, 6, dtype=torch.long), torch.zeros(1, 8, dtype=torch.long)), "patch_ids", id="patch_ids_wrong_length"),
    pytest.param(lambda enc: enc(torch.zeros(1, 6, dtype=torch.long), torch.zeros(6, dtype=torch.long)), "patch_ids", id="patch_ids_1d"),
    pytest.param(lambda enc: enc(torch.zeros(1, 6, dtype=torch.long), torch.tensor([[-1, 0, 1, 1, 2, 2]])), "patch_ids", id="patch_ids_negative"),
    pytest.param(lambda enc: enc(torch.zeros(1, 6, dtype=torch.long), torch.ones(1, 6, dtype=torch.long)), "patch_ids", id="patch_ids_not_starting_at_zero"),
    pytest.param(lambda enc: enc(torch.zeros(1, 6, dtype=torch.long), torch.tensor([[0, 0, 2, 2, 3, 3]])), "patch_ids", id="patch_ids_gap"),
    pytest.param(lambda enc: enc(torch.zeros(1, 6, dtype=torch.long), torch.tensor([[0, 1, 1, 0, 1, 1]])), "patch_ids", id="patch_ids_backwards"),
    pytest.param(lambda enc: enc(torch.zeros(1, 6, dtype=torch.long), _patch_ids(6).float()), "patch_ids", id="patch_ids_float"),
    pytest.param(lambda enc: enc(torch.zeros(2, 4, dtype=torch.long), torch.tensor([[0, 0, 1, 1], [0, 0, 1, 2]])), "same number of patches", id="patch_ids_unequal_patch_counts"),
    pytest.param(lambda enc: enc(torch.zeros(2, 6, dtype=torch.long), _patch_ids(6)), "patch_ids", id="patch_ids_batch_mismatch"),
    pytest.param(lambda enc: enc(byte_ids=torch.zeros(1, 8, dtype=torch.long), byte_embeds=torch.randn(1, 6, LOCAL)), "byte_ids", id="byte_embeds_byte_ids_length_mismatch"),
    pytest.param(
        lambda enc: enc(
            byte_ids=torch.zeros(1, 8, dtype=torch.long),
            patch_ids=_patch_ids(6),
            byte_embeds=torch.randn(1, 6, LOCAL),
        ),
        "byte_ids",
        id="byte_embeds_byte_ids_length_mismatch_with_patch_ids",
    ),
    pytest.param(
        lambda enc: enc(
            byte_ids=torch.zeros(2, 6, dtype=torch.long),
            patch_ids=_patch_ids(6),
            byte_embeds=torch.randn(1, 6, LOCAL),
        ),
        "byte_ids",
        id="byte_embeds_byte_ids_batch_mismatch",
    ),
    pytest.param(lambda enc: enc(torch.zeros(1, 0, dtype=torch.long)), "byte", id="empty_sequence"),
    pytest.param(lambda enc: enc(torch.zeros(1, 0, dtype=torch.long), torch.zeros(1, 0, dtype=torch.long)), "byte", id="empty_sequence_explicit_patch_ids"),
]


@pytest.mark.parametrize("call,expected_message", _ENCODER_BAD_CASES)
def test_encoder_rejects_invalid_inputs(call: Callable[[VocabularyEncoder], object], expected_message: str) -> None:
    with pytest.raises(ValueError, match=expected_message):
        call(VocabularyEncoder(_cfg()).eval())


@pytest.mark.parametrize(
    "position,first_changed_patch",
    [(7, 3), (3, 1)],
    ids=["last_patch", "middle_patch"],
)
def test_encoder_is_causal_across_patches(position: int, first_changed_patch: int) -> None:
    cfg = _cfg()
    enc = VocabularyEncoder(cfg).eval()
    byte_ids = torch.randint(0, VOCAB, (1, 8))
    patch_ids = compute_patch_ids(byte_ids, 2)
    changed = byte_ids.clone()
    changed[0, position] = (changed[0, position] + 1) % VOCAB
    with torch.no_grad():
        base_reps, base_states = enc(byte_ids, patch_ids)
        new_reps, new_states = enc(changed, patch_ids)
    # Everything strictly before the perturbed byte is bit-identical.
    assert torch.equal(base_states[:, :position], new_states[:, :position])
    assert torch.equal(base_reps[:, :first_changed_patch], new_reps[:, :first_changed_patch])
    assert (base_reps[:, first_changed_patch:] - new_reps[:, first_changed_patch:]).abs().max().item() > 0


def test_decoder_shapes() -> None:
    cfg = _cfg()
    dec = VocabularyDecoder(cfg, vocab_size=VOCAB).eval()
    logits = dec(torch.randn(2, 6, LOCAL), torch.randn(2, 3, cfg.hidden_size), _patch_ids(6, batch=2))
    assert tuple(logits.shape) == (2, 6, VOCAB)


@pytest.mark.parametrize(
    "target,ids,num_patches,first_affected",
    [
        ("begin_patch", [1, 1, 0, 0], 1, 2),
        ("patch_0", [0, 0, 1, 1], 1, 2),
        ("patch_1", [0, 0, 1, 1, 2, 2], 2, 4),
    ],
    ids=["begin_patch_is_id_zero", "id_one_reads_patch_zero", "id_two_reads_patch_one"],
)
def test_decoder_reads_key_selected_by_patch_id(target: str, ids: list[int], num_patches: int, first_affected: int) -> None:
    cfg = _cfg()
    dec = VocabularyDecoder(cfg, vocab_size=VOCAB).eval()
    patch_ids = torch.tensor([ids])
    byte_states = torch.randn(1, len(ids), LOCAL)
    patch_reps = torch.randn(1, num_patches, cfg.hidden_size)
    with torch.no_grad():
        base = dec(byte_states, patch_reps, patch_ids)
        if target == "begin_patch":
            dec.begin_patch.add_(1.0)
            try:
                new = dec(byte_states, patch_reps, patch_ids)
            finally:
                dec.begin_patch.sub_(1.0)
        else:
            perturbed = patch_reps.clone()
            perturbed[:, 0 if target == "patch_0" else 1] += 1.0
            new = dec(byte_states, perturbed, patch_ids)
    # Only the bytes whose id selects the perturbed key may change, and they sit
    # after every unaffected byte, so causality keeps the earlier prefix exact.
    assert torch.equal(base[:, :first_affected], new[:, :first_affected])
    changed = base[:, first_affected:] - new[:, first_affected:]
    assert changed.abs().max().item() > 0


_DECODER_BAD_CASES = [
    pytest.param(lambda dec: dec(torch.randn(2, 6, LOCAL // 2), torch.randn(2, 3, HIDDEN), _patch_ids(6, batch=2)), "byte_states", id="byte_states_wrong_width"),
    pytest.param(lambda dec: dec(torch.randn(6, LOCAL), torch.randn(2, 3, HIDDEN), _patch_ids(6, batch=2)), "byte_states", id="byte_states_wrong_dim"),
    pytest.param(lambda dec: dec(torch.randn(2, 6, LOCAL), torch.randn(2, 3, HIDDEN // 2), _patch_ids(6, batch=2)), "patch_reps", id="patch_reps_wrong_width"),
    pytest.param(lambda dec: dec(torch.randn(2, 6, LOCAL), torch.randn(3, HIDDEN), _patch_ids(6, batch=2)), "patch_reps", id="patch_reps_wrong_dim"),
    pytest.param(lambda dec: dec(torch.randn(2, 6, LOCAL), torch.randn(2, 3, HIDDEN), _patch_ids(5, batch=2)), "patch_ids", id="patch_ids_wrong_shape"),
    pytest.param(lambda dec: dec(torch.randn(2, 6, LOCAL), torch.randn(2, 3, HIDDEN), torch.full((2, 6), -1)), "patch_ids", id="patch_ids_negative"),
    pytest.param(lambda dec: dec(torch.randn(2, 6, LOCAL), torch.randn(2, 3, HIDDEN), _patch_ids(6, batch=2) + 2), "patch_ids", id="patch_ids_above_range"),
    pytest.param(lambda dec: dec(torch.randn(2, 6, LOCAL), torch.randn(2, 3, HIDDEN), _patch_ids(6, batch=2).float()), "patch_ids", id="patch_ids_float"),
    pytest.param(lambda dec: dec(torch.randn(2, 6, LOCAL), torch.randn(3, 3, HIDDEN), _patch_ids(6, batch=2)), "batch", id="patch_reps_batch_mismatch"),
]


@pytest.mark.parametrize("call,expected_message", _DECODER_BAD_CASES)
def test_decoder_rejects_invalid_inputs(call: Callable[[VocabularyDecoder], object], expected_message: str) -> None:
    with pytest.raises(ValueError, match=expected_message):
        call(VocabularyDecoder(_cfg(), vocab_size=VOCAB).eval())


@pytest.mark.parametrize(
    "num_masked,max_patch_bytes,first_patch_index",
    [(2, 3, 3), (1, 4, 0), (3, 2, 2)],
    ids=["two_masked_slots_3", "one_masked_slots_4", "three_masked_slots_2"],
)
def test_decode_masked_patches_shapes(num_masked: int, max_patch_bytes: int, first_patch_index: int) -> None:
    cfg = _cfg(blt_max_patch_bytes=max_patch_bytes)
    dec = VocabularyDecoder(cfg, vocab_size=VOCAB).eval()
    masked = torch.randn(2, num_masked, cfg.hidden_size)
    total_patches = first_patch_index + num_masked
    patch_reps = torch.randn(2, total_patches, cfg.hidden_size)
    logits = dec.decode_masked_patches(masked, patch_reps, first_patch_index)
    assert tuple(logits.shape) == (2, num_masked * max_patch_bytes, VOCAB)


def test_decode_masked_patches_isolates_each_masked_latent() -> None:
    cfg = _cfg(blt_max_patch_bytes=3)
    dec = VocabularyDecoder(cfg, vocab_size=VOCAB).eval()
    masked = torch.randn(2, 2, cfg.hidden_size)
    patch_reps = torch.randn(2, 5, cfg.hidden_size)
    slots = cfg.blt_max_patch_bytes
    with torch.no_grad():
        base = dec.decode_masked_patches(masked, patch_reps, first_patch_index=3)
        perturbed = masked.clone()
        perturbed[:, -1] += 1.0
        new = dec.decode_masked_patches(perturbed, patch_reps, first_patch_index=3)
    # The last masked patch owns exactly the trailing `slots` byte slots.
    assert torch.equal(base[:, :-slots], new[:, :-slots])
    assert (base[:, -slots:] - new[:, -slots:]).abs().max().item() > 0


_MASKED_BAD_CASES = [
    pytest.param(lambda dec: dec.decode_masked_patches(torch.randn(2, 0, HIDDEN), torch.randn(2, 5, HIDDEN), 3), "at least one patch", id="no_masked_patches"),
    pytest.param(lambda dec: dec.decode_masked_patches(torch.randn(2, 2, HIDDEN), torch.randn(2, 5, HIDDEN), -1), "first_patch_index", id="negative_first_patch_index"),
    pytest.param(lambda dec: dec.decode_masked_patches(torch.randn(2, 2, HIDDEN), torch.randn(2, 5, HIDDEN), 4), "first_patch_index", id="first_patch_index_overflow"),
    pytest.param(lambda dec: dec.decode_masked_patches(torch.randn(2, 2, HIDDEN // 2), torch.randn(2, 5, HIDDEN), 3), "masked_states", id="masked_states_wrong_width"),
    pytest.param(lambda dec: dec.decode_masked_patches(torch.randn(2, HIDDEN), torch.randn(2, 5, HIDDEN), 3), "masked_states", id="masked_states_wrong_dim"),
    pytest.param(lambda dec: dec.decode_masked_patches(torch.randn(2, 2, HIDDEN), torch.randn(2, 5, HIDDEN // 2), 3), "patch_reps", id="patch_reps_wrong_width"),
    pytest.param(lambda dec: dec.decode_masked_patches(torch.randn(2, 2, HIDDEN), torch.randn(5, HIDDEN), 3), "patch_reps", id="patch_reps_wrong_dim"),
    pytest.param(lambda dec: dec.decode_masked_patches(torch.randn(3, 2, HIDDEN), torch.randn(2, 5, HIDDEN), 3), "batch", id="masked_batch_mismatch"),
]


@pytest.mark.parametrize("call,expected_message", _MASKED_BAD_CASES)
def test_decode_masked_patches_rejects_invalid_inputs(call: Callable[[VocabularyDecoder], object], expected_message: str) -> None:
    with pytest.raises(ValueError, match=expected_message):
        call(VocabularyDecoder(_cfg(), vocab_size=VOCAB).eval())


def test_encoder_gradients_are_finite() -> None:
    enc = VocabularyEncoder(_cfg())
    patch_reps, _ = enc(torch.randint(0, VOCAB, (2, 8)))
    patch_reps.sum().backward()
    params = dict(enc.named_parameters())
    for name in ("byte_embeddings.weight", "patch_out_proj.weight"):
        grad = params[name].grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name


def test_decoder_gradients_are_finite() -> None:
    cfg = _cfg()
    dec = VocabularyDecoder(cfg, vocab_size=VOCAB)
    logits = dec(torch.randn(2, 6, LOCAL), torch.randn(2, 3, cfg.hidden_size), _patch_ids(6, batch=2))
    logits.float().pow(2).sum().backward()
    params = dict(dec.named_parameters())
    for name in ("output.weight", "begin_patch", "patch_proj.weight"):
        grad = params[name].grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name


def test_masked_decode_gradients_reach_slot_embeddings() -> None:
    # `byte_slot_embed` is only used by the masked-patch path, never by forward.
    cfg = _cfg(blt_max_patch_bytes=3)
    dec = VocabularyDecoder(cfg, vocab_size=VOCAB)
    logits = dec.decode_masked_patches(torch.randn(2, 2, cfg.hidden_size), torch.randn(2, 5, cfg.hidden_size), 3)
    logits.float().pow(2).sum().backward()
    grad = dict(dec.named_parameters())["byte_slot_embed.weight"].grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().max().item() > 0


@pytest.mark.parametrize("kind", ["encoder", "decoder"])
def test_reset_parameters_keeps_all_parameters_finite(kind: str) -> None:
    model = VocabularyEncoder(_cfg()) if kind == "encoder" else VocabularyDecoder(_cfg(), vocab_size=VOCAB)
    model.reset_parameters()
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), name


def test_encoder_decoder_roundtrip_shapes() -> None:
    cfg = _cfg()
    enc = VocabularyEncoder(cfg).eval()
    dec = VocabularyDecoder(cfg, vocab_size=VOCAB).eval()
    byte_ids = torch.randint(0, VOCAB, (2, 8))
    patch_ids = compute_patch_ids(byte_ids, cfg.blt_patch_size)
    with torch.no_grad():
        patch_reps, byte_states = enc(byte_ids, patch_ids)
        logits = dec(byte_states, patch_reps, patch_ids)
    assert tuple(logits.shape) == (2, 8, VOCAB)


@pytest.mark.parametrize("kind", ["encoder", "decoder"])
def test_constructor_local_width_override(kind: str) -> None:
    config = _cfg()
    if kind == "encoder":
        encoder = VocabularyEncoder(config, local_hidden_size=8)
        assert encoder.local_hidden_size == 8
        assert encoder.byte_embeddings.embedding_dim == 8
    else:
        decoder = VocabularyDecoder(config, vocab_size=VOCAB, local_hidden_size=8)
        assert decoder.local_hidden_size == 8
        assert decoder.patch_proj.out_features == 8


@pytest.mark.parametrize("kind", ["encoder", "decoder"])
def test_constructor_rejects_indivisible_local_width(kind: str) -> None:
    model_cls = VocabularyEncoder if kind == "encoder" else VocabularyDecoder
    with pytest.raises(ValueError, match="divisible"):
        model_cls(_cfg(), local_hidden_size=5)


@pytest.mark.parametrize("kind", ["encoder", "decoder"])
@pytest.mark.parametrize("vocab_size", [0, -1], ids=["zero", "negative"])
def test_constructor_rejects_non_positive_vocab_size(kind: str, vocab_size: int) -> None:
    model_cls = VocabularyEncoder if kind == "encoder" else VocabularyDecoder
    with pytest.raises(ValueError, match="vocab_size"):
        model_cls(_cfg(), vocab_size=vocab_size)
