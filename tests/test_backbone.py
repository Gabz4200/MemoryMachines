"""Tests for src/memory_machines/models/backbone.py.

Covers CheapLLM (encoder-decoder LM) and ExampleBackbone (flat GQA/SwiGLU stack):
construction validation, forward shapes, encoding, reset_parameters, tie_word_embeddings,
and invalid-encoder-layer rejection.
"""

from typing import cast

import pytest
import torch

from memory_machines.models.backbone import CheapLLM, ExampleBackbone
from memory_machines.models.blocks import (
    CausalEncoderDecoderStack,
    GroupedQueryAttention,
    ParallelAttentionGDN2,
)
from memory_machines.models.io import ModelInput
from memory_machines.utils.config_schema import MemoryMachineConfig


def _ced_cfg(**kw):
    cfg = {
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "ced_num_encoder_layers": 1,
        "ced_num_decoder_layers": 1,
        "csa_top_k": 4,
        "csa_num_output_groups": 2,
        "vocab_size": 32,
    }
    cfg.update(kw)
    return MemoryMachineConfig(**cfg)


def test_cheapllm_builds_from_config():
    cfg = _ced_cfg()
    model = CheapLLM(cfg)
    assert isinstance(model.vocab_size, int) and model.vocab_size == 32
    assert len(model.encoder_layers) == 1
    assert len(model.decoder.layers) == 1


def test_cheapllm_forward_shapes():
    cfg = _ced_cfg()
    model = CheapLLM(cfg).eval()
    x = torch.randint(0, 32, (1, 6))
    out = model(ModelInput(input_ids=x))
    assert tuple(out.logits.shape) == (1, 6, 32)


def test_cheapllm_inputs_embeds_matches_input_ids():
    cfg = _ced_cfg()
    model = CheapLLM(cfg).eval()
    ids = torch.randint(0, 32, (2, 5))
    with torch.no_grad():
        a = model(ModelInput(input_ids=ids)).logits
        b = model(ModelInput(inputs_embeds=model.embed(ids))).logits
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_cheapllm_encode_decodes():
    cfg = _ced_cfg()
    model = CheapLLM(cfg).eval()
    x = torch.randn(2, 4, 16)
    enc, shared = model.encode(x)
    assert enc.shape == (2, 4, 16)
    assert shared is not None


def test_cheapllm_reset_parameters_does_not_error():
    cfg = _ced_cfg()
    model = CheapLLM(cfg)
    model.reset_parameters()


def test_cheapllm_tie_word_embeddings_shares_weight():
    cfg = _ced_cfg(tie_word_embeddings=True)
    model = CheapLLM(cfg)
    assert model.head.weight is model.embed.weight


def test_cheapllm_accessors():
    cfg = _ced_cfg()
    model = CheapLLM(cfg)
    assert model.get_input_embeddings() is model.embed
    new_emb = torch.nn.Embedding(32, 16)
    model.set_input_embeddings(new_emb)
    assert model.get_input_embeddings() is new_emb
    assert model.decoder_layers is model.decoder.layers
    assert model.global_projs is model.decoder.global_projs
    assert model.decoder_ffns is model.decoder.ffns


def test_cheapllm_invalid_vocab_size_raises():
    # Config does not validate vocab_size; the model constructor must fail fast.
    with pytest.raises(ValueError, match="vocab_size"):
        CheapLLM(_ced_cfg(vocab_size=0))


def test_example_backbone_builds_and_forward():
    model = ExampleBackbone(embed_dim=32, num_layers=2, num_heads=4, vocab_size=100)
    out = model(ModelInput(input_ids=torch.randint(0, 100, (1, 6))))
    assert tuple(out.logits.shape) == (1, 6, 100)


def test_example_backbone_inputs_embeds_matches_input_ids():
    model = ExampleBackbone(embed_dim=16, num_layers=2, num_heads=4, vocab_size=32).eval()
    ids = torch.randint(0, 32, (2, 5))
    with torch.no_grad():
        a = model(ModelInput(input_ids=ids)).logits
        b = model(ModelInput(inputs_embeds=model.embed(ids))).logits
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("kwarg,value", [("embed_dim", 0), ("num_layers", 0), ("num_heads", 0), ("vocab_size", 0)])
def test_example_backbone_invalid_dims_raise(kwarg, value):
    cfg = {"embed_dim": 16, "num_layers": 2, "num_heads": 4, "vocab_size": 32}
    cfg[kwarg] = value
    with pytest.raises(ValueError):
        ExampleBackbone(**cfg)


def test_cheapllm_invalid_encoder_layers_raise_at_construction():
    cfg = _ced_cfg(ced_num_encoder_layers=1)
    with pytest.raises(TypeError):
        CheapLLM(cfg, encoder_layers=[torch.nn.Linear(16, 16)])


def test_cheapllm_accepts_bare_tensor_input():
    # D4 regression: CheapLLM.forward now accepts a bare torch.Tensor as a
    # shorthand for ModelInput(input_ids=tensor) — the HF adapter and ad-hoc
    # scripts must not crash with AttributeError on .inputs_embeds.
    cfg = _ced_cfg()
    model = CheapLLM(cfg)
    out = model(torch.randint(0, cfg.vocab_size, (1, 6)))
    assert tuple(out.logits.shape) == (1, 6, cfg.vocab_size)


def test_cheapllm_bare_tensor_matches_modelinput():
    # The coercion must be lossless: a bare tensor and the equivalent
    # ModelInput(input_ids=tensor) produce identical logits.
    cfg = _ced_cfg()
    model = CheapLLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    a = model(ids).logits
    b = model(ModelInput(input_ids=ids)).logits
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_cheapllm_bare_tensor_requires_input_ids_when_embeds_missing():
    # A bare ModelInput with neither input_ids nor inputs_embeds must still
    # fail — the coercion only fills in the input_ids shorthand.
    cfg = _ced_cfg()
    model = CheapLLM(cfg)
    with pytest.raises(ValueError, match="input_ids or inputs_embeds"):
        model(ModelInput())


def test_cheapllm_masked_token_positions_anchor_at_end():
    # D3 regression: Perceiver-AR masked queries must anchor at [N-L, N) by
    # contract (explicit position_ids), not by accident of the fallback
    # arithmetic inside CausalDecoder.decode_full_context. A decoder prompt
    # shorter than the encoder length must NOT shift the masked queries to the
    # encoder tail — the query slice is the last L tokens of the full sequence.
    cfg = _ced_cfg(
        ced_num_encoder_layers=1,
        ced_num_decoder_layers=1,
        ced_masked_query=True,
        ced_masked_token_count=3,
    )
    model = CheapLLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    out = model(ModelInput(input_ids=ids, num_masked_tokens=3))
    # 6 input tokens + 3 masked = 9 total; only the 3 masked positions decode.
    assert tuple(out.logits.shape) == (1, 3, cfg.vocab_size)


def test_cheapllm_masked_token_positions_independent_of_prompt_length():
    # The masked query slice is always the last L tokens of the full
    # sequence, so a 4-token prompt with 2 masked tokens produces 2 logits
    # anchored at positions [4-2, 4) = [2, 4), regardless of the encoder length.
    for prompt_len in (4, 6, 8):
        cfg = _ced_cfg(
            ced_num_encoder_layers=1,
            ced_num_decoder_layers=1,
            ced_masked_query=True,
            ced_masked_token_count=2,
        )
        model = CheapLLM(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, prompt_len))
        out = model(ModelInput(input_ids=ids, num_masked_tokens=2))
        assert tuple(out.logits.shape) == (1, 2, cfg.vocab_size)


@pytest.mark.parametrize("explicit_positions", [False, True], ids=["implicit", "explicit"])
def test_cheapllm_encode_matches_the_stack_encoder_with_shared_layers(explicit_positions: bool) -> None:
    """``CheapLLM.encode`` and ``CausalEncoderDecoderStack.encode`` share one loop."""
    cfg = _ced_cfg(ced_num_encoder_layers=2, ced_encoder_alternating=False)
    torch.manual_seed(0)
    model = CheapLLM(cfg).eval()
    stack = CausalEncoderDecoderStack(cfg, encoder_layers=list(model.encoder_layers)).eval()
    stack.encoder_ffns.load_state_dict(model.encoder_ffns.state_dict())
    x = torch.randn(1, 7, cfg.hidden_size)
    pos = torch.arange(7).unsqueeze(0) if explicit_positions else None
    with torch.no_grad():
        a, shared_a = model.encode(x, pos)
        b, shared_b = stack.encode(x, pos)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert shared_a.total_len == shared_b.total_len
    assert shared_a.compressed_kv is not None and shared_b.compressed_kv is not None
    torch.testing.assert_close(shared_a.compressed_kv, shared_b.compressed_kv, atol=0, rtol=0)


def test_cheapllm_hybrid_encoder_starts_with_plain_swa_then_seeds_the_store() -> None:
    cfg = _ced_cfg(ced_num_encoder_layers=3, ced_encoder_alternating=False)
    model = CheapLLM(cfg)
    assert [layer.mode for layer in model.encoder_layers] == [None, "full", "reuse"]
    first = cast("ParallelAttentionGDN2", model.encoder_layers[0])
    assert type(getattr(first.attention, "inner", None)) is GroupedQueryAttention
    assert len(model.encoder_ffns) == 3


def test_cheapllm_rejects_a_masked_count_above_the_embedding_table() -> None:
    cfg = _ced_cfg(ced_masked_query=True, ced_masked_token_count=2)
    model = CheapLLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    with pytest.raises(ValueError, match="exceeds the table size"):
        model(ModelInput(input_ids=ids, num_masked_tokens=3))


def test_num_masked_tokens_is_ignored_when_masked_query_is_disabled() -> None:
    cfg = _ced_cfg(ced_masked_query=False, ced_masked_token_count=3)
    model = CheapLLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        a = model(ModelInput(input_ids=ids, num_masked_tokens=3)).logits
        b = model(ModelInput(input_ids=ids)).logits
        c = model(ModelInput(input_ids=ids, num_masked_tokens=0)).logits
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(b, c, atol=0, rtol=0)
    assert tuple(a.shape) == (1, 6, cfg.vocab_size)
