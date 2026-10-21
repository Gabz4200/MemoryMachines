"""Perceiver-AR masked-token generation on CED.

Covers mask-embedding append, encoder causal context, decoder queries/KV
from encoder states, CVPE positions, looped decoders, and the legacy path.
"""

import pytest
import torch

from memory_machines.models.backbone import CheapLLM
from memory_machines.models.blocks.ced import CausalDecoder
from memory_machines.models.blocks.positioning import CVPEContextBias
from memory_machines.models.io import ModelInput
from memory_machines.utils.config_schema import MemoryMachineConfig


def _cfg(**kw):
    cfg = {
        "hidden_size": 16,
        "vocab_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "ced_num_encoder_layers": 1,
        "ced_num_decoder_layers": 1,
        "csa_top_k": 4,
        "csa_num_output_groups": 4,
        "ced_masked_token_count": 4,
        "ced_masked_query": False,
    }
    cfg.update(kw)
    return MemoryMachineConfig(**cfg)


def test_config_validation_masked_token_count():
    with pytest.raises(ValueError, match="ced_masked_token_count"):
        MemoryMachineConfig(hidden_size=8, vocab_size=8, num_attention_heads=4, num_key_value_heads=2, ced_masked_token_count=0, csa_num_output_groups=4)


def test_masked_token_embedding_exists():
    cfg = _cfg(ced_masked_token_count=3)
    model = CheapLLM(cfg)
    assert hasattr(model, "masked_token_embed")


def test_masked_generation_shape():
    """Masked generation returns one prediction per mask."""
    cfg = _cfg(ced_masked_token_count=4, ced_masked_query=True)
    torch.manual_seed(0)
    model = CheapLLM(cfg).eval()
    B, Le = 2, 5
    L = 3
    input_ids = torch.randint(0, cfg.vocab_size, (B, Le))
    inputs = ModelInput(input_ids=input_ids, num_masked_tokens=L)
    out = model(inputs)
    assert out.logits.shape == (B, L, cfg.vocab_size), f"got {out.logits.shape}"


def test_masked_encoder_causal():
    """Encoder outputs at a position do not depend on later inputs."""
    cfg = _cfg(ced_masked_token_count=4, ced_masked_query=True)
    torch.manual_seed(42)
    model = CheapLLM(cfg).eval()
    x = torch.randn(1, 6, 16)
    base, _ = model.encode(x)
    changed = x.clone()
    changed[:, -1] += 1.0
    new, _ = model.encode(changed)
    torch.testing.assert_close(base[:, :-1], new[:, :-1])
    assert not torch.allclose(base[:, -1], new[:, -1])


def test_masked_decoder_queries_and_kv_are_encoder_states():
    """Perceiver-AR reads the encoder's enriched representations, not raw inputs."""
    cfg = _cfg(ced_masked_token_count=4, ced_masked_query=True)
    torch.manual_seed(0)
    model = CheapLLM(cfg).eval()
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
            model(ModelInput(input_ids=torch.randint(0, cfg.vocab_size, (2, 5)), num_masked_tokens=L))
    finally:
        del model.decoder.decode_full_context
    assert torch.equal(captured["queries"], captured["encoder"][:, -L:, :])
    assert torch.equal(captured["kv"], captured["encoder"])


def test_masked_decoder_kv_attention_uses_positional_encoding():
    """Masked-query SWA receives CVPE positions for both Q and K streams."""
    torch.manual_seed(0)
    dec = CausalDecoder(_cfg(ced_masked_token_count=4, ced_masked_query=True)).eval()
    # CVPE zero-initializes to a no-op; enable it so positions must matter.
    with torch.no_grad():
        for module in dec.modules():
            if isinstance(module, CVPEContextBias):
                module.gamma_proj.bias.fill_(10.0)
                module.mlp_in.weight.fill_(1.0)
                module.offset_out.weight.fill_(1.0)
    enc = torch.randn(1, 6, 16)
    queries = torch.randn(1, 2, 16)
    kv = torch.randn(1, 6, 16)
    q_pos = torch.tensor([[4, 5]])
    kv_pos_a = torch.arange(6).unsqueeze(0)
    kv_pos_b = kv_pos_a.clone()
    kv_pos_b[0, 2] = 1  # same causal visibility/window, different distance only
    out_a = dec.decode_full_context(queries, enc, q_pos, start_pos=4, kv_states=kv, kv_position_ids=kv_pos_a)
    out_b = dec.decode_full_context(queries, enc, q_pos, start_pos=4, kv_states=kv, kv_position_ids=kv_pos_b)
    assert not torch.allclose(out_a, out_b)


def test_masked_generation_with_looped_decoder():
    """Perceiver-AR keeps working when the decoder is the shared looped layer."""
    cfg = _cfg(ced_masked_token_count=4, ced_masked_query=True, ced_decoder_looping=True, ced_decoder_loop_steps=6)
    torch.manual_seed(0)
    model = CheapLLM(cfg).eval()
    B, Le, L = 2, 5, 3
    input_ids = torch.randint(0, cfg.vocab_size, (B, Le))
    out = model(ModelInput(input_ids=input_ids, num_masked_tokens=L))
    assert out.logits.shape == (B, L, cfg.vocab_size)


def test_negative_masked_token_count_is_rejected():
    model = CheapLLM(_cfg(ced_masked_query=True)).eval()
    with pytest.raises(ValueError, match="must be non-negative"):
        model(ModelInput(input_ids=torch.randint(0, 32, (1, 5)), num_masked_tokens=-1))


def test_cheapllm_forwards_the_kv_stream_to_the_decoder():
    torch.manual_seed(0)
    model = CheapLLM(_cfg()).eval()
    h, enc, kv = torch.randn(1, 2, 16), torch.randn(1, 6, 16), torch.randn(1, 6, 16)
    kv_pos = torch.arange(6).unsqueeze(0)
    via_model = model.decode_full_context(h, enc, kv_states=kv, kv_position_ids=kv_pos)
    via_decoder = model.decoder.decode_full_context(h, enc, kv_states=kv, kv_position_ids=kv_pos)
    torch.testing.assert_close(via_model, via_decoder)


def test_legacy_path_unchanged():
    """When num_masked_tokens is None/0, behavior is unchanged."""
    cfg = _cfg(ced_masked_token_count=4, ced_masked_query=True)
    torch.manual_seed(7)
    model = CheapLLM(cfg).eval()
    B, Le = 2, 5
    input_ids = torch.randint(0, cfg.vocab_size, (B, Le))
    out_legacy = model(ModelInput(input_ids=input_ids))
    assert out_legacy.logits.shape == (B, Le, cfg.vocab_size)


def test_masked_self_compression_anchors_kv_at_zero():
    from memory_machines.models.blocks.shared import CSA2SharedState
    from memory_machines.models.blocks.sparse import CompressedSparseAttention2

    cfg = _cfg(ced_masked_token_count=4, ced_masked_query=True, csa_decoder_compression_ratio=1)
    layer = CompressedSparseAttention2(cfg, compression_ratio=1).eval()
    shared = CSA2SharedState(total_len=4)
    _, shared = layer.forward_shared(
        torch.randn(1, 2, 16),
        shared,
        seeded_store=False,
        kv_states=torch.randn(1, 6, 16),
        kv_position_ids=torch.arange(6).unsqueeze(0),
    )
    assert shared.comp_end_pos is not None
    assert int(shared.comp_end_pos[0]) == 0
