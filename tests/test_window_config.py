"""SWA-by-design window config coverage: config defaults, zero/negative full attention, CED bounded replay, LIV perceiver window override."""

from dataclasses import replace

import torch

from memory_machines.models.blocks import CausalEncoderDecoderStack, GroupedQueryAttention
from memory_machines.models.blocks.liv import LivPerceiver
from memory_machines.utils.config_schema import MemoryMachineConfig
from tests.test_blocks_shapes import tiny_cfg


def test_window_config_defaults():
    cfg = MemoryMachineConfig()
    assert cfg.window_size == 256
    assert cfg.window_size_compressed == 1024  # 256 * 4
    assert cfg.window_size_sparse == 2048  # 256 * 8
    assert cfg.window_size_shared == 4096  # 256 * 16


def test_window_size_zero_runs_full_attention():
    torch.manual_seed(0)
    cfg = MemoryMachineConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        window_size=0,
        csa_num_output_groups=4,
    )
    model = GroupedQueryAttention(cfg).eval()
    x = torch.randn(1, 6, 32)
    out_zero, _, _ = model(x, is_causal=True, output_attentions=True)

    cfg_full = replace(cfg, window_size=6)
    model_full = GroupedQueryAttention(cfg_full).eval()
    model_full.load_state_dict(model.state_dict())
    out_full, _, _ = model_full(x, is_causal=True, output_attentions=True)
    torch.testing.assert_close(out_zero, out_full, atol=1e-6, rtol=1e-6)


def test_window_size_negative_runs_full_attention():
    torch.manual_seed(1)
    base_cfg = MemoryMachineConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        csa_num_output_groups=4,
    )
    x = torch.randn(1, 6, 32)

    model_neg = GroupedQueryAttention(replace(base_cfg, window_size=-1)).eval()
    model_zero = GroupedQueryAttention(replace(base_cfg, window_size=0)).eval()
    model_zero.load_state_dict(model_neg.state_dict())
    out_neg, _, past_neg = model_neg(x, is_causal=True, use_cache=True)
    out_zero, _, past_zero = model_zero(x, is_causal=True, use_cache=True)
    torch.testing.assert_close(out_neg, out_zero, atol=1e-6, rtol=1e-6)
    assert past_neg is not None and past_zero is not None
    assert tuple(past_neg[0].shape) == (1, 2, 6, 8)
    assert tuple(past_zero[0].shape) == (1, 2, 6, 8)


def test_window_size_zero_untruncated_cache():
    torch.manual_seed(2)
    base_cfg = MemoryMachineConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        csa_num_output_groups=4,
    )
    x = torch.randn(1, 6, 32)

    model_zero = GroupedQueryAttention(replace(base_cfg, window_size=0)).eval()
    _, _, past_zero = model_zero(x, is_causal=True, use_cache=True)
    assert past_zero is not None
    assert tuple(past_zero[0].shape) == (1, 2, 6, 8)

    model_win = GroupedQueryAttention(replace(base_cfg, window_size=4)).eval()
    _, _, past_win = model_win(x, is_causal=True, use_cache=True)
    assert past_win is not None
    assert tuple(past_win[0].shape) == (1, 2, 4, 8)


def _ced_cfg(**overrides):
    base = {
        "ced_num_encoder_layers": 1,
        "ced_num_decoder_layers": 1,
        "csa_top_k": 2,
        "csa_num_output_groups": 2,
    }
    base.update(overrides)
    return replace(tiny_cfg(), **base)


def test_ced_window_size_zero_prefill_returns_full_sequence():
    torch.manual_seed(0)
    model = CausalEncoderDecoderStack(_ced_cfg(window_size=0, swa_bounded_replay=True)).eval()
    H = torch.randn(1, 10, 32)
    E = torch.randn(1, 8, 32)
    out = model.prefill(H, E)
    assert tuple(out.shape) == (1, 10, 32)
    full = model.decode_full_context(H, E)
    torch.testing.assert_close(out, full, atol=1e-5, rtol=1e-5)


def test_ced_bounded_replay_disabled_returns_full_context():
    torch.manual_seed(1)
    model = CausalEncoderDecoderStack(_ced_cfg(window_size=4, swa_bounded_replay=False)).eval()
    H = torch.randn(1, 10, 32)
    E = torch.randn(1, 8, 32)
    pos2d = torch.arange(10).unsqueeze(0)
    out = model.prefill(H, E, pos2d)
    assert tuple(out.shape) == (1, 10, 32)
    direct = model.decode_full_context(H, E, pos2d)
    torch.testing.assert_close(out, direct, atol=1e-6, rtol=1e-6)


def test_ced_prefill_window_larger_than_sequence():
    torch.manual_seed(2)
    model = CausalEncoderDecoderStack(_ced_cfg(window_size=16, swa_bounded_replay=True)).eval()
    H = torch.randn(1, 10, 32)
    E = torch.randn(1, 8, 32)
    out = model.prefill(H, E)
    assert tuple(out.shape) == (1, 10, 32)
    full = model.decode_full_context(H, E)
    torch.testing.assert_close(out, full, atol=1e-5, rtol=1e-5)


def test_ced_prefill_matches_decode_of_tail():
    torch.manual_seed(3)
    model = CausalEncoderDecoderStack(_ced_cfg(window_size=4, swa_bounded_replay=True)).eval()
    H = torch.randn(1, 10, 32)
    E = torch.randn(1, 8, 32)

    pos2d = torch.arange(10).unsqueeze(0)
    torch.testing.assert_close(
        model.prefill(H, E, pos2d),
        model.decode_full_context(H[:, -4:, :], E, pos2d[:, -4:]),
        atol=1e-5,
        rtol=1e-5,
    )

    pos1d = torch.arange(10)
    torch.testing.assert_close(
        model.prefill(H, E, pos1d),
        model.decode_full_context(H[:, -4:, :], E, pos1d[-4:]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_livperceiver_window_override_propagates():
    cfg = tiny_cfg()
    assert LivPerceiver(cfg, window_size=5).swa.window_size == 5
    assert LivPerceiver(cfg).swa.window_size == cfg.window_size
    assert LivPerceiver(cfg, window_size=0).swa.window_size == 0


def test_livperceiver_window_disabled_matches_full():
    torch.manual_seed(4)
    cfg = tiny_cfg()
    x = torch.randn(1, 6, 32)

    perceiver_zero = LivPerceiver(cfg, window_size=0).eval()
    perceiver_full = LivPerceiver(cfg, window_size=6).eval()
    perceiver_full.load_state_dict(perceiver_zero.state_dict())
    out_zero = perceiver_zero(x)
    out_full = perceiver_full(x)
    assert tuple(out_zero.shape) == (1, 6, 32)
    assert tuple(out_full.shape) == (1, 6, 32)
    torch.testing.assert_close(out_zero, out_full, atol=1e-6, rtol=1e-6)
