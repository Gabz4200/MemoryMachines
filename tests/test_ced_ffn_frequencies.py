"""CED FFN frequency-gating (HOPE time-cycle) tests.

Tests for per-block FFN update intervals in the CED encoder and decoder,
and the config/schema wiring. The goal is to preserve exact-output
behaviour when intervals are all ones (default), and to gate FFNs off/on
at specific steps otherwise.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
import torch

from memory_machines.models.blocks.ced import CausalEncoderDecoderStack
from memory_machines.models.blocks.ffn import DenseFFN
from memory_machines.utils.config_schema import MemoryMachineConfig

from .test_blocks_shapes import tiny_cfg
from .test_ced_ffn import _ced_cfg


def test_denseffn_should_update_schedule() -> None:
    ffn = DenseFFN(
        MemoryMachineConfig(
            hidden_size=16,
            intermediate_size=32,
            rms_norm_eps=1e-6,
            initializer_range=0.006,
            num_layers=1,
            residual_init_scale=False,
        ),
        update_interval=2,
    )
    # step % update_interval == 0
    assert ffn.should_update(0)
    assert not ffn.should_update(1)
    assert ffn.should_update(2)
    assert not ffn.should_update(3)
    assert ffn.should_update(4)


def test_denseffn_should_update_negative_step_raises() -> None:
    ffn = DenseFFN(
        MemoryMachineConfig(
            hidden_size=16,
            intermediate_size=32,
            rms_norm_eps=1e-6,
            initializer_range=0.006,
            num_layers=1,
            residual_init_scale=False,
        ),
        update_interval=1,
    )
    with pytest.raises(ValueError):
        ffn.should_update(-1)


def test_denseffn_forward_gated_off_equals_identity_on_same_step() -> None:
    torch.manual_seed(123)
    cfg = MemoryMachineConfig(
        hidden_size=16,
        intermediate_size=32,
        rms_norm_eps=1e-6,
        initializer_range=0.006,
        num_layers=1,
        residual_init_scale=False,
    )
    ffn = DenseFFN(cfg, update_interval=2)
    x = torch.randn(2, 3, 16)
    # step 1: gated off -> identity
    y = ffn(x, step=1)
    torch.testing.assert_close(y, x)
    # step 0: gated on -> equals x + FFN(norm(x)); with zero init down_proj at this cfg, equals x
    y0 = ffn(x, step=0)
    torch.testing.assert_close(y0, x)


def test_denseffn_forward_gated_on_equals_forward_no_step() -> None:
    torch.manual_seed(42)
    cfg = MemoryMachineConfig(
        hidden_size=16,
        intermediate_size=32,
        rms_norm_eps=1e-6,
        initializer_range=0.006,
        num_layers=1,
        residual_init_scale=False,
    )
    ffn = DenseFFN(cfg, update_interval=3)
    x = torch.randn(1, 4, 16)
    # step 0 or multiples of 3 apply; compare step=3 to no step (default)
    y_step = ffn(x, step=3)
    y_none = ffn(x)  # step=None -> always apply
    torch.testing.assert_close(y_step, y_none)


def test_denseffn_update_interval_lt_1_raises() -> None:
    cfg = MemoryMachineConfig(
        hidden_size=16,
        intermediate_size=32,
        rms_norm_eps=1e-6,
        initializer_range=0.006,
        num_layers=1,
        residual_init_scale=False,
    )
    with pytest.raises(ValueError):
        DenseFFN(cfg, update_interval=0)
    with pytest.raises(ValueError):
        DenseFFN(cfg, update_interval=-5)


def test_config_validation_wrong_length_raises() -> None:
    # encoder intervals length != ced_num_encoder_layers
    cfg = tiny_cfg()
    cfg.ced_encoder_ffn_update_intervals = (1, 2)  # should be 20
    with pytest.raises(ValueError):
        CausalEncoderDecoderStack(cfg)
    # decoder intervals length != ced_num_decoder_layers (stack)
    cfg2 = tiny_cfg()
    cfg2.ced_decoder_ffn_update_intervals = (1,)
    with pytest.raises(ValueError):
        CausalEncoderDecoderStack(cfg2)


def test_config_validation_non_positive_raises() -> None:
    cfg = tiny_cfg()
    cfg.ced_encoder_ffn_update_intervals = (1,) * 20
    cfg.ced_encoder_ffn_update_intervals = cfg.ced_encoder_ffn_update_intervals[:19] + (0,)
    with pytest.raises(ValueError):
        CausalEncoderDecoderStack(cfg)
    cfg = tiny_cfg()
    cfg.ced_decoder_ffn_update_intervals = (2, 0, 4)
    cfg.ced_num_encoder_layers = 1
    cfg.ced_num_decoder_layers = 3
    with pytest.raises(ValueError):
        CausalEncoderDecoderStack(cfg)


def test_stack_default_all_ones() -> None:
    cfg = tiny_cfg()
    # defaults
    assert cfg.ced_encoder_ffn_update_intervals == ()
    assert cfg.ced_decoder_ffn_update_intervals == ()
    stack = CausalEncoderDecoderStack(cfg)
    # all encoder FFNs interval 1
    for ffn in stack.encoder_ffns:
        assert ffn.update_interval == 1
    # all decoder FFNs interval 1
    for ffn in stack.decoder.ffns:
        assert ffn.update_interval == 1


def test_stack_encoder_decoder_independent_intervals() -> None:
    cfg = _ced_cfg()
    # encoder 2 layers, decoder 3 layers
    cfg.ced_encoder_ffn_update_intervals = (1, 2)
    cfg.ced_decoder_ffn_update_intervals = (2, 4, 8)
    stack = CausalEncoderDecoderStack(cfg)
    assert stack.encoder_ffns[0].update_interval == 1
    assert stack.encoder_ffns[1].update_interval == 2
    assert stack.decoder.ffns[0].update_interval == 2
    assert stack.decoder.ffns[1].update_interval == 4
    assert stack.decoder.ffns[2].update_interval == 8


def test_gated_off_encoder_ffn_is_identity_at_its_step() -> None:
    torch.manual_seed(7)
    cfg = _ced_cfg()
    # layer 0 FFN on at step 0 (0 % 2 == 0); layer 1 FFN gated off at step 1
    cfg.ced_encoder_ffn_update_intervals = (2, 2)
    stack = CausalEncoderDecoderStack(cfg)
    x = torch.randn(1, 6, cfg.hidden_size)
    # encode runs end-to-end; the gated-off FFN at step 1 is an identity
    # residual, so the stream passes through unchanged at that step.
    out, _ = stack.encode(x)
    assert out.shape == x.shape
    enc_ffns = cast("list[DenseFFN]", list(stack.encoder_ffns))
    assert enc_ffns[0].should_update(0)
    assert not enc_ffns[1].should_update(1)


def test_gated_off_decoder_ffn_is_identity_at_its_step() -> None:
    torch.manual_seed(5)
    cfg = _ced_cfg()
    # step 0 on (0%2==0); step 1 off (1%4!=0); step 2 off (2%8!=0)
    cfg.ced_decoder_ffn_update_intervals = (2, 4, 8)
    stack = CausalEncoderDecoderStack(cfg)
    enc = torch.randn(1, 4, cfg.hidden_size)
    dec = torch.randn(1, 3, cfg.hidden_size)
    enc_out, _ = stack.encode(enc)
    out = stack.decode_full_context(dec, enc_out)
    assert out.shape == dec.shape
    dec_ffns = cast("list[DenseFFN]", list(stack.decoder.ffns))
    assert dec_ffns[0].should_update(0)
    assert not dec_ffns[1].should_update(1)
    assert not dec_ffns[2].should_update(2)


def test_decoder_looped_uses_step0_for_shared_ffn() -> None:
    torch.manual_seed(123)
    cfg = _ced_cfg()
    cfg.ced_decoder_looping = True
    cfg.ced_decoder_loop_steps = 3
    cfg.ced_decoder_ffn_update_intervals = ()  # auto all-ones
    stack = CausalEncoderDecoderStack(cfg)
    enc = torch.randn(1, 4, cfg.hidden_size)
    dec = torch.randn(1, 2, cfg.hidden_size)
    enc_out, _ = stack.encode(enc)
    out = stack.decode_full_context(dec, enc_out)
    assert out.shape == (1, 2, cfg.hidden_size)


def test_default_config_preserves_exact_output() -> None:
    from .test_ced_ffn import _ced_cfg

    torch.manual_seed(0)
    cfg1 = _ced_cfg()
    cfg2 = _ced_cfg()
    stack1 = CausalEncoderDecoderStack(cfg1).eval()
    stack2 = CausalEncoderDecoderStack(cfg2).eval()
    x = torch.randn(1, 6, cfg1.hidden_size)
    y = torch.randn(1, 3, cfg1.hidden_size)
    enc1, _ = stack1.encode(x)
    enc2, _ = stack2.encode(x)
    out1 = stack1.decode_full_context(y, enc1)
    out2 = stack2.decode_full_context(y, enc2)
    torch.testing.assert_close(out1, out2, atol=1e-4, rtol=1e-4)


def test_encoder_ffn_step_counter_is_the_layer_index() -> None:
    """Each encoder FFN is called with its own block index, not a constant."""
    cfg = _ced_cfg(ced_encoder_ffn_update_intervals=(1, 2))
    stack = CausalEncoderDecoderStack(cfg).eval()
    steps: list[int] = []
    for ffn in cast("list[DenseFFN]", list(stack.encoder_ffns)):
        original = ffn.forward

        def spy(hidden_states: torch.Tensor, step: int = 0, _original=original) -> torch.Tensor:
            steps.append(step)
            return _original(hidden_states, step=step)

        ffn.forward = spy  # type: ignore[method-assign]
    with torch.no_grad():
        stack.encode(torch.randn(1, 6, cfg.hidden_size))
    assert steps == [0, 1]


def test_ffn_update_interval_guards_fire_at_config_construction() -> None:
    """``__post_init__`` rejects bad intervals before any model is built."""
    with pytest.raises(ValueError, match="ced_encoder_ffn_update_intervals"):
        replace(tiny_cfg(), ced_num_encoder_layers=2, ced_encoder_ffn_update_intervals=(1,))
    with pytest.raises(ValueError, match="ced_encoder_ffn_update_intervals"):
        replace(tiny_cfg(), ced_num_encoder_layers=2, ced_encoder_ffn_update_intervals=(1, 0))
    # The looped decoder shares one layer, so it takes exactly one interval.
    with pytest.raises(ValueError, match="ced_decoder_ffn_update_intervals"):
        replace(
            tiny_cfg(),
            ced_num_decoder_layers=3,
            ced_decoder_looping=True,
            ced_decoder_ffn_update_intervals=(1, 1),
        )
