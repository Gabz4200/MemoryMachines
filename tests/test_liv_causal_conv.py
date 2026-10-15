"""Causal-conv coverage for LivConvBlock (mixes via ShortConvolution).

Complements tests/test_liv_positioning.py (per-head independence, merged/head
adapter equivalence, grad flow) — none of that is duplicated here.
"""

import dataclasses

import pytest
import torch

from memory_machines.models.blocks.causal_conv1d import ShortConvolution
from memory_machines.models.blocks.liv import LivConvBlock
from memory_machines.utils.config_schema import MemoryMachineConfig
from tests.test_blocks_shapes import tiny_cfg


def _cfg_with_kernel(k: int) -> MemoryMachineConfig:
    return dataclasses.replace(tiny_cfg(), conv_kernel_size=k)


def _assert_causal_dirac(block: LivConvBlock) -> None:
    w = block.conv.weight.detach()
    torch.testing.assert_close(w[:, 0, -1], torch.ones_like(w[:, 0, -1]))
    if block.conv.kernel_size[0] > 1:
        torch.testing.assert_close(w[:, 0, :-1], torch.zeros_like(w[:, 0, :-1]))


def test_mix_heads_identity_at_init():
    torch.manual_seed(0)
    block = LivConvBlock(tiny_cfg()).eval()
    x = torch.randn(2, 8, 4, 8)
    with torch.no_grad():
        out = block.mix_heads(x)
    torch.testing.assert_close(out, x)


def test_mix_heads_causality_future_perturb():
    torch.manual_seed(1)
    block = LivConvBlock(tiny_cfg()).eval()
    t = 3
    x = torch.randn(2, 8, 4, 8)
    x2 = x.clone()
    x2[:, t:, ...] = torch.randn(2, 8 - t, 4, 8)
    with torch.no_grad():
        base = block.mix_heads(x)
        pert = block.mix_heads(x2)
    # Past untouched by future perturbation.
    torch.testing.assert_close(base[:, :t], pert[:, :t])
    # Present/future changed.
    assert not torch.allclose(base[:, t:], pert[:, t:], atol=1e-6, rtol=1e-6)


def test_dirac_layout_at_construction():
    block = LivConvBlock(tiny_cfg())
    assert block.conv.kernel_size[0] == tiny_cfg().conv_kernel_size
    _assert_causal_dirac(block)


def test_structural_pin():
    cfg = tiny_cfg()
    block = LivConvBlock(cfg)
    assert isinstance(block.conv, ShortConvolution)
    assert block.conv.activation is None
    assert block.conv.backend == "naive"
    assert block.conv.groups == cfg.hidden_size
    assert block.conv.bias is None
    assert block.conv.kernel_size == (cfg.conv_kernel_size,)


def test_reset_reestablishes_dirac_and_identity():
    torch.manual_seed(4)
    block = LivConvBlock(tiny_cfg()).eval()
    with torch.no_grad():
        block.conv.weight.copy_(torch.randn_like(block.conv.weight))
    assert not torch.allclose(block.conv.weight[:, 0, -1], torch.ones_like(block.conv.weight[:, 0, -1]))
    block.reset_parameters()
    block.eval()
    _assert_causal_dirac(block)
    x = torch.randn(2, 8, 4, 8)
    with torch.no_grad():
        torch.testing.assert_close(block.mix_heads(x), x)


def test_randomized_weight_reads_past_not_future():
    torch.manual_seed(5)
    block = LivConvBlock(tiny_cfg()).eval()
    with torch.no_grad():
        block.conv.weight.copy_(torch.randn_like(block.conv.weight))
    t, L = 3, 6
    x = torch.randn(2, L, 4, 8)
    with torch.no_grad():
        ref = block.mix_heads(x)
        # Zeroing the past (<= t influences position t via kernel width 3).
        xp = x.clone()
        xp[:, :t] = 0
        out_past_zeroed = block.mix_heads(xp)
        # Zeroing the strict future must not touch position t.
        xf = x.clone()
        xf[:, t + 1 :] = 0
        out_fut_zeroed = block.mix_heads(xf)
    assert not torch.allclose(ref[:, t], out_past_zeroed[:, t], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out_fut_zeroed[:, t], ref[:, t], atol=1e-6, rtol=1e-6)


def test_short_sequences_l1_l2():
    torch.manual_seed(6)
    block = LivConvBlock(tiny_cfg()).eval()
    with torch.no_grad():
        for L in (1, 2):
            x = torch.randn(2, L, 4, 8)
            out = block.mix_heads(x)
            assert tuple(out.shape) == (2, L, 4, 8)
            torch.testing.assert_close(out, x)


def test_full_forward_shape_and_eval_determinism():
    torch.manual_seed(7)
    cfg = tiny_cfg()
    block = LivConvBlock(cfg).eval()
    x = torch.randn(2, 8, cfg.hidden_size)
    with torch.no_grad():
        o1 = block(x)
        o2 = block(x)
    assert tuple(o1.shape) == (2, 8, cfg.hidden_size)
    torch.testing.assert_close(o1, o2)


# NOTE: conv_kernel_size must be odd (config_schema validates
# `conv_kernel_size % 2 == 1`), so even K=2/4 cannot be constructed —
# covered by test_even_kernel_size_rejected below. Parametrize odd sizes.
@pytest.mark.parametrize("k", [3, 5])
def test_kernel_size_identity_and_causality(k):
    torch.manual_seed(10 + k)
    block = LivConvBlock(_cfg_with_kernel(k)).eval()
    assert block.conv.kernel_size == (k,)
    x = torch.randn(1, 8, 4, 8)
    with torch.no_grad():
        torch.testing.assert_close(block.mix_heads(x), x)
        t = 4
        x2 = x.clone()
        x2[:, t:, ...] = torch.randn(1, 8 - t, 4, 8)
        base = block.mix_heads(x)
        pert = block.mix_heads(x2)
    torch.testing.assert_close(base[:, :t], pert[:, :t])
    assert not torch.allclose(base[:, t:], pert[:, t:], atol=1e-6, rtol=1e-6)


def test_even_kernel_size_rejected():
    with pytest.raises(ValueError, match="must be odd"):
        _cfg_with_kernel(2)
    with pytest.raises(ValueError, match="must be odd"):
        _cfg_with_kernel(4)
