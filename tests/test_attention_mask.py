"""Attention-mask and is_causal support for GQA/SWA."""

import dataclasses

import pytest
import torch

from memory_machines.models.blocks import GroupedQueryAttention, SlidingWindowAttention
from tests.test_blocks_shapes import tiny_cfg


def plain_cfg():
    return dataclasses.replace(tiny_cfg(), cvpe_enabled=False, frvm_enabled=False)


def test_gqa_is_causal_matches_explicit_tril():
    torch.manual_seed(0)
    m = GroupedQueryAttention(plain_cfg()).eval()
    x = torch.randn(2, 6, 32)
    causal_mask = ~torch.tril(torch.ones(6, 6, dtype=torch.bool))
    with torch.no_grad():
        out_causal, _, _ = m(x, is_causal=True)
        out_mask, _, _ = m(x, attention_mask=causal_mask)
    torch.testing.assert_close(out_causal, out_mask, atol=1e-6, rtol=1e-6)


def test_gqa_key_padding_mask_weights():
    torch.manual_seed(2)
    m = GroupedQueryAttention(plain_cfg()).eval()
    x = torch.randn(2, 6, 32)
    pad = torch.zeros(2, 1, 6, dtype=torch.bool)
    pad[:, :, 4:] = True
    with torch.no_grad():
        out, weights, _ = m(x, attention_mask=pad, output_attentions=True)
    assert weights is not None
    assert bool((weights[:, :, :, 4:] == 0).all())
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 6), atol=1e-5, rtol=1e-5)
    assert torch.isfinite(out).all()


def test_gqa_fully_masked_row_is_zero_not_nan():
    torch.manual_seed(3)
    m = GroupedQueryAttention(plain_cfg()).eval()
    x = torch.randn(1, 4, 32)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    mask[:, 0, :] = True
    with torch.no_grad():
        out, weights, _ = m(x, attention_mask=mask, output_attentions=True)
    assert torch.isfinite(out).all()
    assert weights is not None and torch.isfinite(weights).all()
    torch.testing.assert_close(out[:, 0, :], torch.zeros(1, 32), atol=1e-6, rtol=1e-6)


def test_mask_and_causal_compose():
    torch.manual_seed(0)
    m = GroupedQueryAttention(plain_cfg()).eval()
    x = torch.randn(1, 5, 32)
    pad = torch.zeros(1, 1, 5, dtype=torch.bool)
    pad[:, :, 3:] = True
    tril = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    with torch.no_grad():
        out_both, _, _ = m(x, attention_mask=pad, is_causal=True)
        both = (~tril).unsqueeze(0) | pad.expand(1, 5, 5)
        out_manual, _, _ = m(x, attention_mask=both)
    torch.testing.assert_close(out_both, out_manual, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(3, 6),
        torch.ones(1, 1, 1, 1, 1),
        torch.ones(6, 6, dtype=torch.int32),
    ],
)
def test_gqa_bad_mask_shapes_raise(mask):
    m = GroupedQueryAttention(plain_cfg()).eval()
    with pytest.raises(ValueError):
        m(torch.randn(2, 6, 32), attention_mask=mask)


def test_swa_noncausal_full_window_matches_gqa():
    torch.manual_seed(4)
    cfg = dataclasses.replace(plain_cfg(), swa_window_size=16)
    gqa = GroupedQueryAttention(cfg).eval()
    swa = SlidingWindowAttention(cfg)
    swa.load_state_dict(gqa.state_dict())
    swa.eval()
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        out_gqa, _, _ = gqa(x)
        out_swa, _, _ = swa(x, is_causal=False)
    torch.testing.assert_close(out_swa, out_gqa, atol=1e-5, rtol=1e-5)


def test_swa_noncausal_cache_raises():
    swa = SlidingWindowAttention(plain_cfg()).eval()
    x = torch.randn(1, 4, 32)
    with pytest.raises(ValueError):
        swa(x, use_cache=True, is_causal=False)
    _, _, past = swa(x, use_cache=True)
    with pytest.raises(ValueError):
        swa(x, past_key_value=past, is_causal=False)


def test_gqa_causal_cache_consistency():
    torch.manual_seed(5)
    m = GroupedQueryAttention(tiny_cfg()).eval()
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        o_full, _, _ = m(x, is_causal=True)
        _, _, past = m(x[:, :2, :], use_cache=True, is_causal=True)
        o2, _, _ = m(x[:, 2:, :], past_key_value=past, use_cache=True, is_causal=True)
    torch.testing.assert_close(o2, o_full[:, 2:, :], atol=1e-5, rtol=1e-5)
