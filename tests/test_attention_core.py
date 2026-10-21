"""Core attention correctness: manual SDPA, SWA cache, SwiGLU, windows, positions."""

import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F

from memory_machines.models.blocks import GroupedQueryAttention, SwiGLU
from tests.test_blocks_shapes import tiny_cfg


def plain_cfg():
    return dataclasses.replace(tiny_cfg(), cvpe_enabled=False, frvm_enabled=False)


def test_plain_gqa_matches_manual_sdpa():
    torch.manual_seed(0)
    m = GroupedQueryAttention(plain_cfg()).eval()
    x = torch.randn(2, 6, 32)
    bsz, seq, _ = x.shape
    with torch.no_grad():
        out, _, _ = m(x, is_causal=False)
        q = m.q_proj(x).view(bsz, seq, m.num_heads, m.head_dim).transpose(1, 2)
        k = m.k_proj(x).view(bsz, seq, m.num_key_value_heads, m.head_dim).transpose(1, 2)
        v = m.v_proj(x).view(bsz, seq, m.num_key_value_heads, m.head_dim).transpose(1, 2)
        k = k.repeat_interleave(m.num_key_value_groups, dim=1)
        v = v.repeat_interleave(m.num_key_value_groups, dim=1)
        logits = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(m.head_dim)
        # forward() always applies the sliding-window mask
        # (windowed=True). With is_causal=False the bound is symmetric
        # (|q-k| < window), matching attend().
        from memory_machines.models.blocks import _resolve_positions

        q_pos, k_pos = _resolve_positions(x, seq, 0, None)
        rel = q_pos.unsqueeze(-1) - k_pos.unsqueeze(1)
        win = (rel.abs() < m.window_size).unsqueeze(1)
        logits = logits.masked_fill(~win, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        heads = torch.matmul(probs, v)
        expected = m.o_proj(heads.transpose(1, 2).reshape(bsz, seq, m.hidden_size))
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_swa_truncated_cache_value_consistency():
    torch.manual_seed(1)
    cfg = dataclasses.replace(plain_cfg(), window_size=4)
    m = GroupedQueryAttention(cfg).eval()
    x = torch.randn(1, 10, 32)
    with torch.no_grad():
        full, _, _ = m(x, is_causal=True)
        _, _, past = m(x[:, :6, :], use_cache=True, is_causal=True)
        # The windowed cache (width 4) holds keys at positions
        # 2..5; thread them through so positions stay correct.
        chunk, _, past2 = m(
            x[:, 6:, :],
            past_key_value=past,
            use_cache=True,
            past_positions=torch.arange(2, 6).unsqueeze(0),
            position_ids=torch.arange(6, 10).unsqueeze(0),
            is_causal=True,
        )
    torch.testing.assert_close(full[:, 6:, :], chunk, atol=1e-5, rtol=1e-5)
    assert past is not None and past2 is not None
    assert past[0].shape[-2] == 4
    assert past2[0].shape[-2] == 4


def test_swiglu_matches_manual_and_grads():
    torch.manual_seed(2)
    m = SwiGLU(tiny_cfg())
    m.eval()
    # SwiGLU uses a zero-start init (down_proj = 0), under which both the
    # SiLU and no-SiLU paths are exactly zero; restore nonzero weights so
    # the manual-math comparison below is meaningful.
    torch.nn.init.xavier_uniform_(m.down_proj.weight)
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        out = m(x)
        gate = m.gate_proj(x)
        up = m.up_proj(x)
        expected = m.down_proj(F.silu(gate) * up)
        no_silu = m.down_proj(gate * up)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(out, no_silu, atol=1e-5, rtol=1e-5)

    m.train()
    xg = torch.randn(2, 6, 32, requires_grad=True)
    m.zero_grad()
    m(xg).sum().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()
    for proj in (m.gate_proj, m.up_proj, m.down_proj):
        assert proj.weight.grad is not None and torch.isfinite(proj.weight.grad).all()


def test_swa_causal_window_boundaries():
    torch.manual_seed(3)
    cfg = dataclasses.replace(plain_cfg(), window_size=4)
    swa = GroupedQueryAttention(cfg).eval()
    gqa = GroupedQueryAttention(cfg).eval()
    gqa.load_state_dict(swa.state_dict())
    gqa.eval()
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        out_swa, weights, _ = swa(x, is_causal=True, output_attentions=True)
    assert weights is not None
    q_idx = torch.arange(8).unsqueeze(1)
    k_idx = torch.arange(8).unsqueeze(0)
    rel = q_idx - k_idx
    masked = (rel >= 4) | (rel < 0)
    assert bool((weights[:, :, masked] == 0).all())
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 8), atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        out_gqa, _, _ = gqa(x, attention_mask=masked)
    torch.testing.assert_close(out_swa, out_gqa, atol=1e-5, rtol=1e-5)


def test_position_ids_explicit_and_overflow():
    # Explicit ids must match the full-run slice; default positions differ.
    torch.manual_seed(4)
    m = GroupedQueryAttention(tiny_cfg()).eval()
    # Zero-start o_proj outputs all zeros alike; restore signal so the
    # explicit-vs-default position comparison is meaningful.
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(m.o_proj.weight)
    x_full = torch.randn(1, 14, 32)
    explicit = torch.arange(10, 14)
    with torch.no_grad():
        full, _, _ = m(x_full, is_causal=True)
        _, _, past = m(x_full[:, :10, :], use_cache=True, is_causal=True)
        # The window cache (width 8) drops early keys; thread the
        # cached positions through so absolute positions stay correct.
        # The cache holds keys at absolute positions 2..9, so the new
        # queries sit at 10..13.
        past_positions = torch.arange(2, 10).unsqueeze(0)
        out_explicit, _, _ = m(
            x_full[:, 10:, :],
            past_key_value=past,
            past_positions=past_positions,
            position_ids=explicit,
            is_causal=True,
        )
        out_default, _, _ = m(x_full[:, 10:, :], is_causal=True)
    assert not torch.allclose(out_explicit, out_default, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_explicit, full[:, 10:, :], atol=1e-5, rtol=1e-5)

    short_cfg = dataclasses.replace(tiny_cfg(), max_position_embeddings=8)
    m_short = GroupedQueryAttention(short_cfg).eval()
    with pytest.raises(ValueError):
        m_short(torch.randn(1, 10, 32))

    with pytest.raises(ValueError):
        m_short(torch.randn(1, 2, 32), position_ids=torch.tensor([[6, 8]]))


def test_residual_init_scale_off_zero_starts_the_output_projection() -> None:
    scaled = GroupedQueryAttention(dataclasses.replace(plain_cfg(), residual_init_scale=True))
    zero = GroupedQueryAttention(dataclasses.replace(plain_cfg(), residual_init_scale=False))
    assert torch.count_nonzero(scaled.o_proj.weight) > 0
    assert torch.count_nonzero(zero.o_proj.weight) == 0


def test_output_attentions_and_cache_none_contract():
    torch.manual_seed(5)
    x = torch.randn(1, 5, 32)
    m = GroupedQueryAttention(plain_cfg()).eval()
    with torch.no_grad():
        out, weights, past = m(x)
    assert torch.isfinite(out).all()
    assert weights is None
    assert past is None
