"""SWA-by-design attention: window disable, cache, attend core, replay truncation."""

import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F

from memory_machines.models.blocks import GroupedQueryAttention
from tests.test_blocks_shapes import tiny_cfg


def plain_cfg():
    return dataclasses.replace(tiny_cfg(), cvpe_enabled=False, frvm_enabled=False)


def test_full_attention_window_disabled_matches_manual_sdpa():
    # window_size=0 must run pure full attention: no window mask,
    # matching manual SDPA under both causal and non-causal modes.
    torch.manual_seed(0)
    cfg = dataclasses.replace(plain_cfg(), window_size=0)
    m = GroupedQueryAttention(cfg).eval()
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(m.o_proj.weight)
    x = torch.randn(2, 6, 32)
    bsz, seq, _ = x.shape
    with torch.no_grad():
        out_nc, _, _ = m(x, is_causal=False)
        out_c, _, _ = m(x, is_causal=True)
        q = m.q_proj(x).view(bsz, seq, m.num_heads, m.head_dim).transpose(1, 2)
        k = m.k_proj(x).view(bsz, seq, m.num_key_value_heads, m.head_dim).transpose(1, 2)
        v = m.v_proj(x).view(bsz, seq, m.num_key_value_heads, m.head_dim).transpose(1, 2)
        k = k.repeat_interleave(m.num_key_value_groups, dim=1)
        v = v.repeat_interleave(m.num_key_value_groups, dim=1)
        logits = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(m.head_dim)
        probs = F.softmax(logits, dim=-1)
        heads = torch.matmul(probs, v)
        expected_nc = m.o_proj(heads.transpose(1, 2).reshape(bsz, seq, m.hidden_size))
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        probs_c = F.softmax(logits.masked_fill(~causal, float("-inf")), dim=-1)
        heads_c = torch.matmul(probs_c, v)
        expected_c = m.o_proj(heads_c.transpose(1, 2).reshape(bsz, seq, m.hidden_size))
    torch.testing.assert_close(out_nc, expected_nc, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_c, expected_c, atol=1e-5, rtol=1e-5)


def test_full_attention_cache_untruncated():
    # window_size=0 keeps the full cache: width equals the total
    # sequence length and keeps growing across incremental passes.
    torch.manual_seed(1)
    cfg = dataclasses.replace(plain_cfg(), window_size=0)
    m = GroupedQueryAttention(cfg).eval()
    x1 = torch.randn(1, 10, 32)
    with torch.no_grad():
        _, _, past = m(x1, use_cache=True, is_causal=True)
    assert past is not None
    assert past[0].shape[-2] == 10
    assert past[1].shape[-2] == 10
    x2 = torch.randn(1, 10, 32)
    with torch.no_grad():
        _, _, past2 = m(x2, past_key_value=past, use_cache=True, is_causal=True)
    assert past2 is not None
    assert past2[0].shape[-2] == 20
    assert past2[1].shape[-2] == 20


def test_full_attention_incremental_replay_matches_full_run():
    # With the window disabled, chunk-by-chunk decode against the
    # full (untruncated) cache reproduces the single full run.
    torch.manual_seed(2)
    cfg = dataclasses.replace(plain_cfg(), window_size=0)
    m = GroupedQueryAttention(cfg).eval()
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(m.o_proj.weight)
    x = torch.randn(1, 10, 32)
    with torch.no_grad():
        full, _, _ = m(x, is_causal=True)
        out1, _, past = m(x[:, :6, :], use_cache=True, is_causal=True)
        out2, _, past2 = m(x[:, 6:, :], past_key_value=past, use_cache=True, is_causal=True)
    torch.testing.assert_close(out1, full[:, :6, :], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out2, full[:, 6:, :], atol=1e-5, rtol=1e-5)
    assert past2 is not None and past2[0].shape[-2] == 10


def test_ctor_window_size_overrides_config():
    # window_size=None falls back to config.window_size; an explicit
    # 0 disables the window; a positive value sets a custom width.
    cfg = tiny_cfg()  # config.window_size == 8
    m_default = GroupedQueryAttention(cfg).eval()
    assert m_default.window_size == cfg.window_size == 8
    m_off = GroupedQueryAttention(cfg, window_size=0).eval()
    assert m_off.window_size == 0
    x = torch.randn(1, 6, 32)
    with torch.no_grad():
        _, _, past_off = m_off(x, use_cache=True, is_causal=True)
    assert past_off is not None and past_off[0].shape[-2] == 6  # untruncated
    m_custom = GroupedQueryAttention(cfg, window_size=3).eval()
    assert m_custom.window_size == 3
    with torch.no_grad():
        _, _, past_custom = m_custom(x, use_cache=True, is_causal=True)
    assert past_custom is not None and past_custom[0].shape[-2] == 3


def test_attend_windowed_causal_mask():
    # windowed causal attend: weights vanish outside [q-3, q], rows
    # sum to 1, and within-window weights equal the causal softmax
    # restricted to the window and renormalized over window keys.
    torch.manual_seed(3)
    m = GroupedQueryAttention(dataclasses.replace(plain_cfg(), window_size=4)).eval()
    bsz, lq, lk = 2, 5, 8
    q = torch.randn(bsz, m.num_heads, lq, m.head_dim)
    k = torch.randn(bsz, m.num_key_value_heads, lk, m.head_dim)
    v = torch.randn(bsz, m.num_key_value_heads, lk, m.head_dim)
    vq = torch.randn(bsz, m.num_key_value_heads, lq, m.head_dim)
    q_pos = torch.arange(4, 4 + lq).unsqueeze(0).expand(bsz, -1)
    k_pos = torch.arange(lk).unsqueeze(0).expand(bsz, -1)
    with torch.no_grad():
        out, w = m.attend(q, k, v, vq, q_pos, k_pos, windowed=True, is_causal=True)
    assert out.shape == (bsz, m.num_heads, lq, m.head_dim)
    rel = q_pos.unsqueeze(-1) - k_pos.unsqueeze(1)
    outside = ~((rel >= 0) & (rel < m.window_size))
    assert bool((w[outside.unsqueeze(1).expand_as(w)] == 0).all())
    torch.testing.assert_close(w.sum(-1), torch.ones(bsz, m.num_heads, lq), atol=1e-5, rtol=1e-5)
    # Manual renormalized expectation: causal softmax over all keys,
    # restricted to the window and renormalized over window keys.
    k_exp = k.repeat_interleave(m.num_key_value_groups, dim=1)
    logits = torch.matmul(q, k_exp.transpose(2, 3)) / math.sqrt(m.head_dim)
    causal = (k_pos.unsqueeze(1) <= q_pos.unsqueeze(-1)).unsqueeze(1)
    causal_probs = F.softmax(logits.masked_fill(~causal, float("-inf")), dim=-1)
    win = ((rel >= 0) & (rel < m.window_size)).unsqueeze(1)
    restricted = causal_probs * win
    expected = restricted / restricted.sum(-1, keepdim=True)
    torch.testing.assert_close(w, expected, atol=1e-5, rtol=1e-5)


def test_attend_pure_causal_when_windowed_false():
    # windowed=False, is_causal=True: pure causal attention over all
    # past keys; future keys are zeroed and weights match the manual
    # causal softmax.
    torch.manual_seed(4)
    m = GroupedQueryAttention(dataclasses.replace(plain_cfg(), window_size=4)).eval()
    bsz, lq, lk = 2, 5, 8
    q = torch.randn(bsz, m.num_heads, lq, m.head_dim)
    k = torch.randn(bsz, m.num_key_value_heads, lk, m.head_dim)
    v = torch.randn(bsz, m.num_key_value_heads, lk, m.head_dim)
    vq = torch.randn(bsz, m.num_key_value_heads, lq, m.head_dim)
    q_pos = torch.arange(lq).unsqueeze(0).expand(bsz, -1)
    k_pos = torch.arange(lk).unsqueeze(0).expand(bsz, -1)
    with torch.no_grad():
        out, w = m.attend(q, k, v, vq, q_pos, k_pos, windowed=False, is_causal=True)
    assert out.shape == (bsz, m.num_heads, lq, m.head_dim)
    future = q_pos.unsqueeze(-1) < k_pos.unsqueeze(1)
    assert bool((w[future.unsqueeze(1).expand_as(w)] == 0).all())
    torch.testing.assert_close(w.sum(-1), torch.ones(bsz, m.num_heads, lq), atol=1e-5, rtol=1e-5)
    k_exp = k.repeat_interleave(m.num_key_value_groups, dim=1)
    logits = torch.matmul(q, k_exp.transpose(2, 3)) / math.sqrt(m.head_dim)
    causal = (k_pos.unsqueeze(1) <= q_pos.unsqueeze(-1)).unsqueeze(1)
    expected = F.softmax(logits.masked_fill(~causal, float("-inf")), dim=-1)
    torch.testing.assert_close(w, expected, atol=1e-5, rtol=1e-5)


def test_attend_noncausal_windowed_bidirectional():
    # Non-causal windowed attend keeps keys with |q-k| < window and
    # zeroes the rest; rows still sum to 1.
    torch.manual_seed(5)
    m = GroupedQueryAttention(dataclasses.replace(plain_cfg(), window_size=4)).eval()
    bsz, lq, lk = 2, 5, 8
    q = torch.randn(bsz, m.num_heads, lq, m.head_dim)
    k = torch.randn(bsz, m.num_key_value_heads, lk, m.head_dim)
    v = torch.randn(bsz, m.num_key_value_heads, lk, m.head_dim)
    vq = torch.randn(bsz, m.num_key_value_heads, lq, m.head_dim)
    q_pos = torch.arange(2, 2 + lq).unsqueeze(0).expand(bsz, -1)
    k_pos = torch.arange(lk).unsqueeze(0).expand(bsz, -1)
    with torch.no_grad():
        out, w = m.attend(q, k, v, vq, q_pos, k_pos, windowed=True, is_causal=False)
    assert out.shape == (bsz, m.num_heads, lq, m.head_dim)
    rel = q_pos.unsqueeze(-1) - k_pos.unsqueeze(1)
    outside = rel.abs() >= m.window_size
    assert bool((w[outside.unsqueeze(1).expand_as(w)] == 0).all())
    torch.testing.assert_close(w.sum(-1), torch.ones(bsz, m.num_heads, lq), atol=1e-5, rtol=1e-5)


def test_attend_full_attention_when_window_disabled():
    # windowed=True with window_size=0 degenerates to pure causal
    # attention: identical to windowed=False and to the manual
    # causal softmax over all keys.
    torch.manual_seed(6)
    m = GroupedQueryAttention(dataclasses.replace(plain_cfg(), window_size=0)).eval()
    bsz, lq, lk = 2, 5, 8
    q = torch.randn(bsz, m.num_heads, lq, m.head_dim)
    k = torch.randn(bsz, m.num_key_value_heads, lk, m.head_dim)
    v = torch.randn(bsz, m.num_key_value_heads, lk, m.head_dim)
    vq = torch.randn(bsz, m.num_key_value_heads, lq, m.head_dim)
    q_pos = torch.arange(lq).unsqueeze(0).expand(bsz, -1)
    k_pos = torch.arange(lk).unsqueeze(0).expand(bsz, -1)
    with torch.no_grad():
        _, w_w = m.attend(q, k, v, vq, q_pos, k_pos, windowed=True, is_causal=True)
        _, w_p = m.attend(q, k, v, vq, q_pos, k_pos, windowed=False, is_causal=True)
    assert torch.equal(w_w, w_p)
    k_exp = k.repeat_interleave(m.num_key_value_groups, dim=1)
    logits = torch.matmul(q, k_exp.transpose(2, 3)) / math.sqrt(m.head_dim)
    causal = (k_pos.unsqueeze(1) <= q_pos.unsqueeze(-1)).unsqueeze(1)
    expected = F.softmax(logits.masked_fill(~causal, float("-inf")), dim=-1)
    torch.testing.assert_close(w_w, expected, atol=1e-5, rtol=1e-5)


def test_truncate_for_replay_1d_and_2d():
    # truncate_for_replay keeps the trailing window on the last axis
    # for both 1D and 2D position tensors, and trims K/V on dim 2.
    m = GroupedQueryAttention(dataclasses.replace(plain_cfg(), window_size=4)).eval()
    bsz, lk = 2, 6
    k = torch.randn(bsz, m.num_key_value_heads, lk, m.head_dim)
    v = torch.randn(bsz, m.num_key_value_heads, lk, m.head_dim)
    pos1d = torch.arange(lk)
    tk, tv, tp = m.truncate_for_replay(k, v, pos1d)
    assert tp.shape == (4,)
    assert tp.tolist() == [2, 3, 4, 5]
    assert tk.shape == (bsz, m.num_key_value_heads, 4, m.head_dim)
    assert tv.shape == (bsz, m.num_key_value_heads, 4, m.head_dim)
    assert torch.equal(tk, k[:, :, -4:, :])
    assert torch.equal(tv, v[:, :, -4:, :])
    pos2d = torch.arange(bsz * lk).reshape(bsz, lk)
    tk2, tv2, tp2 = m.truncate_for_replay(k, v, pos2d)
    assert tp2.shape == (bsz, 4)
    assert tp2.tolist() == [[2, 3, 4, 5], [8, 9, 10, 11]]
    assert torch.equal(tk2, tk)
    assert torch.equal(tv2, tv)


def test_truncate_for_replay_noop_when_disabled():
    # window_size=0 returns keys, values, and positions unchanged.
    m = GroupedQueryAttention(dataclasses.replace(plain_cfg(), window_size=0)).eval()
    bsz, lk = 2, 6
    k = torch.randn(bsz, m.num_key_value_heads, lk, m.head_dim)
    v = torch.randn(bsz, m.num_key_value_heads, lk, m.head_dim)
    pos1d = torch.arange(lk)
    tk, tv, tp = m.truncate_for_replay(k, v, pos1d)
    assert tk.shape == k.shape and torch.equal(tk, k)
    assert tv.shape == v.shape and torch.equal(tv, v)
    assert tp.shape == pos1d.shape and torch.equal(tp, pos1d)
    pos2d = torch.arange(bsz * lk).reshape(bsz, lk)
    tk2, tv2, tp2 = m.truncate_for_replay(k, v, pos2d)
    assert tk2.shape == k.shape and torch.equal(tk2, k)
    assert tv2.shape == v.shape and torch.equal(tv2, v)
    assert tp2.shape == pos2d.shape and torch.equal(tp2, pos2d)


def test_past_len_override_matches_expected_width():
    # Truncated-window replay: thread the cached positions and
    # explicit query ids so the incremental output matches the full
    # run slice; the first-pass cache keeps exactly window_size keys.
    torch.manual_seed(7)
    m = GroupedQueryAttention(dataclasses.replace(plain_cfg(), window_size=4)).eval()
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(m.o_proj.weight)
    x = torch.randn(1, 10, 32)
    with torch.no_grad():
        full, _, _ = m(x, is_causal=True)
        _, _, past = m(x[:, :6, :], use_cache=True, is_causal=True)
    assert past is not None and past[0].shape[-2] == 4
    with torch.no_grad():
        out, _, _ = m(
            x[:, 6:, :],
            past_key_value=past,
            past_positions=torch.arange(2, 6).unsqueeze(0),
            position_ids=torch.arange(6, 10).unsqueeze(0),
            is_causal=True,
        )
    torch.testing.assert_close(out, full[:, 6:, :], atol=1e-5, rtol=1e-5)


def test_is_causal_false_cache_raises_window_disabled():
    # The is_causal=False + caching guard is independent of the
    # window: it still raises when window_size=0.
    m = GroupedQueryAttention(dataclasses.replace(plain_cfg(), window_size=0)).eval()
    with pytest.raises(ValueError):
        m(torch.randn(1, 4, 32), is_causal=False, use_cache=True)
