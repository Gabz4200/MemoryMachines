"""Review guards for attention: window boundaries, replay contracts, mask prep.

Covers gaps not in test_attention_core.py / test_attention_windows.py /
test_attention_mask.py. Uses plain (cvpe/frvm-off) configs throughout.
"""

import dataclasses

import pytest
import torch

from memory_machines.models.blocks.attention import (
    GroupedQueryAttention,
    _apply_masks_and_softmax,
    _prepare_attention_mask,
)
from tests.test_blocks_shapes import tiny_cfg


def plain_cfg():
    return dataclasses.replace(tiny_cfg(), cvpe_enabled=False, frvm_enabled=False)


def test_forward_noncausal_windowed_boundary_exactness():
    # w=4, L=8: keep |q-k|<4. Row 0 keeps k=0..3; row 7 keeps k=4..7.
    torch.manual_seed(0)
    cfg = dataclasses.replace(plain_cfg(), window_size=4)
    m = GroupedQueryAttention(cfg).eval()
    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        _, weights, _ = m(x, is_causal=False, output_attentions=True)
    assert weights is not None
    assert tuple(weights.shape) == (1, m.num_heads, 8, 8)
    q_idx = torch.arange(8).unsqueeze(1)
    k_idx = torch.arange(8).unsqueeze(0)
    outside = (q_idx - k_idx).abs() >= 4
    assert bool((weights[:, :, outside] == 0).all())
    torch.testing.assert_close(weights.sum(-1), torch.ones(1, m.num_heads, 8), atol=1e-5, rtol=1e-5)
    assert bool((weights[0, 0, 0, 3] > 0).item())
    assert weights[0, 0, 0, 4].item() == 0
    # Symmetric far edge: |7-4|=3 kept, |7-3|=4 masked.
    assert bool((weights[0, 0, 7, 4] > 0).item())
    assert weights[0, 0, 7, 3].item() == 0


def test_forward_attend_parity_noncausal_windowed():
    torch.manual_seed(1)
    cfg = dataclasses.replace(plain_cfg(), window_size=4)
    m = GroupedQueryAttention(cfg).eval()
    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        out_fwd, w_fwd, _ = m(x, is_causal=False, output_attentions=True)
        q, k, v_current = m._project_qkv(x)
        q_pos = torch.arange(8).unsqueeze(0)
        k_pos = torch.arange(8).unsqueeze(0)
        heads, w_att = m.attend(
            q,
            k,
            v_current,
            v_current,
            q_pos,
            k_pos,
            windowed=True,
            is_causal=False,
        )
        out_att = m._project_heads(heads, 1, 8)
    assert w_fwd is not None
    torch.testing.assert_close(w_att, w_fwd, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_att, out_fwd, atol=1e-5, rtol=1e-5)


def test_apply_masks_and_softmax_four_way():
    attn = torch.zeros(1, 1, 3, 5)
    q_pos = torch.tensor([[4, 5, 6]])
    k_pos = torch.tensor([[0, 1, 2, 3, 4]])
    # (a) windowed + causal: 0 <= q-k < 3.
    out_a = _apply_masks_and_softmax(attn, q_pos, k_pos, 3, True, True)
    keep_a = torch.tensor([[[[False, False, True, True, True], [False, False, False, True, True], [False, False, False, False, True]]]])
    assert bool(torch.isneginf(out_a[~keep_a]).all())
    assert bool((out_a[keep_a] == 0).all())
    # (b) windowed + non-causal: |q-k| < 3 (no future keys here, same keep).
    out_b = _apply_masks_and_softmax(attn, q_pos, k_pos, 3, False, True)
    rel = q_pos.unsqueeze(-1) - k_pos.unsqueeze(1)
    keep_b = (rel.abs() < 3).unsqueeze(1)
    assert torch.equal(keep_b, keep_a)
    assert bool(torch.isneginf(out_b[~keep_b]).all())
    assert bool((out_b[keep_b] == 0).all())
    # (c) full + causal: k <= q (all keep here since k<=4<=q).
    out_c = _apply_masks_and_softmax(attn, q_pos, k_pos, 3, True, False)
    assert bool((out_c == 0).all())
    assert not bool(torch.isneginf(out_c).any())
    # (d) full + non-causal: unchanged.
    out_d = _apply_masks_and_softmax(attn, q_pos, k_pos, 3, False, False)
    assert bool((out_d == 0).all())
    assert not bool(torch.isneginf(out_d).any())


def test_1d_past_positions_replay_parity():
    torch.manual_seed(2)
    cfg = dataclasses.replace(plain_cfg(), window_size=4)
    m = GroupedQueryAttention(cfg).eval()
    g = torch.Generator().manual_seed(2)
    x = torch.randn(1, 10, 32, generator=g)
    with torch.no_grad():
        full, _, _ = m(x, is_causal=True)
        _, _, past = m(x[:, :6, :], use_cache=True, is_causal=True)
        assert past is not None and past[0].shape[-2] == 4
        out, _, _ = m(
            x[:, 6:, :],
            past_key_value=past,
            past_positions=torch.arange(2, 6),
            position_ids=torch.arange(6, 10),
            is_causal=True,
        )
    torch.testing.assert_close(out, full[:, 6:, :], atol=1e-5, rtol=1e-5)


def test_1d_past_positions_broadcast_batch():
    torch.manual_seed(3)
    cfg = dataclasses.replace(plain_cfg(), window_size=4)
    m = GroupedQueryAttention(cfg).eval()
    g = torch.Generator().manual_seed(3)
    x = torch.randn(2, 10, 32, generator=g)
    with torch.no_grad():
        _, _, past = m(x[:, :6, :], use_cache=True, is_causal=True)
        assert past is not None
        out_1d, _, _ = m(
            x[:, 6:, :],
            past_key_value=past,
            past_positions=torch.arange(2, 6),
            position_ids=torch.arange(6, 10),
            is_causal=True,
        )
        out_2d, _, _ = m(
            x[:, 6:, :],
            past_key_value=past,
            past_positions=torch.arange(2, 6).unsqueeze(0).expand(2, -1),
            position_ids=torch.arange(6, 10).unsqueeze(0).expand(2, -1),
            is_causal=True,
        )
    assert tuple(out_1d.shape) == (2, 4, 32)
    torch.testing.assert_close(out_1d, out_2d, atol=1e-6, rtol=1e-6)


def test_past_len_override_cached_width_guard_window_disabled():
    torch.manual_seed(4)
    cfg = dataclasses.replace(plain_cfg(), window_size=0)
    m = GroupedQueryAttention(cfg).eval()
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, 10, 32, generator=g)
    with torch.no_grad():
        full, _, _ = m(x, is_causal=True)
        _, _, past = m(x[:, :6, :], use_cache=True, is_causal=True)
        assert past is not None and past[0].shape[-2] == 6
        with pytest.raises(ValueError, match="cached width"):
            m(
                x[:, 6:, :],
                past_key_value=past,
                past_positions=torch.arange(10),
                past_len_override=10,
                is_causal=True,
            )
        out, _, _ = m(
            x[:, 6:, :],
            past_key_value=past,
            past_positions=torch.arange(6),
            past_len_override=6,
            position_ids=torch.arange(6, 10),
            is_causal=True,
        )
    torch.testing.assert_close(out, full[:, 6:, :], atol=1e-5, rtol=1e-5)


def test_override_positions_disagreement_raises():
    torch.manual_seed(5)
    cfg = dataclasses.replace(plain_cfg(), window_size=4)
    m = GroupedQueryAttention(cfg).eval()
    x = torch.randn(1, 10, 32)
    with torch.no_grad():
        _, _, past = m(x[:, :6, :], use_cache=True, is_causal=True)
        assert past is not None
        with pytest.raises(ValueError, match="disagrees with past_positions"):
            m(
                x[:, 6:, :],
                past_key_value=past,
                past_positions=torch.arange(2, 6).unsqueeze(0),
                past_len_override=10,
                position_ids=torch.arange(6, 10).unsqueeze(0),
                is_causal=True,
            )


def test_nonzero_override_without_cache_raises_zero_succeeds():
    m = GroupedQueryAttention(dataclasses.replace(plain_cfg(), window_size=4)).eval()
    x = torch.randn(1, 4, 32)
    with pytest.raises(ValueError, match="requires past_key_value"):
        m(x, past_len_override=5, is_causal=True)
    with torch.no_grad():
        out, _, _ = m(x, past_len_override=0, is_causal=True)
    assert tuple(out.shape) == (1, 4, 32)
    assert torch.isfinite(out).all()


def test_past_positions_batch_mismatch_raises():
    m = GroupedQueryAttention(dataclasses.replace(plain_cfg(), window_size=4)).eval()
    x = torch.randn(2, 4, 32)
    with pytest.raises(ValueError, match="does not match input batch"):
        m(x, past_positions=torch.zeros(1, 4), is_causal=True)


def test_1d_override_consistent_path():
    torch.manual_seed(6)
    cfg = dataclasses.replace(plain_cfg(), window_size=4)
    m = GroupedQueryAttention(cfg).eval()
    g = torch.Generator().manual_seed(6)
    x = torch.randn(1, 10, 32, generator=g)
    with torch.no_grad():
        _, _, past = m(x[:, :6, :], use_cache=True, is_causal=True)
        assert past is not None
        out_1d, _, _ = m(
            x[:, 6:, :],
            past_key_value=past,
            past_positions=torch.arange(2, 6),
            past_len_override=6,
            position_ids=torch.arange(6, 10),
            is_causal=True,
        )
        out_2d, _, _ = m(
            x[:, 6:, :],
            past_key_value=past,
            past_positions=torch.arange(2, 6).unsqueeze(0),
            past_len_override=6,
            position_ids=torch.arange(6, 10).unsqueeze(0),
            is_causal=True,
        )
    torch.testing.assert_close(out_1d, out_2d, atol=1e-6, rtol=1e-6)


def test_prepare_attention_mask_bool_broadcast_dtype_parity():
    logits = torch.zeros(2, 4, 3, 5, dtype=torch.float16)
    mask_3d = torch.zeros(2, 1, 5, dtype=torch.bool)
    mask_3d[:, :, 3:] = True
    out_3d = _prepare_attention_mask(mask_3d, 2, 3, 5, 4, logits)
    assert out_3d is not None
    assert out_3d.dtype == torch.float16
    assert out_3d.device == logits.device
    assert tuple(out_3d.shape) == (2, 1, 1, 5)
    assert bool(torch.isneginf(out_3d[..., 3:]).all())
    assert bool((out_3d[..., :3] == 0).all())
    mask_4d = torch.zeros(1, 1, 3, 5, dtype=torch.bool)
    mask_4d[..., 0] = True
    out_4d = _prepare_attention_mask(mask_4d, 2, 3, 5, 4, logits)
    assert out_4d is not None
    assert out_4d.dtype == torch.float16
    assert out_4d.device == logits.device
    assert bool(torch.isneginf(out_4d[..., 0]).all())
    assert bool((out_4d[..., 1:] == 0).all())


def test_window_size_1_noncausal_diagonal_only():
    torch.manual_seed(7)
    cfg = dataclasses.replace(plain_cfg(), window_size=1)
    m = GroupedQueryAttention(cfg).eval()
    x = torch.randn(1, 6, 32)
    with torch.no_grad():
        _, weights, _ = m(x, is_causal=False, output_attentions=True)
    assert weights is not None
    eye = torch.eye(6).unsqueeze(0).unsqueeze(0).expand(1, m.num_heads, 6, 6)
    torch.testing.assert_close(weights, eye, atol=1e-5, rtol=1e-5)


def test_implied_len_overflow_raises():
    cfg = dataclasses.replace(plain_cfg(), window_size=4, max_position_embeddings=8)
    m = GroupedQueryAttention(cfg).eval()
    x = torch.randn(1, 2, 32)
    with pytest.raises(ValueError, match="max_position_embeddings"):
        m(x, past_positions=torch.arange(8, 10), is_causal=True)


def test_windowed_replay_without_past_positions_raises():
    torch.manual_seed(10)
    cfg = dataclasses.replace(plain_cfg(), window_size=4)
    m = GroupedQueryAttention(cfg).eval()
    g = torch.Generator().manual_seed(10)
    x = torch.randn(1, 10, 32, generator=g)
    with torch.no_grad():
        _, _, past = m(x[:, :6, :], use_cache=True, is_causal=True)
        assert past is not None
        with pytest.raises(ValueError, match="past_positions"):
            m(x[:, 6:, :], past_key_value=past, is_causal=True)
        with pytest.raises(ValueError, match="past_positions"):
            m(
                x[:, 6:, :],
                past_key_value=past,
                position_ids=torch.arange(6, 10),
                is_causal=True,
            )


def test_prepare_float_additive_passthrough():
    logits = torch.zeros(1, 2, 3, 5, dtype=torch.float32)
    fmask = torch.randn(3, 5, dtype=torch.float32)
    out = _prepare_attention_mask(fmask, 1, 3, 5, 2, logits)
    assert out is not None
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, fmask, atol=0, rtol=0)
    assert not bool(torch.isneginf(out).any())


def test_prepare_bad_3d_shape_raises():
    logits = torch.zeros(1, 4, 3, 5, dtype=torch.float32)
    bad_3d = torch.zeros(1, 2, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="3D attention_mask"):
        _prepare_attention_mask(bad_3d, 1, 3, 5, 4, logits)


def test_prepare_bad_4d_shape_raises():
    logits = torch.zeros(1, 4, 3, 5, dtype=torch.float32)
    bad_4d = torch.zeros(1, 2, 3, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="4D attention_mask"):
        _prepare_attention_mask(bad_4d, 1, 3, 5, 4, logits)


def test_position_ids_batch_mismatch_raises():
    m = GroupedQueryAttention(plain_cfg()).eval()
    x = torch.randn(2, 4, 32)
    with pytest.raises(ValueError, match="batch size"):
        m(x, position_ids=torch.zeros(1, 4, dtype=torch.long), is_causal=True)


def test_past_positions_invalid_ndim_raises():
    torch.manual_seed(11)
    cfg = dataclasses.replace(plain_cfg(), window_size=4)
    m = GroupedQueryAttention(cfg).eval()
    g = torch.Generator().manual_seed(11)
    x = torch.randn(1, 10, 32, generator=g)
    with torch.no_grad():
        _, _, past = m(x[:, :6, :], use_cache=True, is_causal=True)
        assert past is not None
        with pytest.raises(ValueError, match="1D.*2D|past_positions must be"):
            m(
                x[:, 6:, :],
                past_key_value=past,
                past_positions=torch.tensor(5),
                is_causal=True,
            )
        with pytest.raises(ValueError, match="1D.*2D|past_positions must be"):
            m(
                x[:, 6:, :],
                past_key_value=past,
                past_positions=torch.zeros(1, 2, 2),
                is_causal=True,
            )
