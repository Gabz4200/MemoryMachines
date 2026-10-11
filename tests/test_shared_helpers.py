"""Unit tests for src/memory_machines/models/blocks/shared.py helpers."""

import pytest
import torch

from memory_machines.models.blocks.shared import (
    CompressedCache,
    _causal_comp_mask,
    _check_position_bounds,
    _past_total_len,
    _resolve_positions,
    _sinkhorn,
    _update_swa_pos,
    _window_mask,
)


def test_resolve_positions_default_arange():
    torch.manual_seed(0)
    h = torch.randn(2, 3, 32)
    q_pos, k_pos = _resolve_positions(h, 3, 5, None)
    torch.testing.assert_close(q_pos, torch.tensor([[5, 6, 7], [5, 6, 7]]))
    torch.testing.assert_close(k_pos, torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]))


def test_resolve_positions_explicit_2d_slices_last_q_len():
    torch.manual_seed(0)
    h = torch.randn(2, 3, 32)
    position_ids = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]])
    q_pos, k_pos = _resolve_positions(h, 3, 5, position_ids)
    torch.testing.assert_close(q_pos, torch.tensor([[12, 13, 14], [22, 23, 24]]))
    torch.testing.assert_close(
        k_pos,
        torch.tensor([[0, 1, 2, 3, 4, 12, 13, 14], [0, 1, 2, 3, 4, 22, 23, 24]]),
    )


def test_resolve_positions_batch_mismatch_raises():
    h = torch.randn(2, 3, 32)
    position_ids = torch.zeros(3, 3, dtype=torch.long)
    with pytest.raises(ValueError):
        _resolve_positions(h, 3, 5, position_ids)


def test_check_position_bounds_raises_with_explicit_ids():
    q_pos = torch.tensor([[7, 8]])
    k_pos = torch.tensor([[0, 1]])
    with pytest.raises(ValueError):
        _check_position_bounds(q_pos, k_pos, 8, torch.tensor([[0, 1]]))


def test_check_position_bounds_noop_when_none():
    q_pos = torch.tensor([[7, 99]])
    k_pos = torch.tensor([[0, 100]])
    _check_position_bounds(q_pos, k_pos, 8, None)


def test_window_mask():
    q_pos = torch.tensor([[2, 3]])
    k_pos = torch.tensor([[0, 1, 2, 3]])
    mask = _window_mask(q_pos, k_pos, 2)
    expected = torch.tensor([[[False, True, True, False], [False, False, True, True]]])
    torch.testing.assert_close(mask, expected)


def test_causal_comp_mask():
    q_pos = torch.tensor([[1, 3]])
    ends = torch.tensor([0, 2, 4])
    mask = _causal_comp_mask(q_pos, ends)
    expected = torch.tensor([[[True, False, False], [True, True, False]]])
    torch.testing.assert_close(mask, expected)


def test_sinkhorn_rows_cols_approx_one():
    torch.manual_seed(0)
    m = torch.randn(2, 4, 4)
    p = _sinkhorn(m, 50)
    torch.testing.assert_close(p.sum(dim=-1), torch.ones(2, 4), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(p.sum(dim=-2), torch.ones(2, 4), atol=1e-4, rtol=1e-4)


def test_sinkhorn_nan_raises():
    m = torch.tensor([[[1.0, float("nan")], [1.0, 1.0]]])
    with pytest.raises(ValueError):
        _sinkhorn(m, 5)


def test_update_swa_pos_last_width_concat():
    prev = torch.tensor([[0, 1, 2, 3]])
    q = torch.tensor([[4, 5, 6]])
    out = _update_swa_pos(prev, q, 5)
    torch.testing.assert_close(out, torch.tensor([[2, 3, 4, 5, 6]]))


def test_update_swa_pos_none_prev():
    q = torch.tensor([[4, 5, 6]])
    out = _update_swa_pos(None, q, 5)
    torch.testing.assert_close(out, q)


def test_past_total_len_prefers_total_len():
    past = CompressedCache(total_len=10, swa_pos=torch.tensor([[0, 1, 2, 3, 4]]))
    assert _past_total_len(past) == 10


def test_past_total_len_falls_back_to_swa_max_plus_one():
    past = CompressedCache(total_len=None, swa_pos=torch.tensor([[0, 1, 2, 3, 4]]))
    assert _past_total_len(past) == 5


def test_past_total_len_none_past_and_empty():
    assert _past_total_len(None) == 0
    assert _past_total_len(CompressedCache()) is None
