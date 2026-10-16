"""Boundary guards for ``shared._check_position_bounds`` and its GQA caller.

Pins the exclusive upper bound (``max_position_embeddings``: id equal to the
bound and one above raise, one below passes) and the no-``position_ids`` skip,
plus the same failure end-to-end through ``GroupedQueryAttention``.
"""

import dataclasses

import pytest
import torch

from memory_machines.models.blocks import GroupedQueryAttention
from memory_machines.models.blocks.shared import _check_position_bounds
from tests.test_blocks_shapes import tiny_cfg


def plain_cfg():
    """Tiny config with CVPE/FRVM off; position bounds do not depend on either."""
    return dataclasses.replace(tiny_cfg(), cvpe_enabled=False, frvm_enabled=False)


@pytest.mark.parametrize("position", [8, 9], ids=["at_max", "above_max"])
def test_explicit_position_at_or_above_max_raises(position: int) -> None:
    with pytest.raises(ValueError, match="Absolute position id exceeds max_position_embeddings"):
        _check_position_bounds(
            torch.tensor([[position]]),
            torch.tensor([[0]]),
            8,
            torch.tensor([[position]]),
        )


def test_explicit_position_below_max_passes() -> None:
    assert _check_position_bounds(torch.tensor([[7]]), torch.tensor([[7]]), 8, torch.tensor([[7]])) is None


def test_none_position_ids_skips_out_of_range_positions() -> None:
    assert _check_position_bounds(torch.tensor([[99]]), torch.tensor([[100]]), 8, None) is None


def test_gqa_position_ids_at_max_raises_bounds_error() -> None:
    cfg = dataclasses.replace(plain_cfg(), max_position_embeddings=8)
    m = GroupedQueryAttention(cfg).eval()
    with pytest.raises(ValueError, match="Absolute position id exceeds max_position_embeddings"):
        m(torch.randn(1, 2, cfg.hidden_size), position_ids=torch.tensor([[6, 8]]), is_causal=True)
