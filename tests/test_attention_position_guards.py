"""Position/replay guards for GroupedQueryAttention and FRVM gate shape guards.

Covers the ``max_position_embeddings`` sequence-length raise, the
windowed-replay ``past_positions`` requirement, the public
``project_heads`` alias, and the shape-mismatch raises of
``FRVMValueGate.gated_reduction`` (each pinned to its exact message).
"""

import dataclasses

import pytest
import torch

from memory_machines.models.blocks import GroupedQueryAttention
from memory_machines.models.blocks.positioning import FRVMValueGate
from tests.test_blocks_shapes import tiny_cfg


def plain_cfg():
    """Tiny config with CVPE/FRVM off (the guards under test need neither)."""
    return dataclasses.replace(tiny_cfg(), cvpe_enabled=False, frvm_enabled=False)


def test_over_long_sequence_exceeds_max_position_embeddings_raises() -> None:
    cfg = dataclasses.replace(plain_cfg(), max_position_embeddings=8)
    m = GroupedQueryAttention(cfg).eval()
    x = torch.randn(1, 10, cfg.hidden_size)
    with pytest.raises(ValueError, match="exceeds configured max_position_embeddings"):
        m(x, is_causal=True)


def test_windowed_replay_without_past_positions_raises() -> None:
    torch.manual_seed(0)
    cfg = dataclasses.replace(plain_cfg(), window_size=4)
    m = GroupedQueryAttention(cfg).eval()
    x = torch.randn(1, 10, cfg.hidden_size)
    with torch.no_grad():
        _, _, past = m(x[:, :6, :], use_cache=True, is_causal=True)
    assert past is not None
    with pytest.raises(ValueError, match="replaying a past_key_value with window_size > 0 requires past_positions"):
        m(x[:, 6:, :], past_key_value=past, is_causal=True)


def test_project_heads_alias_matches_private_project_heads() -> None:
    torch.manual_seed(0)
    m = GroupedQueryAttention(plain_cfg()).eval()
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(m.o_proj.weight)
    heads = torch.randn(2, m.num_heads, 5, m.head_dim)
    torch.testing.assert_close(m.project_heads(heads, 2, 5), m._project_heads(heads, 2, 5))


def test_project_heads_returns_model_dim_shape() -> None:
    torch.manual_seed(1)
    m = GroupedQueryAttention(plain_cfg()).eval()
    heads = torch.randn(2, m.num_heads, 5, m.head_dim)
    out = m.project_heads(heads, 2, 5)
    assert tuple(out.shape) == (2, 5, m.hidden_size)


def test_gated_reduction_valid_shapes_returns_bh_lq_dh() -> None:
    torch.manual_seed(0)
    gate = FRVMValueGate(head_dim=8, num_heads=2).eval()
    weights = torch.softmax(torch.randn(2, 2, 4, 6), dim=-1)
    v_full = torch.randn(2, 2, 6, 8)
    v_query = torch.randn(2, 2, 4, 8)
    out = gate.gated_reduction(weights, v_full, v_query)
    assert tuple(out.shape) == (2, 2, 4, 8)


@pytest.mark.parametrize(
    ("w_shape", "v_full_shape", "v_query_shape", "match"),
    [
        ((1, 2, 3, 5), (1, 2, 6, 8), (1, 2, 3, 8), "does not match v_full Lk"),
        ((1, 2, 3, 5), (1, 2, 5, 8), (1, 2, 4, 8), "does not match v_query Lq"),
        ((1, 2, 3, 5), (1, 2, 5, 8), (1, 2, 3, 7), "does not match v_query head_dim"),
        ((1, 2, 3, 5), (1, 3, 5, 8), (1, 3, 3, 8), "batch/heads mismatch"),
    ],
    ids=["lk_mismatch", "lq_mismatch", "head_dim_mismatch", "batch_heads_mismatch"],
)
def test_gated_reduction_shape_mismatch_raises(
    w_shape: tuple[int, ...],
    v_full_shape: tuple[int, ...],
    v_query_shape: tuple[int, ...],
    match: str,
) -> None:
    gate = FRVMValueGate(head_dim=8, num_heads=2).eval()
    weights = torch.zeros(*w_shape)
    v_full = torch.zeros(*v_full_shape)
    v_query = torch.zeros(*v_query_shape)
    with pytest.raises(ValueError, match=match):
        gate.gated_reduction(weights, v_full, v_query)
