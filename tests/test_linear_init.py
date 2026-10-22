"""Initialization helpers that keep a factored projection at the dense scale."""

import pytest
import torch
from torch import nn

from memory_machines.models.blocks.linear_init import (
    copy_linear_weight_,
    effective_weight,
    factorize_linear,
    init_linear,
    init_linear_columns,
    linear_dtype,
    linear_in_features,
    linear_out_features,
    zero_linear,
)
from memory_machines.models.blocks.linears import FactoredLinear


def _pair(in_features: int = 128, rank: int = 32, out_features: int = 64) -> tuple[nn.Linear, FactoredLinear]:
    torch.manual_seed(0)
    return nn.Linear(in_features, out_features, bias=False), FactoredLinear(in_features, rank, out_features, bias=False)


@pytest.mark.parametrize("std", [0.006, 0.02], ids=["deepseek", "transformer"])
def test_factored_std_matches_the_dense_draw_it_replaces(std: float) -> None:
    dense, factored = _pair()
    init_linear(dense, std)
    init_linear(factored, std)
    assert effective_weight(factored).std().item() == pytest.approx(dense.weight.std().item(), rel=0.15)


def test_init_linear_zeroes_only_the_output_factor() -> None:
    _, factored = _pair()
    init_linear(factored, 0.0)
    assert effective_weight(factored).abs().max().item() == 0.0
    # The input factor keeps its draw, so the pair still has a gradient path.
    assert factored.in_linear.weight.abs().max().item() > 0.0


def test_init_linear_columns_sets_each_output_column_scale() -> None:
    dense, factored = _pair()
    column_stds = torch.full((128,), 0.001)
    column_stds[:64] = 0.01
    init_linear_columns(dense, column_stds)
    init_linear_columns(factored, column_stds)
    for weight in (dense.weight, effective_weight(factored)):
        assert weight[:, :64].std().item() == pytest.approx(0.01, rel=0.2)
        assert weight[:, 64:].std().item() == pytest.approx(0.001, rel=0.3)


def test_init_linear_columns_rejects_a_mismatched_width() -> None:
    _, factored = _pair()
    with pytest.raises(ValueError, match="column_stds"):
        init_linear_columns(factored, torch.ones(64))


def test_zero_linear_leaves_the_input_factor_usable() -> None:
    _, factored = _pair()
    zero_linear(factored)
    assert effective_weight(factored).abs().max().item() == 0.0
    # A zeroed input factor would never receive gradient; keep it random.
    assert factored.in_linear.weight.abs().max().item() > 0.0


def test_copy_linear_weight_round_trips_a_low_rank_matrix() -> None:
    dense, factored = _pair(in_features=64, rank=8, out_features=32)
    torch.manual_seed(1)
    matrix = torch.randn(32, 8) @ torch.randn(8, 64)
    copy_linear_weight_(dense, matrix)
    copy_linear_weight_(factored, matrix)
    torch.testing.assert_close(dense.weight, matrix)
    torch.testing.assert_close(effective_weight(factored), matrix, atol=1e-4, rtol=1e-4)


def test_copy_linear_weight_rejects_the_wrong_shape() -> None:
    dense, _ = _pair()
    with pytest.raises(ValueError, match="weight shape"):
        copy_linear_weight_(dense, torch.ones(8, 8))


def test_factorize_linear_preserves_per_column_scale_when_asked() -> None:
    torch.manual_seed(0)
    dense = nn.Linear(128, 32, bias=False)
    matrix = torch.randn(32, 128) * torch.cat((torch.full((64,), 0.01), torch.full((64,), 0.0005)))
    copy_linear_weight_(dense, matrix)
    factored = factorize_linear(dense, rank=16, preserve_columns=True)
    weight = effective_weight(factored)
    assert weight[:, :64].std().item() == pytest.approx(0.01, rel=0.35)
    assert weight[:, 64:].std().item() == pytest.approx(0.0005, rel=0.6)


@pytest.mark.parametrize("factory", [lambda: nn.Linear(32, 16), lambda: FactoredLinear(32, 8, 16)], ids=["dense", "factored"])
def test_introspection_helpers_agree_on_both_kinds(factory) -> None:
    linear = factory()
    assert (linear_in_features(linear), linear_out_features(linear)) == (32, 16)
    assert linear_dtype(linear) == torch.float32
