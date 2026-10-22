"""Unit tests for FactoredLinear."""

import pytest
import torch

from memory_machines.models.blocks.linears import FactoredLinear


def test_factored_linear_forward_leading_dims():
    torch.manual_seed(0)
    layer = FactoredLinear(8, rank=4, output_dim=6)
    assert tuple(layer(torch.randn(2, 5, 8)).shape) == (2, 5, 6)
    assert tuple(layer(torch.randn(8)).shape) == (6,)


def test_factored_linear_matches_effective_dense():
    torch.manual_seed(0)
    layer = FactoredLinear(8, rank=4, output_dim=6)
    x = torch.randn(3, 8)
    w_eff = layer.out_linear.weight @ layer.in_linear.weight
    expected = x @ w_eff.t() + layer.out_linear.bias
    torch.testing.assert_close(layer(x), expected)


def test_factored_linear_wrong_last_dim_raises():
    layer = FactoredLinear(8, rank=4, output_dim=6)
    with pytest.raises(ValueError, match="expected last dim 8"):
        layer(torch.randn(2, 7))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_dim": 0, "rank": 4, "output_dim": 6},
        {"input_dim": 8, "rank": 0, "output_dim": 6},
        {"input_dim": 8, "rank": 4, "output_dim": -1},
        {"input_dim": 8, "rank": 2.5, "output_dim": 6},
    ],
)
def test_factored_linear_rejects_non_positive_dims(kwargs):
    with pytest.raises(ValueError, match="must be a positive int"):
        FactoredLinear(**kwargs)


def test_factored_linear_excess_rank_warns():
    with pytest.warns(UserWarning, match="exceeds min"):
        layer = FactoredLinear(4, rank=8, output_dim=6)
    assert layer.rank == 8


def test_factored_linear_reset_parameters():
    torch.manual_seed(0)
    layer = FactoredLinear(8, rank=4, output_dim=6)
    layer.in_linear.weight.data.uniform_(-5, 5)
    layer.out_linear.weight.data.uniform_(-5, 5)
    layer.out_linear.bias.data.uniform_(-5, 5)
    pre_w = layer.out_linear.weight.clone()
    pre_b = layer.out_linear.bias.clone()

    layer.reset_parameters()

    assert tuple(layer.in_linear.weight.shape) == (4, 8)
    assert tuple(layer.out_linear.weight.shape) == (6, 4)
    assert not torch.allclose(layer.out_linear.weight, pre_w)
    assert not torch.allclose(layer.out_linear.bias, pre_b)
    assert torch.count_nonzero(layer.out_linear.bias) == 0


def test_factored_linear_effective_rank_bounded():
    torch.manual_seed(0)
    layer = FactoredLinear(12, rank=4, output_dim=9)
    w = layer.out_linear.weight @ layer.in_linear.weight
    assert tuple(w.shape) == (9, 12)
    assert int(torch.linalg.matrix_rank(w)) <= 4


def test_factored_linear_no_bias():
    torch.manual_seed(0)
    layer = FactoredLinear(8, rank=4, output_dim=6, bias=False)
    assert layer.out_linear.bias is None
    assert tuple(layer(torch.randn(2, 8)).shape) == (2, 6)
