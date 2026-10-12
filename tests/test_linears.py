"""Unit tests for FactoredLinear and ChunkedBottleneckLinear."""

import math
import warnings
from typing import cast

import pytest
import torch
from torch import nn

from memory_machines.models.blocks.linears import ChunkedBottleneckLinear, FactoredLinear


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


def test_factored_linear_valid_rank_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FactoredLinear(8, rank=4, output_dim=6)


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


def test_chunked_bottleneck_forward_leading_dims():
    torch.manual_seed(0)
    layer = ChunkedBottleneckLinear(8, num_blocks=2, output_dim=6)
    assert tuple(layer(torch.randn(2, 5, 8)).shape) == (2, 5, 6)
    assert tuple(layer(torch.randn(8)).shape) == (6,)


def test_chunked_bottleneck_wrong_last_dim_raises():
    layer = ChunkedBottleneckLinear(8, num_blocks=2, output_dim=6)
    with pytest.raises(ValueError, match="expected last dim 8"):
        layer(torch.randn(2, 7))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_dim": 7, "num_blocks": 2, "output_dim": 6},
        {"input_dim": 8, "num_blocks": 2, "output_dim": 7},
    ],
)
def test_chunked_bottleneck_rejects_undivisible_dims(kwargs):
    with pytest.raises(ValueError, match="must be divisible by num_blocks"):
        ChunkedBottleneckLinear(**kwargs)


@pytest.mark.parametrize("bottleneck_dim", [0, -3])
def test_chunked_bottleneck_rejects_non_positive_bottleneck(bottleneck_dim):
    with pytest.raises(ValueError, match="bottleneck_dim"):
        ChunkedBottleneckLinear(8, num_blocks=2, output_dim=6, bottleneck_dim=bottleneck_dim)


def test_chunked_bottleneck_no_bias():
    torch.manual_seed(0)
    layer = ChunkedBottleneckLinear(8, num_blocks=2, output_dim=6, bias=False)
    assert layer.bias is None
    assert tuple(layer(torch.randn(2, 8)).shape) == (2, 6)


def test_chunked_bottleneck_reset_square_mixer_identity():
    torch.manual_seed(0)
    layer = ChunkedBottleneckLinear(8, num_blocks=2, output_dim=6)
    mixer = layer.mixer
    assert isinstance(mixer, nn.Linear)
    mixer.weight.data.uniform_(-3, 3)
    assert layer.bias is not None
    layer.bias.data.uniform_(-3, 3)

    layer.reset_parameters()

    torch.testing.assert_close(mixer.weight, torch.eye(2))
    assert torch.count_nonzero(layer.bias) == 0
    assert tuple(layer.down_weight.shape) == (2, 4)
    assert tuple(layer.up_weight.shape) == (2, 3)


def test_chunked_bottleneck_reset_two_stage_mixer_xavier():
    torch.manual_seed(0)
    layer = ChunkedBottleneckLinear(8, num_blocks=2, output_dim=6, bottleneck_dim=3)
    assert isinstance(layer.mixer, nn.Sequential)
    first = cast(nn.Linear, layer.mixer[0])
    second = cast(nn.Linear, layer.mixer[1])
    pre = first.weight.clone()
    first.weight.data.uniform_(-9, 9)
    second.weight.data.uniform_(-9, 9)

    layer.reset_parameters()

    assert tuple(first.weight.shape) == (3, 2)
    assert tuple(second.weight.shape) == (2, 3)
    assert not torch.allclose(first.weight, pre)
    # xavier_uniform_ bound: gain * sqrt(6 / (fan_in + fan_out)), gain=1.
    bound = math.sqrt(6 / (2 + 3))
    assert torch.all(first.weight.abs() <= bound)
    assert torch.all(second.weight.abs() <= bound)


def test_chunked_bottleneck_chunk_isolation():
    torch.manual_seed(0)
    layer = ChunkedBottleneckLinear(8, num_blocks=2, output_dim=6, bias=False)
    x = torch.randn(1, 8)
    x_perturbed = x.clone()
    x_perturbed[0, :4] += 1.0
    assert not torch.allclose(layer(x), layer(x_perturbed))

    # Identity mixer: output depends only on the pooled scalars, so a
    # within-chunk perturbation orthogonal to the chunk's down_weight
    # leaves the output unchanged.
    mixer = cast(nn.Linear, layer.mixer)
    with torch.no_grad():
        mixer.weight.copy_(torch.eye(2))
    w0 = layer.down_weight.detach()[0]
    delta = torch.tensor([w0[1], -w0[0], 0.0, 0.0])
    x_orth = x.clone()
    x_orth[0, :4] += delta
    torch.testing.assert_close(layer(x), layer(x_orth))


@pytest.mark.parametrize("bottleneck_dim", [None, 3])
def test_chunked_bottleneck_manual_math(bottleneck_dim):
    torch.manual_seed(0)
    layer = ChunkedBottleneckLinear(8, num_blocks=2, output_dim=6, bias=False, bottleneck_dim=bottleneck_dim).eval()
    mixer = layer.mixer
    x = torch.randn(2, 3, 8)
    x_chunked = x.reshape(-1, layer.num_blocks, layer.down_chunk)
    s = torch.einsum("bgc,gc->bg", x_chunked, layer.down_weight)
    h = mixer(s)
    y = torch.einsum("bg,gc->bgc", h, layer.up_weight)
    expected = y.reshape(2, 3, 6)
    torch.testing.assert_close(layer(x), expected)


def test_chunked_bottleneck_manual_math_with_bias():
    torch.manual_seed(0)
    layer = ChunkedBottleneckLinear(8, num_blocks=2, output_dim=6).eval()
    mixer = layer.mixer
    x = torch.randn(2, 3, 8)
    x_chunked = x.reshape(-1, layer.num_blocks, layer.down_chunk)
    s = torch.einsum("bgc,gc->bg", x_chunked, layer.down_weight)
    h = mixer(s)
    y = torch.einsum("bg,gc->bgc", h, layer.up_weight)
    expected = y.reshape(2, 3, 6) + layer.bias
    torch.testing.assert_close(layer(x), expected)
