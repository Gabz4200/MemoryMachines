"""Single-Pass mHC routing primitives (standalone block, not the CED encoder).

Covers two-branch coefficient shapes/constraints, branch-specific routing,
``forward_parallel`` and the zero-branch residual identity, plus the stream
replication/collapse helpers. The CED encoder no longer uses mHC; these pin
the block as a reusable module.
"""

from dataclasses import replace

import pytest
import torch

from memory_machines.models.blocks import SinglePassMHC, from_streams, to_streams
from memory_machines.utils.config_schema import MemoryMachineConfig
from tests.test_blocks_shapes import tiny_cfg

H = 32
N = 2


def _cfg(**kw) -> MemoryMachineConfig:
    return replace(tiny_cfg(), mhc_num_streams=N, **kw)


def test_to_from_streams_roundtrip_and_validation() -> None:
    x = torch.randn(2, 5, H)
    streams = to_streams(x, N)
    assert tuple(streams.shape) == (2, 5, N, H)
    torch.testing.assert_close(streams[:, :, 0], streams[:, :, 1])  # replication
    torch.testing.assert_close(from_streams(streams), x)  # mean of identical streams
    with pytest.raises(ValueError, match="must be"):
        to_streams(torch.randn(5, H), N)
    with pytest.raises(ValueError, match="must be"):
        from_streams(torch.randn(2, 5, H))


def test_parallel_coefficients_shapes_and_constraints() -> None:
    torch.manual_seed(0)
    mhc = SinglePassMHC(_cfg(), parallel_branches=True)
    streams = torch.randn(1, 6, N, H)
    a1, a2, b, c1, c2 = mhc.parallel_coefficients(streams)
    assert tuple(a1.shape) == (1, 6, 1, N)
    assert tuple(a2.shape) == (1, 6, 1, N)
    assert tuple(b.shape) == (1, 6, N, N)
    assert tuple(c1.shape) == (1, 6, N, 1)
    assert tuple(c2.shape) == (1, 6, N, 1)
    for a in (a1, a2):
        assert bool(((a > 0) & (a < 1)).all())
    for c in (c1, c2):
        assert bool(((c > 0) & (c < 2)).all())
    # Sinkhorn residual is doubly stochastic with non-negative entries.
    ones = torch.ones(1, 6, N)
    torch.testing.assert_close(b.sum(-1), ones, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(b.sum(-2), ones, atol=1e-5, rtol=1e-5)
    assert bool((b >= 0).all())


def test_parallel_coefficients_requires_parallel_branches_flag() -> None:
    mhc = SinglePassMHC(_cfg())  # single-branch
    with pytest.raises(RuntimeError, match="parallel_branches"):
        mhc.parallel_coefficients(torch.randn(1, 4, N, H))


def test_parallel_routing_is_branch_specific() -> None:
    torch.manual_seed(0)
    mhc = SinglePassMHC(_cfg(), parallel_branches=True)
    with torch.no_grad():
        mhc.alpha_pre.fill_(1.0)
        mhc.alpha_pre2.fill_(1.0)
        mhc.a_dyn.weight.normal_(std=0.5)
        mhc.a2_dyn.weight.normal_(std=0.5)
    a1, a2, _, _, _ = mhc.parallel_coefficients(torch.randn(1, 6, N, H))
    assert not torch.allclose(a1, a2), "the two branches must learn different routings"


def test_forward_parallel_routes_branches_differently() -> None:
    torch.manual_seed(0)
    mhc = SinglePassMHC(_cfg(), parallel_branches=True)
    with torch.no_grad():
        mhc.alpha_pre.fill_(1.0)
        mhc.alpha_pre2.fill_(1.0)
        mhc.a_dyn.weight.normal_(std=0.5)
        mhc.a2_dyn.weight.normal_(std=0.5)
    streams = torch.randn(1, 5, N, H)
    seen: dict[str, torch.Tensor] = {}

    def branch_a(x: torch.Tensor) -> torch.Tensor:
        seen["a"] = x.detach().clone()
        return x

    def branch_g(x: torch.Tensor) -> torch.Tensor:
        seen["g"] = x.detach().clone()
        return x

    out, a1, a2 = mhc.forward_parallel(streams, branch_a, branch_g)
    assert tuple(out.shape) == (1, 5, N, H)
    assert tuple(a1.shape) == (1, 5, 1, N)
    assert tuple(a2.shape) == (1, 5, 1, N)
    # The routed views differ because A^{a} != A^{g}.
    assert not torch.allclose(seen["a"], seen["g"])
    # Output is the two-branch write-in of the branch outputs.
    _, _, b, c1, c2 = mhc.parallel_coefficients(streams)
    expected = mhc.combine_parallel(streams, b, c1, seen["a"], c2, seen["g"])
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_combine_parallel_residual_term_is_identity_when_branches_are_zero() -> None:
    streams = torch.randn(1, 4, N, H)
    b = torch.eye(N).expand(1, 4, N, N).contiguous()
    zeros = torch.zeros(1, 4, H)
    ones = torch.ones(1, 4, N, 1)
    out = SinglePassMHC.combine_parallel(streams, b, ones, zeros, ones, zeros)
    torch.testing.assert_close(out, streams)
