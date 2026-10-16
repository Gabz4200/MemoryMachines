"""Unit tests for rational Kolmogorov-Arnold Network (rKAN) blocks.

Covers the functional Jacobi basis, the finite/semi-infinite/infinite domain
mappings (paper eqs. 1-5), and the two trainable modules ``JacobiRKAN`` and
``PadeRKAN`` from Aghaei, "rKAN: Rational Kolmogorov-Arnold Networks"
(arXiv:2406.14495).
"""

import math

import pytest
import torch
from torch import nn

from memory_machines.models.blocks.rkan import (
    JacobiRKAN,
    PadeRKAN,
    algebraic_infinite_map,
    algebraic_semi_infinite_map,
    exponential_semi_infinite_map,
    jacobi_basis,
    jacobi_polynomial,
    linear_map,
    logarithmic_infinite_map,
    logarithmic_semi_infinite_map,
    rational_jacobi,
    shifted_jacobi,
)


def _t(*values: float, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype)


# ---------------------------------------------------------------------------
# Domain mappings (paper Section 2, eqs. 1-5)
# ---------------------------------------------------------------------------


def test_linear_map_endpoints_and_midpoint():
    x = _t(2.0, 5.0, 8.0)
    out = linear_map(x, d0=2.0, d1=8.0)
    torch.testing.assert_close(out, _t(-1.0, 0.0, 1.0))


def test_linear_map_rejects_degenerate_interval():
    with pytest.raises(ValueError, match="d1"):
        linear_map(_t(0.0), d0=1.0, d1=1.0)


def test_algebraic_infinite_map_range_and_symmetry():
    iota = torch.tensor(1.5)
    x = _t(-100.0, -1.0, 0.0, 1.0, 100.0)
    out = algebraic_infinite_map(x, iota)
    assert torch.all(out > -1.0) and torch.all(out < 1.0)
    torch.testing.assert_close(out[0], torch.tensor(-1.0, dtype=torch.float64), atol=1e-2, rtol=0.0)
    torch.testing.assert_close(out[4], torch.tensor(1.0, dtype=torch.float64), atol=1e-2, rtol=0.0)
    torch.testing.assert_close(out[2], torch.tensor(0.0, dtype=torch.float64), atol=0.0, rtol=0.0)
    # Odd symmetry: phi(-x) == -phi(x).
    torch.testing.assert_close(out[1], -out[3])


def test_algebraic_infinite_map_rejects_non_positive_iota():
    with pytest.raises(ValueError, match="iota"):
        algebraic_infinite_map(_t(1.0), torch.tensor(0.0))


def test_logarithmic_infinite_map_range():
    iota = torch.tensor(2.0)
    x = _t(-100.0, 0.0, 100.0)
    out = logarithmic_infinite_map(x, iota)
    # tanh saturates to exactly +/-1 in floating point; closed interval is correct.
    assert torch.all(out >= -1.0) and torch.all(out <= 1.0)
    torch.testing.assert_close(out[1], torch.tensor(0.0, dtype=torch.float64), atol=0.0, rtol=0.0)
    torch.testing.assert_close(out[0], torch.tensor(-1.0, dtype=torch.float64), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(out[2], torch.tensor(1.0, dtype=torch.float64), atol=1e-6, rtol=0.0)


def test_semi_infinite_maps_span_minus_one_to_one():
    iota = torch.tensor(1.0)
    # The algebraic map converges as 1 - 2*iota/x, so the limit point is far out.
    x = _t(0.0, 1.0, 1e4)
    for mapping in (logarithmic_semi_infinite_map, algebraic_semi_infinite_map, exponential_semi_infinite_map):
        out = mapping(x, iota)
        torch.testing.assert_close(out[0], torch.tensor(-1.0, dtype=torch.float64), atol=1e-9, rtol=0.0)
        # x = 0 maps to exactly -1; the moderate point stays strictly inside;
        # the far point reaches the +1 limit (possibly saturating in fp).
        assert bool(out[1] > -1.0) and bool(out[1] < 1.0)
        assert bool(out[2] > 0.999) and bool(out[2] <= 1.0)
        # Strictly increasing on (0, inf).
        assert bool(out[0] < out[1] < out[2])


def test_semi_infinite_maps_reject_negative_inputs():
    with pytest.raises(ValueError, match="non-negative"):
        algebraic_semi_infinite_map(_t(-1.0), torch.tensor(1.0))


# ---------------------------------------------------------------------------
# Jacobi basis (paper Section 2)
# ---------------------------------------------------------------------------


def test_jacobi_polynomial_degree_zero_is_one():
    xi = _t(-0.7, 0.0, 0.9)
    torch.testing.assert_close(
        jacobi_polynomial(xi, 0, torch.tensor(1.0), torch.tensor(2.0)),
        torch.ones(3, dtype=torch.float64),
    )


def test_jacobi_polynomial_legendre_case_matches_closed_form():
    # alpha = beta = 0 reduces Jacobi to Legendre: P1(x) = x,
    # P2(x) = (3x^2 - 1) / 2, P3(x) = (5x^3 - 3x) / 2.
    xi = _t(-0.9, -0.2, 0.0, 0.4, 0.99)
    a = torch.tensor(0.0)
    b = torch.tensor(0.0)
    torch.testing.assert_close(jacobi_polynomial(xi, 1, a, b), xi)
    torch.testing.assert_close(jacobi_polynomial(xi, 2, a, b), (3 * xi**2 - 1) / 2)
    torch.testing.assert_close(jacobi_polynomial(xi, 3, a, b), (5 * xi**3 - 3 * xi) / 2)


def test_jacobi_polynomial_first_order_general_ab():
    xi = _t(-0.5, 0.25, 0.75)
    a = torch.tensor(1.5)
    b = torch.tensor(0.5)
    expected = ((a - b) + (a + b + 2.0) * xi) / 2.0
    torch.testing.assert_close(jacobi_polynomial(xi, 1, a, b), expected)


def test_jacobi_polynomial_rejects_non_positive_degree():
    with pytest.raises(ValueError, match="degree"):
        jacobi_polynomial(_t(0.0), -1, torch.tensor(0.0), torch.tensor(0.0))


def test_jacobi_basis_stacks_orders_zero_to_degree():
    xi = _t(-0.4, 0.6)
    a = torch.tensor(0.0)
    b = torch.tensor(0.0)
    stacked = jacobi_basis(xi, 3, a, b)
    assert tuple(stacked.shape) == (2, 4)
    torch.testing.assert_close(stacked[:, 0], torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(stacked[:, 2], (3 * xi**2 - 1) / 2)


def test_shifted_jacobi_matches_reference_expansion():
    # Independent inline copy of the reference rKAN shifted expansion for n = 2
    # pins the port (alpha = beta = 1, zeta = 0.5 on [0, 1]).
    x = _t(0.1, 0.5, 0.9)
    alpha = torch.tensor(1.0)
    beta = torch.tensor(1.0)
    zeta = torch.tensor(0.5)
    xi = 2.0 * x**zeta - 1.0
    z = xi - 1.0
    expected = ((alpha + 1) * (alpha + 2)) / 2 + (alpha + 2) * (3 + alpha + beta) * z / 2 + (3 + alpha + beta) * (4 + alpha + beta) * z**2 / 8
    torch.testing.assert_close(shifted_jacobi(x, 2, alpha, beta, zeta, 0.0, 1.0), expected)


def test_rational_jacobi_matches_reference_expansion():
    # Independent inline copy of the reference rKAN rational expansion for n = 2
    # (algebraic infinite map, zeta = 1).
    x = _t(-2.0, -0.5, 0.0, 1.5, 3.0)
    alpha = torch.tensor(1.0)
    beta = torch.tensor(1.0)
    iota = torch.tensor(1.2)
    u = x / torch.sqrt(x**2 + iota**2)
    z = u - 1.0
    expected = ((alpha + 1) * (alpha + 2)) / 2 + (alpha + 2) * (3 + alpha + beta) * z / 2 + (3 + alpha + beta) * (4 + alpha + beta) * z**2 / 8
    torch.testing.assert_close(rational_jacobi(x, 2, alpha, beta, iota), expected)


def test_rational_jacobi_logarithmic_mapping_bounded_input():
    x = _t(-5.0, 0.0, 5.0)
    out = rational_jacobi(x, 3, torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0), mapping="logarithmic")
    assert torch.all(torch.isfinite(out))


# ---------------------------------------------------------------------------
# JacobiRKAN module (paper Section 3.2)
# ---------------------------------------------------------------------------


def test_algebraic_maps_accept_python_float_iota():
    # torch.hypot requires Tensor operands; float iotas must be converted.
    x = _t(-2.0, 0.0, 2.0)
    torch.testing.assert_close(
        algebraic_infinite_map(x, 1.5),
        algebraic_infinite_map(x, torch.tensor(1.5)),
    )
    torch.testing.assert_close(
        rational_jacobi(x, 2, torch.tensor(1.0), torch.tensor(1.0), 1.2),
        rational_jacobi(x, 2, torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.2)),
    )


def test_jacobi_rkan_rejects_bad_degree():
    for bad in (0, -2, 2.5, "3"):
        with pytest.raises((ValueError, TypeError)):
            JacobiRKAN(bad)  # type: ignore[arg-type]


def test_jacobi_rkan_preserves_shape_and_is_differentiable():
    torch.manual_seed(0)
    layer = JacobiRKAN(3)
    x = torch.randn(2, 5, 4)
    y = layer(x)
    assert tuple(y.shape) == (2, 5, 4)
    assert torch.all(torch.isfinite(y))
    y.sum().backward()
    for name in ("alpha", "beta", "iota"):
        grad = getattr(layer, name).grad
        assert grad is not None and torch.all(torch.isfinite(grad))


def test_jacobi_rkan_matches_functional_with_normalized_params():
    import torch.nn.functional as F

    torch.manual_seed(1)
    layer = JacobiRKAN(4).double()
    x = torch.randn(3, 7, dtype=torch.float64)
    with torch.no_grad():
        expected = rational_jacobi(
            x,
            4,
            F.elu(layer.alpha.double(), 1.0),
            F.elu(layer.beta.double(), 1.0),
            F.softplus(layer.iota.double()),
        )
    torch.testing.assert_close(layer(x), expected)


def test_jacobi_rkan_initializes_to_reference_values():
    layer = JacobiRKAN(2)
    torch.testing.assert_close(layer.alpha, torch.ones(1))
    torch.testing.assert_close(layer.beta, torch.ones(1))
    torch.testing.assert_close(layer.iota, torch.ones(1))


def test_jacobi_rkan_stays_finite_under_extreme_raw_params():
    layer = JacobiRKAN(3)
    with torch.no_grad():
        layer.alpha.fill_(-100.0)  # ELU -> ~-1 (still a valid Jacobi shape)
        layer.iota.fill_(-100.0)  # softplus -> ~0+
    y = layer(torch.tensor([-10.0, 0.0, 10.0]))
    assert torch.all(torch.isfinite(y))


def test_jacobi_rkan_supports_degree_above_reference_cap():
    layer = JacobiRKAN(8)
    y = layer(torch.randn(4))
    assert tuple(y.shape) == (4,)
    assert torch.all(torch.isfinite(y))


# ---------------------------------------------------------------------------
# PadeRKAN module (paper Section 3.1)
# ---------------------------------------------------------------------------


def test_pade_rkan_rejects_bad_degrees():
    for bad in (0, -1, 2.5):
        with pytest.raises((ValueError, TypeError)):
            PadeRKAN(bad, 3)  # type: ignore[arg-type]
        with pytest.raises((ValueError, TypeError)):
            PadeRKAN(3, bad)  # type: ignore[arg-type]


def test_pade_rkan_degree_two_matches_manual_ratio():
    # degree 2 uses no Jacobi terms: p = w0 + w1*sigmoid(x), same for q.
    torch.manual_seed(0)
    layer = PadeRKAN(2, 2, denominator_eps=0.0)
    with torch.no_grad():
        layer.w_p.copy_(torch.tensor([1.0, 2.0]))
        layer.w_q.copy_(torch.tensor([1.0, -0.5]))
    x = torch.tensor([-1.0, 0.0, 1.0])
    s = torch.sigmoid(x)
    expected = (1.0 + 2.0 * s) / (1.0 - 0.5 * s)
    torch.testing.assert_close(layer(x), expected)


def test_pade_rkan_degree_three_includes_shifted_jacobi_term():
    # degree 3 adds one Jacobi term: w2 * J_2(2*sigmoid(x)^zeta - 1)
    # with alpha = beta = ELU(0) = 0 (Legendre) and zeta = sigmoid(0) = 0.5.
    layer = PadeRKAN(3, 2, denominator_eps=0.0)
    with torch.no_grad():
        layer.alpha_p.zero_()
        layer.beta_p.zero_()
        layer.zeta_p.zero_()
        layer.w_p.copy_(torch.tensor([0.5, -1.0, 2.0]))
        layer.w_q.copy_(torch.tensor([1.0, 0.25]))
    x = torch.tensor([-1.0, 0.5, 2.0])
    s = torch.sigmoid(x)
    xi = 2.0 * s**0.5 - 1.0
    j2 = (3.0 * xi**2 - 1.0) / 2.0
    expected = (0.5 - 1.0 * s + 2.0 * j2) / (1.0 + 0.25 * s)
    torch.testing.assert_close(layer(x), expected)


def test_pade_rkan_degree_one_does_not_crash():
    # The reference implementation indexes w[1] unconditionally; degree 1
    # must degrade to a constant numerator instead of raising IndexError.
    layer = PadeRKAN(1, 2, denominator_eps=0.0)
    with torch.no_grad():
        layer.w_p.copy_(torch.tensor([2.0]))
        layer.w_q.copy_(torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(
        layer(torch.tensor([-3.0, 0.0, 3.0])),
        torch.tensor([2.0, 2.0, 2.0]),
    )


def test_pade_rkan_preserves_shape_is_finite_and_differentiable():
    torch.manual_seed(0)
    layer = PadeRKAN(3, 4)
    x = torch.randn(2, 6, 5) * 3.0
    y = layer(x)
    assert tuple(y.shape) == (2, 6, 5)
    assert torch.all(torch.isfinite(y))
    y.sum().backward()
    for name in ("alpha_p", "beta_p", "zeta_p", "w_p", "alpha_q", "beta_q", "zeta_q", "w_q"):
        grad = getattr(layer, name).grad
        assert grad is not None and torch.all(torch.isfinite(grad))


def test_pade_rkan_denominator_guard_kicks_in_near_zero():
    layer = PadeRKAN(2, 2, denominator_eps=1e-3)
    with torch.no_grad():
        layer.w_p.copy_(torch.tensor([1.0, 0.0]))
        layer.w_q.copy_(torch.tensor([0.0, 0.0]))  # q == 0 everywhere
    y = layer(torch.tensor([0.0, 1.0]))
    assert torch.all(torch.isfinite(y))
    torch.testing.assert_close(y, torch.tensor([1000.0, 1000.0]))


def test_pade_rkan_initializes_to_reference_values():
    layer = PadeRKAN(2, 3)
    for name in ("alpha_p", "beta_p", "alpha_q", "beta_q"):
        torch.testing.assert_close(getattr(layer, name), torch.ones(1))
    for name in ("zeta_p", "zeta_q"):
        torch.testing.assert_close(getattr(layer, name), torch.zeros(1))
    torch.testing.assert_close(layer.w_p, torch.ones(2))
    torch.testing.assert_close(layer.w_q, torch.ones(3))


def test_pade_rkan_reset_parameters_restores_init():
    torch.manual_seed(0)
    layer = PadeRKAN(2, 3)
    with torch.no_grad():
        layer.w_p.uniform_(-5, 5)
        layer.zeta_p.fill_(4.0)
    layer.reset_parameters()
    torch.testing.assert_close(layer.w_p, torch.ones(2))
    torch.testing.assert_close(layer.zeta_p, torch.zeros(1))


# ---------------------------------------------------------------------------
# Composition (paper architectures such as [1, 10, 1] = Linear + rKAN)
# ---------------------------------------------------------------------------


def test_rkan_composes_with_linears_like_paper_architectures():
    torch.manual_seed(0)
    seq = nn.Sequential(nn.Linear(1, 16), JacobiRKAN(3), nn.Linear(16, 32), PadeRKAN(2, 6), nn.Linear(32, 1))
    y = seq(torch.randn(8, 1))
    assert tuple(y.shape) == (8, 1)
    assert torch.all(torch.isfinite(y))


def test_rkan_fits_asymptotic_bump():
    # F2(x) = 1 / (1 + x^2) is one of the paper's regression targets; a
    # rational basis should drive the loss down within a few Adam steps.
    torch.manual_seed(0)
    x = torch.linspace(-3, 3, 64).unsqueeze(1)
    target = 1.0 / (1.0 + x**2)
    model = nn.Sequential(nn.Linear(1, 16), PadeRKAN(3, 3), nn.Linear(16, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(300):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), target)
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    assert first is not None and first > 0
    final = nn.functional.mse_loss(model(x), target).item()
    assert final < 0.5 * first
    assert math.isfinite(final)
