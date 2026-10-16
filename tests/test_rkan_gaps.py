"""Gap coverage for rational KAN blocks not exercised by ``test_rkan.py``.

Pins the error branches, ``repr``/``reset_parameters`` contracts, integer-dtype
handling, and denominator-guard behavior of
``memory_machines.models.blocks.rkan``. All tests are deterministic, CPU-only,
and behavior-focused.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from memory_machines.models.blocks.rkan import (
    JacobiRKAN,
    PadeRKAN,
    algebraic_infinite_map,
    exponential_semi_infinite_map,
    jacobi_polynomial,
    linear_map,
    logarithmic_semi_infinite_map,
    rational_jacobi,
    shifted_jacobi,
)


def _t(*values: float, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype)


# ---------------------------------------------------------------------------
# 1. Non-convertible iota raises TypeError ("positive scalar")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_iota", [None, object()])
def test_maps_reject_non_convertible_iota_with_type_error(bad_iota):
    with pytest.raises(TypeError, match="positive scalar"):
        algebraic_infinite_map(_t(1.0), bad_iota)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Semi-infinite maps reject negative inputs (log/exp)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mapping", [logarithmic_semi_infinite_map, exponential_semi_infinite_map])
def test_log_exp_semi_infinite_maps_reject_negative_inputs(mapping):
    with pytest.raises(ValueError, match="non-negative"):
        mapping(_t(-1.0), torch.tensor(1.0))


# ---------------------------------------------------------------------------
# 3-4. shifted_jacobi interval / zeta validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a,b", [(0.0, 0.0), (1.0, 0.0)])
def test_shifted_jacobi_rejects_degenerate_interval(a, b):
    with pytest.raises(ValueError, match="greater than"):
        shifted_jacobi(_t(0.5), 1, torch.tensor(0.0), torch.tensor(0.0), a=a, b=b)


@pytest.mark.parametrize("zeta", [0.0, -0.5])
def test_shifted_jacobi_rejects_non_positive_zeta(zeta):
    with pytest.raises(ValueError, match="zeta"):
        shifted_jacobi(_t(0.5), 1, torch.tensor(0.0), torch.tensor(0.0), zeta)


# ---------------------------------------------------------------------------
# 5. rational_jacobi mapping / zeta validation
# ---------------------------------------------------------------------------


def test_rational_jacobi_rejects_unknown_mapping():
    with pytest.raises(ValueError, match="mapping"):
        rational_jacobi(_t(0.5), 1, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0), mapping="exponential")


@pytest.mark.parametrize("zeta", [0.0, -2.0])
def test_rational_jacobi_rejects_non_positive_zeta(zeta):
    with pytest.raises(ValueError, match="zeta"):
        rational_jacobi(_t(0.5), 1, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0), zeta=zeta)


# ---------------------------------------------------------------------------
# 6-7. Module constructor validation
# ---------------------------------------------------------------------------


def test_jacobi_rkan_rejects_unknown_mapping():
    with pytest.raises(ValueError, match="mapping"):
        JacobiRKAN(2, mapping="foo")


def test_pade_rkan_rejects_negative_denominator_eps():
    with pytest.raises(ValueError, match="denominator_eps"):
        PadeRKAN(2, 2, denominator_eps=-1e-3)


# ---------------------------------------------------------------------------
# 8. Denominator guard, negative side
# ---------------------------------------------------------------------------


def test_pade_rkan_denominator_guard_negative_side():
    eps = 1e-3
    layer = PadeRKAN(2, 2, denominator_eps=eps)
    with torch.no_grad():
        layer.w_p.copy_(torch.tensor([1.0, 0.0]))
        layer.w_q.copy_(torch.tensor([-1.0, -0.5]))  # q = -1 - 0.5*sigmoid(x) < 0
    x = torch.tensor([-2.0, 0.0, 2.0])
    y = layer(x)
    assert torch.all(torch.isfinite(y))
    s = torch.sigmoid(x)
    q = -1.0 - 0.5 * s
    assert torch.all(q < 0)
    torch.testing.assert_close(y, torch.ones(3) / (q - eps))


# ---------------------------------------------------------------------------
# 9. Vector alpha/beta matches per-element scalar calls
# ---------------------------------------------------------------------------


def test_jacobi_polynomial_vector_ab_matches_scalar_calls():
    xi = _t(-0.3, 0.2, 0.7)
    alpha = _t(0.0, 1.0, 0.5)
    beta = _t(0.0, 0.5, 1.5)
    out = jacobi_polynomial(xi, 3, alpha, beta)
    expected = torch.cat([jacobi_polynomial(xi[i : i + 1], 3, alpha[i], beta[i]) for i in range(3)])
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# 10-11. reset_parameters / repr contracts
# ---------------------------------------------------------------------------


def test_jacobi_rkan_reset_parameters_restores_ones():
    layer = JacobiRKAN(2)
    with torch.no_grad():
        layer.alpha.fill_(-3.0)
        layer.beta.fill_(5.0)
        layer.iota.fill_(-2.0)
    layer.reset_parameters()
    torch.testing.assert_close(layer.alpha, torch.ones(1))
    torch.testing.assert_close(layer.beta, torch.ones(1))
    torch.testing.assert_close(layer.iota, torch.ones(1))


def test_jacobi_rkan_repr_mentions_degree_and_mapping():
    text = repr(JacobiRKAN(4, mapping="logarithmic"))
    assert "degree=4" in text
    assert "mapping" in text and "logarithmic" in text


def test_pade_rkan_repr_mentions_degrees_and_eps():
    text = repr(PadeRKAN(2, 3))
    assert "degree_p=2, degree_q=3, denominator_eps=" in text


# ---------------------------------------------------------------------------
# 12. Logarithmic JacobiRKAN matches functional with normalized params
# ---------------------------------------------------------------------------


def test_jacobi_rkan_log_mapping_matches_functional_with_normalized_params():
    torch.manual_seed(1)
    layer = JacobiRKAN(3, mapping="logarithmic").double()
    x = torch.randn(3, 7, dtype=torch.float64)
    with torch.no_grad():
        expected = rational_jacobi(
            x,
            3,
            F.elu(layer.alpha.double(), 1.0),
            F.elu(layer.beta.double(), 1.0),
            F.softplus(layer.iota.double()),
            mapping="logarithmic",
        )
    torch.testing.assert_close(layer(x), expected)


# ---------------------------------------------------------------------------
# 13. Integer-dtype xi
# ---------------------------------------------------------------------------


def test_jacobi_polynomial_integer_xi_returns_default_float():
    xi = torch.tensor([-1, 0, 1])
    out = jacobi_polynomial(xi, 1, torch.tensor(0.0), torch.tensor(0.0))
    assert out.dtype == torch.get_default_dtype()
    # Legendre P1(xi) = xi.
    torch.testing.assert_close(out, torch.tensor([-1.0, 0.0, 1.0]))


# ---------------------------------------------------------------------------
# 14. Bool degrees rejected
# ---------------------------------------------------------------------------


def test_bool_degree_rejected_with_type_error():
    with pytest.raises(TypeError):
        jacobi_polynomial(_t(0.0), True, torch.tensor(0.0), torch.tensor(0.0))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        JacobiRKAN(True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 15. Fractional zeta on negative x stays finite
# ---------------------------------------------------------------------------


def test_rational_jacobi_fractional_zeta_negative_x_stays_finite():
    x = _t(-3.0, -1.0, -0.25, 0.0, 2.0)
    out = rational_jacobi(x, 2, torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.2), zeta=0.5)
    assert torch.all(torch.isfinite(out))
    assert not torch.any(torch.isnan(out))


# ---------------------------------------------------------------------------
# 16. linear_map with d1 < d0
# ---------------------------------------------------------------------------


def test_linear_map_rejects_inverted_interval():
    with pytest.raises(ValueError, match="d1"):
        linear_map(_t(0.0), d0=2.0, d1=1.0)


# ---------------------------------------------------------------------------
# 17. PadeRKAN constant denominator side
# ---------------------------------------------------------------------------


def test_pade_rkan_constant_denominator_matches_ratio():
    layer = PadeRKAN(2, 1, denominator_eps=0.0)
    with torch.no_grad():
        layer.w_p.copy_(torch.tensor([1.0, 3.0]))
        layer.w_q.copy_(torch.tensor([2.0]))
    x = torch.tensor([-1.0, 0.0, 1.0])
    s = torch.sigmoid(x)
    torch.testing.assert_close(layer(x), (1.0 + 3.0 * s) / layer.w_q[0])


# ---------------------------------------------------------------------------
# 18. Shape params stay in the open interval (-1, inf)
# ---------------------------------------------------------------------------


def test_shape_params_stay_above_minus_one_under_extreme_raw():
    layer = JacobiRKAN(3)
    with torch.no_grad():
        layer.alpha.fill_(-1e10)
        layer.beta.fill_(-1e10)
    y = layer(torch.tensor([-10.0, 0.0, 10.0]))
    assert torch.all(torch.isfinite(y))
    # The strict `> -1` invariant itself is only observable through the
    # private helper, hence the narrow import here (behavior is pinned above
    # through the public module surface).
    from memory_machines.models.blocks.rkan import _constrain_shape

    constrained = _constrain_shape(torch.tensor([-1e10, -100.0, 0.0, 5.0]))
    assert torch.all(constrained > -1.0)


def test_rkan_blocks_are_nn_modules():
    assert isinstance(JacobiRKAN(2), nn.Module)
    assert isinstance(PadeRKAN(2, 2), nn.Module)
