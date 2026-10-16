r"""Rational Kolmogorov-Arnold Network (rKAN) blocks.

Implements the two rational-basis approaches from Aghaei, "rKAN: Rational
Kolmogorov-Arnold Networks" (arXiv:2406.14495):

- :class:`JacobiRKAN`: a trainable rational Jacobi function used as an
  elementwise activation. The Jacobi shape parameters ``alpha``/``beta`` are
  constrained with ``ELU(..., kappa=1)`` so they stay in ``(-1, inf)`` (the
  paper's validity range), and the mapping scale ``iota`` is constrained with
  ``softplus`` so it stays positive.
- :class:`PadeRKAN`: a Pade-like rational activation ``P(x) / Q(x)`` where
  ``P`` and ``Q`` are weighted (shifted) Jacobi expansions over
  ``sigmoid(x)``. The fractional orders ``zeta`` are ``sigmoid``-constrained
  to ``(0, 1)``.

The Jacobi polynomials themselves are evaluated with the closed-form sum from
Section 2 of the paper (equivalent to the reference implementation's unrolled
expansions, but valid for any degree)::

    J_n(xi) = C * sum_m C(n, m) * Gamma(a+b+n+m+1) / Gamma(a+m+1) * ((xi-1)/2)^m

over the mapped input ``xi``. Domain mappings: a linear map for finite
intervals, and logarithmic/algebraic/exponential rational maps for
semi-infinite and infinite domains (paper eqs. 1-5).

As in the reference implementation, these modules are elementwise
activations: reproduce a paper architecture such as ``[1, 10, 1]`` by
composing them with :class:`~torch.nn.Linear` layers.

Deviations from the reference implementation
(https://github.com/alirezaafzalaghaei/rKAN), all documented here:

- Any degree ``>= 1`` is supported (the reference hardcodes expansions up to
  degree 6).
- ``PadeRKAN`` with ``degree == 1`` returns the constant term instead of
  raising ``IndexError`` (the reference indexes ``w[1]`` unconditionally).
- ``PadeRKAN`` guards the denominator with ``denominator_eps`` (default
  ``1e-6``); pass ``0.0`` for the exact ``p / q`` ratio.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "JacobiRKAN",
    "PadeRKAN",
    "algebraic_infinite_map",
    "algebraic_semi_infinite_map",
    "exponential_semi_infinite_map",
    "jacobi_basis",
    "jacobi_polynomial",
    "linear_map",
    "logarithmic_infinite_map",
    "logarithmic_semi_infinite_map",
    "rational_jacobi",
    "shifted_jacobi",
]


def _positive_scalar(value: object, name: str) -> torch.Tensor:
    """Validate that ``value`` is a positive scalar, returning it unchanged."""
    probe = value.detach() if isinstance(value, torch.Tensor) else value
    try:
        as_float = float(probe)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a positive scalar, got {value!r}") from None
    if not as_float > 0:
        raise ValueError(f"{name} ({as_float}) must be positive")
    return value  # type: ignore[return-value]


def _constrain_shape(raw: torch.Tensor) -> torch.Tensor:
    """Map a raw shape parameter into the valid Jacobi range ``alpha, beta > -1``.

    ``ELU(..., kappa=1)`` maps ``R -> (-1, inf)`` (paper Section 2), but in
    floating point it saturates at exactly ``-1.0`` for very negative inputs,
    which would make the ``Gamma`` arguments in :func:`jacobi_polynomial`
    non-positive. The clamp restores the open-interval invariant.
    """
    return F.elu(raw, 1.0).clamp_min(-1.0 + 1e-6)


def _validate_degree(value: object, name: str, minimum: int) -> int:
    """Validate an integer polynomial degree, returning it unchanged."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} ({value!r}) must be an int")
    if value < minimum:
        raise ValueError(f"{name} ({value}) must be a degree >= {minimum}")
    return value


def _signed_power(x: torch.Tensor, zeta: torch.Tensor | float) -> torch.Tensor:
    """Odd extension of ``x ** zeta`` that stays real for negative ``x``.

    Identical to ``x ** zeta`` for ``x >= 0`` (and for ``zeta == 1``
    everywhere); avoids ``NaN`` for fractional ``zeta`` on negative inputs.
    """
    return torch.sign(x) * torch.abs(x).pow(zeta)


def _require_non_negative(x: torch.Tensor, name: str) -> None:
    """Reject negative inputs for semi-infinite domain maps."""
    if bool(torch.any(torch.as_tensor(x) < 0)):
        raise ValueError(f"inputs to {name} must be non-negative (semi-infinite domain)")


def _fractional_order(zeta: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
    """Validate a fractional order as a tensor matching ``x``'s dtype/device."""
    zeta_t = torch.as_tensor(zeta, dtype=x.dtype if torch.is_floating_point(x) else None, device=x.device)
    if not bool(torch.all(zeta_t > 0)):
        raise ValueError("zeta (fractional order) must be positive")
    return zeta_t


def linear_map(x: torch.Tensor, d0: float, d1: float) -> torch.Tensor:
    r"""linear_map(x, d0, d1) -> Tensor

    Linear finite-domain map ``[d0, d1] -> [-1, 1]``: ``(2x - d0 - d1)/(d1 - d0)``.

    Examples::

        >>> linear_map(torch.tensor([2.0, 5.0, 8.0]), d0=2.0, d1=8.0)
        tensor([-1.,  0.,  1.])
    """
    if not d1 > d0:
        raise ValueError(f"d1 ({d1}) must be greater than d0 ({d0})")
    return (2.0 * x - d0 - d1) / (d1 - d0)


def algebraic_infinite_map(x: torch.Tensor, iota: torch.Tensor | float, zeta: torch.Tensor | float = 1.0) -> torch.Tensor:
    r"""algebraic_infinite_map(x, iota, zeta=1.0) -> Tensor

    Algebraic map ``(-inf, inf) -> (-1, 1)`` (paper eq. 5):
    ``x^zeta / sqrt(x^{2 zeta} + iota^2)`` with ``iota > 0``.

    Examples::

        >>> algebraic_infinite_map(torch.tensor([0.0]), 1.0)
        tensor([0.])
    """
    iota = _positive_scalar(iota, "iota")
    xz = _signed_power(x, zeta)
    if not isinstance(iota, torch.Tensor):
        iota = torch.as_tensor(iota, dtype=xz.dtype, device=xz.device)
    # hypot (not sqrt(xz^2 + iota^2)) so a denormal iota whose square
    # underflows still yields a nonzero hypotenuse.
    return xz / torch.hypot(xz, iota)


def logarithmic_infinite_map(x: torch.Tensor, iota: torch.Tensor | float) -> torch.Tensor:
    r"""logarithmic_infinite_map(x, iota) -> Tensor

    Logarithmic map ``(-inf, inf) -> (-1, 1)`` (paper eq. 4): ``tanh(x / iota)``.

    Examples::

        >>> logarithmic_infinite_map(torch.tensor([0.0]), 2.0)
        tensor([0.])
    """
    iota = _positive_scalar(iota, "iota")
    return torch.tanh(x / iota)


def logarithmic_semi_infinite_map(x: torch.Tensor, iota: torch.Tensor | float) -> torch.Tensor:
    r"""logarithmic_semi_infinite_map(x, iota) -> Tensor

    Logarithmic map ``(0, inf) -> (-1, 1)`` (paper eq. 1): ``2 tanh(x/iota) - 1``.

    Examples::

        >>> logarithmic_semi_infinite_map(torch.tensor([0.0, 100.0]), 1.0)
        tensor([-1.,  1.])
    """
    iota = _positive_scalar(iota, "iota")
    _require_non_negative(x, "logarithmic_semi_infinite_map")
    return 2.0 * torch.tanh(x / iota) - 1.0


def algebraic_semi_infinite_map(x: torch.Tensor, iota: torch.Tensor | float) -> torch.Tensor:
    r"""algebraic_semi_infinite_map(x, iota) -> Tensor

    Algebraic map ``(0, inf) -> (-1, 1)`` (paper eq. 2): ``(x - iota)/(x + iota)``.

    Examples::

        >>> algebraic_semi_infinite_map(torch.tensor([0.0, 1.0]), 1.0)
        tensor([-1.,  0.])
    """
    iota = _positive_scalar(iota, "iota")
    _require_non_negative(x, "algebraic_semi_infinite_map")
    return (x - iota) / (x + iota)


def exponential_semi_infinite_map(x: torch.Tensor, iota: torch.Tensor | float) -> torch.Tensor:
    r"""exponential_semi_infinite_map(x, iota) -> Tensor

    Exponential map ``(0, inf) -> (-1, 1)`` (paper eq. 3): ``1 - 2 exp(-x/iota)``.

    Examples::

        >>> exponential_semi_infinite_map(torch.tensor([0.0]), 1.0)
        tensor([-1.])
    """
    iota = _positive_scalar(iota, "iota")
    _require_non_negative(x, "exponential_semi_infinite_map")
    return 1.0 - 2.0 * torch.exp(-x / iota)


def jacobi_polynomial(xi: torch.Tensor, n: int, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    r"""jacobi_polynomial(xi, n, alpha, beta) -> Tensor

    Jacobi polynomial ``J_n(xi)`` of order ``n >= 0`` via the paper's
    closed-form sum (Section 2). ``alpha``/``beta`` must satisfy ``> -1`` and
    broadcast against ``xi``; gradients flow to all tensor inputs.

    Examples::

        >>> jacobi_polynomial(torch.tensor([-1.0, 0.0, 1.0]), 1, torch.tensor(0.0), torch.tensor(0.0))
        tensor([-1.,  0.,  1.])
    """
    _validate_degree(n, "degree", minimum=0)
    if n == 0:
        return torch.ones_like(xi)
    float_dtype = xi.dtype if torch.is_floating_point(xi) else torch.get_default_dtype()
    m = torch.arange(n + 1, dtype=float_dtype, device=xi.device)
    a, b = torch.broadcast_tensors(alpha, beta)
    keep = (1,) * len(a.shape) + (n + 1,)
    aa = a.reshape(a.shape + (1,))
    bb = b.reshape(b.shape + (1,))
    mm = m.reshape(keep)
    log_comb = math.lgamma(n + 1) - torch.lgamma(mm + 1) - torch.lgamma(n - mm + 1)
    log_pref = torch.lgamma(aa + n + 1) - math.lgamma(n + 1) - torch.lgamma(aa + bb + n + 1)
    log_num = torch.lgamma(aa + bb + n + mm + 1) - torch.lgamma(aa + mm + 1)
    coeff = torch.exp(log_pref + log_comb + log_num)
    z = (xi - 1.0) / 2.0
    powers = z.unsqueeze(-1).pow(mm)
    return (coeff * powers).sum(-1)


def jacobi_basis(xi: torch.Tensor, degree: int, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    r"""jacobi_basis(xi, degree, alpha, beta) -> Tensor

    Stacked Jacobi orders ``J_0..J_degree`` with shape ``(*xi.shape, degree + 1)``.

    Examples::

        >>> jacobi_basis(torch.tensor([0.0]), 1, torch.tensor(0.0), torch.tensor(0.0)).shape
        torch.Size([1, 2])
    """
    _validate_degree(degree, "degree", minimum=0)
    return torch.stack([jacobi_polynomial(xi, k, alpha, beta) for k in range(degree + 1)], dim=-1)


def shifted_jacobi(
    x: torch.Tensor,
    n: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    zeta: torch.Tensor | float = 1.0,
    a: float = 0.0,
    b: float = 1.0,
) -> torch.Tensor:
    r"""shifted_jacobi(x, n, alpha, beta, zeta=1.0, a=0.0, b=1.0) -> Tensor

    Jacobi polynomial evaluated at the finite-interval map
    ``xi = (2 x^zeta - a - b) / (b - a)``. With the defaults
    (``zeta=1, a=0, b=1``) this maps ``sigmoid`` outputs in ``[0, 1]`` to
    ``[-1, 1]``, as used by :class:`PadeRKAN`.
    """
    _validate_degree(n, "degree", minimum=0)
    if not b > a:
        raise ValueError(f"b ({b}) must be greater than a ({a})")
    zeta_t = _fractional_order(zeta, x)
    xi = (2.0 * _signed_power(x, zeta_t) - a - b) / (b - a)
    return jacobi_polynomial(xi, n, alpha, beta)


def rational_jacobi(
    x: torch.Tensor,
    n: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    iota: torch.Tensor | float,
    zeta: torch.Tensor | float = 1.0,
    mapping: str = "algebraic",
) -> torch.Tensor:
    r"""rational_jacobi(x, n, alpha, beta, iota, zeta=1.0, mapping="algebraic") -> Tensor

    Jacobi polynomial evaluated at a rational map of ``x`` into ``(-1, 1)``:
    ``"algebraic"`` (paper eq. 5, the reference implementation's choice) or
    ``"logarithmic"`` (paper eq. 4). Used by :class:`JacobiRKAN`.
    """
    _validate_degree(n, "degree", minimum=0)
    if mapping not in ("algebraic", "logarithmic"):
        raise ValueError(f"mapping ({mapping!r}) must be 'algebraic' or 'logarithmic'")
    iota = _positive_scalar(iota, "iota")
    if mapping == "algebraic":
        zeta_t = _fractional_order(zeta, x)
        xz = _signed_power(x, zeta_t)
        other = iota if isinstance(iota, torch.Tensor) else torch.as_tensor(iota, dtype=xz.dtype, device=xz.device)
        u = xz / torch.hypot(xz, other)
    else:
        u = torch.tanh(x / iota)
    return jacobi_polynomial(u, n, alpha, beta)


class JacobiRKAN(nn.Module):
    r"""JacobiRKAN(degree, mapping="algebraic")

    Trainable rational Jacobi activation (paper Section 3.2): applies a single
    rational Jacobi function ``J_degree(phi(x))`` elementwise, where ``phi``
    is the ``"algebraic"`` (eq. 5, reference default) or ``"logarithmic"``
    (eq. 4) infinite-domain map. ``alpha``/``beta`` are ELU-constrained to
    ``(-1, inf)`` and ``iota`` is softplus-constrained to ``(0, inf)``.

    Shape:
        - Input: :math:`(*)`, any shape.
        - Output: same shape as input.

    Args:
        degree (int): Jacobi order; must be a positive int.
        mapping (str): ``"algebraic"`` or ``"logarithmic"``.

    Examples::

        >>> import torch
        >>> layer = JacobiRKAN(3)
        >>> layer(torch.randn(2, 5)).shape
        torch.Size([2, 5])
    """

    def __init__(self, degree: int, mapping: str = "algebraic") -> None:
        super().__init__()
        _validate_degree(degree, "degree", minimum=1)
        if mapping not in ("algebraic", "logarithmic"):
            raise ValueError(f"mapping ({mapping!r}) must be 'algebraic' or 'logarithmic'")
        self.degree = degree
        self.mapping = mapping
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.iota = nn.Parameter(torch.ones(1))

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Restore the reference initialization: ``alpha = beta = iota = 1``.
        """
        with torch.no_grad():
            self.alpha.fill_(1.0)
            self.beta.fill_(1.0)
            self.iota.fill_(1.0)

    def extra_repr(self) -> str:
        return f"degree={self.degree}, mapping={self.mapping!r}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x) -> Tensor

        Apply the rational Jacobi function elementwise.

        Args:
            x (Tensor): input of any shape.

        Returns:
            Tensor: output of the same shape as ``x``.
        """
        alpha = _constrain_shape(self.alpha)
        beta = _constrain_shape(self.beta)
        iota = F.softplus(self.iota)
        return rational_jacobi(x, self.degree, alpha, beta, iota, mapping=self.mapping)


class PadeRKAN(nn.Module):
    r"""PadeRKAN(degree_p, degree_q, denominator_eps=1e-6)

    Pade-like rational activation (paper Section 3.1): ``P(x) / Q(x)`` where
    ``P``/``Q`` are weighted shifted-Jacobi expansions over ``sigmoid(x)``::

        P(x) = w_0 + w_1 s + sum_{d=2}^{p-1} w_d J_d(2 s^zeta_p - 1)

    with ``s = sigmoid(x)``, trainable weights ``w``, ELU-constrained Jacobi
    shapes ``alpha``/``beta``, and sigmoid-constrained fractional orders
    ``zeta`` in ``(0, 1)``. A ``degree`` of 1 degrades to the constant term.

    Shape:
        - Input: :math:`(*)`, any shape.
        - Output: same shape as input.

    Args:
        degree_p (int): number of numerator weights; must be a positive int.
        degree_q (int): number of denominator weights; must be a positive int.
        denominator_eps (float): ``|Q|`` is clamped away from zero by this
            amount to avoid ``inf``/``NaN``; ``0.0`` recovers exact ``P / Q``.

    Examples::

        >>> import torch
        >>> layer = PadeRKAN(2, 6)
        >>> layer(torch.randn(2, 5)).shape
        torch.Size([2, 5])
    """

    def __init__(self, degree_p: int, degree_q: int, denominator_eps: float = 1e-6) -> None:
        super().__init__()
        _validate_degree(degree_p, "degree_p", minimum=1)
        _validate_degree(degree_q, "degree_q", minimum=1)
        if not denominator_eps >= 0:
            raise ValueError(f"denominator_eps ({denominator_eps}) must be non-negative")
        self.degree_p = degree_p
        self.degree_q = degree_q
        self.denominator_eps = float(denominator_eps)
        self.alpha_p = nn.Parameter(torch.ones(1))
        self.beta_p = nn.Parameter(torch.ones(1))
        self.zeta_p = nn.Parameter(torch.zeros(1))
        self.w_p = nn.Parameter(torch.ones(degree_p))
        self.alpha_q = nn.Parameter(torch.ones(1))
        self.beta_q = nn.Parameter(torch.ones(1))
        self.zeta_q = nn.Parameter(torch.zeros(1))
        self.w_q = nn.Parameter(torch.ones(degree_q))

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Restore the reference initialization: shape parameters to ones,
        fractional orders to zeros, and weights to ones.
        """
        with torch.no_grad():
            for name in ("alpha_p", "beta_p", "alpha_q", "beta_q"):
                getattr(self, name).fill_(1.0)
            for name in ("zeta_p", "zeta_q"):
                getattr(self, name).zero_()
            self.w_p.fill_(1.0)
            self.w_q.fill_(1.0)

    def extra_repr(self) -> str:
        return f"degree_p={self.degree_p}, degree_q={self.degree_q}, denominator_eps={self.denominator_eps}"

    def _poly(
        self,
        s: torch.Tensor,
        w: torch.Tensor,
        degree: int,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        zeta: torch.Tensor,
    ) -> torch.Tensor:
        if degree == 1:
            return w[0] + torch.zeros_like(s)
        out = w[0] + w[1] * s
        for d in range(2, degree):
            out = out + w[d] * shifted_jacobi(s, d, alpha, beta, zeta, 0.0, 1.0)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x) -> Tensor

        Apply the Pade rational function elementwise.

        Args:
            x (Tensor): input of any shape.

        Returns:
            Tensor: output of the same shape as ``x``.
        """
        s = torch.sigmoid(x)
        p = self._poly(s, self.w_p, self.degree_p, _constrain_shape(self.alpha_p), _constrain_shape(self.beta_p), torch.sigmoid(self.zeta_p))
        q = self._poly(s, self.w_q, self.degree_q, _constrain_shape(self.alpha_q), _constrain_shape(self.beta_q), torch.sigmoid(self.zeta_q))
        if self.denominator_eps > 0:
            q = torch.where(q >= 0, q + self.denominator_eps, q - self.denominator_eps)
        return p / q
