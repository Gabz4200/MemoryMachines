"""Dynamic Tanh normalization (:class:`~memory_machines.models.blocks.dyt.DyT`).

Pins the raw-torch equivalence of ``forward`` (``tanh(x * saturation + shift) *
layer_scale`` over the trailing axis), broadcasting over leading axes, the
neutral ``reset_parameters`` configuration, and gradient flow to all three
learned parameters.
"""

import torch

from memory_machines.models.blocks.dyt import DyT

H = 8


def _dyt_reference(norm: DyT, x: torch.Tensor) -> torch.Tensor:
    """Raw-op reference reading the module's own parameters."""
    return torch.tanh(x * norm.saturation + norm.shift) * norm.layer_scale


def _perturb(norm: DyT) -> None:
    """Give saturation/shift/layer_scale non-trivial values (arrange helper)."""
    with torch.no_grad():
        norm.saturation.normal_(std=1.0)
        norm.shift.normal_(std=1.0)
        norm.layer_scale.normal_(std=1.0)


def test_forward_matches_raw_tanh_reference() -> None:
    torch.manual_seed(0)
    norm = DyT(H)
    _perturb(norm)
    x = torch.randn(2, 5, H)
    torch.testing.assert_close(norm(x), _dyt_reference(norm, x), atol=1e-6, rtol=1e-6)


def test_forward_broadcasts_over_leading_dimensions() -> None:
    torch.manual_seed(0)
    norm = DyT(H)
    _perturb(norm)
    x = torch.randn(2, 3, 5, H)
    out = norm(x)
    assert tuple(out.shape) == (2, 3, 5, H)
    torch.testing.assert_close(out, _dyt_reference(norm, x), atol=1e-6, rtol=1e-6)


def test_reset_parameters_restores_neutral_configuration() -> None:
    torch.manual_seed(0)
    norm = DyT(H)
    _perturb(norm)
    x = torch.randn(2, 5, H)
    norm.reset_parameters()
    torch.testing.assert_close(norm(x), DyT(H)(x), atol=0.0, rtol=0.0)


def test_gradients_reach_all_three_parameters() -> None:
    torch.manual_seed(0)
    norm = DyT(H)
    x = torch.randn(2, 5, H)
    norm(x).sum().backward()
    for name, param in norm.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum().item() > 0.0, name
