r"""Dynamic Tanh normalization.

:class:`DyT` replaces the statistics of RMSNorm/LayerNorm with a learnable
tanh saturation, ``tanh(saturation * x + shift) * layer_scale``. It is wired
into :class:`~memory_machines.models.blocks.ffn.DenseFFN` through
:func:`~memory_machines.models.blocks.ffn.build_norm`.
"""

import torch
from torch import nn


class DyT(nn.Module):
    r"""DyT(hidden_dim) — elementwise dynamic Tanh normalization.

    Args:
        hidden_dim (int): size of the (trailing) normalized axis.

    Examples::

        >>> from memory_machines.models.blocks.dyt import DyT
        >>> DyT(32)(torch.randn(2, 5, 32)).shape
        torch.Size([2, 5, 32])
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.saturation = nn.Parameter(torch.ones(1, hidden_dim))
        self.layer_scale = nn.Parameter(torch.ones(1, hidden_dim))
        self.shift = nn.Parameter(torch.zeros(1, hidden_dim))

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None — unit gain and zero shift."""
        nn.init.ones_(self.saturation)
        nn.init.ones_(self.layer_scale)
        nn.init.zeros_(self.shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x) -> Tensor — saturate, shift, then scale the trailing axis.

        Args:
            x (Tensor): input ``(..., hidden_dim)``.

        Returns:
            Tensor: output with the same shape as ``x``.
        """
        return torch.tanh(x * self.saturation + self.shift) * self.layer_scale
