r"""ShortConvolution(config, ...)

Depthwise causal Conv1d block with optional activation and state caching,
mirroring the ``fla.modules.causal_conv1d.ShortConvolution`` API subset
needed here. The compute runs through the pure-PyTorch reference oracle
until the compiled kernel lands (see ``memory_machines.ops.causal_conv1d``).

Weight layout is ``nn.Conv1d``-native ``(D, 1, W)`` so a future kernel can
``rearrange(weight, "d 1 w -> d w")`` exactly like FLA.
"""

import torch
from torch import nn

from memory_machines.ops.references.causal_conv1d_reference import (
    reference_causal_conv1d_fwd,
    reference_causal_conv1d_update,
    reference_causal_conv1d_update_states,
)

_ACTIVATIONS = (None, "silu", "swish")


class ShortConvolution(nn.Conv1d):
    r"""ShortConvolution(hidden_size, kernel_size, ...)

    Depthwise causal convolution with optional activation and cache.

    Args:
        hidden_size (int): number of input/output channels.
        kernel_size (int): convolution width.
        bias (bool): whether to learn a channel bias. Default: ``False``.
        activation (str | None): ``"silu"``/``"swish"`` or ``None``.
            Default: ``"silu"``.
        backend (str): only ``"naive"`` until the kernel lands; any other
            value raises. Default: ``"naive"``.

    Examples::

        >>> import torch
        >>> from memory_machines.models.blocks.causal_conv1d import ShortConvolution
        >>> conv = ShortConvolution(hidden_size=8, kernel_size=3)
        >>> y, _ = conv(torch.randn(2, 5, 8))
        >>> y.shape
        torch.Size([2, 5, 8])
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = "silu",
        backend: str = "naive",
    ) -> None:
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=0,
            groups=hidden_size,
            bias=bias,
        )
        self.hidden_size = hidden_size
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {_ACTIVATIONS}, got {activation!r}")
        self.activation = activation
        if backend != "naive":
            raise ValueError(f"backend {backend!r} has no kernel yet; use backend='naive'")
        self.backend = backend

    def extra_repr(self) -> str:
        s = super().extra_repr()
        return f"{s}, activation={self.activation}, backend={self.backend}"

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size[0]

    def _weight_dw(self) -> torch.Tensor:
        return self.weight.reshape(self.hidden_size, self.kernel_size[0])

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""forward(x, ...) -> (y, final_state)

        Args:
            x: ``[B, T, D]`` dense, or packed ``[1, L, D]``/``[L, D]`` with
                ``cu_seqlens``.
            residual: optional ``[B, T, D]`` added to the activated output.
            mask: optional ``[B, T]`` multiplicative mask for padded positions.
            cache: optional ``[N, D, W]`` input history (slot 0 is padding,
                matching FLA). Single-token no-grad calls update it in place.
            output_final_state: whether to return the ``[N, D, W]`` state.
            cu_seqlens: optional ``[N+1]`` cumulative lengths for packed input.

        Returns:
            ``(y, final_state)`` with ``y`` shaped like ``x`` and
            ``final_state`` ``None`` unless requested (or a decode cache).
        """
        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError("`mask` and `cu_seqlens` cannot be provided at the same time")
            x = x * mask.unsqueeze(-1).to(x.dtype)

        n = x.shape[0] if cu_seqlens is None else int(len(cu_seqlens) - 1)
        # Single-token-per-sequence inference (grad off) always takes the
        # decode path; step() creates the cache when None, like FLA.
        # The dim()==3 gate keeps 2D [L, D] inputs on the grad-carrying path.
        if x.dim() == 3 and not torch.is_grad_enabled() and x.numel() // self.hidden_size == n:
            return self.step(x, residual, cache, output_final_state, cu_seqlens)

        y = reference_causal_conv1d_fwd(
            x=x,
            weight=self._weight_dw(),
            bias=self.bias,
            residual=residual,
            initial_state=cache,
            activation=self.activation,
            cu_seqlens=cu_seqlens,
        )
        final_state = None
        if output_final_state:
            final_state = reference_causal_conv1d_update_states(
                x=x,
                state_len=self.kernel_size[0],
                initial_state=cache,
                cu_seqlens=cu_seqlens,
            )
        return y, final_state

    @torch.no_grad()
    def step(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cache: torch.Tensor | None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""step(x, residual, cache, ...) -> (y, cache')

        Single-token decode: exactly one token per sequence. ``cache`` is
        created when ``None`` and updated in place.
        """
        n = x.shape[0] if cu_seqlens is None else int(len(cu_seqlens) - 1)
        if cache is None:
            cache = x.new_zeros(n, self.hidden_size, self.kernel_size[0])
        y, cache = reference_causal_conv1d_update(
            x=x,
            cache=cache,
            weight=self._weight_dw(),
            bias=self.bias,
            residual=residual,
            activation=self.activation,
        )
        return y, cache
