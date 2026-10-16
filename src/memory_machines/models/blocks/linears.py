r"""Cheap linear ablations: low-rank factorization and chunked bottleneck mixing.

:class:`FactoredLinear` is a standard low-rank factorization
:math:`W \approx W_{out} W_{in}`. :class:`ChunkedBottleneckLinear` is a more
aggressive structured alternative: each input chunk is pooled to a single
scalar, the scalars are mixed through a small dense bottleneck, and each mixed
scalar is broadcast back out to an output chunk.
"""

import math
import warnings

import torch
from torch import nn

__all__ = ["ChunkedBottleneckLinear", "FactoredLinear"]


class FactoredLinear(nn.Module):
    r"""FactoredLinear(input_dim, rank, output_dim, bias=True)

    Low-rank factorization of a linear map: :math:`y = W_{out}(W_{in} x) + b`,
    so the effective matrix :math:`W = W_{out} W_{in}` has maximum rank
    ``rank``.

    Both factors are standard :class:`~torch.nn.Linear` layers, so all
    computation runs as dense GEMM kernels and leading dimensions are handled
    automatically. See :class:`~memory_machines.models.blocks.ChunkedBottleneckLinear`
    for a cheaper structured alternative with a stronger inductive bias.

    Shape:
        - Input: :math:`(*, \text{input\_dim})`, where :math:`*` means any number of
          leading dimensions.
        - Output: :math:`(*, \text{output\_dim})`.

    Args:
        input_dim (int): input feature dimension; must be positive.
        rank (int): bottleneck width; must be positive.
        output_dim (int): output feature dimension; must be positive.
        bias (bool, optional): per-output bias. Default: ``True``.

    .. warning::
        A ``rank`` larger than ``min(input_dim, output_dim)`` costs more
        parameters than a dense :class:`~torch.nn.Linear` with no expressivity
        gain. A warning is issued in that case.

    Examples::

        >>> import torch
        >>> layer = FactoredLinear(8, rank=4, output_dim=6)
        >>> layer(torch.randn(2, 5, 8)).shape
        torch.Size([2, 5, 6])
    """

    def __init__(self, input_dim: int, rank: int, output_dim: int, bias: bool = True) -> None:
        super().__init__()
        for name, value in (("input_dim", input_dim), ("rank", rank), ("output_dim", output_dim)):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} ({value!r}) must be a positive int")
        if rank > min(input_dim, output_dim):
            warnings.warn(
                f"rank ({rank}) exceeds min(input_dim, output_dim) ({min(input_dim, output_dim)}); a dense nn.Linear would use fewer parameters",
                stacklevel=2,
            )
        self.input_dim = input_dim
        self.rank = rank
        self.output_dim = output_dim
        self.in_linear = nn.Linear(in_features=input_dim, out_features=rank, bias=False)
        self.out_linear = nn.Linear(in_features=rank, out_features=output_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Variance-matched truncated normal (Glorot & Bengio, 2010):
        with ``Var(A) = 1/fan\_in`` and ``Var(B) = 1/rank``, the
        effective map ``W = BA`` preserves input variance, since
        ``Var(y) = fan\_in * rank * Var(A) * Var(B) * Var(x) =
        Var(x)``. Weights are drawn from a truncated normal clipped
        to ``[-2s, 2s]`` so no single factor starts with an outlier
        scale. Bias is zeroed. This contrasts with LoRA (Hu et al.,
        2021), where ``B = 0`` is correct for a residual *adapter*
        but would kill the signal of this standalone map.
        """
        in_std = 1 / math.sqrt(self.input_dim)
        out_std = 1 / math.sqrt(self.rank)
        nn.init.trunc_normal_(self.in_linear.weight, mean=0.0, std=in_std, a=-2 * in_std, b=2 * in_std)
        nn.init.trunc_normal_(self.out_linear.weight, mean=0.0, std=out_std, a=-2 * out_std, b=2 * out_std)
        if self.out_linear.bias is not None:
            nn.init.zeros_(self.out_linear.bias)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, rank={self.rank}, output_dim={self.output_dim}, bias={self.out_linear.bias is not None}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x) -> Tensor

        Apply the factored map to the last dimension.

        Args:
            x (Tensor): input of shape :math:`(*, \text{input\_dim})`.

        Returns:
            Tensor: output of shape :math:`(*, \text{output\_dim})`.
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected last dim {self.input_dim}, got {x.shape[-1]}")
        return self.out_linear(self.in_linear(x))


class ChunkedBottleneckLinear(nn.Module):
    r"""ChunkedBottleneckLinear(input_dim, num_blocks, output_dim, bias=True, bottleneck_dim=None)

    Structured linear map with :math:`O(D_{in} + D_{out} + R^2)` parameters:
    split the input into ``num_blocks`` chunks, pool each chunk to one scalar
    with its own weight vector, mix the scalars through a small dense
    bottleneck, then broadcast each mixed scalar back to an output chunk.

    .. math::
        s_g = w^{down}_g \cdot x_g, \quad h = M(s), \quad y_g = h_g w^{up}_g

    .. note::
        Cross-chunk interaction happens only in the bottleneck ``M``, so this
        layer is less expressive than
        :class:`~memory_machines.models.blocks.FactoredLinear` of the same width
        but far cheaper when ``num_blocks`` is large.

    Shape:
        - Input: :math:`(*, \text{input\_dim})`, where :math:`*` means any number of
          leading dimensions, and ``input_dim`` is divisible by ``num_blocks``.
        - Output: :math:`(*, \text{output\_dim})`, where ``output_dim`` is divisible
          by ``num_blocks``.

    Args:
        input_dim (int): input features; must be divisible by ``num_blocks``.
        num_blocks (int): number of independent chunks ``G``; must be positive.
        output_dim (int): output features; must be divisible by ``num_blocks``.
        bias (bool, optional): independent per-output bias of shape
            :math:`(\text{output\_dim})`. Default: ``True``.
        bottleneck_dim (int, optional): width ``R`` of the dense mixer. If ``None``,
            a single ``GxG`` mixer is used; otherwise a two-stage ``G -> R -> G``
            mixer. Must be positive. Default: ``None``.

    Examples::

        >>> import torch
        >>> layer = ChunkedBottleneckLinear(8, num_blocks=2, output_dim=6)
        >>> layer(torch.randn(2, 5, 8)).shape
        torch.Size([2, 5, 6])
    """

    def __init__(
        self,
        input_dim: int,
        num_blocks: int,
        output_dim: int,
        bias: bool = True,
        bottleneck_dim: int | None = None,
    ) -> None:
        super().__init__()
        for name, value in (("input_dim", input_dim), ("num_blocks", num_blocks), ("output_dim", output_dim)):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} ({value!r}) must be a positive int")
        if input_dim % num_blocks != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by num_blocks ({num_blocks})")
        if output_dim % num_blocks != 0:
            raise ValueError(f"output_dim ({output_dim}) must be divisible by num_blocks ({num_blocks})")
        bottleneck = num_blocks if bottleneck_dim is None else bottleneck_dim
        if not isinstance(bottleneck, int) or bottleneck <= 0:
            raise ValueError(f"bottleneck_dim ({bottleneck_dim!r}) must be a positive int")

        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.output_dim = output_dim
        self.bottleneck_dim = bottleneck
        self.down_chunk = input_dim // num_blocks
        self.up_chunk = output_dim // num_blocks

        self.down_weight = nn.Parameter(torch.empty(num_blocks, self.down_chunk))
        self.up_weight = nn.Parameter(torch.empty(num_blocks, self.up_chunk))
        if bottleneck == num_blocks:
            self.mixer = nn.Linear(num_blocks, num_blocks, bias=False)
        else:
            self.mixer = nn.Sequential(nn.Linear(num_blocks, bottleneck, bias=False), nn.Linear(bottleneck, num_blocks, bias=False))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Down-projection uses a LeCun truncated normal with std
        ``1/sqrt(chunk)`` so each pooled scalar preserves input
        variance; the up-projection uses the matching
        ``1/sqrt(up_chunk)`` scale for symmetry-breaking broadcast
        weights; a square mixer is set to the identity (an
        orthogonal, norm-preserving map per Saxe et al., 2013) so
        chunks pass through unbiased at init, while a two-stage
        mixer uses ``1/sqrt(num_blocks)`` truncated normal; bias is
        zeroed.
        """
        down_std = 1 / math.sqrt(self.down_chunk)
        up_std = 1 / math.sqrt(self.up_chunk)
        nn.init.trunc_normal_(self.down_weight, mean=0.0, std=down_std, a=-2 * down_std, b=2 * down_std)
        nn.init.trunc_normal_(self.up_weight, mean=0.0, std=up_std, a=-2 * up_std, b=2 * up_std)
        if isinstance(self.mixer, nn.Linear):
            nn.init.eye_(self.mixer.weight)
        else:
            # Two-stage G->R->G: first factor 1/sqrt(G) (fan-in G), second
            # 1/sqrt(R) (fan-in R) so product variance is 1/G (variance-matched).
            in_std = 1 / math.sqrt(self.num_blocks)
            out_std = 1 / math.sqrt(self.bottleneck_dim)
            mods = [m for m in self.mixer if isinstance(m, nn.Linear)]
            nn.init.trunc_normal_(mods[0].weight, mean=0.0, std=in_std, a=-2 * in_std, b=2 * in_std)
            nn.init.trunc_normal_(mods[1].weight, mean=0.0, std=out_std, a=-2 * out_std, b=2 * out_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, num_blocks={self.num_blocks}, output_dim={self.output_dim}, bias={self.bias is not None}, bottleneck_dim={self.bottleneck_dim}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x) -> Tensor

        Pool each input chunk to a scalar, mix, and broadcast back out.

        Args:
            x (Tensor): input of shape :math:`(*, \text{input\_dim})`.

        Returns:
            Tensor: output of shape :math:`(*, \text{output\_dim})`.
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected last dim {self.input_dim}, got {x.shape[-1]}")
        out_leading = x.shape[:-1]
        x_chunked = x.reshape(-1, self.num_blocks, self.down_chunk)
        x_down = torch.einsum("bgc,gc->bg", x_chunked, self.down_weight)
        x_mixed = self.mixer(x_down)
        x_up = torch.einsum("bg,gc->bgc", x_mixed, self.up_weight)

        x_out = x_up.reshape(-1, self.output_dim)
        if self.bias is not None:
            x_out = x_out + self.bias
        return x_out.reshape(*out_leading, self.output_dim)
