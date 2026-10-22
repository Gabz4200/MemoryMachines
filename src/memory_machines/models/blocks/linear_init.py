r"""Initialize and inspect a projection that may be dense or factored.

The helpers take ``nn.Linear`` and :class:`~memory_machines.models.blocks.linears.FactoredLinear`
alike and put the scale where it belongs: for a factored layer the product
``W = B A`` is what the rest of the block sees, so initialization variance-matches
that product rather than either factor.
"""

import math

import torch
from torch import nn

from .linears import FactoredLinear

__all__ = [
    "copy_linear_weight_",
    "effective_weight",
    "factorize_linear",
    "init_linear",
    "init_linear_columns",
    "init_linear_xavier",
    "linear_dtype",
    "linear_in_features",
    "linear_out_features",
    "zero_linear",
]


def linear_in_features(linear: nn.Module) -> int:
    r"""linear_in_features(linear) -> int

    Input width of a dense or factored projection.

    Examples::

        >>> linear_in_features(nn.Linear(8, 6))
        8
        >>> linear_in_features(FactoredLinear(8, 2, 6))
        8
    """
    if isinstance(linear, FactoredLinear):
        return linear.input_dim
    if isinstance(linear, nn.Linear):
        return linear.in_features
    raise TypeError(f"expected nn.Linear or FactoredLinear, got {type(linear).__name__}")


def linear_out_features(linear: nn.Module) -> int:
    r"""linear_out_features(linear) -> int

    Output width of a dense or factored projection.

    Examples::

        >>> linear_out_features(nn.Linear(8, 6))
        6
        >>> linear_out_features(FactoredLinear(8, 2, 6))
        6
    """
    if isinstance(linear, FactoredLinear):
        return linear.output_dim
    if isinstance(linear, nn.Linear):
        return linear.out_features
    raise TypeError(f"expected nn.Linear or FactoredLinear, got {type(linear).__name__}")


def linear_dtype(linear: nn.Module) -> torch.dtype:
    r"""linear_dtype(linear) -> torch.dtype

    Parameter dtype of a dense or factored projection, without materializing
    the effective matrix.

    Args:
        linear (nn.Module): a dense or factored projection.

    Returns:
        torch.dtype: dtype of the stored weights.

    Examples::

        >>> linear_dtype(FactoredLinear(8, 2, 6)) is torch.float32
        True
    """
    if isinstance(linear, FactoredLinear):
        return linear.in_linear.weight.dtype
    if isinstance(linear, nn.Linear):
        return linear.weight.dtype
    raise TypeError(f"expected nn.Linear or FactoredLinear, got {type(linear).__name__}")


def effective_weight(linear: nn.Module) -> torch.Tensor:
    r"""effective_weight(linear) -> Tensor

    Materialize the matrix a projection applies: the stored weight for
    :class:`~torch.nn.Linear`, ``W_out @ W_in`` for :class:`FactoredLinear`.
    Intended for inspection and tests, not for the forward path.

    Args:
        linear (nn.Module): a dense or factored projection.

    Returns:
        Tensor: matrix of shape ``(out_features, in_features)``.

    Examples::

        >>> effective_weight(FactoredLinear(8, 2, 6)).shape
        torch.Size([6, 8])
    """
    if isinstance(linear, FactoredLinear):
        return linear.out_linear.weight @ linear.in_linear.weight
    if isinstance(linear, nn.Linear):
        return linear.weight
    raise TypeError(f"expected nn.Linear or FactoredLinear, got {type(linear).__name__}")


def init_linear(linear: nn.Module, std: float, *, truncated: bool = True) -> None:
    r"""init_linear(linear, std, truncated=True) -> None

    Initialize a dense or factored projection so the map it applies has
    element standard deviation ``std``.

    For :class:`FactoredLinear` the factors are variance-matched instead of
    using a single matrix draw: with ``Var(A) = 1/in_features`` and
    ``Var(B) = std^2 * in_features / rank`` the product ``B A`` has variance
    ``std^2``, so swapping a dense projection for a factored one preserves the
    activation scale the rest of the block was tuned for. Only the input factor
    is clipped when ``truncated``; clipping both would compound the shrinkage
    and leave the product below the dense draw it replaces. ``std == 0`` zeroes
    the map (the neutral-start case) while leaving the input factor usable, so
    it still receives gradient once the output factor moves. Bias is zeroed.

    Args:
        linear (nn.Module): a dense or factored projection.
        std (float): target standard deviation of the effective weights.
        truncated (bool, optional): draw from a truncated normal clipped to
            ``[-2 std, 2 std]``. Default: ``True``.

    Raises:
        ValueError: if ``std`` is negative.
    """
    if std < 0:
        raise ValueError(f"std ({std}) must be non-negative")
    in_dim = linear_in_features(linear)
    if isinstance(linear, nn.Linear):
        if std == 0:
            nn.init.zeros_(linear.weight)
        elif truncated:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        else:
            nn.init.normal_(linear.weight, mean=0.0, std=std)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)
        return
    assert isinstance(linear, FactoredLinear)
    out_std = std * math.sqrt(in_dim / linear.rank) if std > 0 else 0.0
    # Only the input factor is clipped. Clipping both would multiply the two
    # truncation shrinkages (0.88 * 0.88), leaving the product ~12% below the
    # dense draw this replaces; one clipped factor matches it exactly.
    for factor, scale, clip in (
        (linear.in_linear, 1 / math.sqrt(in_dim), truncated),
        (linear.out_linear, out_std, False),
    ):
        if scale == 0:
            nn.init.zeros_(factor.weight)
        elif clip:
            nn.init.trunc_normal_(factor.weight, mean=0.0, std=scale, a=-2 * scale, b=2 * scale)
        else:
            nn.init.normal_(factor.weight, mean=0.0, std=scale)
    if linear.out_linear.bias is not None:
        nn.init.zeros_(linear.out_linear.bias)


def init_linear_columns(linear: nn.Module, column_stds: torch.Tensor) -> None:
    r"""init_linear_columns(linear, column_stds) -> None

    Initialize a projection per input column, so each column of the effective
    weight gets its own standard deviation.

    Used where a single global scale cannot describe the matrix because it is
    built from blocks with different scales, such as the parallel-hybrid
    output projection whose attention half is standard-init and whose memory
    half starts near zero. Nothing about the matrix is truncated, so the split
    survives exactly, unlike an SVD of the mixed matrix.

    For a factored projection the scale goes entirely into the input factor
    (``A[:, j] ~ N(0, column_stds[j]^2)``) and the output factor gets
    ``N(0, 1/rank)``, so ``Var(W[:, j]) = column_stds[j]^2``.

    Args:
        linear (nn.Module): a dense or factored projection.
        column_stds (Tensor): target standard deviation of each input column,
            shape ``(in_features,)``.

    Raises:
        ValueError: if ``column_stds`` has the wrong length.
    """
    in_dim = linear_in_features(linear)
    if tuple(column_stds.shape) != (in_dim,):
        raise ValueError(f"expected column_stds shape ({in_dim},), got {tuple(column_stds.shape)}")
    if isinstance(linear, nn.Linear):
        noise = torch.randn_like(linear.weight)
        with torch.no_grad():
            linear.weight.copy_(noise * column_stds.unsqueeze(0).to(linear.weight.dtype))
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)
        return
    assert isinstance(linear, FactoredLinear)
    noise = torch.randn(linear.in_linear.weight.shape, dtype=linear.in_linear.weight.dtype, device=linear.in_linear.weight.device)
    with torch.no_grad():
        linear.in_linear.weight.copy_(noise * column_stds.to(linear.in_linear.weight.dtype).unsqueeze(0))
    nn.init.normal_(linear.out_linear.weight, mean=0.0, std=1 / math.sqrt(linear.rank))
    if linear.out_linear.bias is not None:
        nn.init.zeros_(linear.out_linear.bias)


def init_linear_xavier(linear: nn.Module, gain: float) -> None:
    r"""init_linear_xavier(linear, gain) -> None

    Xavier-uniform equivalent for a dense or factored projection: the effective
    map gets variance ``gain^2 * 2 / (in_features + out_features)``.

    Args:
        linear (nn.Module): a dense or factored projection.
        gain (float): Xavier gain.
    """
    if isinstance(linear, nn.Linear):
        nn.init.xavier_uniform_(linear.weight, gain=gain)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)
        return
    std = gain * math.sqrt(2.0 / (linear_in_features(linear) + linear_out_features(linear)))
    init_linear(linear, std, truncated=False)


def zero_linear(linear: nn.Module) -> None:
    r"""zero_linear(linear) -> None

    Zero the map, including its bias, for a dense or factored projection.

    Args:
        linear (nn.Module): a dense or factored projection.
    """
    if isinstance(linear, FactoredLinear):
        nn.init.zeros_(linear.out_linear.weight)
        if linear.out_linear.bias is not None:
            nn.init.zeros_(linear.out_linear.bias)
        return
    assert isinstance(linear, nn.Linear)
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def copy_linear_weight_(linear: nn.Module, weight: torch.Tensor) -> None:
    r"""copy_linear_weight_(linear, weight) -> None

    Load a full dense matrix into a dense or factored projection.

    For :class:`FactoredLinear` the matrix is truncated to its leading ``rank``
    singular triplets, i.e. the minimum-Frobenius-norm rank-``rank``
    approximation (Eckart-Young). Used where a projection starts from a
    structured matrix the statistics of a single global standard deviation
    cannot reproduce, such as the two-scale parallel-hybrid output projection.

    Args:
        linear (nn.Module): a dense or factored projection.
        weight (Tensor): matrix of shape ``(out_features, in_features)``.

    Raises:
        ValueError: if ``weight`` has the wrong shape.
    """
    expected = (linear_out_features(linear), linear_in_features(linear))
    if tuple(weight.shape) != expected:
        raise ValueError(f"expected weight shape {expected}, got {tuple(weight.shape)}")
    if isinstance(linear, nn.Linear):
        with torch.no_grad():
            linear.weight.copy_(weight)
        return
    assert isinstance(linear, FactoredLinear)
    u, s, vh = torch.linalg.svd(weight.detach().to(torch.float32), full_matrices=False)
    rank = linear.rank
    root = s[:rank].clamp_min(0).sqrt()
    with torch.no_grad():
        linear.in_linear.weight.copy_((root.unsqueeze(1) * vh[:rank]).to(linear.in_linear.weight.dtype))
        linear.out_linear.weight.copy_((u[:, :rank] * root.unsqueeze(0)).to(linear.out_linear.weight.dtype))


def factorize_linear(linear: nn.Linear, rank: int, *, preserve_columns: bool = False) -> FactoredLinear:
    r"""factorize_linear(linear, rank, preserve_columns=False) -> FactoredLinear

    Replace a dense projection with a rank-``rank`` factorization of it.

    By default the new factors reproduce the dense weight's single element
    standard deviation, which is what a from-scratch initialization needs.
    ``preserve_columns=True`` reproduces each input column's own standard
    deviation instead, preserving structured init layouts (block-wise scales,
    zero blocks) without the tail loss of an SVD of the mixed matrix.

    Args:
        linear (nn.Linear): dense projection to replace.
        rank (int): bottleneck width.
        preserve_columns (bool, optional): per-column init instead of one
            global scale. Default: ``False``.

    Returns:
        FactoredLinear: the replacement module.
    """
    factored = FactoredLinear(linear.in_features, rank, linear.out_features, bias=linear.bias is not None)
    # Meta-device construction (HF ``low_cpu_mem_usage``) has no values to match:
    # the placeholder parameters are replaced by the checkpoint before any forward.
    if not linear.weight.is_meta:
        if preserve_columns:
            init_linear_columns(factored, linear.weight.detach().to(torch.float32).std(dim=0))
        else:
            init_linear(factored, linear.weight.detach().to(torch.float32).std().item())
        if linear.bias is not None:
            assert factored.out_linear.bias is not None
            with torch.no_grad():
                factored.out_linear.bias.copy_(linear.bias)
    factored.train(linear.training)
    return factored
