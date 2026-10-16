r"""Dense feed-forward sub-layers.

:class:`SwiGLU` is the gated dense feed-forward block and
:class:`ClassicMLP` the ungated SiLU variant; :class:`DenseFFN` wraps either
(or any other registered kind) with a pre-norm residual so a token mixer plus
one :class:`DenseFFN` forms a full transformer layer. Kinds are looked up by
name through :func:`build_ffn` and the pre-norm through :func:`build_norm`,
which are the single seams for swapping the dense FFN of the CED encoder or
decoder without touching their wiring.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from memory_machines.utils.config_schema import MemoryMachineConfig

from .dyt import DyT
from .shared import residual_init_scale


class SwiGLU(nn.Module):
    r"""SwiGLU(config, intermediate_size=None)

    SwiGLU gated dense feed-forward block.

    .. math::
        \mathrm{SwiGLU}(x) = W_{down}(\mathrm{SiLU}(W_{gate}x) \odot W_{up}x)

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``hidden_size``, ``intermediate_size``, and
            ``initializer_range``.
        intermediate_size (int, optional): explicit FFN hidden width,
            overriding ``config.intermediate_size`` (used when the CED
            encoder and decoder are given independent widths). Default:
            ``None``.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> mlp = SwiGLU(MemoryMachineConfig(hidden_size=32, intermediate_size=64))
        >>> mlp(torch.randn(2, 5, 32)).shape
        torch.Size([2, 5, 32])
    """

    def __init__(self, config: MemoryMachineConfig, intermediate_size: int | None = None):
        super().__init__()
        resolved = intermediate_size if intermediate_size is not None else config.intermediate_size
        if resolved % 2 != 0:
            raise ValueError(f"SwiGLU needs even intermediate_size, got {resolved}")
        self.hidden_size = config.hidden_size
        self.intermediate_size = resolved
        self.initializer_range: float = config.initializer_range
        self.num_layers: int = config.num_layers
        self.residual_init_scale: bool = config.residual_init_scale

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        SwiGLU.reset_parameters(self)

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Gate/up projections use truncated normal
        ``N(0, initializer_range^2)`` (DeepSeek-style,
        ``initializer_range = 0.006``). The down projection
        is either zero-initialized (identity residual start,
        Fixup/ReZero) or depth-scaled to
        ``initializer_range / sqrt(2N)`` (Megatron-LM/Qwen
        residual scaling) per ``config.residual_init_scale``.
        The SiLU-gated product has roughly half the variance
        of a plain linear path, so a variance-matched input
        init plus a small/zero-start output keeps early-step
        activations bounded.
        """
        std = self.initializer_range
        scale = residual_init_scale(std, self.num_layers, self.residual_init_scale)
        nn.init.trunc_normal_(self.gate_proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.trunc_normal_(self.up_proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        if scale is None:
            nn.init.zeros_(self.down_proj.weight)
        else:
            nn.init.trunc_normal_(self.down_proj.weight, mean=0.0, std=scale, a=-2 * scale, b=2 * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x) -> Tensor

        Apply the gated feed-forward transformation.

        Args:
            x (Tensor): input of shape ``(B, L, hidden)``.

        Returns:
            Tensor: output with the same shape ``(B, L, hidden)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> SwiGLU(MemoryMachineConfig(hidden_size=16, intermediate_size=32))(torch.randn(1, 4, 16)).shape
            torch.Size([1, 4, 16])
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        swish_gate = F.silu(gate)
        return self.down_proj(swish_gate * up)


class ClassicMLP(nn.Module):
    r"""ClassicMLP(config, intermediate_size=None)

    Ungated two-matrix dense feed-forward block.

    .. math::
        \mathrm{ClassicMLP}(x) = W_{down}\,\mathrm{SiLU}(W_{up}x)

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``hidden_size``, ``intermediate_size``, and
            ``initializer_range``.
        intermediate_size (int, optional): explicit FFN hidden width,
            overriding ``config.intermediate_size``. Default: ``None``.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> mlp = ClassicMLP(MemoryMachineConfig(hidden_size=32, intermediate_size=64))
        >>> mlp(torch.randn(2, 5, 32)).shape
        torch.Size([2, 5, 32])
    """

    def __init__(self, config: MemoryMachineConfig, intermediate_size: int | None = None):
        super().__init__()
        resolved = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.hidden_size = config.hidden_size
        self.intermediate_size = resolved
        self.initializer_range: float = config.initializer_range
        self.num_layers: int = config.num_layers
        self.residual_init_scale: bool = config.residual_init_scale

        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        ClassicMLP.reset_parameters(self)

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Up projection uses truncated normal ``N(0, initializer_range^2)``; the
        down projection is zero- or depth-scaled per
        ``config.residual_init_scale`` (see :meth:`SwiGLU.reset_parameters`).
        """
        std = self.initializer_range
        scale = residual_init_scale(std, self.num_layers, self.residual_init_scale)
        nn.init.trunc_normal_(self.up_proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        if scale is None:
            nn.init.zeros_(self.down_proj.weight)
        else:
            nn.init.trunc_normal_(self.down_proj.weight, mean=0.0, std=scale, a=-2 * scale, b=2 * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x) -> Tensor — apply the SiLU feed-forward transformation.

        Args:
            x (Tensor): input of shape ``(B, L, hidden)``.

        Returns:
            Tensor: output with the same shape ``(B, L, hidden)``.
        """
        return self.down_proj(F.silu(self.up_proj(x)))


# One entry per dense FFN kind; every kind takes (config, intermediate_size=...)
# and defines ``reset_parameters``.
_FFN_REGISTRY: dict[str, Callable[..., nn.Module]] = {"swiglu": SwiGLU, "classic_mlp": ClassicMLP}


def build_ffn(config: MemoryMachineConfig, kind: str, intermediate_size: int | None = None) -> nn.Module:
    r"""build_ffn(config, kind, intermediate_size=None) -> nn.Module

    Construct a dense FFN of the named ``kind``.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            passed through to the FFN constructor.
        kind (str): registered FFN kind (currently ``"swiglu"`` or
            ``"classic_mlp"``).
        intermediate_size (int, optional): explicit hidden width; ``None``
            lets the FFN fall back to ``config.intermediate_size``.

    Returns:
        nn.Module: the constructed FFN.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> build_ffn(MemoryMachineConfig(hidden_size=32, intermediate_size=64), "swiglu").intermediate_size
        64
    """
    if kind not in _FFN_REGISTRY:
        raise ValueError(f"unknown FFN kind {kind!r}; known kinds: {sorted(_FFN_REGISTRY)}")
    return _FFN_REGISTRY[kind](config, intermediate_size=intermediate_size)


_NORMALIZATION_REGISTRY: dict[str, Callable[..., nn.Module]] = {"rmsnorm": nn.RMSNorm, "layernorm": nn.LayerNorm, "dyT": DyT}


def build_norm(kind: str, hidden_size: int, eps: float | None = None) -> nn.Module:
    r"""build_norm(kind, hidden_size, eps=None) -> nn.Module

    Construct a normalization layer of the named ``kind``.

    Args:
        kind (str): registered normalization kind (currently ``"rmsnorm"``,
            ``"layernorm"``, or ``"dyT"``).
        hidden_size (int): trailing dimension size for the normalization.
        eps (float, optional): epsilon for numerical stability; if ``None``,
            the default for the selected kind is used.
    Returns:
        nn.Module: the constructed normalization layer.

    Examples::

        >>> build_norm("rmsnorm", 32).weight.shape
        torch.Size([32])
        >>> build_norm("layernorm", 32).weight.shape
        torch.Size([32])
        >>> build_norm("dyT", 32).saturation.shape
        torch.Size([1, 32])
    """
    if kind not in _NORMALIZATION_REGISTRY:
        raise ValueError(f"unknown normalization kind {kind!r}; known kinds: {sorted(_NORMALIZATION_REGISTRY)}")
    # DyT takes no epsilon; the other kinds accept it.
    if eps is None or kind == "dyT":
        return _NORMALIZATION_REGISTRY[kind](hidden_size)
    return _NORMALIZATION_REGISTRY[kind](hidden_size, eps=eps)


class DenseFFN(nn.Module):
    r"""DenseFFN(config, kind="swiglu", norm_kind="dyT", intermediate_size=None)

    Dense (non-MoE) feed-forward sub-layer with a pre-norm residual, applied
    after each CED *encoder* and *decoder* layer.

    .. math::
        \mathrm{DenseFFN}(x) = x + \mathrm{FFN}(\mathrm{norm}(x))

    The inner FFN is selected by ``kind`` (:func:`build_ffn`) and the pre-norm
    by ``norm_kind`` (:func:`build_norm`); the encoder and decoder each own
    their own instance, so they may use different kinds and widths. A single
    weight-shared FFN is sufficient because the block is pointwise across
    positions and — on the mHC path — shared across the residual streams.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``hidden_size``, ``intermediate_size``, and
            ``rms_norm_eps``.
        kind (str, optional): dense FFN kind (see :func:`build_ffn`).
            Default: ``"swiglu"``.
        norm_kind (str, optional): pre-norm kind (see :func:`build_norm`).
            Default: ``"dyT"``.
        intermediate_size (int, optional): explicit FFN hidden width;
            ``None`` falls back to ``config.intermediate_size``. Default:
            ``None``.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> ffn = DenseFFN(MemoryMachineConfig(hidden_size=32, intermediate_size=64))
        >>> ffn(torch.randn(2, 5, 32)).shape
        torch.Size([2, 5, 32])
    """

    def __init__(self, config: MemoryMachineConfig, kind: str = "swiglu", norm_kind: str = "dyT", intermediate_size: int | None = None):
        super().__init__()
        self.kind = kind
        self.norm = build_norm(norm_kind, config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = build_ffn(config, kind, intermediate_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Reset the pre-norm to its neutral configuration and re-apply the
        wrapped FFN init, so a freshly built FFN starts as an identity residual
        branch (zero-start or depth-scaled output projection).
        """
        # Registry kinds define reset_parameters, but ``nn.Module`` does not
        # declare it, so resolve dynamically instead of asserting on the type.
        for sublayer in (self.norm, self.ffn):
            reset = getattr(sublayer, "reset_parameters", None)
            if callable(reset):
                reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x) -> Tensor

        Apply the pre-norm dense FFN residual.

        Args:
            x (Tensor): input ``(..., hidden)``; positions and, on the mHC
                path, residual streams are both accepted (the norm and the
                projections act on the trailing hidden axis).

        Returns:
            Tensor: output with the same shape as ``x``.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> ffn = DenseFFN(MemoryMachineConfig(hidden_size=16, intermediate_size=32))
            >>> ffn(torch.randn(1, 4, 16)).shape
            torch.Size([1, 4, 16])
        """
        return x + self.ffn(self.norm(x))
