"""Pre-norm / FFN registries and the :class:`DenseFFN` wrapper.

Covers ``build_norm`` kind dispatch and epsilon handling, the ``x +
ffn(norm(x))`` residual identity against raw-op references, the depth-scaled
residual output init, ``build_ffn``'s ``classic_mlp`` kind and width override,
and the odd-width guard on ``SwiGLU``.
"""

import math
from dataclasses import replace
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from memory_machines.models.blocks.dyt import DyT
from memory_machines.models.blocks.ffn import (
    ClassicMLP,
    DenseFFN,
    SwiGLU,
    build_ffn,
    build_norm,
)
from tests.conftest import make_inputs, make_tiny_cfg

H = 32


def _inner_swiglu(ffn: DenseFFN) -> SwiGLU:
    """The wrapped SwiGLU (cast for the type checker)."""
    assert isinstance(ffn.ffn, SwiGLU)
    return cast("SwiGLU", ffn.ffn)


def _dense_ffn_reference(ffn: DenseFFN, x: torch.Tensor) -> torch.Tensor:
    """Raw-op ``x + SwiGLU(norm(x))`` using the module's own weights."""
    norm = ffn.norm
    if isinstance(norm, DyT):
        hn = torch.tanh(x * norm.saturation + norm.shift) * norm.layer_scale
    elif isinstance(norm, nn.RMSNorm):
        hn = F.rms_norm(x, (x.shape[-1],), norm.weight, norm.eps)
    else:
        assert isinstance(norm, nn.LayerNorm)
        hn = F.layer_norm(x, (x.shape[-1],), norm.weight, norm.bias, norm.eps)
    inner = _inner_swiglu(ffn)
    return x + F.linear(F.silu(F.linear(hn, inner.gate_proj.weight)) * F.linear(hn, inner.up_proj.weight), inner.down_proj.weight)


# --- build_norm --------------------------------------------------------------
@pytest.mark.parametrize(
    "kind,expected",
    [("rmsnorm", nn.RMSNorm), ("layernorm", nn.LayerNorm), ("dyT", DyT)],
    ids=["rmsnorm", "layernorm", "dyT"],
)
def test_build_norm_dispatches_on_kind(kind: str, expected: type) -> None:
    assert type(build_norm(kind, 16)) is expected


@pytest.mark.parametrize("kind", ["rmsnorm", "layernorm"], ids=["rmsnorm", "layernorm"])
def test_build_norm_honors_explicit_eps(kind: str) -> None:
    assert build_norm(kind, 16, eps=1e-5).eps == 1e-5


def test_build_norm_rmsnorm_default_eps_matches_class_default() -> None:
    assert build_norm("rmsnorm", 16).eps == nn.RMSNorm(16).eps


def test_build_norm_dyt_accepts_epsilon() -> None:
    assert isinstance(build_norm("dyT", 16, eps=1e-5), DyT)


def test_build_norm_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown normalization kind"):
        build_norm("bogus", 16)


# --- DenseFFN pre-norm -------------------------------------------------------
def test_dense_ffn_default_pre_norm_is_dyt() -> None:
    assert isinstance(DenseFFN(make_tiny_cfg()).norm, DyT)


@pytest.mark.parametrize("norm_kind,expected", [("rmsnorm", nn.RMSNorm), ("layernorm", nn.LayerNorm)], ids=["rmsnorm", "layernorm"])
def test_dense_ffn_norm_kind_selects_pre_norm(norm_kind: str, expected: type) -> None:
    assert type(DenseFFN(make_tiny_cfg(), norm_kind=norm_kind).norm) is expected


@pytest.mark.parametrize("norm_kind", ["dyT", "rmsnorm", "layernorm"], ids=["dyT", "rmsnorm", "layernorm"])
def test_dense_ffn_forward_matches_raw_reference(norm_kind: str) -> None:
    torch.manual_seed(0)
    ffn = DenseFFN(make_tiny_cfg(), norm_kind=norm_kind).eval()
    with torch.no_grad():
        _inner_swiglu(ffn).down_proj.weight.normal_(std=0.5)
    x = make_inputs(B=2, T=5, D=H)
    with torch.no_grad():
        out = ffn(x)
        assert (out - x).abs().amax().item() > 1e-4  # non-trivial residual signal
        torch.testing.assert_close(out, _dense_ffn_reference(ffn, x), atol=1e-6, rtol=1e-5)


# --- residual init -----------------------------------------------------------
@pytest.mark.parametrize("kind", ["swiglu", "classic_mlp"], ids=["swiglu", "classic_mlp"])
def test_dense_ffn_down_proj_is_depth_scaled(kind: str) -> None:
    torch.manual_seed(0)
    cfg = replace(make_tiny_cfg(), residual_init_scale=True)
    inner = cast("SwiGLU | ClassicMLP", DenseFFN(cfg, kind=kind).ffn)
    down = inner.down_proj.weight
    expected = cfg.initializer_range / math.sqrt(2 * cfg.num_layers)
    assert down.std().item() == pytest.approx(expected, rel=0.25)


# --- build_ffn / ClassicMLP --------------------------------------------------
def test_build_ffn_classic_mlp_returns_classic_mlp() -> None:
    assert type(build_ffn(make_tiny_cfg(), "classic_mlp")) is ClassicMLP


@pytest.mark.parametrize("explicit,expected", [(None, 64), (48, 48)], ids=["falls_back_to_config", "explicit_overrides"])
def test_build_ffn_intermediate_size(explicit: int | None, expected: int) -> None:
    assert build_ffn(make_tiny_cfg(), "classic_mlp", intermediate_size=explicit).intermediate_size == expected


def test_classic_mlp_forward_matches_raw_reference() -> None:
    torch.manual_seed(0)
    mlp = cast("ClassicMLP", build_ffn(make_tiny_cfg(), "classic_mlp"))
    with torch.no_grad():
        mlp.up_proj.weight.normal_(std=0.5)
        mlp.down_proj.weight.normal_(std=0.5)
    x = make_inputs(B=2, T=5, D=H)
    with torch.no_grad():
        expected = F.linear(F.silu(F.linear(x, mlp.up_proj.weight)), mlp.down_proj.weight)
        torch.testing.assert_close(mlp(x), expected, atol=1e-6, rtol=1e-6)


def test_swiglu_rejects_odd_intermediate_size() -> None:
    with pytest.raises(ValueError, match="SwiGLU needs even intermediate_size"):
        SwiGLU(make_tiny_cfg(), intermediate_size=63)
