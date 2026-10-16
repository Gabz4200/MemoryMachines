"""Hope / ContinuumMemorySystem (Sec. 7-8) paper-parity tests.

Each test ties to a paper equation. Deterministic (fixed seeds), tiny dims,
independent references with raw torch ops. CMS levels are
:class:`~memory_machines.models.blocks.ffn.DenseFFN` sub-layers (default
SwiGLU FFN behind a DyT pre-norm), so the raw references mirror that kind and
normalization.
"""

import math
from dataclasses import replace
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from memory_machines.models.blocks.dyt import DyT
from memory_machines.models.blocks.fast_weights import (
    ContinuumMemorySystem,
    HopeAttentionBlock,
    HopeBlock,
)
from memory_machines.models.blocks.ffn import ClassicMLP, DenseFFN, SwiGLU
from tests.test_blocks_shapes import tiny_cfg

D = 16


def _cfg(**kw):
    """Tiny config at width ``D`` with 2 attention heads plus overrides."""
    base = {"hidden_size": D, "num_attention_heads": 2, "num_key_value_heads": 2}
    base.update(kw)
    return replace(tiny_cfg(), **base)


def _levels(cms: ContinuumMemorySystem) -> list[DenseFFN]:
    """The DenseFFN levels (cast for the type checker)."""
    return cast("list[DenseFFN]", list(cms.levels))


def _inner(level: DenseFFN) -> SwiGLU:
    """The wrapped SwiGLU of a default CMS level (cast for the type checker)."""
    assert isinstance(level.ffn, SwiGLU)
    return cast("SwiGLU", level.ffn)


def _dyt(level: DenseFFN) -> DyT:
    """The DyT pre-norm of a default CMS level (cast for the type checker)."""
    assert isinstance(level.norm, DyT)
    return cast("DyT", level.norm)


def _level_branch(level: DenseFFN, h: torch.Tensor) -> torch.Tensor:
    """Raw-op SwiGLU branch of one default CMS level, pre-normed by DyT."""
    n, inner = _dyt(level), _inner(level)
    hn = torch.tanh(h * n.saturation + n.shift) * n.layer_scale
    return F.linear(F.silu(F.linear(hn, inner.gate_proj.weight)) * F.linear(hn, inner.up_proj.weight), inner.down_proj.weight)


def _cms_raw(cms, x, order=None):
    """Hand-rolled Eq.70 chain with plain ops (no cms.forward/level calls)."""
    h = x.clone()
    lvls = list(range(cms.num_levels)) if order is None else list(order)
    for l in lvls:
        h = h + _level_branch(_levels(cms)[l], h)
    return h


def _amplify(cms, std=0.5):
    """Give the levels a non-trivial residual magnitude for signal tests.

    The default (depth-scaled) FFN output init is tiny, so magnitude-sensitive
    assertions restore signal the way ``tests/test_ced_ffn.py`` does.
    """
    with torch.no_grad():
        for level in _levels(cms):
            inner = _inner(level)
            for w in (inner.gate_proj.weight, inner.up_proj.weight, inner.down_proj.weight):
                w.normal_(std=std)


def test_cms_levels_are_dense_ffns():
    """CMS builds one DenseFFN per level; FFN kind and norm come from config."""
    cms = ContinuumMemorySystem(_cfg())
    assert all(isinstance(level, DenseFFN) and isinstance(_inner(level), SwiGLU) and isinstance(_dyt(level), DyT) for level in _levels(cms))
    alt = ContinuumMemorySystem(_cfg(cms_ffn="classic_mlp", cms_norm="layernorm"))
    assert all(isinstance(level.ffn, ClassicMLP) and isinstance(level.norm, nn.LayerNorm) for level in _levels(alt))


def test_cms_equals_manual_chain_eq70():
    """Eq.70: y = FFN_fk(...FFN_f1(x)) with per-level pre-norm + residual."""
    torch.manual_seed(0)
    cms = ContinuumMemorySystem(_cfg(), num_levels=3).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 4, D)
    with torch.no_grad():
        torch.testing.assert_close(cms(x), _cms_raw(cms, x), atol=1e-6, rtol=1e-5)


def test_cms_is_sequential_not_aggregation():
    """Levels compose sequentially; a mean-aggregation over levels differs."""
    torch.manual_seed(0)
    cms = ContinuumMemorySystem(_cfg(), num_levels=3).eval()
    _amplify(cms)
    torch.manual_seed(1)
    x = torch.randn(1, 4, D)
    with torch.no_grad():
        out = cms(x)
        # Independent mean-aggregation ref: mean_l(level_l(x)) + x.
        agg = x + torch.stack([_level_branch(_levels(cms)[l], x) for l in range(cms.num_levels)]).mean(0)
        assert (out - agg).abs().amax().item() > 1e-4
        # Order matters for a chain: reversed level order gives another output.
        rev = _cms_raw(cms, x, order=reversed(range(cms.num_levels)))
        assert (out - rev).abs().amax().item() > 1e-4


def test_hope_is_titan_then_cms_eq94_97():
    """Eq.94-97: HopeBlock = SelfRefTitan(x) then CMS chain, shared weights."""
    torch.manual_seed(0)
    hope = HopeBlock(_cfg(), num_heads=2, memory_depth=1, num_levels=2, use_short_conv=False).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 4, D)
    with torch.no_grad():
        o, _ = hope.titan(x)
        ref = hope.cms(o)
        y, _ = hope(x)
        torch.testing.assert_close(y, ref, atol=1e-6, rtol=1e-5)


def test_hope_block_num_heads_default_from_config():
    """HopeBlock derives width and head count from the config when unspecified."""
    hope = HopeBlock(_cfg())
    assert hope.titan.num_heads == 2
    assert hope.titan.hidden_size == D
    assert hope.cms.hidden_size == D


def _hope_attn():
    torch.manual_seed(0)
    return HopeAttentionBlock(_cfg(), num_heads=2, num_levels=2).eval()


def test_hope_attention_causality_all_positions():
    """Causal attention: future perturbation leaves the strict past bit-exact."""
    blk = _hope_attn()
    torch.manual_seed(1)
    x = torch.randn(1, 5, D)
    with torch.no_grad():
        base = blk(x)
        for fut in range(1, 5):
            x2 = x.clone()
            x2[:, fut] += 10.0
            delta = blk(x2) - base
            torch.testing.assert_close(delta[:, :fut], torch.zeros_like(delta[:, :fut]), atol=0.0, rtol=0.0)
            assert delta[:, fut:].abs().amax().item() > 0.0


def test_attention_mixes_past():
    """Perturbing position 0 changes later outputs (a per-position map would not)."""
    blk = _hope_attn()
    torch.manual_seed(1)
    x = torch.randn(1, 5, D)
    with torch.no_grad():
        base = blk(x)
        x2 = x.clone()
        x2[:, 0] += 5.0
        delta = (blk(x2) - base).abs().amax(-1)[0]
    assert (delta[1:] > 1e-6).all()


def _attn_only_ref(blk, x):
    """Hand-rolled causal attention + o_proj + residual with raw ops (no CMS)."""
    b, t, _ = x.shape
    n = blk.attn_norm
    hn = F.rms_norm(x, (blk.hidden_size,), n.weight.detach().clone(), eps=n.eps)
    q = F.linear(hn, blk.q_proj.weight.detach().clone()).view(b, t, blk.num_heads, blk.head_dim).transpose(1, 2)
    k = F.linear(hn, blk.k_proj.weight.detach().clone()).view(b, t, blk.num_heads, blk.head_dim).transpose(1, 2)
    v = F.linear(hn, blk.v_proj.weight.detach().clone()).view(b, t, blk.num_heads, blk.head_dim).transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(blk.head_dim)
    mask = torch.ones(t, t, dtype=torch.bool).triu(1)
    scores = scores.masked_fill(mask, float("-inf"))
    a = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(b, t, blk.hidden_size)
    return F.linear(a, blk.o_proj.weight.detach().clone()) + x


def test_cms_contributes_in_attention_variant():
    """Zeroed CMS FFNs reduce the block to attention-only; default init differs."""
    torch.manual_seed(1)
    x = torch.randn(1, 4, D)
    blk = _hope_attn()
    _amplify(blk.cms)
    with torch.no_grad():
        ref = _attn_only_ref(blk, x)
        assert (blk(x) - ref).abs().amax().item() > 1e-6  # CMS active: differs
        for level in _levels(blk.cms):
            _inner(level).down_proj.weight.zero_()
        torch.testing.assert_close(blk(x), ref, atol=1e-6, rtol=1e-5)


def test_single_level_cms_matches_transformer_mlp():
    """num_levels=1: single pre-norm + FFN + residual."""
    torch.manual_seed(0)
    cms = ContinuumMemorySystem(_cfg(), num_levels=1).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 4, D)
    with torch.no_grad():
        torch.testing.assert_close(cms(x), _cms_raw(cms, x), atol=1e-6, rtol=1e-5)


def test_should_update_schedule_eq71():
    """Eq.71: level l updates iff step % C^(l) == 0; step 0 always updates."""
    cms = ContinuumMemorySystem(_cfg(), num_levels=3, update_intervals=(1, 2, 4))
    period = math.lcm(1, 2, 4)
    for step in range(2 * period):
        for l, c in enumerate((1, 2, 4)):
            assert cms.should_update(l, step) == (step % c == 0)
    assert all(cms.should_update(l, 0) for l in range(3))


def test_cms_validation_errors():
    """Bad levels/steps/intervals/expansion and non-divisible heads fail loudly."""
    cms = ContinuumMemorySystem(_cfg(), num_levels=2)
    with pytest.raises(ValueError, match="level"):
        cms.should_update(-1, 0)
    with pytest.raises(ValueError, match="level"):
        cms.should_update(2, 0)
    with pytest.raises(ValueError, match="non-negative"):
        cms.should_update(0, -1)
    with pytest.raises(ValueError, match="positive"):
        ContinuumMemorySystem(_cfg(), num_levels=2, update_intervals=(1,))
    with pytest.raises(ValueError, match="positive"):
        ContinuumMemorySystem(_cfg(), num_levels=2, update_intervals=(1, 0))
    with pytest.raises(ValueError, match="positive"):
        ContinuumMemorySystem(_cfg(), num_levels=2, update_intervals=(1, -2))
    with pytest.raises(ValueError, match="positive"):
        ContinuumMemorySystem(_cfg(), num_levels=2, expansion=0.0)
    with pytest.raises(ValueError, match="divisible"):
        HopeAttentionBlock(_cfg(hidden_size=30, csa_num_output_groups=2), num_heads=4)


def _ffn_src(cfg, hidden, val):
    src = SwiGLU(cfg, intermediate_size=hidden)
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(val)
    return src


def test_load_pretrained_copies_and_skips():
    """Matching names+shapes copy over (bare FFN and DenseFFN); mismatches skip."""
    cfg = _cfg()
    cms = ContinuumMemorySystem(cfg, num_levels=2, expansion=2.0)
    hidden = round(D * 2.0)
    dense_src = DenseFFN(cfg, "swiglu", "dyT", intermediate_size=hidden)
    srcs = [_ffn_src(cfg, hidden, 0.5), dense_src]
    with torch.no_grad():
        for p in _inner(dense_src).parameters():
            p.fill_(-0.25)
    cms.load_pretrained_into_levels(srcs)
    for level, src in zip(_levels(cms), srcs):
        src_params = dict(src.named_parameters())
        for name, p in _inner(level).named_parameters():
            donor = src_params.get(name)
            if donor is None:
                donor = src_params.get(f"ffn.{name}")  # DenseFFN source: skip its "ffn." prefix
            assert donor is not None and donor.shape == p.shape
            torch.testing.assert_close(p.detach(), donor.detach(), atol=0.0, rtol=0.0)
    # Wrong-shape source: nothing copied, dst unchanged.
    before = [p.detach().clone() for p in _inner(_levels(cms)[0]).parameters()]
    bad = SwiGLU(cfg, intermediate_size=hidden + 2)
    cms.load_pretrained_into_levels([bad, bad])
    for old, p in zip(before, _inner(_levels(cms)[0]).parameters()):
        torch.testing.assert_close(p.detach(), old, atol=0.0, rtol=0.0)
    with pytest.raises(ValueError, match="expected 2 modules"):
        cms.load_pretrained_into_levels([srcs[0]])


def test_invalid_shapes_raise():
    """Non-[B,T,D] inputs fail loudly on all three modules."""
    torch.manual_seed(0)
    cms = ContinuumMemorySystem(_cfg(), num_levels=2).eval()
    hope = HopeBlock(_cfg(), num_heads=2, memory_depth=1, num_levels=2, use_short_conv=False).eval()
    attn = HopeAttentionBlock(_cfg(), num_heads=2, num_levels=2).eval()
    for bad in (torch.randn(1, D), torch.randn(1, 4, D - 1), torch.randn(1, 4, D + 1)):
        with pytest.raises(ValueError, match=r"\[B, T, D="):
            cms(bad)
        with pytest.raises(ValueError, match=r"\[B, T, D="):
            hope(bad)
        with pytest.raises(ValueError, match=r"\[B, T, D="):
            attn(bad)


def test_default_intervals_powers_of_two():
    """Default C^(l) = 2^l."""
    cms = ContinuumMemorySystem(_cfg(), num_levels=4)
    assert cms.update_intervals == (1, 2, 4, 8)


def test_zeroed_cms_is_identity():
    """Zeroed CMS FFN outputs reduce the chain to the identity."""
    torch.manual_seed(0)
    cms = ContinuumMemorySystem(_cfg(), num_levels=2).eval()
    with torch.no_grad():
        for level in _levels(cms):
            _inner(level).down_proj.weight.zero_()
    x = torch.randn(1, 4, D)
    torch.testing.assert_close(cms(x), x, atol=0.0, rtol=0.0)


def test_hope_cache_passthrough():
    """output_final_state returns self_ref state plus CMS schedule metadata."""
    torch.manual_seed(0)
    hope = HopeBlock(_cfg(), num_heads=2, memory_depth=1, num_levels=3, update_intervals=(1, 2, 4), use_short_conv=False).eval()
    x = torch.randn(1, 4, D)
    with torch.no_grad():
        _, cache = hope(x, output_final_state=True)
    assert cache is not None
    assert cache["self_ref"] is not None
    assert cache["cms"] == {"num_levels": 3, "update_intervals": (1, 2, 4)}
    _, nocache = hope(x)
    assert nocache is None
