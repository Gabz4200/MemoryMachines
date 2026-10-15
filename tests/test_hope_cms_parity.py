"""Hope / ContinuumMemorySystem (Sec. 7-8) paper-parity tests.

Each test ties to a paper equation. Deterministic (fixed seeds), tiny dims,
independent references with raw torch ops.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from memory_machines.models.blocks.fast_weights import (
    ContinuumMemorySystem,
    HopeAttentionBlock,
    HopeBlock,
)

D = 16


def _cms_raw(cms, x, order=None):
    """Hand-rolled Eq.70 chain with plain ops (no cms.forward/norm/mlp calls)."""
    h = x.clone()
    lvls = list(range(cms.num_levels)) if order is None else list(order)
    for l in lvls:
        n = cms.norms[l]
        w1, b1 = cms.mlps[l][0].weight.detach().clone(), cms.mlps[l][0].bias.detach().clone()
        w2, b2 = cms.mlps[l][2].weight.detach().clone(), cms.mlps[l][2].bias.detach().clone()
        wgt = n.weight.detach().clone()
        hn = F.rms_norm(h, (cms.hidden_size,), wgt, eps=n.eps)
        h = h + F.linear(F.silu(F.linear(hn, w1, b1)), w2, b2)
    return h


def test_cms_equals_manual_chain_eq70():
    """Eq.70: y = MLP_fk(...MLP_f1(x)) with per-level RMSNorm + residual."""
    torch.manual_seed(0)
    cms = ContinuumMemorySystem(hidden_size=D, num_levels=3).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 4, D)
    with torch.no_grad():
        torch.testing.assert_close(cms(x), _cms_raw(cms, x), atol=1e-6, rtol=1e-5)


def test_cms_is_sequential_not_aggregation():
    """Levels compose sequentially; a mean-aggregation over levels differs."""
    torch.manual_seed(0)
    cms = ContinuumMemorySystem(hidden_size=D, num_levels=3).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 4, D)
    with torch.no_grad():
        out = cms(x)
        # Independent mean-aggregation ref: mean_l(mlp_l(norm_l(x))) + x.
        branches = []
        for l in range(cms.num_levels):
            n = cms.norms[l]
            w1, b1 = cms.mlps[l][0].weight.detach().clone(), cms.mlps[l][0].bias.detach().clone()
            w2, b2 = cms.mlps[l][2].weight.detach().clone(), cms.mlps[l][2].bias.detach().clone()
            hn = F.rms_norm(x, (D,), n.weight.detach().clone(), eps=n.eps)
            branches.append(F.linear(F.silu(F.linear(hn, w1, b1)), w2, b2))
        agg = x + torch.stack(branches).mean(0)
        assert (out - agg).abs().amax().item() > 1e-4
        # Order matters for a chain: reversed level order gives another output.
        rev = _cms_raw(cms, x, order=reversed(range(cms.num_levels)))
        assert (out - rev).abs().amax().item() > 1e-4


def test_hope_is_titan_then_cms_eq94_97():
    """Eq.94-97: HopeBlock = SelfRefTitan(x) then CMS chain, shared weights."""
    torch.manual_seed(0)
    hope = HopeBlock(hidden_size=D, num_heads=2, memory_depth=1, num_levels=2, use_short_conv=False).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 4, D)
    with torch.no_grad():
        o, _ = hope.titan(x)
        ref = hope.cms(o)
        y, _ = hope(x)
        torch.testing.assert_close(y, ref, atol=1e-6, rtol=1e-5)


def _hope_attn():
    torch.manual_seed(0)
    return HopeAttentionBlock(hidden_size=D, num_heads=2, num_levels=2).eval()


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
    """Zeroed CMS MLPs reduce the block to attention-only; default init differs."""
    torch.manual_seed(1)
    x = torch.randn(1, 4, D)
    blk = _hope_attn()
    with torch.no_grad():
        ref = _attn_only_ref(blk, x)
        assert (blk(x) - ref).abs().amax().item() > 1e-6  # CMS active: differs
        for seq in blk.cms.mlps:
            for lin in (seq[0], seq[2]):
                lin.weight.zero_()
                if lin.bias is not None:
                    lin.bias.zero_()
        torch.testing.assert_close(blk(x), ref, atol=1e-6, rtol=1e-5)


def test_single_level_cms_is_transformer_mlp():
    """num_levels=1: single norm + MLP + residual."""
    torch.manual_seed(0)
    cms = ContinuumMemorySystem(hidden_size=D, num_levels=1).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 4, D)
    with torch.no_grad():
        torch.testing.assert_close(cms(x), _cms_raw(cms, x), atol=1e-6, rtol=1e-5)


def test_should_update_schedule_eq71():
    """Eq.71: level l updates iff step % C^(l) == 0; step 0 always updates."""
    torch.manual_seed(0)
    cms = ContinuumMemorySystem(hidden_size=D, num_levels=3, update_intervals=(1, 2, 4))
    period = math.lcm(1, 2, 4)
    for step in range(2 * period):
        for l, c in enumerate((1, 2, 4)):
            assert cms.should_update(l, step) == (step % c == 0)
    assert all(cms.should_update(l, 0) for l in range(3))


def test_cms_validation_errors():
    """Bad levels/steps/intervals and non-divisible heads fail loudly."""
    cms = ContinuumMemorySystem(hidden_size=D, num_levels=2)
    with pytest.raises(ValueError, match="level"):
        cms.should_update(-1, 0)
    with pytest.raises(ValueError, match="level"):
        cms.should_update(2, 0)
    with pytest.raises(ValueError, match="non-negative"):
        cms.should_update(0, -1)
    with pytest.raises(ValueError, match="positive"):
        ContinuumMemorySystem(hidden_size=D, num_levels=2, update_intervals=(1,))
    with pytest.raises(ValueError, match="positive"):
        ContinuumMemorySystem(hidden_size=D, num_levels=2, update_intervals=(1, 0))
    with pytest.raises(ValueError, match="positive"):
        ContinuumMemorySystem(hidden_size=D, num_levels=2, update_intervals=(1, -2))
    with pytest.raises(ValueError, match="divisible"):
        HopeAttentionBlock(hidden_size=30, num_heads=4)


def _const_src(d, h, val):
    from torch import nn

    src = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, d))
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(val)
    return src


def test_load_pretrained_copies_and_skips():
    """Matching shapes copy over; wrong shapes are skipped; wrong count raises."""
    from torch import nn

    torch.manual_seed(0)
    cms = ContinuumMemorySystem(hidden_size=8, num_levels=2, expansion=2.0)
    hidden = round(8 * 2.0)
    srcs = [_const_src(8, hidden, 0.5), _const_src(8, hidden, -0.25)]
    cms.load_pretrained_into_levels(srcs)
    for dst_seq, src in zip(cms.mlps, srcs):
        for (n, p), q in zip(dst_seq.named_parameters(), src.parameters()):
            _ = n
            torch.testing.assert_close(p.detach(), q.detach(), atol=0.0, rtol=0.0)
    # Wrong-shape source: nothing copied, dst unchanged.
    before = [p.detach().clone() for seq in cms.mlps for p in seq.parameters()]
    bad = nn.Sequential(nn.Linear(9, hidden + 1), nn.SiLU(), nn.Linear(hidden + 1, 9))
    cms.load_pretrained_into_levels([bad, bad])
    for b, seq in zip([before[:4], before[4:]], cms.mlps):
        for old, p in zip(b, seq.parameters()):
            torch.testing.assert_close(p.detach(), old, atol=0.0, rtol=0.0)
    with pytest.raises(ValueError, match="expected 2 modules"):
        cms.load_pretrained_into_levels([srcs[0]])


def test_invalid_shapes_raise():
    """Non-[B,T,D] inputs fail loudly on all three modules."""
    torch.manual_seed(0)
    cms = ContinuumMemorySystem(hidden_size=D, num_levels=2).eval()
    hope = HopeBlock(hidden_size=D, num_heads=2, memory_depth=1, num_levels=2, use_short_conv=False).eval()
    attn = HopeAttentionBlock(hidden_size=D, num_heads=2, num_levels=2).eval()
    for bad in (torch.randn(1, D), torch.randn(1, 4, D - 1), torch.randn(1, 4, D + 1)):
        with pytest.raises(ValueError, match=r"\[B, T, D="):
            cms(bad)
        with pytest.raises(ValueError, match=r"\[B, T, D="):
            hope(bad)
        with pytest.raises(ValueError, match=r"\[B, T, D="):
            attn(bad)


def test_default_intervals_powers_of_two():
    """Default C^(l) = 2^l."""
    cms = ContinuumMemorySystem(hidden_size=D, num_levels=4)
    assert cms.update_intervals == (1, 2, 4, 8)


def test_zeroed_cms_is_identity():
    """Zeroed CMS MLPs reduce the chain to the identity."""
    torch.manual_seed(0)
    cms = ContinuumMemorySystem(hidden_size=D, num_levels=2).eval()
    with torch.no_grad():
        for seq in cms.mlps:
            for lin in (seq[0], seq[2]):
                lin.weight.zero_()
                if lin.bias is not None:
                    lin.bias.zero_()
    x = torch.randn(1, 4, D)
    torch.testing.assert_close(cms(x), x, atol=0.0, rtol=0.0)


def test_hope_cache_passthrough():
    """output_final_state returns self_ref state plus CMS schedule metadata."""
    torch.manual_seed(0)
    hope = HopeBlock(hidden_size=D, num_heads=2, memory_depth=1, num_levels=3, update_intervals=(1, 2, 4), use_short_conv=False).eval()
    x = torch.randn(1, 4, D)
    with torch.no_grad():
        _, cache = hope(x, output_final_state=True)
    assert cache is not None
    assert cache["self_ref"] is not None
    assert cache["cms"] == {"num_levels": 3, "update_intervals": (1, 2, 4)}
    _, nocache = hope(x)
    assert nocache is None
