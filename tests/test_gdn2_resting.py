"""Tests for Gated Delta Rule-2 with learnable resting state.

Covers the centered reparameterization wrapping the canonical kernels:
``z_tilde = z - S0^T e`` pre-processing, unmodified scan on the centered
state, ``o = o_tilde + S0^T q`` post-processing. Tiny shapes, float64.
"""

import pytest
import torch

from memory_machines.models.blocks import gdn2 as G


def _inputs(B=1, T=6, H=1, HV=2, K=4, V=3, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, dtype=torch.float64)
    k = torch.randn(B, T, H, K, dtype=torch.float64)
    v = torch.randn(B, T, HV, V, dtype=torch.float64)
    g = -torch.rand(B, T, H, K, dtype=torch.float64)
    b = torch.rand(B, T, HV, K, dtype=torch.float64)
    w = torch.rand(B, T, HV, V, dtype=torch.float64)
    s0 = torch.randn(HV, K, V, dtype=torch.float64) * 0.5
    return q, k, v, g, b, w, s0


def _direct_resting_oracle(q, k, v, g, b, w, s0, scale=1.0):
    """Inline Eq. 18: bar = S0 + D (S - S0), then the canonical edit."""
    B, T, _, K = k.shape
    HV = v.shape[2]
    S = torch.zeros(B, HV, K, v.shape[-1], dtype=torch.float64)
    outs = []
    IK = torch.eye(K, dtype=torch.float64)
    for t in range(T):
        D = torch.diag_embed(torch.exp(g[:, t].expand(-1, HV, -1)))
        Sbar = (IK - D) @ s0.unsqueeze(0).expand(B, -1, -1, -1) + D @ S
        e = (b[:, t] * k[:, t].expand(-1, HV, -1)).unsqueeze(-1)
        kt = k[:, t].expand(-1, HV, -1).unsqueeze(-1)
        zt = (w[:, t] * v[:, t]).unsqueeze(-2)
        rt = torch.matmul(Sbar.transpose(-1, -2), e).transpose(-1, -2)
        S = Sbar + torch.matmul(kt, zt - rt)
        outs.append(torch.matmul(S.transpose(-1, -2), q[:, t].expand(-1, HV, -1).unsqueeze(-1)).squeeze(-1))
    return torch.stack(outs, dim=1) * scale, S


def test_zero_resting_matches_canonical():
    """S0 = 0 reproduces the canonical paths exactly."""
    q, k, v, g, b, w, _ = _inputs()
    zero = torch.zeros(2, 4, 3, dtype=torch.float64)
    o_canon, _ = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, scale=1.0)
    o_rec, s_rec = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, scale=1.0, resting_state=zero, output_final_state=True)
    o_wy, s_wy = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, scale=1.0, chunk_size=2, resting_state=zero, output_final_state=True)
    torch.testing.assert_close(o_rec, o_canon, atol=0.0, rtol=0.0)
    torch.testing.assert_close(o_wy, o_canon, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(s_rec, s_wy, atol=1e-10, rtol=1e-10)


def test_reparam_matches_direct_oracle_both_paths():
    """Centered wrapper equals the direct Eq. 18 recurrence on both paths."""
    q, k, v, g, b, w, s0 = _inputs(seed=1)
    expected_o, expected_s = _direct_resting_oracle(q, k, v, g, b, w, s0)
    o_rec, s_rec = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, scale=1.0, resting_state=s0, output_final_state=True)
    torch.testing.assert_close(o_rec, expected_o, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(s_rec, expected_s, atol=1e-10, rtol=1e-10)
    o_wy, s_wy = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, scale=1.0, chunk_size=2, resting_state=s0, output_final_state=True)
    torch.testing.assert_close(o_wy, expected_o, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(s_wy, expected_s, atol=1e-10, rtol=1e-10)


def test_chunk_vs_recurrent_agree_with_resting_and_grad():
    """Paths agree with nonzero resting; grads flow to all inputs + S0."""
    q, k, v, g, b, w, s0 = _inputs(seed=2)
    o_rec, _ = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, resting_state=s0)
    o_wy, _ = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=3, resting_state=s0)
    torch.testing.assert_close(o_wy, o_rec, atol=1e-10, rtol=1e-10)
    rec = [t.clone().detach().requires_grad_(True) for t in (q, k, v, g, b, w)] + [s0.clone().detach().requires_grad_(True)]
    o, _ = G.gated_delta_rule_2_chunk_forward(rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], chunk_size=2, resting_state=rec[6])
    grads = torch.autograd.grad(o.sum(), rec)
    for t in grads:
        assert t is not None and torch.isfinite(t).all()
    assert grads[-1].abs().sum() > 0


def test_idle_state_relaxes_to_resting():
    """With no input and full decay, the state converges to S0."""
    torch.manual_seed(3)
    B, HV, K, V, T = 1, 2, 4, 3, 6
    s0 = torch.randn(HV, K, V, dtype=torch.float64)
    q = torch.randn(B, T, 1, K, dtype=torch.float64)
    kw = {
        "k": torch.zeros(B, T, 1, K, dtype=torch.float64),
        "v": torch.zeros(B, T, HV, V, dtype=torch.float64),
        "g": torch.full((B, T, 1, K), -5.0, dtype=torch.float64),
        "b": torch.zeros(B, T, HV, K, dtype=torch.float64),
        "w": torch.zeros(B, T, HV, V, dtype=torch.float64),
    }
    _, s_rec = G.gated_delta_rule_2_recurrent_forward(q, output_final_state=True, resting_state=s0, scale=1.0, **kw)
    _, s_wy = G.gated_delta_rule_2_chunk_forward(q, output_final_state=True, resting_state=s0, scale=1.0, chunk_size=2, **kw)
    assert s_rec is not None and s_wy is not None
    torch.testing.assert_close(s_rec[0], s0, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(s_wy[0], s0, atol=1e-9, rtol=1e-9)


def test_output_decomposition_helpers():
    """o = o_tilde + S0^T q via the public pre/post helpers."""
    q, k, v, g, b, w, s0 = _inputs(seed=4)
    scale = 4**-0.5
    # Manual rows (defaults off: plain scale only); helpers act per head row.
    kr = k.expand(-1, -1, 2, -1)
    e = b * kr
    z = w * v
    z_tilde = torch.stack([G.resting_corrected_write(z[:, :, hv], e[:, :, hv], s0[hv]) for hv in range(2)], dim=2)
    torch.testing.assert_close(z_tilde, z - torch.stack([e[:, :, hv] @ s0[hv] for hv in range(2)], dim=2), atol=0.0, rtol=0.0)
    o_rest, _ = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, scale=scale, resting_state=s0)
    # Re-run the centered system explicitly: v' carries z_tilde with w'=1.
    one = torch.ones_like(w)
    o_tilde, _ = G.gated_delta_rule_2_recurrent_forward(q, k, z_tilde, g, b, one, scale=scale, initial_state=-s0.unsqueeze(0))
    o_static = torch.stack([G.add_resting_projection(o_tilde[:, :, hv], (q * scale)[:, :, 0], s0[hv]) for hv in range(2)], dim=2)
    torch.testing.assert_close(o_rest, o_static, atol=1e-10, rtol=1e-10)


def test_resting_incremental_decode_matches_full():
    """Cache handoff carries the total state, so split runs match."""
    q, k, v, g, b, w, s0 = _inputs(T=8, seed=5)
    kw = {"resting_state": s0, "output_final_state": True}
    o_full, _ = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, **kw)
    o1, s1 = G.gated_delta_rule_2_recurrent_forward(q[:, :5], k[:, :5], v[:, :5], g[:, :5], b[:, :5], w[:, :5], **kw)
    o2, _ = G.gated_delta_rule_2_recurrent_forward(q[:, 5:], k[:, 5:], v[:, 5:], g[:, 5:], b[:, 5:], w[:, 5:], resting_state=s0, initial_state=s1)
    torch.testing.assert_close(torch.cat([o1, o2], dim=1), o_full, atol=1e-10, rtol=1e-10)


def test_resting_bad_shape_raises():
    """resting_state must be [HV, K, V]."""
    q, k, v, g, b, w, _ = _inputs()
    with pytest.raises(ValueError):
        G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, resting_state=torch.zeros(2, 4, 2, dtype=torch.float64))
    with pytest.raises(ValueError):
        G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=2, resting_state=torch.zeros(4, 3, dtype=torch.float64))


def test_block_resting_opt_in():
    """use_resting_state=False has no prior param; True defaults to canonical."""
    torch.manual_seed(6)
    plain = G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, head_dim_v=4, use_short_conv=False).double()
    assert plain.resting_state is None
    blk = G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, head_dim_v=4, use_short_conv=False, use_resting_state=True).double()
    assert blk.resting_state is not None and torch.equal(blk.resting_state, torch.zeros_like(blk.resting_state))
    blk.load_state_dict(plain.state_dict(), strict=False)
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    o_rest, _, _ = blk(x)
    o_plain, _, _ = plain(x)
    torch.testing.assert_close(o_rest, o_plain, atol=0.0, rtol=0.0)
    # Nonzero prior changes the output, stays finite, and gets grads.
    with torch.no_grad():
        blk.resting_state.add_(torch.randn_like(blk.resting_state) * 0.5)
    o_nz, _, cache = blk(x, use_cache=True)
    assert torch.isfinite(o_nz).all() and not torch.allclose(o_nz, o_plain)
    assert cache is not None and torch.isfinite(cache["recurrent_state"]).all()
    blk.zero_grad()
    o_nz.sum().backward()
    assert blk.resting_state.grad is not None and torch.isfinite(blk.resting_state.grad).all()
    # Step-by-step replay through the block cache matches with resting on.
    o1, _, c1 = blk(x[:, :2], use_cache=True)
    o2, _, _ = blk(x[:, 2:], past_key_values=c1, use_cache=True)
    torch.testing.assert_close(torch.cat([o1, o2], dim=1), o_nz, atol=1e-8, rtol=1e-8)
