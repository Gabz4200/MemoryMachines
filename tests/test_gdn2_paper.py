"""Paper-parity tests for Gated DeltaNet-2 (arXiv:2605.22791) reference.

Covers ``memory_machines.models.blocks.gdn2`` against the paper spec:
KDA tied-gate reduction (Eq. 10 with ``b = w = beta * 1``), the scalar-decay
Gated DeltaNet tie, chunk-vs-recurrent gradient agreement, gradcheck on the
recurrent oracle, the beta-guard projection, gate ranges (Eq. 11-12), the
Eq-48 tied-gate gradient relation, masking/packing edge cases, conv-cache
decode, and the documented error paths. Complements ``tests/test_gdn2.py``.

Conventions: tiny shapes (B<=2, T<=10, K/V<=8), float64,
``torch.testing.assert_close`` with tight tolerances, deterministic seeds.
"""

import pytest
import torch
import torch.nn.functional as F

from memory_machines.models.blocks import gdn2 as G


def _inputs(B=1, T=7, H=2, HV=2, K=4, V=6, seed=0):
    """Build float64 (q, k, v, g, b, w); g <= 0, b/w in [0, 1]."""
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, dtype=torch.float64)
    k = torch.randn(B, T, H, K, dtype=torch.float64)
    v = torch.randn(B, T, HV, V, dtype=torch.float64)
    g = -torch.rand(B, T, H, K, dtype=torch.float64)
    b = torch.rand(B, T, HV, K, dtype=torch.float64)
    w = torch.rand(B, T, HV, V, dtype=torch.float64)
    return q, k, v, g, b, w


def _kda_reference(q, k, v, g, beta, scale=1.0):
    """Inline scalar delta-rule loop; broadcast g rows give the Gated DeltaNet tie."""
    B, T, H, K = k.shape
    S = torch.zeros(B, H, K, v.shape[-1], dtype=torch.float64)
    outs = []
    for t in range(T):
        Sb = torch.matmul(torch.diag_embed(torch.exp(g[:, t])), S)
        kt = k[:, t].unsqueeze(-1)
        vt = v[:, t].unsqueeze(-2)
        bt = beta[:, t].unsqueeze(-1)
        S = Sb - bt * torch.matmul(kt, torch.matmul(kt.transpose(-1, -2), Sb)) + bt * torch.matmul(kt, vt)
        outs.append(torch.matmul(S.transpose(-1, -2), q[:, t].unsqueeze(-1)).squeeze(-1))
    o = torch.stack(outs, dim=1)
    return o * scale if scale != 1.0 else o


def _tied_beta(B, T, H, K, V):
    """Broadcast scalar gate to ``b = w = beta * 1`` (KDA tying)."""
    beta = torch.rand(B, T, 1, 1, dtype=torch.float64)
    return beta, beta.expand(B, T, H, K).contiguous(), beta.expand(B, T, H, V).contiguous()


def test_kda_tied_reduction_chunk_path():
    """chunk path with b = w = beta*1 matches the KDA scalar loop."""
    torch.manual_seed(11)
    B, T, H, K, V = 1, 6, 1, 4, 3
    q = torch.randn(B, T, H, K, dtype=torch.float64)
    k = torch.randn(B, T, H, K, dtype=torch.float64)
    v = torch.randn(B, T, H, V, dtype=torch.float64)
    g = -torch.rand(B, T, H, K, dtype=torch.float64)
    beta, b, w = _tied_beta(B, T, H, K, V)
    o, _ = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, scale=1.0, chunk_size=2)
    torch.testing.assert_close(o, _kda_reference(q, k, v, g, beta), atol=1e-10, rtol=1e-10)


def test_scalar_decay_tie_both_paths():
    """scalar g over K + tied b = w; both paths match scalar-decay ref."""
    torch.manual_seed(12)
    B, T, H, K, V = 1, 6, 1, 4, 3
    q = torch.randn(B, T, H, K, dtype=torch.float64)
    k = torch.randn(B, T, H, K, dtype=torch.float64)
    v = torch.randn(B, T, H, V, dtype=torch.float64)
    g0 = -torch.rand(B, T, H, 1, dtype=torch.float64)
    g = g0.expand(B, T, H, K).contiguous()
    beta, b, w = _tied_beta(B, T, H, K, V)
    expected = _kda_reference(q, k, v, g, beta)
    o_rec, _ = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, scale=1.0)
    o_wy, _ = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, scale=1.0, chunk_size=2)
    torch.testing.assert_close(o_rec, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(o_wy, expected, atol=1e-10, rtol=1e-10)


def test_chunk_vs_recurrent_grad_agreement():
    """autograd grads w.r.t. q/k/v/g/b/w agree across paths."""
    q, k, v, g, b, w = _inputs(B=1, T=6, H=1, HV=2, K=4, V=3, seed=13)
    rec = [t.clone().detach().requires_grad_(True) for t in (q, k, v, g, b, w)]
    wy = [t.clone().detach().requires_grad_(True) for t in (q, k, v, g, b, w)]
    o_rec, _ = G.gated_delta_rule_2_recurrent_forward(*rec)
    o_wy, _ = G.gated_delta_rule_2_chunk_forward(*wy, chunk_size=2)
    gr = torch.autograd.grad(o_rec.sum(), rec)
    gw = torch.autograd.grad(o_wy.sum(), wy)
    for a, c in zip(gr, gw):
        torch.testing.assert_close(c, a, rtol=1e-8, atol=1e-8)


def test_gradcheck_recurrent():
    """torch.autograd.gradcheck on the recurrent functional (tiny)."""
    torch.manual_seed(14)
    B, T, H, K, V = 1, 3, 1, 3, 2
    ins = tuple(
        (torch.randn(*s, dtype=torch.float64, requires_grad=True) * 0.5)
        for s in (
            (B, T, H, K),
            (B, T, H, K),
            (B, T, H, V),
            (B, T, H, K),
            (B, T, H, K),
            (B, T, H, V),
        )
    )
    ins[3].data.mul_(-1.0).sub_(0.1)  # g <= 0 like the oracle expects
    ins[4].data.sigmoid_()
    ins[5].data.sigmoid_()

    def fn(q, k, v, g, b, w):
        return G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, scale=1.0)[0]

    assert torch.autograd.gradcheck(fn, ins, eps=1e-6, atol=1e-4)


def _contrast_r2(b, k):
    k2 = k * k
    n = k2.sum(-1, keepdim=True)
    a = (b * k2).sum(-1, keepdim=True)
    nu = (b * b * k2).sum(-1, keepdim=True)
    return torch.clamp(n * nu - a * a, min=0.0)


def test_beta_guard_unsafe_row_modification():
    """Unsafe rows shrink (contrast strictly drops); safe rows bitwise identical.

    One-hot keys give r2 = 0 (provably safe), so the unsafe case uses two-hot keys.
    """
    B, T, H, K = 1, 4, 1, 4
    k_onehot = torch.zeros(B, T, H, K, dtype=torch.float64)
    for t in range(T):
        k_onehot[0, t, 0, t % K] = 1.0
    b_any = torch.tensor([[[[0.2, 0.8, 0.3, 0.9]]]], dtype=torch.float64).expand(B, T, H, K).contiguous()
    g0 = torch.zeros(B, T, H, K, dtype=torch.float64)
    torch.testing.assert_close(G.apply_beta_guard(b_any, k_onehot, g0), b_any, atol=0.0, rtol=0.0)
    # Unsafe rows: two-hot keys + alternating b, g = 0 -> budget r2_crit = 0.
    k = torch.zeros(B, T, H, K, dtype=torch.float64)
    k[..., 0] = 1.0 / 2**0.5
    k[..., 1] = 1.0 / 2**0.5
    b = torch.tensor([[0.0, 1.0, 0.3, 0.7], [1.0, 0.0, 0.9, 0.2]], dtype=torch.float64).repeat(2, 1).reshape(B, T, H, K)
    bg = G.apply_beta_guard(b, k, g0)
    assert not torch.allclose(bg, b)
    r2_before = _contrast_r2(b, k)
    r2_after = _contrast_r2(bg, k)
    assert (r2_before > 0).all()
    assert (r2_after < r2_before).all()
    # Safe rows (constant b = 0.5 -> r2 = 0) pass through bitwise identical.
    b_safe = torch.full((B, T, H, K), 0.5, dtype=torch.float64)
    assert torch.equal(G.apply_beta_guard(b_safe, k, g0), b_safe)


def test_allow_neg_eigval_effect():
    """same weights, allow_neg_eigval False vs True differ, both finite."""
    torch.manual_seed(15)
    kw = {"hidden_size": 16, "num_heads": 2, "head_dim_k": 4, "head_dim_v": 4, "use_short_conv": False, "backend": "naive", "chunk_size": 4}
    blk_a = G.GatedDeltaNet2(**kw, allow_neg_eigval=False).double()
    blk_b = G.GatedDeltaNet2(**kw, allow_neg_eigval=True)
    blk_b.load_state_dict(blk_a.state_dict())
    blk_b = blk_b.double()
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    oa, _, _ = blk_a(x)
    ob, _, _ = blk_b(x)
    assert torch.isfinite(oa).all() and torch.isfinite(ob).all()
    assert not torch.allclose(oa, ob)


def test_attention_mask_zero_padding():
    """padded outputs exactly 0; kept slice matches; all-masked safe."""
    torch.manual_seed(16)
    blk = G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, head_dim_v=4, use_short_conv=False, backend="naive", chunk_size=4).double()
    x = torch.randn(2, 6, 16, dtype=torch.float64)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    o, _, _ = blk(x, attention_mask=mask)
    assert torch.equal(o[0, 3:], torch.zeros_like(o[0, 3:]))
    plain, _, _ = blk(x[0:1, :3])
    torch.testing.assert_close(o[0:1, :3], plain, atol=1e-12, rtol=1e-12)
    o_all, _, _ = blk(x, attention_mask=torch.zeros(2, 6, dtype=torch.long))
    assert torch.equal(o_all, torch.zeros_like(o_all))


def test_empty_sequence():
    """T = 0 returns (B, 0, HV, V) on both paths without exception."""
    q, k, v, g, b, w = _inputs(B=2, T=0, H=1, HV=2, K=4, V=3, seed=17)
    o_rec, _ = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w)
    o_wy, _ = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=2)
    assert o_rec.shape == (2, 0, 2, 3)
    assert o_wy.shape == (2, 0, 2, 3)


def test_beta_guard_requires_l2norm():
    """beta_guard without use_qk_l2norm raises on both entry points."""
    q, k, v, g, b, w = _inputs(B=1, T=3, H=1, HV=1, K=4, V=3, seed=18)
    with pytest.raises(ValueError):
        G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, beta_guard=True)
    with pytest.raises(ValueError):
        G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, beta_guard=True, chunk_size=2)


def test_block_gate_ranges():
    """Eq. 11-12 ranges — b, w in [0, 1] (sigmoid), g <= 0."""
    torch.manual_seed(19)
    blk = G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, head_dim_v=4, use_short_conv=False, backend="naive").double()
    x = torch.randn(2, 5, 16, dtype=torch.float64)
    captured = {}

    def hook(name):
        def fn(_m, _i, out):
            captured[name] = out.detach()

        return fn

    blk.b_proj.register_forward_hook(hook("b"))
    blk.w_proj.register_forward_hook(hook("w"))
    o, _, _ = blk(x)
    assert o.shape == (2, 5, 16)
    b = captured["b"].sigmoid()
    w = captured["w"].sigmoid()
    assert ((b >= 0) & (b <= 1)).all()
    assert ((w >= 0) & (w <= 1)).all()
    A = blk.A_log.float().exp().repeat_interleave(blk.head_k_dim)
    g = (-A * F.softplus(blk.f_proj(x).float() + blk.dt_bias)).to(torch.float64)
    assert (g <= 0).all()


def test_tied_gate_gradient_relation():
    """Eq-48 — sum(db) + sum(dw) equals grad w.r.t. leaf beta."""
    torch.manual_seed(20)
    B, T, H, K, V = 1, 4, 1, 3, 2
    q = torch.randn(B, T, H, K, dtype=torch.float64)
    k = torch.randn(B, T, H, K, dtype=torch.float64)
    v = torch.randn(B, T, H, V, dtype=torch.float64)
    g = -torch.rand(B, T, H, K, dtype=torch.float64)
    beta = torch.rand(B, T, 1, 1, dtype=torch.float64).requires_grad_(True)
    b = beta.expand(B, T, H, K)
    w = beta.expand(B, T, H, V)
    o, _ = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, scale=1.0)
    db, dw = torch.autograd.grad(o.sum(), (b, w), retain_graph=False)
    lhs = db.sum(-1) + dw.sum(-1)  # [B, T, H]
    beta2 = beta.clone().detach().requires_grad_(True)
    o2, _ = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, beta2.expand(B, T, H, K), beta2.expand(B, T, H, V), scale=1.0)
    (rhs,) = torch.autograd.grad(o2.sum(), (beta2,))
    torch.testing.assert_close(lhs, rhs.squeeze(-1).expand(B, T, H), atol=1e-10, rtol=1e-10)


def test_guard_mean_preservation_and_range():
    """key-weighted mean preserved to 1e-12; b' within [b, mu]."""
    torch.manual_seed(21)
    B, T, H, K = 1, 5, 1, 4
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, dtype=torch.float64), dim=-1)
    b = torch.rand(B, T, H, K, dtype=torch.float64)
    g = -torch.rand(B, T, H, K, dtype=torch.float64)
    bg = G.apply_beta_guard(b, k, g)
    k2 = k * k
    n = k2.sum(-1, keepdim=True)
    mu = (b * k2).sum(-1, keepdim=True) / n.clamp(min=1e-20)
    mu_after = (bg * k2).sum(-1, keepdim=True) / n.clamp(min=1e-20)
    torch.testing.assert_close(mu_after, mu, atol=1e-12, rtol=1e-12)
    lo = torch.minimum(b, mu)
    hi = torch.maximum(b, mu)
    assert ((bg >= lo - 1e-12) & (bg <= hi + 1e-12)).all()


def test_default_scale():
    """scale=None equals K**-0.5; scale=1.0 differs."""
    q, k, v, g, b, w = _inputs(B=1, T=5, H=1, HV=1, K=4, V=3, seed=22)
    K = q.shape[-1]
    o_none, _ = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=2)
    o_exp, _ = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, scale=K**-0.5, chunk_size=2)
    torch.testing.assert_close(o_none, o_exp, atol=1e-12, rtol=1e-12)
    o_one, _ = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, scale=1.0, chunk_size=2)
    assert not torch.allclose(o_none, o_one)


def test_chunk_size_variants():
    """T = 7 with C in (1, 3, 7, 64) matches recurrent out + state."""
    q, k, v, g, b, w = _inputs(B=2, T=7, H=2, HV=2, K=4, V=5, seed=23)
    o_rec, s_rec = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, output_final_state=True)
    for c in (1, 3, 7, 64):
        o_wy, s_wy = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=c, output_final_state=True)
        torch.testing.assert_close(o_wy, o_rec, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(s_wy, s_rec, atol=1e-10, rtol=1e-10)


def test_packed_empty_segment():
    """cu_seqlens = [0, 0, 5] matches independent per-segment runs."""
    torch.manual_seed(24)
    B, H, HV, K, V, T = 1, 1, 1, 4, 3, 5
    q = torch.randn(B, T, H, K, dtype=torch.float64)
    k = torch.randn(B, T, H, K, dtype=torch.float64)
    v = torch.randn(B, T, HV, V, dtype=torch.float64)
    g = -torch.rand(B, T, H, K, dtype=torch.float64)
    b = torch.rand(B, T, HV, K, dtype=torch.float64)
    w = torch.rand(B, T, HV, V, dtype=torch.float64)
    cu = torch.tensor([0, 0, 5])
    o_pack, _ = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=2, cu_seqlens=cu)
    o_empty, _ = G.gated_delta_rule_2_chunk_forward(q[:, :0], k[:, :0], v[:, :0], g[:, :0], b[:, :0], w[:, :0], chunk_size=2)
    assert o_empty.shape == (1, 0, 1, 3)
    o_full, _ = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=2)
    torch.testing.assert_close(o_pack, o_full, atol=1e-10, rtol=1e-10)


def test_conv_cache_decode_equivalence():
    """short-conv prefix cache replay matches full forward."""
    torch.manual_seed(25)
    conv_size = 4
    blk = G.GatedDeltaNet2(
        hidden_size=16,
        num_heads=2,
        head_dim_k=4,
        head_dim_v=4,
        use_short_conv=True,
        conv_size=conv_size,
        backend="naive",
        chunk_size=4,
    ).double()
    x = torch.randn(1, 7, 16, dtype=torch.float64)
    o, _, cache = blk(x, use_cache=True)
    o1, _, c1 = blk(x[:, :4], use_cache=True)
    o2, _, _ = blk(x[:, 4:], past_key_values=c1, use_cache=True)
    torch.testing.assert_close(torch.cat([o1, o2], dim=1), o, atol=1e-8, rtol=1e-8)
    assert cache["recurrent_state"].shape == (1, 2, 4, 4)
    for s, dim in zip(cache["conv_state"], (blk.key_dim, blk.key_dim, blk.value_dim)):
        assert s.shape == (1, dim, conv_size)


def test_error_paths():
    """documented error paths raise."""
    q, k, v, g, b, w = _inputs(B=1, T=4, H=1, HV=2, K=4, V=3, seed=26)
    with pytest.raises(ValueError):
        G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=0)
    with pytest.raises(ValueError):
        G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=-3)
    with pytest.raises(ValueError):
        G.gated_delta_rule_2(q, k, v, g, b, w, backend="flash")
    with pytest.raises(ValueError):
        G.GatedDeltaNet2(hidden_size=16, num_heads=2, backend="flash")
    with pytest.raises(NotImplementedError):  # reserved compiled name, never silent
        G.gated_delta_rule_2(q, k, v, g, b, w, backend="cpu")
    with pytest.raises(ValueError):  # k shape mismatch
        G.gated_delta_rule_2_recurrent_forward(q, k[..., :2], v, g, b, w)
    with pytest.raises(ValueError):  # g shape mismatch
        G.gated_delta_rule_2_recurrent_forward(q, k, v, g[..., :2], b, w)
    with pytest.raises(ValueError):  # b must be [B, T, HV, K]
        G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b[:, :, :1], w)
    with pytest.raises(ValueError):  # v/w mismatch
        G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w[..., :2])
    q2 = torch.randn(1, 4, 2, 4, dtype=torch.float64)
    v3 = torch.randn(1, 4, 3, 3, dtype=torch.float64)
    with pytest.raises(ValueError):  # HV=3 not a multiple of H=2
        G.gated_delta_rule_2_recurrent_forward(
            q2,
            torch.randn(1, 4, 2, 4, dtype=torch.float64),
            v3,
            torch.zeros(1, 4, 2, 4, dtype=torch.float64),
            torch.rand(1, 4, 3, 4, dtype=torch.float64),
            torch.rand(1, 4, 3, 3, dtype=torch.float64),
        )
    with pytest.raises(ValueError):  # initial_state wrong shape
        G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, initial_state=torch.zeros(1, 2, 4, 2, dtype=torch.float64))
    q_b2, k_b2, v_b2, g_b2, b_b2, w_b2 = _inputs(B=2, T=4, H=1, HV=1, K=4, V=3, seed=27)
    with pytest.raises(ValueError):  # packing requires batch 1
        G.gated_delta_rule_2_chunk_forward(q_b2, k_b2, v_b2, g_b2, b_b2, w_b2, chunk_size=2, cu_seqlens=torch.tensor([0, 2, 4]))
    blk = G.GatedDeltaNet2(hidden_size=16, num_heads=2, use_short_conv=False)
    with pytest.raises(ValueError):  # hidden_states must be 3D
        blk(torch.randn(5, 16))
    with pytest.raises(ValueError):  # mask + cu_seqlens together
        blk(
            torch.randn(1, 4, 16),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            cu_seqlens=torch.tensor([0, 4]),
        )


def test_rejects_cache_batch_mismatch():
    """Cached recurrent_state batch must match the input batch."""
    blk = G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, head_dim_v=4, use_short_conv=False)
    x = torch.randn(1, 4, 16)
    bad = {"recurrent_state": torch.zeros(2, 2, 4, 4), "conv_state": None}
    with pytest.raises(ValueError):
        blk(x, past_key_values=bad, use_cache=True)


def test_single_token_closed_form():
    """T = 1 matches S1 = k z^T, o1 = scale * (k.q) z (b vacuous from S0 = 0)."""
    torch.manual_seed(28)
    B, H, K, V = 1, 1, 4, 3
    q = torch.randn(B, 1, H, K, dtype=torch.float64)
    k = torch.randn(B, 1, H, K, dtype=torch.float64)
    v = torch.randn(B, 1, H, V, dtype=torch.float64)
    g = torch.zeros(B, 1, H, K, dtype=torch.float64)
    _, b, w = _tied_beta(B, 1, H, K, V)
    scale = K**-0.5
    z = (w * v)[:, 0]  # [B, H, V]
    s1 = torch.matmul(k[:, 0].unsqueeze(-1), z.unsqueeze(-2))  # [B, H, K, V]
    expected = torch.matmul(s1.transpose(-1, -2), (q[:, 0] * scale).unsqueeze(-1)).squeeze(-1).unsqueeze(1)
    o_rec, _ = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w)
    o_wy, _ = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=1)
    torch.testing.assert_close(o_rec, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(o_wy, expected, atol=1e-12, rtol=1e-12)


def test_output_final_state_flag_and_initial_state():
    """output_final_state=False gives None; nonzero s0 changes output."""
    q, k, v, g, b, w = _inputs(B=1, T=5, H=1, HV=1, K=4, V=3, seed=29)
    o, s = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w)
    assert s is None
    o_wy, s_wy = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=2)
    assert s_wy is None
    torch.testing.assert_close(o_wy, o, atol=1e-12, rtol=1e-12)
    s0 = torch.ones(1, 1, 4, 3, dtype=torch.float64)
    o_s0, s_out = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, initial_state=s0, output_final_state=True)
    assert s_out is not None and s_out.shape == (1, 1, 4, 3)
    assert not torch.allclose(o_s0, o)


def test_block_bf16_smoke():
    """Float block on bf16 input stays finite under CPU autocast (no internal cast)."""
    torch.manual_seed(30)
    blk = G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, head_dim_v=4, use_short_conv=False, backend="naive", chunk_size=4)
    x = torch.randn(1, 5, 16).to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        o, _, _ = blk(x)
    assert torch.isfinite(o.float()).all()
