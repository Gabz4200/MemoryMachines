"""Regression tests for the Gated Delta Rule-2 resting-state feature.

Covers dispatcher threading, nonzero initial+resting oracles, chunk-size
variants, GVA groups, norm/guard combos, packed sequences, grad flow, helper
values, block registration/round-trips, dtype casting, T=0, bad shapes,
conv+GVA blocks, masking, cache replay, and output_final_state=False.

Tiny shapes (B<=2, T<=10, K/V<=8); float64; deterministic seeds.
"""

import pytest
import torch

from memory_machines.models.blocks import gdn2 as G


def _mk(B=1, T=6, H=1, HV=2, K=4, V=3, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, dtype=torch.float64)
    k = torch.randn(B, T, H, K, dtype=torch.float64)
    v = torch.randn(B, T, HV, V, dtype=torch.float64)
    g = -torch.rand(B, T, H, K, dtype=torch.float64)
    b = torch.rand(B, T, HV, K, dtype=torch.float64)
    w = torch.rand(B, T, HV, V, dtype=torch.float64)
    s0 = torch.randn(HV, K, V, dtype=torch.float64) * 0.5
    return q, k, v, g, b, w, s0


def _oracle(q, k, v, g, b, w, s0, scale=None, s_init=None):
    """Direct Eq.18 recurrence with resting prior, seeded from s_init."""
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    if scale is None:
        scale = K**-0.5
    group = HV // H
    qe = q.repeat_interleave(group, dim=2) * scale
    ke = k.repeat_interleave(group, dim=2)
    ge = g.repeat_interleave(group, dim=2)
    if s_init is None:
        s = torch.zeros(B, HV, K, V, dtype=torch.float64)
    else:
        s = s_init.clone()
    ik = torch.eye(K, dtype=torch.float64)
    outs = []
    for t in range(T):
        d = torch.diag_embed(torch.exp(ge[:, t]))
        sbar = (ik - d) @ s0.unsqueeze(0).expand(B, -1, -1, -1) + d @ s
        e = (b[:, t] * ke[:, t]).unsqueeze(-1)
        kt = ke[:, t].unsqueeze(-1)
        zt = (w[:, t] * v[:, t]).unsqueeze(-2)
        rt = torch.matmul(sbar.transpose(-1, -2), e).transpose(-1, -2)
        s = sbar + torch.matmul(kt, zt - rt)
        outs.append(torch.matmul(s.transpose(-1, -2), qe[:, t].unsqueeze(-1)).squeeze(-1))
    o = torch.stack(outs, dim=1) if T else torch.empty(B, 0, HV, V, dtype=torch.float64)
    return o, s


def test_01_dispatcher_threads_resting_both_backends():
    q, k, v, g, b, w, s0 = _mk(seed=10)
    kw = {"resting_state": s0, "output_final_state": True}
    o_rec, s_rec = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, **kw)
    o_ch, s_ch = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=2, **kw)
    o_naive, s_naive = G.gated_delta_rule_2(q, k, v, g, b, w, backend="naive", **kw)
    o_chunk, s_chunk = G.gated_delta_rule_2(q, k, v, g, b, w, backend="chunk", chunk_size=2, **kw)
    torch.testing.assert_close(o_naive, o_rec, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(s_naive, s_rec, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(o_chunk, o_ch, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(s_chunk, s_ch, atol=1e-12, rtol=1e-12)


def test_02_nonzero_init_plus_resting_oracle():
    q, k, v, g, b, w, s0 = _mk(seed=11)
    torch.manual_seed(12)
    s_init = torch.randn(1, 2, 4, 3, dtype=torch.float64) * 0.5
    kw = {"resting_state": s0, "initial_state": s_init, "output_final_state": True}
    o_rec, st_rec = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, **kw)
    o_ch, st_ch = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=2, **kw)
    torch.testing.assert_close(o_rec, o_ch, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(st_rec, st_ch, atol=1e-10, rtol=1e-10)
    exp_o, exp_s = _oracle(q, k, v, g, b, w, s0, s_init=s_init)
    torch.testing.assert_close(o_rec, exp_o, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(st_rec, exp_s, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("chunk_size", [1, 3, 8, 64])
def test_03_chunk_size_variants_with_resting(chunk_size):
    q, k, v, g, b, w, s0 = _mk(T=8, seed=13)
    kw = {"resting_state": s0, "output_final_state": True}
    o_rec, s_rec = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, **kw)
    o_ch, s_ch = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=chunk_size, **kw)
    torch.testing.assert_close(o_ch, o_rec, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(s_ch, s_rec, atol=1e-10, rtol=1e-10)


def test_04_resting_gva_group_one():
    q, k, v, g, b, w, s0 = _mk(H=2, HV=2, seed=14)
    kw = {"resting_state": s0, "output_final_state": True}
    o_rec, s_rec = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, **kw)
    o_ch, s_ch = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=2, **kw)
    torch.testing.assert_close(o_ch, o_rec, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(s_ch, s_rec, atol=1e-10, rtol=1e-10)
    exp_o, exp_s = _oracle(q, k, v, g, b, w, s0)
    torch.testing.assert_close(o_rec, exp_o, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(s_rec, exp_s, atol=1e-10, rtol=1e-10)


def test_05_resting_larger_gva_group():
    q, k, v, g, b, w, s0 = _mk(H=1, HV=4, K=4, V=3, seed=15)
    kw = {"resting_state": s0, "output_final_state": True}
    o_rec, s_rec = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, **kw)
    o_ch, s_ch = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=3, **kw)
    torch.testing.assert_close(o_ch, o_rec, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(s_ch, s_rec, atol=1e-10, rtol=1e-10)


def test_06_resting_norm_guard_agree():
    q, k, v, g, b, w, s0 = _mk(seed=16)
    kw = {"resting_state": s0, "output_final_state": True, "use_qk_l2norm": True, "beta_guard": True}
    o_rec, s_rec = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, **kw)
    o_ch, s_ch = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=2, **kw)
    torch.testing.assert_close(o_ch, o_rec, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(s_ch, s_rec, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("backend", ["recurrent", "chunk"])
def test_07_resting_packed_matches_segments(backend):
    torch.manual_seed(17)
    H, HV, K, V = 1, 2, 4, 3
    lens = [5, 3]
    T = sum(lens)
    q = torch.randn(1, T, H, K, dtype=torch.float64)
    k = torch.randn(1, T, H, K, dtype=torch.float64)
    v = torch.randn(1, T, HV, V, dtype=torch.float64)
    g = -torch.rand(1, T, H, K, dtype=torch.float64)
    b = torch.rand(1, T, HV, K, dtype=torch.float64)
    w = torch.rand(1, T, HV, V, dtype=torch.float64)
    s0 = torch.randn(HV, K, V, dtype=torch.float64) * 0.5
    torch.manual_seed(18)
    s_init = torch.randn(2, HV, K, V, dtype=torch.float64) * 0.5
    cu = torch.tensor([0, lens[0], T])
    fwd = G.gated_delta_rule_2_recurrent_forward if backend == "recurrent" else (lambda *a, **kw: G.gated_delta_rule_2_chunk_forward(*a, chunk_size=2, **kw))
    o_pack, s_pack = fwd(q, k, v, g, b, w, cu_seqlens=cu, initial_state=s_init, resting_state=s0, output_final_state=True)
    assert s_pack is not None
    for i, (s, e) in enumerate([(0, 5), (5, 8)]):
        oi, si = fwd(q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], b[:, s:e], w[:, s:e], initial_state=s_init[i : i + 1], resting_state=s0, output_final_state=True)
        torch.testing.assert_close(o_pack[0, s:e], oi[0], atol=1e-10, rtol=1e-10)
        assert si is not None
        torch.testing.assert_close(s_pack[i : i + 1], si, atol=1e-10, rtol=1e-10)


def test_08_recurrent_grad_to_resting():
    q, k, v, g, b, w, s0 = _mk(T=4, seed=19)
    req = [t.clone().detach().requires_grad_(True) for t in (q, k, v, g, b, w, s0)]
    o, _ = G.gated_delta_rule_2_recurrent_forward(req[0], req[1], req[2], req[3], req[4], req[5], resting_state=req[6])
    grads = torch.autograd.grad(o.sum(), req)
    assert len(grads) == 7
    for t in grads:
        assert t is not None and torch.isfinite(t).all()
    assert grads[6].abs().sum() > 0


def test_09_helper_unit_values_and_grad():
    e = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    q = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    z = torch.tensor([[5.0, 6.0]], dtype=torch.float64)
    ot = torch.tensor([[7.0, 8.0]], dtype=torch.float64)
    s0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    torch.testing.assert_close(G.resting_corrected_write(z, e, s0), torch.tensor([[4.0, 4.0]], dtype=torch.float64), atol=0.0, rtol=0.0)
    torch.testing.assert_close(G.add_resting_projection(ot, q, s0), torch.tensor([[10.0, 12.0]], dtype=torch.float64), atol=0.0, rtol=0.0)
    for args, fn in [((z, e, s0), G.resting_corrected_write), ((ot, q, s0), G.add_resting_projection)]:
        req = [t.clone().detach().requires_grad_(True) for t in args]
        out = fn(*req)
        grads = torch.autograd.grad(out.sum(), req)
        assert len(grads) == 3
        for t in grads:
            assert t is not None and torch.isfinite(t).all() and t.abs().sum() > 0


def test_10_block_reexports():
    import memory_machines.models.blocks as B
    from memory_machines.models.blocks import add_resting_projection, resting_corrected_write

    assert "resting_corrected_write" in B.__all__
    assert "add_resting_projection" in B.__all__
    assert resting_corrected_write is G.resting_corrected_write
    assert add_resting_projection is G.add_resting_projection


def test_11_param_registration():
    torch.manual_seed(20)
    off = G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, head_dim_v=4, use_short_conv=False, use_resting_state=False)
    assert "resting_state" not in off.state_dict()
    assert off.resting_state is None
    on = G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, head_dim_v=4, use_short_conv=False, use_resting_state=True)
    assert "resting_state" in on.state_dict()
    assert tuple(on.state_dict()["resting_state"].shape) == (2, 4, 4)


def test_12_zero_init_preserved():
    torch.manual_seed(21)
    blk = G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, head_dim_v=4, use_short_conv=False, use_resting_state=True)
    assert torch.equal(blk.resting_state, torch.zeros_like(blk.resting_state))
    blk.apply(blk._initialize_weights)
    assert torch.equal(blk.resting_state, torch.zeros_like(blk.resting_state))


def test_13_block_chunk_backend_matches_naive():
    torch.manual_seed(22)
    kw = {"hidden_size": 16, "num_heads": 2, "head_dim_k": 4, "head_dim_v": 4, "use_short_conv": False, "use_resting_state": True}
    naive = G.GatedDeltaNet2(backend="naive", **kw).double()  # type: ignore[arg-type]
    chunk = G.GatedDeltaNet2(backend="chunk", chunk_size=2, **kw).double()  # type: ignore[arg-type]
    chunk.load_state_dict(naive.state_dict())
    with torch.no_grad():
        naive.resting_state.add_(torch.randn_like(naive.resting_state) * 0.5)
    chunk.load_state_dict(naive.state_dict())
    torch.manual_seed(23)
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    o_n, _, _ = naive(x)
    o_c, _, _ = chunk(x)
    assert o_n.shape == (1, 5, 16)
    assert torch.isfinite(o_n).all() and torch.isfinite(o_c).all()
    torch.testing.assert_close(o_c, o_n, atol=1e-8, rtol=1e-8)


def test_14_state_dict_round_trip():
    torch.manual_seed(24)
    blk = G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, head_dim_v=4, use_short_conv=False, use_resting_state=True).double()
    with torch.no_grad():
        blk.resting_state.add_(torch.randn_like(blk.resting_state) * 0.5)
    sd = blk.state_dict()
    torch.manual_seed(25)
    blk2 = G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, head_dim_v=4, use_short_conv=False, use_resting_state=True).double()
    blk2.load_state_dict(sd)
    torch.testing.assert_close(blk2.resting_state, blk.resting_state, atol=0.0, rtol=0.0)
    torch.manual_seed(26)
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    o1, _, _ = blk(x)
    o2, _, _ = blk2(x)
    torch.testing.assert_close(o2, o1, atol=0.0, rtol=0.0)


def test_15_mixed_dtype_casts_to_input_dtype():
    # NOTE: _run_path computes in float64 when any input is float64 but casts
    # outputs/state back to q.dtype; assert that actual contract.
    q, k, v, g, b, w, s0 = _mk(seed=27)
    o, s = G.gated_delta_rule_2_recurrent_forward(q.float(), k.float(), v.float(), g.float(), b.float(), w.float(), resting_state=s0, output_final_state=True)
    assert s is not None
    assert o.dtype == torch.float32 and s.dtype == torch.float32
    assert torch.isfinite(o).all()
    o2, s2 = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, resting_state=s0.float(), output_final_state=True)
    assert s2 is not None
    assert o2.dtype == torch.float64 and s2.dtype == torch.float64
    assert torch.isfinite(o2).all()
    o3, _ = G.gated_delta_rule_2_chunk_forward(q.float(), k.float(), v.float(), g.float(), b.float(), w.float(), chunk_size=2, resting_state=s0)
    torch.testing.assert_close(o3.double(), o.double(), atol=1e-5, rtol=1e-5)


def test_16_t_zero_with_resting_both_paths():
    torch.manual_seed(28)
    B, HV, K, V = 1, 2, 4, 3
    q = torch.empty(B, 0, 1, K, dtype=torch.float64)
    k = torch.empty(B, 0, 1, K, dtype=torch.float64)
    v = torch.empty(B, 0, HV, V, dtype=torch.float64)
    g = torch.empty(B, 0, 1, K, dtype=torch.float64)
    b = torch.empty(B, 0, HV, K, dtype=torch.float64)
    w = torch.empty(B, 0, HV, V, dtype=torch.float64)
    s0 = torch.randn(HV, K, V, dtype=torch.float64) * 0.5
    for fn in (G.gated_delta_rule_2_recurrent_forward, lambda *a, **kw: G.gated_delta_rule_2_chunk_forward(*a, chunk_size=4, **kw)):
        o, s = fn(q, k, v, g, b, w, resting_state=s0, output_final_state=True)
        assert o.shape == (B, 0, HV, V)
        torch.testing.assert_close(s, torch.zeros(1, HV, K, V, dtype=torch.float64), atol=0.0, rtol=0.0)
        torch.manual_seed(29)
        init = torch.randn(1, HV, K, V, dtype=torch.float64)
        o2, s2 = fn(q, k, v, g, b, w, resting_state=s0, initial_state=init, output_final_state=True)
        assert o2.shape == (B, 0, HV, V)
        torch.testing.assert_close(s2, init, atol=1e-12, rtol=1e-12)


def test_17_bad_shapes_raise():
    q, k, v, g, b, w, s0 = _mk(seed=30)
    HV, K, V = 2, 4, 3
    bad4d = torch.zeros(1, HV, K, V, dtype=torch.float64)
    wrong_k = torch.zeros(HV, K + 1, V, dtype=torch.float64)
    wrong_v = torch.zeros(HV, K, V + 1, dtype=torch.float64)
    with pytest.raises(ValueError):
        G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, resting_state=bad4d)
    with pytest.raises(ValueError):
        G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=2, resting_state=wrong_k)
    with pytest.raises(ValueError):
        G.gated_delta_rule_2(q, k, v, g, b, w, backend="chunk", chunk_size=2, resting_state=wrong_v)
    with pytest.raises(ValueError):
        G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, resting_state=s0, beta_guard=True)
    with pytest.raises(ValueError):
        G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=0, resting_state=s0)


def test_18_block_conv_gva_resting():
    torch.manual_seed(31)
    blk = G.GatedDeltaNet2(hidden_size=16, num_heads=1, head_dim_k=4, head_dim_v=4, num_v_heads=2, use_short_conv=True, conv_size=3, use_resting_state=True).double()
    with torch.no_grad():
        blk.resting_state.add_(torch.randn_like(blk.resting_state) * 0.5)
    torch.manual_seed(32)
    twin = G.GatedDeltaNet2(hidden_size=16, num_heads=1, head_dim_k=4, head_dim_v=4, num_v_heads=2, use_short_conv=True, conv_size=3, use_resting_state=True).double()
    twin.load_state_dict(blk.state_dict())
    with torch.no_grad():
        twin.resting_state.zero_()
    torch.manual_seed(33)
    x = torch.randn(1, 6, 16, dtype=torch.float64)
    o, _, cache = blk(x, use_cache=True)
    assert o.shape == (1, 6, 16) and torch.isfinite(o).all()
    assert cache is not None and cache["recurrent_state"].shape == (1, 2, 4, 4)
    o_twin, _, _ = twin(x)
    assert not torch.allclose(o, o_twin)


def test_19_masked_forward_with_resting():
    torch.manual_seed(34)
    blk = G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, head_dim_v=4, use_short_conv=False, use_resting_state=True).double()
    with torch.no_grad():
        blk.resting_state.add_(torch.randn_like(blk.resting_state) * 0.5)
    torch.manual_seed(35)
    x = torch.randn(2, 5, 16, dtype=torch.float64)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 1, 0, 1]])
    o, _, _ = blk(x, attention_mask=mask)
    assert o.shape == (2, 5, 16) and torch.isfinite(o).all()
    for i in range(2):
        keep = mask[i].to(torch.bool)
        assert torch.equal(o[i : i + 1, ~keep], torch.zeros_like(o[i : i + 1, ~keep]))
        o_compact, _, _ = blk(x[i : i + 1, keep])
        torch.testing.assert_close(o[i : i + 1, keep], o_compact, atol=1e-10, rtol=1e-10)
    # All-masked single-row batch is safe (returns full-width zeros, no crash).
    o_empty, _, _ = blk(x[0:1], attention_mask=torch.zeros(1, 5, dtype=torch.long))
    assert tuple(o_empty.shape) == (1, 5, 16) and torch.isfinite(o_empty).all()
    assert o_empty.abs().amax().item() == 0.0


def test_20_block_cache_replay_chunk_backend_resting():
    torch.manual_seed(36)
    blk = G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, head_dim_v=4, use_short_conv=False, backend="chunk", chunk_size=2, use_resting_state=True).double()
    with torch.no_grad():
        blk.resting_state.add_(torch.randn_like(blk.resting_state) * 0.5)
    torch.manual_seed(37)
    x = torch.randn(1, 6, 16, dtype=torch.float64)
    o_full, _, _ = blk(x)
    o1, _, c1 = blk(x[:, :2], use_cache=True)
    o2, _, _ = blk(x[:, 2:], past_key_values=c1, use_cache=True)
    torch.testing.assert_close(torch.cat([o1, o2], dim=1), o_full, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("backend", ["naive", "recurrent", "chunk", "wy_fast"])
def test_21_no_final_state_returns_none(backend):
    q, k, v, g, b, w, s0 = _mk(seed=38)
    o, s = G.gated_delta_rule_2(q, k, v, g, b, w, backend=backend, chunk_size=2, resting_state=s0, output_final_state=False)
    assert s is None
    assert o.shape == (1, 6, 2, 3)


def test_22_t_one_closed_form_with_resting():
    # Hand computation for T=1, zero init: Sbar=(I-D)S0, S=Sbar+k(z-r)^T, o=S^T q.
    k = torch.tensor([[[[0.5, -0.25]]]], dtype=torch.float64)
    q = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float64)
    v = torch.tensor([[[[1.0, -1.0]]]], dtype=torch.float64)
    g = torch.tensor([[[[-0.5, -1.0]]]], dtype=torch.float64)
    b = torch.tensor([[[[0.75, 0.5]]]], dtype=torch.float64)
    w = torch.tensor([[[[1.0, 0.5]]]], dtype=torch.float64)
    s0 = torch.tensor([[[0.5, 1.0], [-0.5, 2.0]]], dtype=torch.float64)
    d = torch.exp(g[0, 0, 0])
    sbar = (1.0 - d).unsqueeze(-1) * s0[0]
    e = (b * k)[0, 0, 0]
    z = (w * v)[0, 0, 0]
    r = sbar.t() @ e
    s = sbar + torch.outer(k[0, 0, 0], z - r)
    expected = (s.t() @ q[0, 0, 0]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    for fn in (G.gated_delta_rule_2_recurrent_forward, lambda *a, **kw: G.gated_delta_rule_2_chunk_forward(*a, chunk_size=1, **kw)):
        o, st = fn(q, k, v, g, b, w, scale=1.0, resting_state=s0, output_final_state=True)
        torch.testing.assert_close(o, expected, atol=1e-12, rtol=1e-12)
        assert st is not None
        torch.testing.assert_close(st[0, 0], s, atol=1e-12, rtol=1e-12)


def test_23_default_scale_with_resting():
    q, k, v, g, b, w, s0 = _mk(seed=39)
    o_rec, s_rec = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, resting_state=s0, output_final_state=True)
    o_ch, s_ch = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=2, resting_state=s0, output_final_state=True)
    torch.testing.assert_close(o_ch, o_rec, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(s_ch, s_rec, atol=1e-10, rtol=1e-10)


def test_24_batch_two_with_resting():
    q, k, v, g, b, w, s0 = _mk(B=2, seed=40)
    kw = {"resting_state": s0, "output_final_state": True}
    o_rec, s_rec = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, **kw)
    o_ch, s_ch = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=2, **kw)
    torch.testing.assert_close(o_ch, o_rec, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(s_ch, s_rec, atol=1e-10, rtol=1e-10)
    assert s_rec is not None
    for i in range(2):
        oi, si = G.gated_delta_rule_2_recurrent_forward(q[i : i + 1], k[i : i + 1], v[i : i + 1], g[i : i + 1], b[i : i + 1], w[i : i + 1], **kw)
        torch.testing.assert_close(oi[0], o_rec[i], atol=1e-12, rtol=1e-12)
        assert si is not None
        torch.testing.assert_close(si[0], s_rec[i], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("backend", ["naive", "chunk"])
def test_25_large_resting_stays_finite(backend):
    q, k, v, g, b, w, _ = _mk(seed=41)
    s0 = torch.randn(2, 4, 3, dtype=torch.float64) * 1e3
    g = torch.full_like(g, -5.0)
    o, s = G.gated_delta_rule_2(q, k, v, g, b, w, backend=backend, chunk_size=2, resting_state=s0, output_final_state=True)
    assert s is not None
    assert torch.isfinite(o).all() and torch.isfinite(s).all()
