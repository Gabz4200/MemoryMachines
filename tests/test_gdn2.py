"""Tests for Gated DeltaNet-2 (Gated Delta Rule-2) reference.

Covers the pure-PyTorch oracle in
``memory_machines.models.blocks.gdn2``: recurrent vs WY-chunkwise
equivalence, KDA/Gated-DeltaNet tied-gate reductions, GVA, caching /
incremental decode, autograd, and the Block forward. Tiny shapes
(B<=2, T<=10, K/V<=8); float64 + tight tolerances for parity.
"""

import pytest
import torch

from memory_machines.models.blocks import gdn2 as G


def _inputs(B=1, T=7, H=2, HV=2, K=4, V=6):
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, dtype=torch.float64)
    k = torch.randn(B, T, H, K, dtype=torch.float64)
    v = torch.randn(B, T, HV, V, dtype=torch.float64)
    g = -torch.rand(B, T, H, K, dtype=torch.float64)  # log-decay <= 0
    b = torch.rand(B, T, HV, K, dtype=torch.float64)
    w = torch.rand(B, T, HV, V, dtype=torch.float64)
    return q, k, v, g, b, w


def test_functional_recurrent_matches_chunk_wy():
    q, k, v, g, b, w = _inputs()
    o_rec, s_rec = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, output_final_state=True)
    o_wy, s_wy = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=4, output_final_state=True)
    torch.testing.assert_close(o_rec, o_wy, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(s_rec, s_wy, atol=1e-10, rtol=1e-10)


def test_dispatcher_matches_both_paths():
    q, k, v, g, b, w = _inputs()
    o, _ = G.gated_delta_rule_2(q, k, v, g, b, w, chunk_size=4)
    o_rec, _ = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w)
    torch.testing.assert_close(o, o_rec, atol=1e-10, rtol=1e-10)
    o_chunk, _ = G.gated_delta_rule_2(q, k, v, g, b, w, chunk_size=4, backend="chunk")
    o_wy, _ = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=4)
    torch.testing.assert_close(o_chunk, o_wy, atol=1e-10, rtol=1e-10)


def test_tied_gates_recover_kda_scalar_update():
    torch.manual_seed(1)
    B, T, H, K, V = 1, 5, 1, 4, 3
    q = torch.randn(B, T, H, K, dtype=torch.float64)
    k = torch.randn(B, T, H, K, dtype=torch.float64)
    v = torch.randn(B, T, H, V, dtype=torch.float64)
    g = -torch.rand(B, T, H, K, dtype=torch.float64)
    beta = torch.rand(B, T, 1, 1, dtype=torch.float64)
    b = beta.expand(B, T, H, K).contiguous()
    w = beta.expand(B, T, H, V).contiguous()
    o, _ = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, scale=1.0)
    # Inline KDA scalar reference: S=(I-beta k k^T)Diag(alpha)S + beta k v^T.
    S = torch.zeros(B, H, K, V, dtype=torch.float64)
    outs = []
    for t in range(T):
        D = torch.diag_embed(torch.exp(g[:, t]))  # [B,H,K,K]
        Sb = torch.matmul(D, S)
        kt = k[:, t].unsqueeze(-1)  # [B,H,K,1]
        vt = v[:, t].unsqueeze(-2)  # [B,H,1,V]
        bt = beta[:, t].unsqueeze(-1)  # [B,1,1,1]
        S = Sb - bt * torch.matmul(kt, torch.matmul(kt.transpose(-1, -2), Sb)) + bt * torch.matmul(kt, vt)
        outs.append(torch.matmul(S.transpose(-1, -2), q[:, t].unsqueeze(-1)).squeeze(-1))
    expected = torch.stack(outs, dim=1)
    torch.testing.assert_close(o, expected, atol=1e-10, rtol=1e-10)


def test_gva_repeat_matches_expanded_call():
    q, k, v, g, b, w = _inputs(H=1, HV=2)
    # GVA path: key-side heads (H=1) are repeated internally to HV=2.
    o_gva, _ = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w)
    # Same math with the repetition done explicitly outside (H=HV=2).
    rep = 2
    o_plain, _ = G.gated_delta_rule_2_recurrent_forward(
        q.expand(-1, -1, rep, -1).contiguous(),
        k.expand(-1, -1, rep, -1).contiguous(),
        v,
        g.expand(-1, -1, rep, -1).contiguous(),
        b,
        w,
    )
    torch.testing.assert_close(o_gva, o_plain, atol=1e-12, rtol=1e-12)
    assert o_gva.shape == (1, 7, 2, 6)


def test_incremental_decode_matches_full():
    q, k, v, g, b, w = _inputs(T=8)
    o_full, _ = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w)
    o1, s1 = G.gated_delta_rule_2_recurrent_forward(
        q[:, :5],
        k[:, :5],
        v[:, :5],
        g[:, :5],
        b[:, :5],
        w[:, :5],
        output_final_state=True,
    )
    o2, _ = G.gated_delta_rule_2_recurrent_forward(
        q[:, 5:],
        k[:, 5:],
        v[:, 5:],
        g[:, 5:],
        b[:, 5:],
        w[:, 5:],
        initial_state=s1,
    )
    torch.testing.assert_close(torch.cat([o1, o2], dim=1), o_full, atol=1e-10, rtol=1e-10)


def test_grad_flows_to_all_inputs():
    q, k, v, g, b, w = _inputs(T=4)
    for t in (q, k, v, g, b, w):
        t.requires_grad_(True)
    o, _ = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=2)
    o.sum().backward()
    for t in (q, k, v, g, b, w):
        assert t.grad is not None and torch.isfinite(t.grad).all()


def test_block_forward_shape_and_step_equivalence():
    torch.manual_seed(3)
    blk = G.GatedDeltaNet2(
        hidden_size=16,
        num_heads=2,
        head_dim_k=4,
        head_dim_v=4,
        use_short_conv=False,
        backend="naive",
        chunk_size=4,
    ).double()
    x = torch.randn(1, 7, 16, dtype=torch.float64)
    o, _, cache = blk(x, use_cache=True)
    assert o.shape == (1, 7, 16)
    assert cache is not None
    # Step-by-step replay from the cache-less prefix must match the suffix.
    o1, _, c1 = blk(x[:, :4], use_cache=True)
    o2, _, _ = blk(x[:, 4:], past_key_values=c1, use_cache=True)
    torch.testing.assert_close(torch.cat([o1, o2], dim=1), o, atol=1e-8, rtol=1e-8)


def test_beta_guard_and_l2norm_paths_agree():
    q, k, v, g, b, w = _inputs()
    kw = {"use_qk_l2norm": True, "beta_guard": True, "output_final_state": True}
    o_rec, s_rec = G.gated_delta_rule_2_recurrent_forward(q, k, v, g, b, w, **kw)
    o_wy, s_wy = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=4, **kw)
    torch.testing.assert_close(o_rec, o_wy, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(s_rec, s_wy, atol=1e-10, rtol=1e-10)
    # Guard leaves safe rows unchanged and only shrinks contrast.
    ks = torch.nn.functional.normalize(k, dim=-1)
    bg = G.apply_beta_guard(torch.ones_like(b) * 0.5, ks, g)
    torch.testing.assert_close(bg, torch.ones_like(b) * 0.5, atol=1e-10, rtol=1e-10)


def test_packed_cu_seqlens_matches_padded():
    torch.manual_seed(5)
    B, H, HV, K, V = 2, 1, 1, 4, 3
    lens = [5, 3]
    T = max(lens)
    q = torch.randn(B, T, H, K, dtype=torch.float64)
    k = torch.randn(B, T, H, K, dtype=torch.float64)
    v = torch.randn(B, T, HV, V, dtype=torch.float64)
    g = -torch.rand(B, T, H, K, dtype=torch.float64)
    b = torch.rand(B, T, HV, K, dtype=torch.float64)
    w = torch.rand(B, T, HV, V, dtype=torch.float64)
    packed = torch.cat([q[i : i + 1, : lens[i]] for i in range(B)], dim=1)
    pk = torch.cat([k[i : i + 1, : lens[i]] for i in range(B)], dim=1)
    pv = torch.cat([v[i : i + 1, : lens[i]] for i in range(B)], dim=1)
    pg = torch.cat([g[i : i + 1, : lens[i]] for i in range(B)], dim=1)
    pb = torch.cat([b[i : i + 1, : lens[i]] for i in range(B)], dim=1)
    pw = torch.cat([w[i : i + 1, : lens[i]] for i in range(B)], dim=1)
    cu = torch.tensor([0, lens[0], lens[0] + lens[1]])
    o_pack, _ = G.gated_delta_rule_2_chunk_forward(packed, pk, pv, pg, pb, pw, chunk_size=4, cu_seqlens=cu)
    for i in range(B):
        oi, _ = G.gated_delta_rule_2_chunk_forward(
            q[i : i + 1, : lens[i]],
            k[i : i + 1, : lens[i]],
            v[i : i + 1, : lens[i]],
            g[i : i + 1, : lens[i]],
            b[i : i + 1, : lens[i]],
            w[i : i + 1, : lens[i]],
            chunk_size=4,
        )
        torch.testing.assert_close(o_pack[0, sum(lens[:i]) : sum(lens[: i + 1])], oi[0], atol=1e-10, rtol=1e-10)


def test_block_with_conv_and_gva():
    torch.manual_seed(7)
    blk = G.GatedDeltaNet2(
        hidden_size=16,
        num_heads=1,
        head_dim_k=4,
        num_v_heads=2,
        use_short_conv=True,
        conv_size=3,
        backend="naive",
        chunk_size=4,
    ).double()
    x = torch.randn(2, 6, 16, dtype=torch.float64)
    o, _, cache = blk(x, use_cache=True)
    assert o.shape == (2, 6, 16)
    assert cache["recurrent_state"].shape == (2, 2, 4, 4)
    assert torch.isfinite(o).all()


def test_block_rejects_compiled_backend():
    with pytest.raises((ValueError, NotImplementedError)):
        G.GatedDeltaNet2(hidden_size=16, num_heads=2, backend="compiled")
    with pytest.raises((ValueError, NotImplementedError)):
        q, k, v, g, b, w = _inputs(T=2)
        G.gated_delta_rule_2(q, k, v, g, b, w, backend="compiled")
