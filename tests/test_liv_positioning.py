"""Coverage for liv.py head-factorization + positioning.py chunking/disable paths."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from memory_machines.models.blocks import GroupedQueryAttention
from memory_machines.models.blocks.liv import LivConvBlock, LivPerceiver
from memory_machines.models.blocks.positioning import FRVMSpatialOV, FRVMValueGate
from tests.test_blocks_shapes import tiny_cfg


def test_livconv_mix_heads_per_head_independence():
    torch.manual_seed(0)
    block = LivConvBlock(tiny_cfg()).eval()
    xg = torch.randn(1, 6, 4, 8)
    base = block.mix_heads(xg)
    xg2 = xg.clone()
    xg2[:, :, 1, :] = torch.randn(1, 6, 8)
    pert = block.mix_heads(xg2)
    for h in (0, 2, 3):
        torch.testing.assert_close(base[:, :, h, :], pert[:, :, h, :], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(base[:, :, 1, :], pert[:, :, 1, :], atol=1e-6, rtol=1e-6)


def test_frvm_chunked_matches_unchunked():
    torch.manual_seed(1)
    gate = FRVMValueGate(head_dim=8, num_heads=2).eval()
    w = torch.softmax(torch.randn(1, 2, 5, 7), dim=-1)
    vf = torch.randn(1, 2, 7, 8)
    vq = torch.randn(1, 2, 5, 8)
    ref = gate.gated_reduction(w, vf, vq, chunk_size=1000)
    small = gate.gated_reduction(w, vf, vq, chunk_size=2)
    tiny = gate.gated_reduction(w, vf, vq, chunk_size=1)
    torch.testing.assert_close(small, ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(tiny, ref, atol=1e-5, rtol=1e-5)


def test_gqa_disable_paths_change_output():
    torch.manual_seed(2)
    model = GroupedQueryAttention(tiny_cfg()).eval()
    # Zero-init CVPE would make the disable test vacuous.
    with torch.no_grad():
        cvpe = model.cvpe_bias
        assert cvpe is not None
        for p in cvpe.parameters():
            torch.nn.init.normal_(p)
        # Zero-start o_proj would zero all outputs alike; restore signal.
        torch.nn.init.xavier_uniform_(model.o_proj.weight)
    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        out_on = model(x)[0]
        saved_cvpe = model.cvpe_bias
        model.cvpe_bias = None
        try:
            out_no_cvpe = model(x)[0]
        finally:
            model.cvpe_bias = saved_cvpe
        saved_frvm = model.frvm_gate
        model.frvm_gate = None
        try:
            out_no_frvm = model(x)[0]
        finally:
            model.frvm_gate = saved_frvm
    assert not torch.allclose(out_on, out_no_cvpe, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(out_on, out_no_frvm, atol=1e-6, rtol=1e-6)


def test_frvm_cross_heads_projects_each_stream_with_its_own_map() -> None:
    torch.manual_seed(5)
    gate = FRVMValueGate(head_dim=4, num_heads=2, cross_heads=True)
    assert gate.phi_q_weight is not None and gate.phi_k_weight is not None
    with torch.no_grad():
        gate.phi_weight.zero_()  # shared map disabled: only Phi_q / Phi_k act
        gate.phi_q_weight.copy_(torch.randn(2, 4, 4))
        gate.phi_k_weight.copy_(torch.randn(2, 4, 4))
    vq, vk = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 5, 4)
    phi_vq, phi_vk = gate.project_qk(vq, vk)
    torch.testing.assert_close(phi_vq, torch.einsum("bhld,hed->bhle", vq, gate.phi_q_weight))
    torch.testing.assert_close(phi_vk, torch.einsum("bhld,hed->bhle", vk, gate.phi_k_weight))
    assert torch.count_nonzero(gate._phi(vq)) == 0
    same = torch.randn(1, 2, 3, 4)
    assert not torch.allclose(gate._phi_q(same), gate._phi_k(same))


@pytest.mark.parametrize("stream", ["query", "key"], ids=["phi-q", "phi-k"])
def test_frvm_cross_head_streams_reject_bad_shapes(stream: str) -> None:
    gate = FRVMValueGate(head_dim=4, num_heads=2, cross_heads=True)
    phi = gate._phi_q if stream == "query" else gate._phi_k
    with pytest.raises(ValueError, match="expected"):
        phi(torch.randn(1, 3, 3, 4))  # 3 heads, gate has 2


def test_gqa_frvm_cross_heads_routes_distinct_query_and_key_maps() -> None:
    torch.manual_seed(6)
    attn = GroupedQueryAttention(replace(tiny_cfg(), window_size=0, frvm_cross_heads=True)).eval()
    gate = attn.frvm_gate
    assert gate is not None and gate.cross_heads
    phi_q, phi_k = gate.phi_q_weight, gate.phi_k_weight
    assert phi_q is not None and phi_k is not None
    with torch.no_grad():
        gate.phi_weight.zero_()
        phi_q.normal_(0.0, 0.3)
        phi_k.normal_(0.0, 0.3)
        torch.nn.init.xavier_uniform_(attn.o_proj.weight)  # zero-start o_proj would hide the change
    xq, xkv = torch.randn(1, 4, 32), torch.randn(1, 7, 32)
    with torch.no_grad():
        out_dual, _, _ = attn(xq, encoder_hidden_states=xkv, bidirectional=True)
        saved = phi_k.detach().clone()
        phi_k.zero_()
        out_no_phi_k, _, _ = attn(xq, encoder_hidden_states=xkv, bidirectional=True)
        phi_k.copy_(saved)
    assert tuple(out_dual.shape) == (1, 4, 32)
    assert torch.isfinite(out_dual).all()
    assert not torch.allclose(out_dual, out_no_phi_k, atol=1e-6, rtol=1e-6)


def test_frvm_spatial_ov_maps_offsets_to_head_dim() -> None:
    ov = FRVMSpatialOV(spatial_dims=3, head_dim=8)
    grid = torch.randn(2, 4, 5, 3)
    assert torch.count_nonzero(ov.weight) == 0  # zero-start: spatial term disabled
    torch.testing.assert_close(ov(grid), torch.zeros(2, 4, 5, 8))
    with torch.no_grad():
        ov.weight.normal_(0.0, 0.5)
        ov.proj.bias.normal_()
    torch.testing.assert_close(ov(grid), F.linear(grid, ov.weight, ov.proj.bias))


def test_liv_merged_compat_adapters_match_head_views():
    torch.manual_seed(3)
    block = LivConvBlock(tiny_cfg()).eval()
    x = torch.randn(2, 6, 32)
    pre_m, post_m, proj_m, gated_m = block.split_gates(x)
    pre_h, post_h, proj_h, gated_h = block.split_gates_heads(x)
    for merged, heads in (
        (pre_m, pre_h),
        (post_m, post_h),
        (proj_m, proj_h),
        (gated_m, gated_h),
    ):
        torch.testing.assert_close(merged, heads.reshape(2, 6, 32), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        block.token_mix(gated_m),
        block.mix_heads(gated_h).reshape(2, 6, 32),
        atol=1e-6,
        rtol=1e-6,
    )
    mix_m = block.mix_heads(gated_h).reshape(2, 6, 32)
    mix_h = block.mix_heads(gated_h)
    torch.testing.assert_close(
        block.finish(post_m, mix_m),
        block.finish_heads(post_h, mix_h),
        atol=1e-6,
        rtol=1e-6,
    )


def test_liv_blocks_grad_flow():
    torch.manual_seed(4)
    cfg = tiny_cfg()
    conv_block = LivConvBlock(cfg).train()
    perceiver = LivPerceiver(cfg).train()

    xc = torch.randn(2, 8, 32, requires_grad=True)
    out_c = conv_block(xc)
    out_c.sum().backward()
    assert xc.grad is not None and torch.isfinite(xc.grad).all()
    assert conv_block.conv.weight.grad is not None
    assert torch.isfinite(conv_block.conv.weight.grad).all()

    xp = torch.randn(2, 8, 32, requires_grad=True)
    out_p = perceiver(xp)
    out_p.sum().backward()
    assert xp.grad is not None and torch.isfinite(xp.grad).all()
    for g in (
        perceiver.swa.q_proj.weight.grad,
        perceiver.latent_queries.grad,
        perceiver.latent_values.grad,
    ):
        assert g is not None
        assert torch.isfinite(g).all()
