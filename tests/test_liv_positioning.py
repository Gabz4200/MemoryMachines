"""Coverage for liv.py head-factorization + positioning.py chunking/disable paths."""

import torch

from memory_machines.models.blocks import GroupedQueryAttention
from memory_machines.models.blocks.liv import LivConvBlock, LivPerceiver
from memory_machines.models.blocks.positioning import FRVMValueGate
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
