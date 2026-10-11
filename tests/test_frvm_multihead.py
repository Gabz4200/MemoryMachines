"""Per-head FRVM geometry: semantics, independence, end-to-end grad flow."""

import torch
import torch.nn.functional as F

from memory_machines.models.blocks import GroupedQueryAttention
from memory_machines.models.blocks.positioning import FRVMValueGate
from tests.test_blocks_shapes import tiny_cfg


def test_tied_slices_match_shared_reference():
    torch.manual_seed(0)
    g = FRVMValueGate(8, 2).eval()
    w_mat = torch.randn(8, 8)
    with torch.no_grad():
        g.phi_weight.copy_(w_mat.unsqueeze(0).expand(2, 8, 8))
        w = torch.softmax(torch.randn(1, 2, 5, 7), dim=-1)
        vf, vq = torch.randn(1, 2, 7, 8), torch.randn(1, 2, 5, 8)
        out = g.gated_reduction(w, vf, vq, chunk_size=1000)
        pvq = F.linear(vq, w_mat)
        pvk = F.linear(vf, w_mat)
        delta = pvq.unsqueeze(3) - pvk.unsqueeze(2)
        gated = vf.unsqueeze(2) * F.silu(vf.unsqueeze(2) + delta)
        ref = (w.unsqueeze(-1) * gated).sum(dim=3)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_heads_are_independent():
    torch.manual_seed(1)
    g = FRVMValueGate(8, 3).eval()
    with torch.no_grad():
        w = torch.softmax(torch.randn(1, 3, 4, 6), dim=-1)
        vf = torch.randn(1, 3, 6, 8)
        vq = torch.randn(1, 3, 4, 8)
        base = g.gated_reduction(w, vf, vq)
        vf2, vq2 = vf.clone(), vq.clone()
        vf2[:, 1, :, :] = torch.randn(1, 6, 8)
        vq2[:, 1, :, :] = torch.randn(1, 4, 8)
        pert = g.gated_reduction(w, vf2, vq2)
    torch.testing.assert_close(base[:, 0], pert[:, 0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(base[:, 2], pert[:, 2], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(base[:, 1], pert[:, 1], atol=1e-6, rtol=1e-6)


def test_gqa_frvm_end_to_end():
    torch.manual_seed(2)
    m = GroupedQueryAttention(tiny_cfg()).train()  # FRVM + CVPE on
    x = torch.randn(2, 8, 32, requires_grad=True)
    out, _, _ = m(x)
    assert tuple(out.shape) == (2, 8, 32)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.isfinite(m.frvm_gate.phi_weight.grad).all()
