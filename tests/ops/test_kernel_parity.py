"""Parity: naive reference == compiled op == Taichi kernel.

Taichi picks its architecture itself (CUDA, Vulkan, or CPU) and ti.cpu runs
without a GPU, so this test is never skipped on a CPU-only machine — it is the
main check that the compiled path matches the oracle everywhere. An iGPU counts:
TARCH=vulkan exercises the GPU kernel path on a machine with no CUDA at all.
"""

import torch

from project.ops.dispatch import dispatch_dw_scale


def test_parity():
    x = torch.randn(3, 5)
    w = torch.randn(3, 5)
    ref = dispatch_dw_scale(x, w, backend="naive")
    got = dispatch_dw_scale(x, w, backend="gpu")
    torch.testing.assert_close(ref, got, rtol=1e-5, atol=1e-5)


def test_gradcheck():
    def f(a, b):
        return dispatch_dw_scale(a, b, backend="gpu")

    a = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (a, b), eps=1e-6, atol=1e-4)
