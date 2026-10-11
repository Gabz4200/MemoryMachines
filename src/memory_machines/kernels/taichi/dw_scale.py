"""Taichi kernels for dw_scale. One @ti.kernel runs on CUDA, Vulkan, or CPU.

Taichi resolves `ti.types.ndarray(ndim=2)` at decoration time, so the
annotation is a call, not a type — every type checker flags it. Dropping the
annotation breaks Taichi, so each line carries a local suppression instead.

The kernel body is backend-agnostic; which backend runs is chosen by
ensure_taichi() (TARCH env var, default cuda-if-present else cpu), not here.
"""

import taichi as ti  # pyrefly: ignore  # type: ignore[import-untyped]


@ti.kernel
def dw_scale_forward_kernel(
    x: ti.types.ndarray(ndim=2),  # pyrefly: ignore
    w: ti.types.ndarray(ndim=2),  # pyrefly: ignore
    out: ti.types.ndarray(ndim=2),  # pyrefly: ignore
):
    for i, j in out:  # auto-parallelized, element-level, readable
        out[i, j] = x[i, j] * w[i, j]


@ti.kernel
def dw_scale_backward_kernel(
    grad_out: ti.types.ndarray(ndim=2),  # pyrefly: ignore
    x: ti.types.ndarray(ndim=2),  # pyrefly: ignore
    w: ti.types.ndarray(ndim=2),  # pyrefly: ignore
    grad_x: ti.types.ndarray(ndim=2),  # pyrefly: ignore
    grad_w: ti.types.ndarray(ndim=2),  # pyrefly: ignore
):
    for i, j in grad_out:
        grad_x[i, j] = grad_out[i, j] * w[i, j]
        grad_w[i, j] = grad_out[i, j] * x[i, j]
