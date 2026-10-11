"""Public op facade: torch.library custom operator + fake + autograd wiring."""

import torch


@torch.library.custom_op("project::dw_scale", mutates_args=())
def dw_scale(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if x.device.type in ("cpu", "cuda"):
        from project.kernels.taichi import ensure_taichi
        from project.kernels.taichi.dw_scale import dw_scale_forward_kernel

        ensure_taichi()
        out = torch.empty_like(x)
        dw_scale_forward_kernel(x, weight, out)
        return out
    raise NotImplementedError(f"no compiled kernel for device {x.device.type}; use backend='naive'")


@torch.library.custom_op("project::dw_scale_backward", mutates_args=())
def dw_scale_backward(
    grad_out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    from project.kernels.taichi import ensure_taichi
    from project.kernels.taichi.dw_scale import dw_scale_backward_kernel

    ensure_taichi()
    grad_x = torch.empty_like(x)
    grad_w = torch.empty_like(weight)
    dw_scale_backward_kernel(grad_out, x, weight, grad_x, grad_w)
    return grad_x, grad_w


@torch.library.register_fake("project::dw_scale")
def _dw_scale_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    torch._check(x.shape == weight.shape)
    torch._check(x.dtype == weight.dtype)
    torch._check(x.device == weight.device)
    return torch.empty_like(x)


@torch.library.register_fake("project::dw_scale_backward")
def _dw_scale_backward_fake(
    grad_out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(weight)


def _setup_context(ctx, inputs, output):
    x, weight = inputs
    ctx.save_for_backward(x, weight)


def _backward(ctx, grad_output):
    x, weight = ctx.saved_tensors
    return torch.ops.project.dw_scale_backward(grad_output, x, weight)


torch.library.register_autograd("project::dw_scale", _backward, setup_context=_setup_context)
