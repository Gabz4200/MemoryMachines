"""Public op facade: torch.library custom operator + fake + autograd wiring."""

import torch


def _check_dw_scale_inputs(x: torch.Tensor, weight: torch.Tensor) -> None:
    if x.dim() != 2:
        raise ValueError(f"dw_scale requires 2D inputs, got x:{tuple(x.shape)} weight:{tuple(weight.shape)}")
    if x.shape != weight.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(weight.shape)}")
    if x.dtype != weight.dtype:
        raise ValueError(f"dtype mismatch: {x.dtype} vs {weight.dtype} (require equal dtypes; no implicit promotion in the compiled path)")
    if x.device != weight.device:
        raise ValueError(f"device mismatch: {x.device} vs {weight.device}")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("dw_scale requires contiguous inputs (call .contiguous() first); strided views may misinterpret offsets in the Taichi kernel")


def _check_dw_scale_backward_inputs(grad_out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor) -> None:
    for name, t in (("grad_out", grad_out), ("x", x), ("weight", weight)):
        if t.dim() != 2:
            raise ValueError(f"dw_scale_backward requires 2D inputs, got {name}:{tuple(t.shape)}")
    if not (grad_out.shape == x.shape == weight.shape):
        raise ValueError(f"shape mismatch: grad_out {tuple(grad_out.shape)} vs x {tuple(x.shape)} vs weight {tuple(weight.shape)}")
    if not (grad_out.dtype == x.dtype == weight.dtype):
        raise ValueError(f"dtype mismatch: {grad_out.dtype} vs {x.dtype} vs {weight.dtype} (require equal dtypes)")
    if not (grad_out.device == x.device == weight.device):
        raise ValueError(f"device mismatch: {grad_out.device} vs {x.device} vs {weight.device}")
    if not (grad_out.is_contiguous() and x.is_contiguous() and weight.is_contiguous()):
        raise ValueError("dw_scale_backward requires contiguous inputs (call .contiguous() first)")


@torch.library.custom_op("memory_machines::dw_scale", mutates_args=())
def dw_scale(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    _check_dw_scale_inputs(x, weight)
    if x.device.type in ("cpu", "cuda"):
        from memory_machines.kernels.taichi import ensure_taichi
        from memory_machines.kernels.taichi.dw_scale import dw_scale_forward_kernel

        ensure_taichi()
        out = torch.empty_like(x)
        dw_scale_forward_kernel(x, weight, out)
        return out
    raise NotImplementedError(f"no compiled kernel for device {x.device.type}; use dispatch_dw_scale(x, weight, backend='naive')")


@torch.library.custom_op("memory_machines::dw_scale_backward", mutates_args=())
def dw_scale_backward(grad_out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _check_dw_scale_backward_inputs(grad_out, x, weight)
    from memory_machines.kernels.taichi import ensure_taichi
    from memory_machines.kernels.taichi.dw_scale import dw_scale_backward_kernel

    ensure_taichi()
    grad_x = torch.empty_like(x)
    grad_w = torch.empty_like(weight)
    dw_scale_backward_kernel(grad_out, x, weight, grad_x, grad_w)
    return grad_x, grad_w


@torch.library.register_fake("memory_machines::dw_scale")
def _dw_scale_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    torch._check(x.dim() == 2)
    torch._check(x.shape == weight.shape)
    torch._check(x.dtype == weight.dtype)
    torch._check(x.device == weight.device)
    return torch.empty_like(x)


@torch.library.register_fake("memory_machines::dw_scale_backward")
def _dw_scale_backward_fake(grad_out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check(grad_out.shape == x.shape)
    torch._check(x.shape == weight.shape)
    return torch.empty_like(x), torch.empty_like(weight)


def _setup_context(ctx, inputs, output):
    x, weight = inputs
    ctx.save_for_backward(x, weight)


def _backward(ctx, grad_output):
    x, weight = ctx.saved_tensors
    # Single-backward only: dw_scale_backward has no autograd entry for double-backward.
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    needs_x = ctx.needs_input_grad[0] if hasattr(ctx, "needs_input_grad") else True
    needs_w = ctx.needs_input_grad[1] if hasattr(ctx, "needs_input_grad") else True
    grad_x, grad_w = torch.ops.memory_machines.dw_scale_backward(grad_output, x, weight)
    return (grad_x if needs_x else None, grad_w if needs_w else None)


torch.library.register_autograd("memory_machines::dw_scale", _backward, setup_context=_setup_context)
