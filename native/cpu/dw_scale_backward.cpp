// CPU backward implementation for dw_scale.
#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor> dw_scale_cpu_backward(
    const at::Tensor& grad_out, const at::Tensor& x, const at::Tensor& weight) {
    TORCH_CHECK(grad_out.is_contiguous() && x.is_contiguous() && weight.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(grad_out.device().is_cpu() && x.device().is_cpu() && weight.device().is_cpu(),
                "tensors must be on CPU");

    auto grad_x = torch::empty_like(x);
    auto grad_w = torch::empty_like(weight);
    const float* g_ptr = grad_out.data_ptr<float>();
    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    float* gx_ptr = grad_x.data_ptr<float>();
    float* gw_ptr = grad_w.data_ptr<float>();

#pragma omp parallel for
    for (int64_t i = 0; i < x.numel(); i++) {
        gx_ptr[i] = g_ptr[i] * w_ptr[i];
        gw_ptr[i] = g_ptr[i] * x_ptr[i];
    }
    return {grad_x, grad_w};
}
