// First-class CPU implementation of dw_scale (native C++/ATen + OpenMP/SIMD).
// Reach for GGML only when it is advantageous (tensor/fused ops with
// zero-copy PyTorch tensor views).
#include <torch/extension.h>

at::Tensor dw_scale_cpu_forward(const at::Tensor& x, const at::Tensor& weight) {
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(x.device().is_cpu() && weight.device().is_cpu(), "tensors must be on CPU");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");

    auto out = torch::empty_like(x);
    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

#pragma omp parallel for
    for (int64_t i = 0; i < x.numel(); i++) {
        out_ptr[i] = x_ptr[i] * w_ptr[i];
    }
    return out;
}
