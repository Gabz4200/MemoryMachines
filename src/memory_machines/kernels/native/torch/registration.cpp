// TORCH_LIBRARY / TORCH_LIBRARY_IMPL bindings for the project custom ops.
//
// The Python facades (memory_machines.ops.causal_conv1d / attention) register
// their custom_op + fake + autograd wiring via torch.library.custom_op at
// import time. This file is the C++ manifest and the future home of the
// compiled CPU kernels (TORCH_LIBRARY_IMPL + fake / autograd bindings) once
// they land. Keeping the registered op names in sync with memory_machines.ops.registry
// ensures the Python and C++ namespaces agree.
#include <torch/library.h>

// TORCH_LIBRARY(memory_machines, m) {
//   m.def("causal_conv1d(Tensor x, Tensor weight, Tensor? bias, Tensor? residual, "
//         "Tensor? initial_state, bool output_final_state, str? activation, "
//         "Tensor? cu_seqlens) -> Tensor");
//   m.def("causal_conv1d_update(Tensor x, Tensor cache, Tensor weight, Tensor? bias, "
//         "Tensor? residual, str? activation) -> (Tensor, Tensor)");
//   m.def("universal_pe_attention(Tensor query, Tensor key, Tensor value, "
//         "Tensor? cvpe_attn_bias, Tensor? phi_vq, Tensor? phi_vk, "
//         "Tensor? spatial_grid, Tensor? ov_spatial_weight, Tensor? attn_mask, "
//         "float dropout_p, bool is_causal, bool bidirectional, float? scale, "
//         "bool enable_gqa, int frvm_chunk_size) -> Tensor");
// }
