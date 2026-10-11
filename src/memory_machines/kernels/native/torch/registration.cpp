// TORCH_LIBRARY / TORCH_LIBRARY_IMPL bindings for the project custom ops.
// Mirrors the torch.library registration in src/project/ops/dw_scale.py; use one
// or the other, not both, for a given op.
#include <torch/library.h>

// TORCH_LIBRARY(project, m) {
//   m.def("dw_scale(Tensor x, Tensor weight) -> Tensor");
//   m.def("dw_scale_backward(Tensor grad_out, Tensor x, Tensor weight) -> (Tensor, Tensor)");
// }
