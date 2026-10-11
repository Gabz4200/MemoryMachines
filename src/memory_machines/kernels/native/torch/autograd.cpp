// Custom autograd glue (only needed when the backward formula cannot be
// expressed via torch.library.register_autograd decompositions).
#include <torch/library.h>
