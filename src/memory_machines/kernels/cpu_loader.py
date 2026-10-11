"""Lazy loader for the compiled CPU extension (JIT during development, AoT in production).

Nothing compiles at package import; the extension loads on first use.
"""

from torch.utils.cpp_extension import load

_ext = None


def get_cpu_extension():
    global _ext
    if _ext is None:
        _ext = load(
            name="memory_machines_cpu_kernels",
            sources=[
                "src/memory_machines/kernels/native/cpu/dw_scale.cpp",
                "src/memory_machines/kernels/native/cpu/dw_scale_backward.cpp",
            ],
            verbose=False,
        )
    return _ext
