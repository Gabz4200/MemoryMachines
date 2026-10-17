"""Lazy loader for the compiled CPU extension (JIT during development, AoT in production).

Nothing compiles at package import; the extension loads on first use.
"""

from pathlib import Path

from torch.utils.cpp_extension import load

# Resolve sources relative to this file so the loader works no matter the
# current working directory (the old repo-relative paths broke any install that
# was not run from the repository root).
_NATIVE_DIR = Path(__file__).resolve().parent / "native" / "cpu"

_ext = None


def get_cpu_extension():
    global _ext
    if _ext is None:
        _ext = load(
            name="memory_machines_cpu_kernels",
            sources=[
                str(_NATIVE_DIR / "dw_scale.cpp"),
                str(_NATIVE_DIR / "dw_scale_backward.cpp"),
            ],
            verbose=False,
        )
    return _ext
