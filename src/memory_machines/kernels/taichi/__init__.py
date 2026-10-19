"""Taichi lazy loader. ti.init() runs on first kernel use, never at import."""

import importlib
import os
from threading import Lock

_ti = None
_lock = Lock()
_initialized = False


def _import_taichi():
    """Import taichi lazily; raise a clear error if it is not installed."""
    global _ti
    if _ti is None:
        try:
            import taichi as ti  # pyrefly: ignore  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("taichi is not installed; install it to use compiled Taichi kernels. The pure-PyTorch 'naive' path works without it.") from exc
        _ti = ti
    return _ti


def ensure_taichi():
    """Initialize taichi once; idempotent. Only runs when a kernel is requested."""
    global _initialized
    if not _initialized:
        with _lock:
            if not _initialized:
                ti = _import_taichi()
                # ti.cpu needs no GPU, so the compiled path is exercisable in CI
                # and on a laptop. TARCH=vulkan selects a Vulkan backend, which
                # an Intel/AMD iGPU provides without CUDA — verified: parity and
                # float64 gradcheck both pass on an Intel UHD 620 via Mesa
                # Vulkan, still driving ordinary CPU tensors.
                arch = os.environ.get("TARCH", "").strip()
                if not arch:
                    arch = "cuda" if torch.cuda.is_available() else "cpu"
                ti.init(arch=getattr(ti, arch))
                _initialized = True


def get_taichi_kernel(op_name: str):
    """Lazily import ``memory_machines.kernels.taichi.<op_name>`` and return its kernel callable.

    Returns ``None`` if the module or the ``kernel`` attribute does not exist,
    so callers can fall back to the naive path instead of raising. ``ti.init``
    is only triggered once a real kernel is actually requested.
    """
    try:
        mod = importlib.import_module(f"memory_machines.kernels.taichi.{op_name}")
    except ImportError:
        return None
    return getattr(mod, "kernel", None)


__all__ = ["ensure_taichi", "get_taichi_kernel"]
