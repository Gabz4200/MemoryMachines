"""Lazy loader for the compiled CPU cpp extension (JIT during development, AoT in production).

Nothing compiles at package import; the extension loads on first use.

Sources are auto-discovered under ``native/cpu/`` relative to this file: every
``*.cpp`` file there is compiled into a single extension module, and each op's
kernel is reachable as ``get_cpu_kernel(<op_name>)`` (the extension exposes
one function per op, conventionally named ``<op_name>_forward``).  Adding a new
CPU kernel therefore only requires dropping a ``<op>.cpp`` file into
``native/cpu/`` and registering the op in the op registry — no edits here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from torch.utils.cpp_extension import load

# Resolve sources relative to this file so the loader works no matter the
# current working directory (old repo-relative paths broke any install that
# was not run from the repository root).
_NATIVE_DIR = Path(__file__).resolve().parent / "native" / "cpu"

_ext: Optional[Any] = None


def _discover_sources() -> list[str]:
    if not _NATIVE_DIR.is_dir():
        return []
    return sorted(str(p) for p in _NATIVE_DIR.glob("*.cpp"))


def get_cpu_extension():
    """Load (once, cached) the CPU cpp extension from ``native/cpu/*.cpp``."""
    global _ext
    if _ext is not None:
        return _ext
    sources = _discover_sources()
    if not sources:
        return None
    _ext = load(
        name="memory_machines_cpu_kernels",
        sources=sources,
        verbose=False,
    )
    return _ext


def get_cpu_kernel(op_name: str):
    """Return the compiled CPU kernel callable for ``op_name``.

    Returns ``None`` if no cpp extension is built yet or the extension exposes
    no such symbol, so callers can fall back to the naive path instead of
    raising.
    """
    ext = get_cpu_extension()
    if ext is None:
        return None
    # Conventional symbol name: "<op_name>_forward".  Fall back to bare
    # op_name so either convention works.
    for attr in (f"{op_name}_forward", op_name):
        fn = getattr(ext, attr, None)
        if fn is not None:
            return fn
    return None


__all__ = ["get_cpu_extension", "get_cpu_kernel"]
