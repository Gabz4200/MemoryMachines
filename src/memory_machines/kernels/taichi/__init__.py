"""Taichi lazy loader. ti.init() runs on first kernel use, never at import."""

import os
from threading import Lock

# Taichi ships no type stubs or py.typed marker, so every checker reports the
# bare import. Suppression is local to this line.
import taichi as ti  # pyrefly: ignore  # type: ignore[import-untyped]
import torch

_lock = Lock()
_initialized = False


def ensure_taichi():
    global _initialized
    if not _initialized:
        with _lock:
            if not _initialized:
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
