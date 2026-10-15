"""Shared test fixtures (DRY home for helpers cloned across test modules).

New tests should import from here instead of cloning ``_inputs``/``_tiny``/
``tiny_cfg`` per file. Legacy modules still import ``tiny_cfg`` from
``tests.test_blocks_shapes`` for backward compatibility; both resolve to the
same config (see ``make_tiny_cfg``).
"""

import torch

from tests.test_blocks_shapes import tiny_cfg as _legacy_tiny_cfg


def make_tiny_cfg():
    """Canonical tiny config shared by all test modules."""
    return _legacy_tiny_cfg()


def make_inputs(B=1, T=6, D=32, seed=0):
    """Deterministic (module, x) pair from a local Generator (no global RNG coupling)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, D, generator=g)


__all__ = ["make_inputs", "make_tiny_cfg"]
