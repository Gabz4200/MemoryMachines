"""Base-class coverage for :class:`LivDoubleGatedBase` (shared double-gated LIV wrapper).

Exercises the inherited helpers directly on the base class rather than through a
subclass: the per-head gate split (``split_gates_heads``), the merge/output stage
(``finish_heads``), and the ``mix_heads`` seam. Subclass mixers themselves are
covered by tests/test_liv_causal_conv.py and tests/test_liv_positioning.py.
"""

import pytest
import torch

from memory_machines.models.blocks.liv import LivDoubleGatedBase
from tests.conftest import make_inputs, make_tiny_cfg


def test_split_gates_heads_returns_gated_product_of_pre_gate_and_projection() -> None:
    torch.manual_seed(0)
    block = LivDoubleGatedBase(make_tiny_cfg())
    x = make_inputs(B=1, T=6, D=32)
    pre_gate, post_gate, x_proj, gated = block.split_gates_heads(x)
    for view in (pre_gate, post_gate, x_proj, gated):
        assert tuple(view.shape) == (1, 6, 4, 8)
    torch.testing.assert_close(gated, pre_gate * x_proj, atol=1e-6, rtol=1e-6)


def test_finish_heads_projects_gated_heads_back_to_hidden_size() -> None:
    torch.manual_seed(1)
    block = LivDoubleGatedBase(make_tiny_cfg())
    post_gate = torch.randn(2, 5, 4, 8)
    mixed = torch.randn(2, 5, 4, 8)
    out = block.finish_heads(post_gate, mixed)
    assert tuple(out.shape) == (2, 5, 32)


def test_base_mix_heads_requires_subclass() -> None:
    block = LivDoubleGatedBase(make_tiny_cfg())
    with pytest.raises(NotImplementedError):
        block.mix_heads(torch.randn(1, 4, 4, 8))
