"""Shape guards of ``gdn2._check_shapes`` reached through the public dispatcher.

Covers the ``q/k/g/b must be 4D`` branch for the key-side tensors ``k``/``g``/
``b`` and the ``k/g must match q`` branch, both via :func:`gated_delta_rule_2`
(not the raw per-path forwards). The 3D-``q``/3D-``v`` spellings of the same
guards already live in ``tests/test_gdn2_validation.py`` and are not repeated.
Small shapes (B=1, T=4, H=1, HV=1, K=4, V=4).
"""

import pytest
import torch

from memory_machines.models.blocks import gdn2 as G


def _inputs(B=1, T=4, H=1, HV=1, K=4, V=4, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, HV, V)
    g = -torch.rand(B, T, H, K)
    b = torch.rand(B, T, HV, K)
    w = torch.rand(B, T, HV, V)
    return q, k, v, g, b, w


@pytest.mark.parametrize("bad", ["k", "g", "b"], ids=["k_3d", "g_3d", "b_3d"])
def test_gated_delta_rule_2_non_4d_key_side_tensor_raises(bad: str) -> None:
    q, k, v, g, b, w = _inputs()
    tensors = {"k": k, "g": g, "b": b}
    tensors[bad] = tensors[bad][0]
    with pytest.raises(ValueError, match=r"q/k/g/b must be 4D"):
        G.gated_delta_rule_2(q, tensors["k"], v, tensors["g"], tensors["b"], w)


@pytest.mark.parametrize("bad", ["k", "g"], ids=["k_time_mismatch", "g_head_mismatch"])
def test_gated_delta_rule_2_key_side_shape_mismatch_raises(bad: str) -> None:
    q, k, v, g, b, w = _inputs()
    if bad == "k":
        k = torch.randn(1, 3, 1, 4)
    else:
        g = -torch.rand(1, 4, 2, 4)
    with pytest.raises(ValueError, match=r"k/g must match q"):
        G.gated_delta_rule_2(q, k, v, g, b, w)
