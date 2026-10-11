"""Naive reference oracle test for dw_scale."""

import torch

from project.ops.dispatch import dispatch_dw_scale


def test_naive_reference():
    x = torch.randn(3, 5)
    w = torch.randn(3, 5)
    out = dispatch_dw_scale(x, w, backend="naive")
    torch.testing.assert_close(out, x * w)
