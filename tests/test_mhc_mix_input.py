"""Direct coverage for :meth:`SinglePassMHC.mix_input` (the mHC input-mixing step).

``mix_input`` is the static ``[B, L, 1, n] @ [B, L, n, d] -> [B, L, 1, d]``
mixer used by ``SinglePassMHC.forward``/``forward_parallel`` and therefore by
``MHCAttentionGDN2``, but it had no test of its own. These pin the matmul
contract, the equal-weight stream-mean property, and the coefficient-shape
guard. No model is constructed (the mixer is a pure staticmethod).
"""

import pytest
import torch

from memory_machines.models.blocks.memory import SinglePassMHC


def test_mix_input_equals_matmul_over_streams() -> None:
    torch.manual_seed(0)
    streams = torch.randn(2, 5, 3, 4)
    a = torch.softmax(torch.randn(2, 5, 1, 3), dim=-1)
    mixed = SinglePassMHC.mix_input(streams, a)
    assert tuple(mixed.shape) == (2, 5, 1, 4)
    torch.testing.assert_close(mixed, torch.matmul(a, streams), atol=1e-6, rtol=1e-6)


def test_mix_input_unit_weights_return_stream_mean() -> None:
    torch.manual_seed(1)
    streams = torch.randn(1, 6, 4, 8)
    a = torch.full((1, 6, 1, 4), 0.25)
    mixed = SinglePassMHC.mix_input(streams, a)
    torch.testing.assert_close(mixed.squeeze(2), streams.mean(dim=2), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "a_shape",
    [(2, 5, 1, 3), (1, 6, 1, 3), (1, 5, 1, 4), (1, 5, 3, 3)],
    ids=["batch_mismatch", "length_mismatch", "stream_count_mismatch", "singleton_axis_mismatch"],
)
def test_mix_input_incompatible_coefficients_raise(a_shape: tuple[int, ...]) -> None:
    torch.manual_seed(2)
    streams = torch.randn(1, 5, 3, 4)
    a = torch.randn(*a_shape)
    with pytest.raises(ValueError, match="incompatible with streams"):
        SinglePassMHC.mix_input(streams, a)
