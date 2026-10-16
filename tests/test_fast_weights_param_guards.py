"""Constructor parameter guards of the Titans/HOPE fast-weight memories.

Covers the shared ``hidden_size/num_heads must be positive`` guard in
``TitansFastWeightsMemory.__init__`` and ``SelfReferentialTitanMemory.__init__``,
asserted by no test today. The neighbouring ``must be divisible by num_heads``
guard is already asserted in
``tests/test_titans_paper_parity.py::test_error_paths`` and
``tests/test_hope_selfref_parity.py::test_selfref_error_paths``, so it is not
duplicated here.
"""

import pytest

from memory_machines.models.blocks.fast_weights import (
    SelfReferentialTitanMemory,
    TitansFastWeightsMemory,
)

_NON_POSITIVE_DIMS = [(0, 4), (32, 0)]
_NON_POSITIVE_IDS = ["hidden_size_zero", "num_heads_zero"]


@pytest.mark.parametrize(("hidden_size", "num_heads"), _NON_POSITIVE_DIMS, ids=_NON_POSITIVE_IDS)
def test_titans_fast_weights_memory_non_positive_dims_raise(hidden_size: int, num_heads: int) -> None:
    with pytest.raises(ValueError, match=r"hidden_size/num_heads must be positive"):
        TitansFastWeightsMemory(hidden_size=hidden_size, num_heads=num_heads)


@pytest.mark.parametrize(("hidden_size", "num_heads"), _NON_POSITIVE_DIMS, ids=_NON_POSITIVE_IDS)
def test_self_referential_titan_memory_non_positive_dims_raise(hidden_size: int, num_heads: int) -> None:
    with pytest.raises(ValueError, match=r"hidden_size/num_heads must be positive"):
        SelfReferentialTitanMemory(hidden_size=hidden_size, num_heads=num_heads)
