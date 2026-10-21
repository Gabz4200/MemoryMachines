"""Constructor parameter guards of the Titans/HOPE fast-weight memories.

Covers the shared ``hidden_size/num_heads must be positive`` guard in
``TitansFastWeightsMemory.__init__`` and ``SelfReferentialTitanMemory.__init__``,
asserted by no test today. The neighbouring ``must be divisible by num_heads``
guard is already asserted in
``tests/test_titans_paper_parity.py::test_error_paths`` and
``tests/test_hope_selfref_parity.py::test_selfref_error_paths``, so it is not
duplicated here.

Also covers the unknown-backend / non-positive-chunk-size constructor guards of
both memories, the ``self_referential_titan`` dispatcher's unknown backend, the
``ContinuumMemorySystem`` level-count guard, and the self-referential forward
guard rejecting a ``conv_state`` when the short conv is disabled.
"""

import pytest
import torch

from memory_machines.models.blocks.fast_weights import (
    ContinuumMemorySystem,
    SelfReferentialTitanMemory,
    TitansFastWeightsMemory,
    self_referential_titan,
)
from memory_machines.utils.config_schema import MemoryMachineConfig

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


@pytest.mark.parametrize("backend", ["bogus", "scan"], ids=["typo", "unsupported"])
def test_unknown_backend_raises(backend: str) -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        TitansFastWeightsMemory(hidden_size=16, num_heads=2, backend=backend)
    with pytest.raises(ValueError, match="Unknown backend"):
        SelfReferentialTitanMemory(hidden_size=16, num_heads=2, backend=backend)


@pytest.mark.parametrize("chunk_size", [0, -1], ids=["zero", "negative"])
def test_non_positive_chunk_size_raises(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        TitansFastWeightsMemory(hidden_size=16, num_heads=2, chunk_size=chunk_size)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        SelfReferentialTitanMemory(hidden_size=16, num_heads=2, chunk_size=chunk_size)


def test_self_referential_titan_dispatcher_unknown_backend_raises() -> None:
    mem = SelfReferentialTitanMemory(hidden_size=16, num_heads=2, memory_depth=1, use_short_conv=False)
    state = mem._init_state(1, torch.device("cpu"), torch.float32)
    heads = torch.randn(1, 2, 2, 8)
    q_all = torch.randn(1, 2, 2, 8)
    with pytest.raises(ValueError, match="Unknown backend"):
        self_referential_titan(heads, q_all, state, backend="bogus")


def test_continuum_memory_system_zero_levels_raises() -> None:
    cfg = MemoryMachineConfig(hidden_size=16, num_attention_heads=2, num_key_value_heads=2, csa_num_output_groups=2)
    with pytest.raises(ValueError, match="num_levels >= 1"):
        ContinuumMemorySystem(cfg, num_levels=0)


def test_self_referential_forward_rejects_conv_state_without_short_conv() -> None:
    mem = SelfReferentialTitanMemory(hidden_size=16, num_heads=2, memory_depth=1, use_short_conv=False)
    with pytest.raises(ValueError, match="conv_state given but use_short_conv=False"):
        mem(torch.randn(1, 3, 16), state={"conv_state": torch.zeros(1)})
