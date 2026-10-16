"""Config <-> blocks coverage guards.

``MemoryMachineConfig`` is the single source of truth for the block library:
every ``config.<field>`` a block reads must exist, and every field must have a
consumer (no dead knobs). These guards fail loudly if a block and the schema
drift apart.
"""

import dataclasses
import pathlib
import re

import pytest

from memory_machines.utils.config_schema import MemoryMachineConfig

BLOCKS = pathlib.Path(__file__).resolve().parents[1] / "src/memory_machines/models/blocks"
SRC = pathlib.Path(__file__).resolve().parents[1] / "src/memory_machines"

# Generation/IO-level HF-compat flags with no block-level consumer.
_HF_COMPAT = {
    "bos_token_id",
    "eos_token_id",
    "output_attentions",
    "output_hidden_states",
    "tie_word_embeddings",
    "use_cache",
}
# Resolved dynamically in mhc_parallel via getattr(config, _variant_window_attr).
_DYNAMIC = {"window_size_sparse", "window_size_shared"}


def _fields() -> set[str]:
    return {f.name for f in dataclasses.fields(MemoryMachineConfig)}


def _referenced(root: pathlib.Path) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {}
    for path in root.rglob("*.py"):
        if path.name == "config_schema.py":
            continue
        for match in re.finditer(r"(?:config|cfg)\.([a-z_][a-z0-9_]*)", path.read_text()):
            refs.setdefault(match.group(1), set()).add(str(path))
    return refs


def test_block_config_references_are_real_fields():
    fields = _fields()
    unknown = {k: sorted(v) for k, v in _referenced(BLOCKS).items() if k not in fields}
    assert not unknown, f"blocks read config fields absent from MemoryMachineConfig: {unknown}"


def test_no_dead_config_fields():
    dead = _fields() - set(_referenced(SRC)) - _HF_COMPAT - _DYNAMIC
    assert not dead, f"MemoryMachineConfig fields with no consumer: {sorted(dead)}"


def test_titan_knobs_are_config_controlled():
    from memory_machines.models.blocks.fast_weights import SelfReferentialTitanMixer

    cfg = MemoryMachineConfig(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        csa_num_output_groups=2,
        titan_memory_depth=1,
        titan_fast_expansion=2.0,
        titan_objective="dot",
        titan_use_short_conv=False,
        titan_chunk_size=4,
    )
    mixer = SelfReferentialTitanMixer(cfg)
    assert mixer.memory.memory_depth == 1
    assert mixer.memory.objective == "dot"
    assert mixer.memory.use_short_conv is False
    assert mixer.memory.fast_dim == 2 * mixer.memory.head_dim
    assert mixer.chunk_size == 4


def test_config_rejects_bad_titan_fields():
    with pytest.raises(ValueError, match="titan_memory_depth"):
        MemoryMachineConfig(titan_memory_depth=0)
    with pytest.raises(ValueError, match="titan_objective"):
        MemoryMachineConfig(titan_objective="bogus")
    with pytest.raises(ValueError, match="titan_chunk_size"):
        MemoryMachineConfig(titan_chunk_size=-1)


def test_cms_ffn_and_norm_are_config_controlled():
    from torch import nn

    from memory_machines.models.blocks.dyt import DyT
    from memory_machines.models.blocks.fast_weights import ContinuumMemorySystem, HopeAttentionBlock, HopeBlock
    from memory_machines.models.blocks.ffn import ClassicMLP, SwiGLU

    default = ContinuumMemorySystem(MemoryMachineConfig(hidden_size=16))
    assert all(isinstance(level.ffn, SwiGLU) and isinstance(level.norm, DyT) for level in default.levels)

    cfg = MemoryMachineConfig(hidden_size=16, cms_ffn="classic_mlp", cms_norm="rmsnorm")
    cms = ContinuumMemorySystem(cfg)
    assert all(level.kind == "classic_mlp" and isinstance(level.ffn, ClassicMLP) and isinstance(level.norm, nn.RMSNorm) for level in cms.levels)

    tiny = MemoryMachineConfig(hidden_size=16, num_attention_heads=2, num_key_value_heads=2, csa_num_output_groups=2)
    assert HopeBlock(tiny).cms.hidden_size == 16
    assert HopeAttentionBlock(tiny, num_heads=2).cms.hidden_size == 16


def test_config_rejects_bad_cms_fields():
    with pytest.raises(ValueError, match="cms_ffn"):
        MemoryMachineConfig(cms_ffn="bogus")
    with pytest.raises(ValueError, match="cms_norm"):
        MemoryMachineConfig(cms_norm="bogus")
