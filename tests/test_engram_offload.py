"""Tests for automatic SSD offloading of the Engram hash tables.

Pins the invariants of the offload path: SSD-backed tables produce
the same forward as RAM tables (same seed), gradients scatter back
into the memmap weight, the "auto" mode picks SSD only above the
RAM threshold, and the config rejects bad offload settings.
"""

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from memory_machines.models.blocks.memory import EngramMemory, _SsdTable
from memory_machines.utils.config_schema import MemoryMachineConfig
from tests.test_blocks_shapes import tiny_cfg


def _engram_cfg(**overrides) -> MemoryMachineConfig:
    base = replace(
        tiny_cfg(),
        engram_orders=(2, 3),
        engram_hash_heads=2,
        engram_dim_per_order=8,
        engram_num_entries=1024,
    )
    return replace(base, **overrides)


def test_ssd_forward_matches_ram():
    """SSD-backed tables must match RAM tables for the same seed."""
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as d:
        cfg_ram = _engram_cfg(engram_offload="ram", seed=42)
        cfg_ssd = _engram_cfg(engram_offload="ssd", seed=42, engram_offload_dir=d)
        mem_ram = EngramMemory(cfg_ram).eval()
        mem_ssd = EngramMemory(cfg_ssd).eval()
        ids = torch.randint(0, 500, (2, 7))
        hidden = torch.randn(2, 7, cfg_ram.hidden_size)
        out_ram = mem_ram(ids, hidden, layer_id=0)
        out_ssd = mem_ssd(ids, hidden, layer_id=0)
        torch.testing.assert_close(out_ram, out_ssd, atol=1e-5, rtol=1e-5)


def test_ssd_gradients_scatter_to_memmap():
    """The embedding weight must receive a dense scatter gradient."""
    torch.manual_seed(1)
    with tempfile.TemporaryDirectory() as d:
        cfg = _engram_cfg(engram_offload="ssd", engram_offload_dir=d)
        mem = EngramMemory(cfg).eval()
        torch.nn.init.ones_(mem.value_proj.weight)
        ids = torch.randint(0, 500, (1, 5))
        hidden = torch.randn(1, 5, cfg.hidden_size)
        out = mem(ids, hidden, layer_id=0)
        out.sum().backward()
        grad = mem.multi_head_embedding.weight.grad
        assert grad is not None
        assert tuple(grad.shape) == (sum(mem.primes), mem.dim)
        assert bool(grad.abs().sum() > 0), "gathered rows must receive nonzero gradient"


def test_ssd_weight_is_trainable_parameter():
    """The memmap weight must be a Parameter in module.parameters()."""
    with tempfile.TemporaryDirectory() as d:
        cfg = _engram_cfg(engram_offload="ssd", engram_offload_dir=d)
        mem = EngramMemory(cfg)
        table = mem.multi_head_embedding
        assert isinstance(table.weight, torch.nn.Parameter), "weight must be a Parameter"
        assert any(p is table.weight for p in mem.parameters()), "SSD table must appear in module.parameters() so the optimizer updates it"
        # Shared storage: the Parameter must alias the memmap, not copy it.
        table._mmap[0, 0] = 123.0
        assert float(table.weight.detach()[0, 0]) == 123.0

        # An optimizer step through the training path writes through to the file.
        torch.nn.init.ones_(mem.value_proj.weight)
        ids = torch.randint(0, 500, (1, 3))
        hidden = torch.randn(1, 3, cfg.hidden_size)
        out = mem(ids, hidden, layer_id=0)
        out.sum().backward()
        opt = torch.optim.SGD([table.weight], lr=1.0)
        before = table.weight.detach().clone()
        opt.step()
        assert not torch.equal(table.weight.detach(), before), "optimizer step must change weights"


def test_auto_offload_threshold():
    """ "auto" must offload only above the RAM threshold."""
    with tempfile.TemporaryDirectory() as d:
        tiny_threshold = _engram_cfg(engram_offload="auto", engram_offload_ram_threshold=1, engram_offload_dir=d)
        assert isinstance(EngramMemory(tiny_threshold).multi_head_embedding, _SsdTable)
        huge_threshold = _engram_cfg(engram_offload="auto", engram_offload_ram_threshold=10**12, engram_offload_dir=d)
        assert isinstance(EngramMemory(huge_threshold).multi_head_embedding, torch.nn.Embedding)


def test_ssd_table_persists_and_reloads():
    """A reused SSD file must keep its contents across loads."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "t.bin"
        table = _SsdTable(path, 8, 4)
        with torch.no_grad():
            table.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4))
        del table
        reloaded = _SsdTable(path, 8, 4)
        expected = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        torch.testing.assert_close(reloaded.weight.detach(), expected)


def test_bad_offload_config_rejected():
    """Invalid offload settings must fail at config validation."""
    for bad in ("disk", "", None):
        try:
            _engram_cfg(engram_offload=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"engram_offload={bad!r} must be rejected")
    try:
        _engram_cfg(engram_offload_ram_threshold=0)
    except ValueError:
        pass
    else:
        raise AssertionError("threshold <= 0 must be rejected")
