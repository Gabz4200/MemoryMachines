"""Chunked-vs-full equivalence for the sparse forward path."""

import torch

from memory_machines.models.blocks import CompressedSparseAttention
from memory_machines.utils.config_schema import MemoryMachineConfig
from tests.test_blocks_shapes import tiny_cfg


def test_chunked_vs_full_equivalence():
    torch.manual_seed(0)
    cfg: MemoryMachineConfig = tiny_cfg()
    layer = CompressedSparseAttention(cfg).eval()
    x = torch.randn(1, 8, cfg.hidden_size)
    out_full, _, _ = layer(x)
    out1, _, past = layer(x[:, :4, :], use_cache=True)
    assert past is not None
    out2, _, _ = layer(x[:, 4:, :], past=past)
    out_chunked = torch.cat([out1, out2], dim=1)
    torch.testing.assert_close(out_full, out_chunked, atol=1e-5, rtol=1e-5)
