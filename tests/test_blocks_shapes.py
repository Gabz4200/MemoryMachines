"""Shape and cache-plumbing smoke tests; tiny_cfg() is shared by other test modules."""

import torch

from memory_machines.models.blocks import (
    BlockCompressor,
    GroupedQueryAttention,
)
from memory_machines.utils.config_schema import MemoryMachineConfig


def tiny_cfg() -> MemoryMachineConfig:
    return MemoryMachineConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_perceiver_query=2,
        window_size=8,
        csa_compression_ratio=2,
        csa_top_k=2,
        csa_num_output_groups=2,
        csa_output_group_dim=8,
        csa_indexer_heads=4,
        csa_indexer_dim=8,
        csa_hier_pool_blocks=4,
        csa_hier_pool_positions=16,
        csa_hier_block_size=2,
    )


def test_block_compressor_shapes_and_block_ends():
    cfg = tiny_cfg()
    comp = BlockCompressor(cfg.hidden_size, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads, 2)
    x = torch.randn(1, 7, cfg.hidden_size)
    c, ends = comp(x)
    assert tuple(c.shape) == (1, 4, cfg.num_key_value_heads, 8)
    assert ends.tolist() == [1, 3, 5, 7]


def test_gqa_forward_cache_grows():
    cfg = tiny_cfg()
    model = GroupedQueryAttention(cfg).eval()
    x1 = torch.randn(1, 5, cfg.hidden_size)
    out1, _, past = model(x1, use_cache=True)
    assert tuple(out1.shape) == (1, 5, cfg.hidden_size)
    assert past is not None
    assert tuple(past[0].shape) == (1, cfg.num_key_value_heads, 5, 8)
    x2 = torch.randn(1, 3, cfg.hidden_size)
    out2, _, past2 = model(
        x2,
        past_key_value=past,
        use_cache=True,
        past_positions=torch.arange(5).unsqueeze(0),
        position_ids=torch.arange(5, 8).unsqueeze(0),
    )
    assert tuple(out2.shape) == (1, 3, cfg.hidden_size)
    assert past2 is not None
    assert tuple(past2[0].shape) == (1, cfg.num_key_value_heads, 8, 8)
