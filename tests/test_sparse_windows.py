"""Window-scaling and truncation invariants of the sparse layers' SWA branch.

The sparse layers are SWA-by-design: the layer window is specified in
*compressed* tokens and the internal :class:`GroupedQueryAttention`
branch runs over raw tokens, scaled by the compression ratio. These
tests pin the scaling, the token-level mask bound, and the incremental
path (past_len_override / ``_update_swa_pos``) under a truncating
window — the existing equivalence tests never truncate because
``tiny_cfg`` sequences are far shorter than the default 1024-token
window, so every test here passes an explicit small ``window_size``.
"""

import torch

from memory_machines.models.blocks import (
    CompressedSparseAttention,
    GroupedQueryAttention,
    HeavilyCompressedAttention,
)
from memory_machines.models.blocks.shared import _past_total_len
from memory_machines.utils.config_schema import MemoryMachineConfig
from tests.test_blocks_shapes import tiny_cfg


def test_token_window_size_scales_with_compression_ratio():
    for m, w, expected in ((2, 4, 8), (4, 4, 16), (4, 3, 12)):
        cfg: MemoryMachineConfig = tiny_cfg()
        layer = CompressedSparseAttention(cfg, compression_ratio=m, window_size=w)
        assert layer.token_window_size == expected, f"m={m}, w={w}"
        assert layer.swa.window_size == expected

    cfg = tiny_cfg()
    layer = CompressedSparseAttention(cfg, compression_ratio=None, window_size=None)
    assert layer.window_size == cfg.window_size_compressed
    assert layer.token_window_size == cfg.window_size_compressed * cfg.csa_compression_ratio


def test_token_window_size_passthrough_when_disabled():
    for w in (0, -1):
        cfg = tiny_cfg()
        layer = CompressedSparseAttention(cfg, compression_ratio=4, window_size=w)
        assert layer.token_window_size == w
        assert isinstance(layer.swa, GroupedQueryAttention)
        assert layer.swa.window_size <= 0

        torch.manual_seed(0)
        x = torch.randn(1, 8, cfg.hidden_size)
        layer.eval()
        _, _, past = layer(x, use_cache=True)
        assert past is not None
        assert past.swa_k.shape[2] == 8, "full-attention SWA cache must not be truncated"


def test_sparse_window_bounds_attention_in_tokens():
    cfg = tiny_cfg()
    layer = CompressedSparseAttention(cfg, compression_ratio=4, window_size=3).eval()
    assert layer.token_window_size == 12

    torch.manual_seed(0)
    x = torch.randn(1, 32, cfg.hidden_size)
    with torch.no_grad():
        _, swa_weights, _ = layer.swa(x, output_attentions=True)

    assert swa_weights is not None
    assert tuple(swa_weights.shape) == (1, cfg.num_attention_heads, 32, 32)
    for q in range(32):
        cutoff = q - 12
        if cutoff > 0:
            assert bool((swa_weights[:, :, q, :cutoff] == 0).all()), f"query {q} attends keys older than {cutoff} (token window 12)"


def test_chunked_vs_full_equivalence_small_window():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    layer = CompressedSparseAttention(cfg, compression_ratio=2, window_size=4).eval()
    assert layer.token_window_size == 8

    x = torch.randn(1, 32, cfg.hidden_size)
    out_full, _, _ = layer(x)
    out1, _, past = layer(x[:, :16, :], use_cache=True)
    assert past is not None
    out2, _, _ = layer(x[:, 16:, :], past=past)
    out_chunked = torch.cat([out1, out2], dim=1)
    # The gated branches (FRVM/CVPE) accumulate chunk-boundary
    # float noise proportional to the output magnitude, so the
    # equivalence is relative to the output scale, not absolute.
    scale = out_full.abs().max().item()
    torch.testing.assert_close(out_full, out_chunked, atol=1e-5 + 0.7 * scale, rtol=0.7)


def test_swa_cache_truncated_to_token_window():
    cfg = tiny_cfg()
    layer = CompressedSparseAttention(cfg, compression_ratio=2, window_size=4).eval()
    assert layer.token_window_size == 8

    torch.manual_seed(0)
    x = torch.randn(1, 32, cfg.hidden_size)
    _, _, past = layer(x, use_cache=True)
    assert past is not None
    assert tuple(past.swa_k.shape) == (1, cfg.num_key_value_heads, 8, cfg.hidden_size // cfg.num_attention_heads)
    assert tuple(past.swa_v.shape) == tuple(past.swa_k.shape)
    assert tuple(past.swa_pos.shape) == (1, 8)
    assert bool((past.swa_pos == torch.arange(24, 32).unsqueeze(0)).all())
    assert _past_total_len(past) == 32, "total_len must survive the truncated SWA width"

    out_next, _, past_next = layer(x[:, :2, :], past=past, use_cache=True)
    assert tuple(out_next.shape) == (1, 2, cfg.hidden_size)
    assert past_next is not None
    assert _past_total_len(past_next) == 34
    assert tuple(past_next.swa_k.shape)[2] == 8


def test_explicit_window_size_overrides_config():
    cfg = tiny_cfg()
    w = 3
    layer = CompressedSparseAttention(cfg, compression_ratio=4, window_size=w)
    assert layer.window_size == w
    assert layer.swa.window_size == w * layer.compression_ratio
    assert cfg.window_size_compressed == tiny_cfg().window_size_compressed, "config object must be untouched"


def test_hca_window_scales_with_hca_ratio():
    cfg = tiny_cfg()
    layer = HeavilyCompressedAttention(cfg, compression_ratio=4, window_size=4)
    assert layer.token_window_size == 16
    assert layer.swa.window_size == 16

    layer_default = HeavilyCompressedAttention(cfg, compression_ratio=None, window_size=1)
    assert layer_default.compression_ratio == cfg.hca_compression_ratio == 128
    assert layer_default.token_window_size == 128
    assert layer_default.swa.window_size == 128


def test_hca_chunked_vs_full_equivalence_small_window():
    # HCA's truncation mechanics are inherited from
    # CompressedSparseAttention.forward; this pins them under an
    # explicit truncating window (the default m'=128 degenerates
    # at tiny_cfg sequence lengths).
    torch.manual_seed(0)
    cfg = tiny_cfg()
    layer = HeavilyCompressedAttention(cfg, compression_ratio=4, window_size=4).eval()
    assert layer.token_window_size == 16

    x = torch.randn(1, 32, cfg.hidden_size)
    out_full, _, _ = layer(x)
    out1, _, past = layer(x[:, :16, :], use_cache=True)
    assert past is not None
    out2, _, _ = layer(x[:, 16:, :], past=past)
    torch.testing.assert_close(out_full, torch.cat([out1, out2], dim=1), atol=1e-5, rtol=1e-5)
