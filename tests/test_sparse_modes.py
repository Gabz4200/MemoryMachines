"""CSA2 Full/Reindex/Reuse modes and HCA/NSA chunked-vs-full equivalence."""

import pytest
import torch

from memory_machines.models.blocks import (
    CompressedSparseAttention2,
    HeavilyCompressedAttention,
    NativeSparseAttention,
)
from memory_machines.models.blocks.shared import CSA2SharedState
from tests.test_blocks_shapes import tiny_cfg


def _make_csa2() -> CompressedSparseAttention2:
    torch.manual_seed(0)
    cfg = tiny_cfg()
    return CompressedSparseAttention2(cfg, compression_ratio=2, top_k=2).eval(), cfg


def test_csa2_full_reindex_reuse_chain_contract():
    layer, cfg = _make_csa2()
    torch.manual_seed(0)
    x = torch.randn(1, 8, cfg.hidden_size)
    shared = CSA2SharedState()

    layer.mode = "full"
    out_full, shared = layer.forward_shared(x, shared, use_cache=True)
    assert shared.compressed_kv is not None
    assert shared.topk_indices is not None
    assert shared.pool_indices is not None
    full_comp = shared.compressed_kv.clone()
    full_topk = shared.topk_indices.clone()
    full_pool = shared.pool_indices.clone()

    layer.mode = "reindex"
    out_reindex, shared = layer.forward_shared(x, shared, use_cache=True)
    assert shared.compressed_kv is not None
    assert shared.topk_indices is not None
    torch.testing.assert_close(shared.compressed_kv, full_comp, atol=1e-5, rtol=1e-5)
    # Reindex topk rows are subsets of the corresponding pool rows.
    reindex_topk = shared.topk_indices
    assert reindex_topk.shape == full_topk.shape
    for q in range(reindex_topk.shape[1]):
        for kid in reindex_topk[0, q].tolist():
            assert kid in full_pool[0, q].tolist()
    reindex_topk_saved = reindex_topk.clone()

    layer.mode = "reuse"
    out_reuse, shared = layer.forward_shared(x, shared, use_cache=True)
    torch.testing.assert_close(shared.topk_indices, reindex_topk_saved, atol=0, rtol=0)

    for out in (out_full, out_reindex, out_reuse):
        assert tuple(out.shape) == (1, 8, cfg.hidden_size)
        assert bool(torch.isfinite(out).all())


@pytest.mark.parametrize("layer_kind", ["hca", "nsa"])
def test_hca_nsa_chunked_vs_full_equivalence(layer_kind):
    torch.manual_seed(0)
    cfg = tiny_cfg()
    if layer_kind == "hca":
        # hca_compression_ratio=128 degenerates at L=8.
        layer = HeavilyCompressedAttention(cfg, compression_ratio=2).eval()
    else:
        layer = NativeSparseAttention(cfg).eval()
    x = torch.randn(1, 8, cfg.hidden_size)
    out_full, _, _ = layer(x)
    out1, _, past = layer(x[:, :4, :], use_cache=True)
    assert past is not None
    out2, _, _ = layer(x[:, 4:, :], past=past)
    torch.testing.assert_close(torch.cat([out1, out2], dim=1), out_full, atol=1e-5, rtol=1e-5)


def test_csa2_reuse_decode_expands_last_topk():
    layer, cfg = _make_csa2()
    torch.manual_seed(0)
    x = torch.randn(1, 8, cfg.hidden_size)
    shared = CSA2SharedState()

    layer.mode = "full"
    _, shared = layer.forward_shared(x, shared, use_cache=True)
    assert shared.topk_indices is not None
    full_topk = shared.topk_indices.clone()

    layer.mode = "reuse"
    captured = {}
    orig_gather = type(layer)._gather_selected

    def spy(full_comp, topk):
        captured["topk"] = topk.clone()
        return orig_gather(full_comp, topk)

    layer._gather_selected = spy
    try:
        out_decode, shared = layer.forward_shared(x[:, -1:, :], shared, use_cache=True)
    finally:
        del layer._gather_selected
    assert "topk" in captured
    expected = full_topk[:, -1:, :].expand(1, 1, -1)
    torch.testing.assert_close(captured["topk"], expected, atol=0, rtol=0)
    assert tuple(out_decode.shape) == (1, 1, cfg.hidden_size)
    assert bool(torch.isfinite(out_decode).all())


def test_csa2_reindex_topk_subset_of_pool_and_causal():
    layer, cfg = _make_csa2()
    torch.manual_seed(0)
    x = torch.randn(1, 8, cfg.hidden_size)
    shared = CSA2SharedState()

    layer.mode = "full"
    _, shared = layer.forward_shared(x, shared, use_cache=True)
    assert shared.pool_indices is not None
    assert shared.comp_end_pos is not None
    pool = shared.pool_indices.clone()
    comp_end = shared.comp_end_pos.clone()
    past_len = shared.total_len if shared.total_len is not None else 0

    layer.mode = "reindex"
    _, shared = layer.forward_shared(x, shared, use_cache=True)
    assert shared.topk_indices is not None
    reindex_topk = shared.topk_indices
    q_pos = torch.arange(past_len, past_len + x.shape[1])
    ends = comp_end.reshape(-1)
    for q in range(reindex_topk.shape[1]):
        pool_row = pool[0, q].tolist()
        for kid in reindex_topk[0, q].tolist():
            assert kid in pool_row
            assert int(ends[kid]) <= int(q_pos[q])
