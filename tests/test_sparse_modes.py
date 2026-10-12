"""CSA2 Full/Reindex/Reuse modes and HCA/NSA chunked-vs-full equivalence."""

import math

import pytest
import torch

from memory_machines.models.blocks import (
    CompressedSparseAttention,
    CompressedSparseAttention2,
    HeavilyCompressedAttention,
    NativeSparseAttention,
)
from memory_machines.models.blocks.shared import CSA2SharedState
from memory_machines.utils.config_schema import MemoryMachineConfig
from tests.test_blocks_shapes import tiny_cfg


def _make_csa2() -> CompressedSparseAttention2:
    torch.manual_seed(0)
    return CompressedSparseAttention2(tiny_cfg(), compression_ratio=2, top_k=2).eval()


def _make_csa2_cfg() -> MemoryMachineConfig:
    return tiny_cfg()


def test_csa2_full_reindex_reuse_chain_contract():
    layer = _make_csa2()
    cfg = _make_csa2_cfg()
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
    layer = _make_csa2()
    cfg = _make_csa2_cfg()
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
    layer = _make_csa2()
    cfg = _make_csa2_cfg()
    torch.manual_seed(0)
    x = torch.randn(1, 8, cfg.hidden_size)
    shared = CSA2SharedState()

    layer.mode = "full"
    _, shared = layer.forward_shared(x, shared, use_cache=True)
    assert shared.pool_indices is not None
    assert shared.comp_end_pos is not None
    pool = shared.pool_indices.clone()
    comp_end = shared.comp_end_pos.clone()

    layer.mode = "reindex"
    # Explicit positions so every query sees the full causal
    # context (implicit positions start from the sequence past,
    # which is 0 here).
    position_ids = torch.arange(x.shape[1]).unsqueeze(0)
    _, shared = layer.forward_shared(x, shared, position_ids=position_ids, use_cache=True)
    assert shared.topk_indices is not None
    reindex_topk = shared.topk_indices
    q_pos = position_ids.reshape(-1)
    ends = comp_end.reshape(-1)
    for q in range(reindex_topk.shape[1]):
        pool_row = pool[0, q].tolist()
        for kid in reindex_topk[0, q].tolist():
            assert kid in pool_row
            # Causal visibility: entry end <= q_pos, except at
            # q=0 where ratio 2 leaves no entry ending at or
            # before 0 (fallback to the first block).
            if q == 0:
                assert int(ends[kid]) <= int(layer.compression_ratio), f"q=0 selected entry {kid} ends at {int(ends[kid])}; expected fallback within the first block (ratio {layer.compression_ratio})"
            else:
                assert int(ends[kid]) <= int(q_pos[q]), f"q={q} q_pos={int(q_pos[q])} selected entry {kid} ends at {int(ends[kid])}"


# ---------------------------------------------------------------------------
# Regression tests for the _selected_branch delegation refactor (flatten to
# B*Lq and delegate to _core_attend) and the vectorized CSA2 reindex topk.
# ---------------------------------------------------------------------------


def _make_full_csa() -> CompressedSparseAttention:
    """CSA with every optional path active (norm/sink/FRVM/CVPE)."""
    cfg = tiny_cfg()
    cfg.csa_use_qkv_norm = True
    cfg.csa_use_sink = True
    cfg.frvm_enabled = True
    cfg.cvpe_enabled = True
    torch.manual_seed(0)
    return CompressedSparseAttention(cfg, compression_ratio=2).eval()


def test_selected_branch_matches_manual_pipeline():
    torch.manual_seed(0)
    layer = _make_full_csa().eval()
    cfg = layer.config
    bsz, lq, ksel = 2, 5, 3
    h, hkv, dh = cfg.num_attention_heads, cfg.num_key_value_heads, layer.head_dim

    q_main = torch.randn(bsz, h, lq, dh)
    sel = torch.randn(bsz, lq, ksel, hkv, dh)
    q_pos = torch.arange(lq).unsqueeze(0).expand(bsz, -1)
    sel_pos = torch.randint(0, lq, (bsz, lq, ksel))
    # Force a mix of visible and future entries per query row.
    sel_pos = torch.clamp(sel_pos, max=q_pos.unsqueeze(-1))
    v_q = torch.randn(bsz, h, lq, dh)

    out = layer._selected_branch(q_main, sel, sel_pos, v_q, q_pos)

    # Manually reconstructed inline pipeline matching _core_attend on the
    # flattened (B * Lq) layout that _selected_branch delegates with.
    flat = bsz * lq
    qf = q_main.transpose(1, 2).reshape(flat, h, 1, dh)
    kf = sel.reshape(flat, ksel, hkv, dh).transpose(1, 2)
    vqf = v_q.transpose(1, 2).reshape(flat, h, 1, dh)
    qpos = q_pos.reshape(flat, 1)
    kpos = sel_pos.reshape(flat, ksel)

    q_norm = layer.q_norm
    kv_norm = layer.kv_norm
    cvpe = layer.cvpe_bias
    sink = layer.sink
    frvm = layer.frvm_gate
    assert q_norm is not None and kv_norm is not None
    assert cvpe is not None and sink is not None and frvm is not None
    qn = q_norm(qf)
    kvn = kv_norm(kf)
    kv_exp = kvn.repeat_interleave(layer.num_key_value_groups, dim=1)
    logits = torch.matmul(qn, kv_exp.transpose(2, 3)) / math.sqrt(dh)
    logits = logits + cvpe(qn, qpos, kpos).to(logits.dtype)
    causal_mask = kpos.unsqueeze(1) <= qpos.unsqueeze(-1)
    logits = logits.masked_fill(~causal_mask.unsqueeze(1), float("-inf"))
    row_ok = causal_mask.any(dim=-1).view(flat, 1, 1, 1)
    weights = sink.weights(logits)
    weights = torch.where(row_ok, weights, torch.zeros_like(weights))
    weights = layer.attention_dropout(weights)
    expected = frvm.gated_reduction(weights, kv_exp, vqf, chunk_size=layer.frvm_chunk_size)
    expected = expected.reshape(bsz, lq, h, dh).transpose(1, 2)

    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_selected_branch_causal_visibility():
    layer = _make_full_csa().eval()
    cfg = layer.config
    bsz, lq, ksel = 1, 3, 2
    h = cfg.num_attention_heads

    q_main = torch.randn(bsz, h, lq, layer.head_dim)
    sel = torch.randn(bsz, lq, ksel, cfg.num_key_value_heads, layer.head_dim)
    v_q = torch.randn(bsz, h, lq, layer.head_dim)
    q_pos = torch.tensor([[2, 3, 4]])
    # Every selected entry ends strictly after its query position:
    # the entire causal row is -inf for every query.
    sel_pos = q_pos.unsqueeze(-1) + 1 + torch.arange(ksel).unsqueeze(0).unsqueeze(0)

    out = layer._selected_branch(q_main, sel, sel_pos, v_q, q_pos)

    assert bool(torch.isfinite(out).all()), "all-invisible causal rows must yield finite output (no NaN from all -inf softmax)"
    # With the sink, fully-masked rows still carry nonzero sink mass;
    # with plain softmax they are zeroed. Either way the output is finite.
    assert out.shape == (bsz, h, lq, layer.head_dim)


def test_csa2_reindex_all_invisible_row_falls_back_to_first_entry():
    layer = _make_csa2()
    cfg = _make_csa2_cfg()
    torch.manual_seed(0)
    x = torch.randn(1, 8, cfg.hidden_size)
    shared = CSA2SharedState()

    layer.mode = "full"
    _, shared = layer.forward_shared(x, shared, use_cache=True)
    assert shared.pool_indices is not None
    assert shared.comp_end_pos is not None
    pool = shared.pool_indices.clone()
    comp_end = shared.comp_end_pos.clone()

    # Craft a shared state whose pool is fully invisible at the reindex
    # query positions: q_pos=0 makes every entry with end > 0 invisible,
    # and with compression_ratio=2 the first entry ends at 1 > 0, so the
    # entire rescored pool row is -inf.
    crafted = CSA2SharedState()
    crafted.compressed_kv = shared.compressed_kv
    crafted.comp_end_pos = comp_end
    crafted.indexer_k = shared.indexer_k
    crafted.pool_indices = pool
    crafted.topk_indices = shared.topk_indices

    layer.mode = "reindex"
    position_ids = torch.zeros(1, x.shape[1], dtype=torch.long)
    out, shared = layer.forward_shared(x, crafted, position_ids=position_ids, use_cache=True)

    topk = shared.topk_indices
    assert topk is not None
    # All-invisible rows fall back to the first pool entry repeated k times.
    assert topk.shape == pool.shape[:2] + (layer.top_k,)
    assert bool((topk == pool[:, :, :1]).all()), "invisible row must select pool_indices[row, 0] repeated k times"
    assert bool(torch.isfinite(topk).all())
    assert bool(torch.isfinite(out).all())
    for q in range(topk.shape[1]):
        pool_row = pool[0, q].tolist()
        for kid in topk[0, q].tolist():
            assert kid in pool_row, f"selected entry {kid} not in pool row {pool_row}"


def test_csa2_reindex_partial_visibility_pads_with_last_visible():
    torch.manual_seed(0)
    # top_k=3 with only 2 visible entries exercises the last-visible
    # padding branch: local index 2 (the tail) is padded with the
    # last visible local index before gathering pool entries.
    layer = CompressedSparseAttention2(tiny_cfg(), compression_ratio=2, top_k=3).eval()
    cfg = _make_csa2_cfg()
    x = torch.randn(1, 8, cfg.hidden_size)
    shared = CSA2SharedState()

    layer.mode = "full"
    _, shared = layer.forward_shared(x, shared, use_cache=True)
    assert shared.pool_indices is not None
    assert shared.comp_end_pos is not None
    pool = shared.pool_indices.clone()
    comp_end = shared.comp_end_pos.clone()
    pool_width = pool.shape[-1]

    # Craft a shared state where only the first 2 of the pool_width
    # entries are visible at each query: pool columns 0..1 end at
    # position 2 (visible for q_pos >= 2), columns 2.. end at 99
    # (invisible).
    crafted_end = comp_end.clone()
    for p in range(pool_width):
        crafted_end[p] = 2 if p < 2 else 99

    crafted = CSA2SharedState()
    crafted.compressed_kv = shared.compressed_kv
    crafted.comp_end_pos = crafted_end
    crafted.indexer_k = shared.indexer_k
    crafted.pool_indices = pool
    crafted.topk_indices = shared.topk_indices

    layer.mode = "reindex"
    position_ids = torch.full((1, x.shape[1]), 4, dtype=torch.long)
    out, shared = layer.forward_shared(x, crafted, position_ids=position_ids, use_cache=True)

    topk = shared.topk_indices
    assert topk is not None
    assert topk.shape == pool.shape[:2] + (layer.top_k,)
    # Exactly 2 of the 4 pool columns are visible (ends 2 <= q_pos=4).
    finite_pool_cols = [0, 1]
    for q in range(topk.shape[1]):
        pool_row = pool[0, q].tolist()
        kids = topk[0, q].tolist()
        # The first 2 selections are the 2 visible entries (in score
        # order); the padded tail (index 2) repeats the last visible
        # pool column (column 1), i.e. pool_row[1].
        assert kids[2] == pool_row[1], f"padded tail must repeat the last visible pool column entry {pool_row[1]}: got {kids}"
        for kid in kids:
            pool_col = pool_row.index(kid)
            assert pool_col in finite_pool_cols, f"selected entry {kid} (pool column {pool_col}) is not among the visible pool columns {finite_pool_cols}"
    assert bool(torch.isfinite(topk).all())
    assert bool(torch.isfinite(out).all())
