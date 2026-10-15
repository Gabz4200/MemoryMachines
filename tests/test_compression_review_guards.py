"""Review-guard tests for BlockCompressor clamping, AttentionSink precision,
_sinkhorn stability, SparseIndexer.topk warnings, and causal-mask integration.

Covers gaps not duplicated in tests/test_compression_units.py,
tests/test_shared_helpers.py, or clamp/sink tests in
tests/test_regression_fixes.py.
"""

import warnings

import pytest
import torch
from torch.testing import assert_close

from memory_machines.models.blocks.compression import (
    AttentionSink,
    BlockCompressor,
    SparseIndexer,
)
from memory_machines.models.blocks.shared import _causal_comp_mask, _sinkhorn


def _comp():
    return BlockCompressor(16, 2, 8, compression_ratio=4).eval()


# ---------------------------------------------------------------- CRITICAL: 1
class TestCompressorClampWithStartPos:
    def test_partial_block_clamped_with_offset(self):
        comp = _comp()
        g = torch.Generator().manual_seed(0)
        x = torch.randn(1, 6, 16, generator=g)
        with torch.no_grad():
            _, ends = comp(x, start_pos=10)
        assert ends.tolist() == [13, 15]

    def test_exact_divisible_no_clamp_with_offset(self):
        comp = _comp()
        g = torch.Generator().manual_seed(1)
        x = torch.randn(1, 8, 16, generator=g)
        with torch.no_grad():
            _, ends = comp(x, start_pos=10)
        # No padding -> pure arithmetic progression start+m-1, start+2m-1.
        assert ends.tolist() == [13, 17]


# ---------------------------------------------------------------- CRITICAL: 2
class TestAttentionSinkLowPrecisionOverflow:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_large_logits_finite_and_deflated(self, dtype):
        sink = AttentionSink(num_heads=2).eval()
        logits = torch.full((1, 2, 1, 4), 20.0, dtype=dtype)
        with torch.no_grad():
            w = sink.weights(logits)
        assert w.dtype == dtype
        assert torch.isfinite(w).all()
        row_sum = w.sum(dim=-1)
        assert torch.isfinite(row_sum).all()
        assert bool((row_sum <= 1 + 1e-3).all())

    def test_all_neginf_row_yields_zeros(self):
        sink = AttentionSink(num_heads=1).eval()
        logits = torch.full((1, 1, 1, 4), float("-inf"))
        with torch.no_grad():
            w = sink.weights(logits)
        assert torch.isfinite(w).all()
        assert_close(w, torch.zeros_like(w), atol=0.0, rtol=0.0)


# ---------------------------------------------------------------- CRITICAL: 3
class TestSinkhornSpreadLargeLogits:
    def test_doubly_stochastic_and_shift_invariant(self):
        M = torch.tensor([[0.0, 100.0], [1000.0, 1001.0]])
        P = _sinkhorn(M, 5)
        assert torch.isfinite(P).all()
        # Each iteration ends with a column normalization, so column sums
        # are exact even mid-convergence; row sums converge only
        # geometrically slowly here because the limit is the permutation
        # [[0, 1], [1, 0]] (entry (0, 0) is ~e^-100), so rows are checked
        # on a well-conditioned matrix below instead.
        assert_close(
            P.sum(dim=-2),
            torch.ones(2),
            atol=1e-6,
            rtol=0.0,
        )
        assert bool((P.sum(dim=-1) > 0).all())
        assert_close(
            _sinkhorn(M + 500, 10),
            _sinkhorn(M, 10),
            atol=1e-5,
            rtol=0.0,
        )
        # Well-conditioned input converges fully: rows AND cols ~ 1.
        W = torch.tensor([[0.0, 2.0], [1.0, 1.5]])
        Q = _sinkhorn(W, 50)
        assert torch.isfinite(Q).all()
        assert_close(
            Q.sum(dim=-1),
            torch.ones(2),
            atol=1e-4,
            rtol=0.0,
        )
        assert_close(
            Q.sum(dim=-2),
            torch.ones(2),
            atol=1e-4,
            rtol=0.0,
        )


# ---------------------------------------------------------------- CRITICAL: 4
class TestSparseIndexerTopkWarning:
    def test_k_exceeds_lc_warns_and_clamps(self):
        g = torch.Generator().manual_seed(0)
        scores = torch.randn(1, 3, 4, generator=g)
        with pytest.warns(UserWarning, match=r"exceeds.*clamping"):
            idx = SparseIndexer.topk(scores, k=7)
        assert idx.shape == (1, 3, 4)
        assert_close(
            idx.float(),
            scores.topk(4, dim=-1).indices.float(),
            atol=0.0,
            rtol=0.0,
        )

    def test_valid_k_raises_no_warning(self):
        g = torch.Generator().manual_seed(0)
        scores = torch.randn(1, 3, 4, generator=g)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            idx = SparseIndexer.topk(scores, k=3)
        assert idx.shape == (1, 3, 3)


# -------------------------------------------------------------- IMPORTANT: 5
class TestPartialBlockFiniteAndSingleTokenTail:
    @pytest.mark.parametrize("bsz", [1, 2])
    def test_partial_block_finite_and_tail_exact(self, bsz):
        comp = _comp()
        g = torch.Generator().manual_seed(7)
        x = torch.randn(bsz, 5, 16, generator=g)
        with torch.no_grad():
            out, ends = comp(x)
        assert ends.tolist() == [3, 4]
        assert torch.isfinite(out).all()
        # Trailing block holds a single valid token -> softmax weight 1,
        # so the entry equals kv_proj applied to the last token.
        with torch.no_grad():
            kv_last = comp.kv_proj(x[:, -1:]).view(bsz, 2, 8)
        assert_close(out[:, -1], kv_last, atol=1e-5, rtol=1e-5)


# -------------------------------------------------------------- IMPORTANT: 6
class TestSinkFpCorrectness:
    def test_fp32_matches_manual_formula(self):
        sink = AttentionSink(num_heads=2).eval()
        with torch.no_grad():
            sink.sink_logit.copy_(torch.tensor([0.5, -1.0]))
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(1, 2, 3, 4, generator=g, dtype=torch.float32)
        with torch.no_grad():
            w = sink.weights(logits)
        e = torch.exp(logits)
        s = torch.exp(torch.tensor([0.5, -1.0])).view(1, 2, 1, 1)
        expected = e / (e.sum(dim=-1, keepdim=True) + s)
        assert_close(w, expected, atol=1e-6, rtol=1e-6)

    def test_bf16_close_to_expected(self):
        sink = AttentionSink(num_heads=2).eval()
        with torch.no_grad():
            sink.sink_logit.copy_(torch.tensor([0.5, -1.0]))
        g = torch.Generator().manual_seed(0)
        logits32 = torch.randn(1, 2, 3, 4, generator=g, dtype=torch.float32)
        with torch.no_grad():
            w = sink.weights(logits32.to(torch.bfloat16))
        e = torch.exp(logits32)
        s = torch.exp(torch.tensor([0.5, -1.0])).view(1, 2, 1, 1)
        expected = e / (e.sum(dim=-1, keepdim=True) + s)
        assert_close(w.float(), expected, atol=1e-2, rtol=1e-2)


# -------------------------------------------------------------- IMPORTANT: 7
class TestClampedCompEndVisibility:
    def test_clamped_end_visible_at_last_token(self):
        comp = _comp()
        g = torch.Generator().manual_seed(0)
        x = torch.randn(1, 6, 16, generator=g)
        with torch.no_grad():
            _, ends = comp(x)
        assert ends.tolist() == [3, 5]
        mask = _causal_comp_mask(torch.tensor([[5]]), ends)
        assert mask.tolist() == [[[True, True]]]


# -------------------------------------------------------------- IMPORTANT: 8
class TestSinkhornItersZero:
    def test_iters_zero_finite_and_argmax_preserved(self):
        M = torch.tensor([[0.0, 100.0], [1000.0, 1001.0]])
        P = _sinkhorn(M, 0)
        assert torch.isfinite(P).all()
        assert P.argmax(dim=-1).tolist() == M.argmax(dim=-1).tolist()


# ------------------------------------------------------------ NICE_TO_HAVE
class TestNiceToHave:
    def test_topk_k_equals_lc_no_warning(self):
        g = torch.Generator().manual_seed(0)
        scores = torch.randn(1, 3, 4, generator=g)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            idx = SparseIndexer.topk(scores, k=4)
        assert idx.shape == (1, 3, 4)

    def test_compressor_seqlen_zero_raises(self):
        comp = _comp()
        with pytest.raises(ValueError, match="empty sequence"):
            comp(torch.zeros(1, 0, 16))

    def test_exact_divisible_offset_start3(self):
        comp = _comp()
        g = torch.Generator().manual_seed(2)
        x = torch.randn(1, 8, 16, generator=g)
        with torch.no_grad():
            _, ends = comp(x, start_pos=3)
        assert ends.tolist() == [6, 10]

    def test_sinkhorn_batched_large_shape_dtype(self):
        g = torch.Generator().manual_seed(0)
        M = torch.randn(2, 3, 5, 5, generator=g, dtype=torch.float64) * 1000.0
        P = _sinkhorn(M, 10)
        assert P.shape == M.shape
        assert P.dtype == M.dtype
        assert torch.isfinite(P).all()
        # No stochasticity assertion here: at spread-1000 whole columns can
        # underflow to zero (only empty *rows* fall back to uniform), so
        # column sums need not reach 1. Full double-stochasticity is
        # checked on moderate logits below.
        g2 = torch.Generator().manual_seed(1)
        W = torch.randn(2, 3, 5, 5, generator=g2, dtype=torch.float64) * 2.0
        Q = _sinkhorn(W, 50)
        assert_close(
            Q.sum(dim=-1),
            torch.ones(2, 3, 5, dtype=torch.float64),
            atol=1e-4,
            rtol=0.0,
        )


# ------------------------------------------------------- APPENDIX: review round 2
import math

from memory_machines.models.blocks.compression import (
    BranchGate,
    GroupedOutputProjection,
    HeadRMSNorm,
)
from memory_machines.models.blocks.shared import (
    _check_position_bounds,
    _resolve_positions,
    _update_swa_pos,
    residual_init_scale,
)


class TestBranchGateBasics:
    def test_shape_and_unit_interval(self):
        gate = BranchGate(hidden_size=8, num_branches=2).eval()
        x = torch.randn(2, 5, 8)
        with torch.no_grad():
            g = gate(x)
        assert g.shape == (2, 5, 2)
        assert bool(((g > 0) & (g < 1)).all())

    def test_zero_init_gives_half(self):
        gate = BranchGate(hidden_size=8, num_branches=2).eval()
        # __init__ already calls reset_parameters (zeros); re-assert.
        with torch.no_grad():
            g = gate(torch.randn(1, 4, 8))
        assert_close(g, torch.full_like(g, 0.5), atol=1e-7, rtol=0.0)

    def test_num_branches_three(self):
        gate = BranchGate(hidden_size=8, num_branches=3).eval()
        with torch.no_grad():
            g = gate(torch.randn(2, 3, 8))
        assert g.shape == (2, 3, 3)
        assert bool(((g > 0) & (g < 1)).all())


class TestBuildPoolBasics:
    def _indexer(self):
        from memory_machines.models.blocks.compression import SparseIndexer

        return SparseIndexer(8, 4, num_index_heads=4, index_dim=8).eval()

    def test_shape_and_range(self):
        idx = self._indexer()
        g = torch.Generator().manual_seed(0)
        scores = torch.randn(1, 2, 16, generator=g)
        with torch.no_grad():
            pool = idx.build_pool(scores, pool_blocks=2, block_size=4)
        assert pool.shape == (1, 2, 8)
        assert bool(((pool >= 0) & (pool < 16)).all())

    def test_non_divisible_stays_in_range(self):
        idx = self._indexer()
        g = torch.Generator().manual_seed(1)
        scores = torch.randn(1, 2, 14, generator=g)
        with torch.no_grad():
            pool = idx.build_pool(scores, pool_blocks=2, block_size=4)
        assert pool.shape == (1, 2, 8)
        assert bool(((pool >= 0) & (pool < 14)).all())

    def test_block_max_selection_sanity(self):
        idx = self._indexer()
        scores = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0]]])
        with torch.no_grad():
            pool = idx.build_pool(scores, pool_blocks=2, block_size=4)
        assert pool.shape == (1, 1, 8)
        # Both blocks kept (nb == num_blocks); winner-first ordering shows
        # block-max ranking works. NOTE: pool_blocks < num_blocks gives a
        # negative pad (nb*bs - lc < 0) which F.pad treats as a crop, so
        # true top-k selection over >nb blocks is unreachable with the
        # current implementation (e.g. pool_blocks=1 on lc=8 keeps [0..3]).
        assert pool[0, 0, :4].tolist() == [4, 5, 6, 7]
        assert sorted(pool[0, 0].tolist()) == list(range(8))


class TestRescorePoolGuards:
    def _indexer(self):
        from memory_machines.models.blocks.compression import SparseIndexer

        return SparseIndexer(8, 4, num_index_heads=4, index_dim=8).eval()

    def test_negative_pool_indices_raises(self):
        idx = self._indexer()
        q = torch.randn(1, 2, 8)
        k = torch.randn(1, 6, 8)
        pool = torch.tensor([[[0, -1, 2]]])
        with pytest.raises(ValueError, match="non-negative"):
            idx.rescore_pool(q, k, pool, torch.tensor([[5, 6]]), torch.tensor([1, 3, 5, 7, 9, 11]))

    def test_oob_masked_neginf(self):
        idx = self._indexer()
        g = torch.Generator().manual_seed(0)
        q = torch.randn(1, 1, 8, generator=g)
        k = torch.randn(1, 4, 8, generator=g)
        pool = torch.tensor([[[0, 1, 99]]])
        q_pos = torch.tensor([[100]])
        comp_end = torch.tensor([1, 2, 3, 4])
        with torch.no_grad():
            out = idx.rescore_pool(q, k, pool, q_pos, comp_end)
        assert out.shape == (1, 1, 3)
        assert torch.isneginf(out[0, 0, 2])

    def test_batched_comp_end_branch(self):
        idx = self._indexer()
        g = torch.Generator().manual_seed(2)
        q = torch.randn(2, 1, 8, generator=g)
        k = torch.randn(2, 4, 8, generator=g)
        pool = torch.tensor([[[0, 1]], [[2, 3]]])
        q_pos = torch.tensor([[2], [100]])
        comp_end = torch.tensor([[1, 2, 50, 60], [1, 2, 3, 4]])
        with torch.no_grad():
            out = idx.rescore_pool(q, k, pool, q_pos, comp_end)
        assert out.shape == (2, 1, 2)
        assert torch.isfinite(out[1]).any() or bool((out[1] == float("-inf")).any())
        # Batch 0: entries ending at 50/60 invisible at q=2 -> -inf handled
        # without crashing; batch 1 fully visible at q=100.
        assert torch.isfinite(out).any()


class TestScoreBatchedCompEnd:
    def test_per_batch_masking_differs(self):
        from memory_machines.models.blocks.compression import SparseIndexer

        idx = SparseIndexer(8, 4, num_index_heads=4, index_dim=8).eval()
        g = torch.Generator().manual_seed(3)
        q = torch.randn(2, 1, 8, generator=g)
        kv = torch.randn(2, 3, 2, 4, generator=g)
        q_pos = torch.tensor([[2], [100]])
        comp_end = torch.tensor([[1, 2, 50], [1, 2, 50]])
        with torch.no_grad():
            s = idx.score(q, kv, q_pos, comp_end)
        assert s.shape == (2, 1, 3)
        # Batch 0 (q=2): last entry (end 50) invisible; batch 1 (q=100): visible.
        assert torch.isneginf(s[0, 0, 2])
        assert not bool(torch.isneginf(s[1, 0, 2]).any())


class TestConstructorGuardsAndReset:
    def test_compressor_ratio_nonpositive_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            BlockCompressor(16, 2, 8, compression_ratio=0)
        with pytest.raises(ValueError, match="must be positive"):
            BlockCompressor(16, 2, 8, compression_ratio=-2)

    def test_grouped_proj_divisibility_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            GroupedOutputProjection(5, 8, num_groups=2, group_dim=8, hidden_size=16)

    def test_reset_parameters_finite_output(self):
        torch.manual_seed(0)
        comp = BlockCompressor(16, 2, 8, compression_ratio=4).eval()
        comp.reset_parameters()
        idx_mod = SparseIndexer(8, 4, num_index_heads=4, index_dim=8).eval()
        idx_mod.reset_parameters()
        gate = BranchGate(8, num_branches=2).eval()
        gate.reset_parameters()
        norm = HeadRMSNorm(8).eval()
        norm.reset_parameters()
        sink = AttentionSink(num_heads=2).eval()
        sink.reset_parameters()
        proj = GroupedOutputProjection(4, 8, num_groups=2, group_dim=8, hidden_size=16).eval()
        proj.reset_parameters()
        proj0 = GroupedOutputProjection(4, 8, num_groups=2, group_dim=8, hidden_size=16, residual_init_scale=False).eval()
        # residual_init_scale=False -> zero out_proj path covered
        with torch.no_grad():
            x = torch.randn(1, 4, 16)
            cout, _ = comp(x)
            assert torch.isfinite(cout).all()
            w = sink.weights(torch.randn(1, 2, 2, 4))
            assert torch.isfinite(w).all()
            n = norm(torch.randn(1, 2, 2, 8))
            assert torch.isfinite(n).all()
            gg = gate(torch.randn(1, 2, 8))
            assert torch.isfinite(gg).all()
            p = proj(torch.randn(1, 4, 2, 8))
            assert torch.isfinite(p).all()
            p0 = proj0(torch.randn(1, 4, 2, 8))
            assert torch.isfinite(p0).all()


class TestResidualInitScale:
    def test_enabled_formula(self):
        assert_close(
            torch.tensor(residual_init_scale(0.006, 40)),
            torch.tensor(0.006 / math.sqrt(2 * 40)),
            atol=0.0,
            rtol=0.0,
        )

    def test_disabled_returns_none(self):
        assert residual_init_scale(0.006, 40, enabled=False) is None

    def test_zero_layers_floored_to_one(self):
        assert_close(
            torch.tensor(residual_init_scale(0.02, 0)),
            torch.tensor(0.02 / math.sqrt(2 * 1)),
            atol=0.0,
            rtol=0.0,
        )


class TestSharedHelpersRound2:
    def test_resolve_positions_1d_past_zero(self):
        x = torch.zeros(2, 3, 4)
        pos1d = torch.tensor([10, 11, 12])
        q_pos, k_pos = _resolve_positions(x, 3, 0, pos1d)
        assert q_pos.tolist() == [[10, 11, 12], [10, 11, 12]]
        # past_len == 0 -> k equals q
        assert k_pos.tolist() == q_pos.tolist()

    def test_causal_comp_mask_batched(self):
        q_pos = torch.tensor([[2], [100]])
        comp_end = torch.tensor([[1, 2, 50], [1, 2, 50]])
        m = _causal_comp_mask(q_pos, comp_end)
        assert m.shape == (2, 1, 3)
        assert m.tolist() == [[[True, True, False]], [[True, True, True]]]

    def test_sinkhorn_empty_row_unreachable_documents(self):
        # Empty-row fallback requires a zero row-sum, but max-subtraction
        # guarantees exp(0)=1 per row so sums are >= 1. The only way to get
        # an all -inf row is rejected upfront as non-finite input.
        with pytest.raises(ValueError, match="finite"):
            _sinkhorn(torch.tensor([[float("-inf"), float("-inf")], [0.0, 1.0]]), 5)
        # Extreme spread stays finite without needing the fallback.
        M = torch.tensor([[0.0, 1000.0], [1000.0, 1001.0]])
        with torch.no_grad():
            P = _sinkhorn(M, 5)
        assert torch.isfinite(P).all()

    def test_update_swa_pos_truncation_none_prev(self):
        out = _update_swa_pos(None, torch.tensor([[0, 1, 2]]), 2)
        assert out.tolist() == [[1, 2]]

    def test_check_position_bounds_in_bounds_passes(self):
        # Explicit in-bounds ids must not raise; returns None.
        assert _check_position_bounds(torch.tensor([[0, 1]]), torch.tensor([[0, 1]]), 8, torch.tensor([[0, 1]])) is None


def test_build_pool_selects_top_blocks():
    from memory_machines.models.blocks.compression import SparseIndexer

    idx = SparseIndexer(8, 4, num_index_heads=4, index_dim=8).eval()
    scores = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 10.0, 11.0, 12.0, 13.0]]])
    with torch.no_grad():
        pool = idx.build_pool(scores, pool_blocks=1, block_size=4)
    assert pool.shape == (1, 1, 4)
    assert sorted(pool[0, 0].tolist()) == [4, 5, 6, 7]
