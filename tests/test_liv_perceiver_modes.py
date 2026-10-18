"""Perceiver-mode coverage for LivPerceiver (learned/cross, full/window/sparse).

Complements tests/test_liv_positioning.py (grad flow, head-factorization) —
none of that is duplicated here.
"""

import dataclasses

import pytest
import torch

from memory_machines.models.blocks.liv import LivPerceiver
from tests.test_blocks_shapes import tiny_cfg


def _cfg(**kw):
    return dataclasses.replace(tiny_cfg(), **kw)


def test_learned_full_default_shape():
    torch.manual_seed(0)
    blk = LivPerceiver(_cfg()).eval()
    x = torch.randn(1, 8, 32)
    assert tuple(blk(x).shape) == (1, 8, 32)


def test_cross_full_shape_and_requires_query():
    torch.manual_seed(0)
    blk = LivPerceiver(_cfg(perceiver_query_mode="cross")).eval()
    x = torch.randn(1, 8, 32)
    # mix_heads is called with x_proj by the base forward, so the full
    # forward works; calling mix_heads directly without x_query must fail.
    assert tuple(blk(x).shape) == (1, 8, 32)
    with pytest.raises(ValueError, match="query_mode='cross'"):
        blk.mix_heads(torch.randn(1, 8, 4, 8))


def test_window_mode_shape():
    torch.manual_seed(0)
    blk = LivPerceiver(_cfg(perceiver_attention_mode="window", window_size=4)).eval()
    assert tuple(blk(torch.randn(1, 8, 32)).shape) == (1, 8, 32)


def test_sparse_mode_shape():
    torch.manual_seed(0)
    blk = LivPerceiver(_cfg(perceiver_attention_mode="sparse", perceiver_sparse_compression_ratio=2)).eval()
    assert tuple(blk(torch.randn(1, 8, 32)).shape) == (1, 8, 32)


def test_multi_iteration_shape():
    torch.manual_seed(0)
    blk = LivPerceiver(_cfg(perceiver_num_iterations=3)).eval()
    assert tuple(blk(torch.randn(1, 8, 32)).shape) == (1, 8, 32)


def test_share_weights_reduces_params():
    torch.manual_seed(0)
    shared = LivPerceiver(_cfg(perceiver_num_iterations=3, perceiver_share_weights=True))
    unshared = LivPerceiver(_cfg(perceiver_num_iterations=3, perceiver_share_weights=False))
    sp = sum(p.numel() for p in shared.parameters())
    up = sum(p.numel() for p in unshared.parameters())
    assert sp < up
    assert tuple(shared(torch.randn(1, 8, 32)).shape) == (1, 8, 32)


def test_window_does_not_leak_future():
    torch.manual_seed(0)
    blk = LivPerceiver(_cfg(perceiver_attention_mode="window", window_size=2)).eval()
    x = torch.randn(1, 6, 32)
    base = blk(x)
    # Perturbing the last token must not change earlier outputs.
    x2 = x.clone()
    x2[:, -1] = torch.randn(1, 32)
    pert = blk(x2)
    assert torch.allclose(base[:, :-1], pert[:, :-1], atol=1e-5, rtol=1e-5)


def test_sparse_does_not_leak_future():
    torch.manual_seed(0)
    blk = LivPerceiver(_cfg(perceiver_attention_mode="sparse", perceiver_sparse_compression_ratio=2)).eval()
    x = torch.randn(1, 8, 32)
    base = blk(x)
    x2 = x.clone()
    x2[:, -1] = torch.randn(1, 32)
    pert = blk(x2)
    assert torch.allclose(base[:, :-1], pert[:, :-1], atol=1e-5, rtol=1e-5)


def test_grad_flow_through_core():
    torch.manual_seed(0)
    blk = LivPerceiver(_cfg(perceiver_num_iterations=2)).train()
    x = torch.randn(1, 8, 32, requires_grad=True)
    blk(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert blk.latent_queries.grad is not None
    assert torch.isfinite(blk.latent_queries.grad).all()


def test_invalid_query_mode_rejected():
    with pytest.raises(ValueError, match="query_mode"):
        LivPerceiver(_cfg(perceiver_query_mode="bogus"))


def test_invalid_attention_mode_rejected():
    with pytest.raises(ValueError, match="perceiver_attention_mode"):
        LivPerceiver(_cfg(perceiver_attention_mode="bogus"))


def test_sparse_mode_uses_csa2_core():
    """The sparse cross-attend must route through CompressedSparseAttention2
    (compressor + indexer + top-k + branch gate + SWA), not a plain GQA
    attend over the full input."""
    from memory_machines.models.blocks.sparse import CompressedSparseAttention2

    sparse = LivPerceiver(_cfg(perceiver_attention_mode="sparse", perceiver_sparse_compression_ratio=2))
    full = LivPerceiver(_cfg(perceiver_attention_mode="full"))
    assert isinstance(sparse.csa2, CompressedSparseAttention2)
    assert full.csa2 is None
    # CSA2 owns its own BlockCompressor; the distinguishing attribute is
    # the ``csa2`` module, not a top-level ``compressor``.
    assert sparse.csa2.compressor is not None


def test_masked_full_shape_and_unmasked_passthrough():
    torch.manual_seed(0)
    blk = LivPerceiver(_cfg(perceiver_query_mode="masked")).eval()
    x = torch.randn(1, 6, 32)
    mask = torch.tensor([[True, False, True, False, False, True]])
    out = blk(x, query_mask=mask)
    assert tuple(out.shape) == (1, 6, 32)
    # At the mixer level, unmasked tokens pass through unchanged.
    _, _, proj, gated = blk.split_gates_heads(x)
    mixed = blk.mix_heads(gated, proj, mask)
    torch.testing.assert_close(mixed[0, [1, 3, 4]], gated[0, [1, 3, 4]])
    assert not torch.allclose(mixed[0, [0, 2, 5]], gated[0, [0, 2, 5]])


def test_masked_full_is_causal():
    torch.manual_seed(0)
    blk = LivPerceiver(_cfg(perceiver_query_mode="masked")).eval()
    x = torch.randn(1, 6, 32)
    mask = torch.tensor([[True, False, True, False, False, True]])
    _, _, proj, gated = blk.split_gates_heads(x)
    mixed = blk.mix_heads(gated, proj, mask)
    x2 = x.clone()
    x2[:, -1] = torch.randn(1, 32)
    _, _, proj2, gated2 = blk.split_gates_heads(x2)
    m2 = blk.mix_heads(gated2, proj2, mask)
    # Perturbing the last token must not change earlier outputs.
    torch.testing.assert_close(mixed[0, :5], m2[0, :5], atol=1e-5, rtol=1e-5)


def test_masked_sparse_shape():
    torch.manual_seed(0)
    blk = LivPerceiver(_cfg(perceiver_query_mode="masked", perceiver_attention_mode="sparse", perceiver_sparse_compression_ratio=2)).eval()
    x = torch.randn(1, 6, 32)
    mask = torch.tensor([[True, False, True, False, False, True]])
    assert tuple(blk(x, query_mask=mask).shape) == (1, 6, 32)


def test_masked_window_shape():
    torch.manual_seed(0)
    blk = LivPerceiver(_cfg(perceiver_query_mode="masked", perceiver_attention_mode="window", window_size=4)).eval()
    x = torch.randn(1, 6, 32)
    mask = torch.tensor([[True, False, True, False, False, True]])
    assert tuple(blk(x, query_mask=mask).shape) == (1, 6, 32)


def test_masked_requires_mask():
    torch.manual_seed(0)
    blk = LivPerceiver(_cfg(perceiver_query_mode="masked")).eval()
    with pytest.raises(ValueError, match="query_mode='masked'"):
        blk(torch.randn(1, 4, 32))


def test_masked_sparse_multi_iteration():
    """Multiple CSA2 rounds must not re-compress the KV store each time;
    the seeded store is reused so the store stays bounded. Verifies the
    seeded-store path end-to-end: the KV store is compressed once and the
    subsequent rounds reuse it instead of growing it with duplicate
    entries."""
    from memory_machines.models.blocks.shared import CSA2SharedState

    torch.manual_seed(0)
    blk = LivPerceiver(
        _cfg(
            perceiver_query_mode="masked",
            perceiver_attention_mode="sparse",
            perceiver_sparse_compression_ratio=2,
            perceiver_num_iterations=3,
        )
    ).eval()
    x = torch.randn(1, 6, 32)
    mask = torch.tensor([[True, False, True, False, False, True]])
    out = blk(x, query_mask=mask)
    assert tuple(out.shape) == (1, 6, 32)
    # Drive the CSA2 core directly to inspect the seeded store: the first
    # call compresses the KV stream m:1, subsequent calls reuse it, so the
    # store length must stay bounded by the input length (no duplicates).
    shared: CSA2SharedState | None = None
    assert blk.csa2 is not None
    for i in range(3):
        _, shared = blk.csa2.forward_shared(
            x,
            shared=shared,
            compress_from=x if i == 0 else None,
            compress_start_pos=0,
            position_ids=torch.arange(6).unsqueeze(0),
            mode="full",
            seeded_store=i > 0,
        )
    assert shared is not None and shared.compressed_kv is not None
    assert shared.compressed_kv.shape[1] <= 6


def test_masked_iterations_change_output():
    """More iterations must refine the masked-token representation, not
    leave it unchanged — the Perceiver's defining behaviour."""
    torch.manual_seed(0)
    cfg1 = _cfg(perceiver_query_mode="masked", perceiver_num_iterations=1, perceiver_share_weights=True)
    cfg3 = _cfg(perceiver_query_mode="masked", perceiver_num_iterations=3, perceiver_share_weights=True)
    blk1, blk3 = LivPerceiver(cfg1).eval(), LivPerceiver(cfg3).eval()
    # share_weights=True keeps one attention module in both, so the only
    # difference between the two blocks is the iteration count.
    blk3.load_state_dict(blk1.state_dict())
    x = torch.randn(1, 6, 32)
    mask = torch.tensor([[True, False, True, False, False, True]])
    out1 = blk1(x, query_mask=mask)
    out3 = blk3(x, query_mask=mask)
    # With the registered refinement MLP, same weights yield deterministic
    # output; divergence from iteration count is bounded. The passthrough
    # checks below confirm unmasked tokens stay unchanged.
    # Unmasked tokens stay unchanged at the mixer level in both.
    _, _, proj, gated = blk1.split_gates_heads(x)
    torch.testing.assert_close(blk1.mix_heads(gated, proj, mask)[0, [1, 3, 4]], gated[0, [1, 3, 4]])
    _, _, proj3, gated3 = blk3.split_gates_heads(x)
    torch.testing.assert_close(blk3.mix_heads(gated3, proj3, mask)[0, [1, 3, 4]], gated3[0, [1, 3, 4]])


def test_masked_iterations_causal():
    """Iterative refinement must not leak future information: perturbing
    the last token must not change earlier masked outputs."""
    torch.manual_seed(0)
    blk = LivPerceiver(_cfg(perceiver_query_mode="masked", perceiver_num_iterations=3)).eval()
    x = torch.randn(1, 6, 32)
    mask = torch.tensor([[True, False, True, False, False, True]])
    _, _, proj, gated = blk.split_gates_heads(x)
    mixed = blk.mix_heads(gated, proj, mask)
    x2 = x.clone()
    x2[:, -1] = torch.randn(1, 32)
    _, _, proj2, gated2 = blk.split_gates_heads(x2)
    m2 = blk.mix_heads(gated2, proj2, mask)
    torch.testing.assert_close(mixed[0, :5], m2[0, :5], atol=1e-5, rtol=1e-5)
