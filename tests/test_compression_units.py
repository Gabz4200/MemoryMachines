"""Unit tests for sparse compression building blocks."""

import torch

from memory_machines.models.blocks.compression import (
    AttentionSink,
    BlockCompressor,
    GroupedOutputProjection,
    HeadRMSNorm,
    SparseIndexer,
)


def test_compressor_uneven_tail_ignores_pad():
    torch.manual_seed(0)
    hidden, nheads, hd, m = 16, 2, 8, 2
    comp_mod = BlockCompressor(hidden, nheads, hd, compression_ratio=m).eval()
    x7 = torch.randn(1, 7, hidden)
    extra = torch.randn(1, 1, hidden)
    x8 = torch.cat([x7, extra], dim=1)

    with torch.no_grad():
        comp7, end7 = comp_mod(x7)
        comp8, end8 = comp_mod(x8)

    torch.testing.assert_close(comp7[:, :3], comp8[:, :3])
    # Single-valid-entry tail collapses softmax to weight 1 on the real row.
    with torch.no_grad():
        expected_tail = comp_mod.kv_proj(x7[:, -1]).view(1, nheads, hd)
    torch.testing.assert_close(comp7[:, -1], expected_tail, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(end7, torch.tensor([1, 3, 5, 7]))
    torch.testing.assert_close(end8, torch.tensor([1, 3, 5, 7]))


def test_indexer_score_and_rescore_are_causal():
    torch.manual_seed(0)
    indexer = SparseIndexer(q_latent_dim=16, kv_head_dim=8).eval()
    comp_end = torch.tensor([1, 3, 5, 7])
    q_pos = torch.tensor([[1, 5]])
    query_latent = torch.randn(1, 2, 16)
    compressed_kv = torch.randn(1, 4, 2, 8)

    with torch.no_grad():
        scores = indexer.score(query_latent, compressed_kv, q_pos, comp_end)
    assert scores.shape == (1, 2, 4)
    assert torch.isfinite(scores[0, 0, 0])
    assert bool((scores[0, 0, 1:] == float("-inf")).all())
    assert bool(torch.isfinite(scores[0, 1, :3]).all())
    assert scores[0, 1, 3] == float("-inf")

    with torch.no_grad():
        indexer_k = indexer.k_index_proj(compressed_kv.mean(dim=2))
        rescored = indexer.rescore_pool(
            query_latent[:, :1],
            indexer_k,
            torch.tensor([[[3, 0]]]),
            torch.tensor([[1]]),
            comp_end,
        )
    assert rescored.shape == (1, 1, 2)
    assert rescored[0, 0, 0] == float("-inf")
    assert torch.isfinite(rescored[0, 0, 1])


def test_attention_sink_reduces_weight_mass():
    torch.manual_seed(0)
    sink = AttentionSink(num_heads=2).eval()
    logits = torch.zeros(1, 2, 1, 4)
    with torch.no_grad():
        w = sink.weights(logits)
    # sink_logit=0 gives weight 1/5, row sum 0.8.
    torch.testing.assert_close(w.sum(-1), torch.full((1, 2, 1), 0.8))

    masses = []
    with torch.no_grad():
        for s in (0.0, 1.0, 3.0):
            sink.sink_logit.fill_(s)
            masses.append(float(sink.weights(logits).sum()))
    assert masses[0] > masses[1] > masses[2]

    # Per-head sinks must not broadcast one value over all heads.
    with torch.no_grad():
        sink.sink_logit.copy_(torch.tensor([0.0, 2.0]))
        w2 = sink.weights(logits)
    head_mass = w2.sum(-1)
    assert abs(float(head_mass[0, 0, 0]) - 0.8) < 1e-6
    assert float(head_mass[0, 1, 0]) < float(head_mass[0, 0, 0]) - 0.1


def test_grouped_out_proj_group_isolation():
    torch.manual_seed(0)
    nheads, hd, groups, gdim, hidden = 4, 8, 2, 16, 32
    proj = GroupedOutputProjection(nheads, hd, groups, gdim, hidden).eval()
    with torch.no_grad():
        # Zero-start out_proj would zero every output alike; restore signal
        # before masking the second group so isolation is observable.
        torch.nn.init.xavier_uniform_(proj.out_proj.weight)
        proj.out_proj.weight[:, gdim:] = 0.0
    x0 = torch.randn(1, 4, 3, 8)
    with torch.no_grad():
        out0 = proj(x0)
        x_other = x0.clone()
        x_other[:, 2:, :, :] += torch.randn(1, 2, 3, 8)
        torch.testing.assert_close(proj(x_other), out0)
        x_same = x0.clone()
        x_same[:, :2, :, :] += torch.randn(1, 2, 3, 8)
        assert float((proj(x_same) - out0).abs().max()) > 1e-5


def test_head_rmsnorm_is_not_layernorm():
    torch.manual_seed(0)
    norm = HeadRMSNorm(head_dim=8).eval()
    x = 2.0 * torch.ones(1, 2, 4, 8)
    with torch.no_grad():
        out = norm(x)
    # Constant-2 RMS is 2, output ~1 (LayerNorm would give ~0).
    torch.testing.assert_close(out, torch.ones_like(out), atol=1e-5, rtol=1e-5)
    assert abs(float(out.mean()) - 1.0) < 1e-5
    rms = out.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        torch.testing.assert_close(norm(4.0 * torch.ones(1, 2, 4, 8)), out)

    norm2 = HeadRMSNorm(head_dim=8)
    xg = (2.0 * torch.ones(1, 2, 4, 8)).requires_grad_(True)
    norm2(xg).sum().backward()
    assert norm2.weight.grad is not None
    assert tuple(norm2.weight.grad.shape) == (8,)
    assert bool((norm2.weight.grad != 0).any())
