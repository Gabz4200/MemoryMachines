"""Titans LMM (arXiv:2501.00663) paper-accuracy tests (RED phase)."""

import pytest
import torch

from memory_machines.models.blocks.fast_weights import (
    TitansFastWeightsMemory,
    TitansMAGBlock,
)


def _tiny(**kw):
    base = {"hidden_size": 32, "num_heads": 4, "memory_depth": 2, "num_persistent_tokens": 2}
    base.update(kw)
    return base


def _tnt(**kw):
    """Single-layer fast weights at width 16 (chunk/TNT backend parity)."""
    base = {"hidden_size": 16, "num_heads": 2, "memory_depth": 1, "num_persistent_tokens": 2, "use_short_conv": False}
    base.update(kw)
    return base


def test_output_shapes_and_residual():
    torch.manual_seed(0)
    mem = TitansFastWeightsMemory(**_tiny()).eval()
    x = torch.randn(1, 7, 32)
    out, state = mem(x, output_final_state=True)
    assert tuple(out.shape) == (1, 7, 32)
    assert state is not None
    assert "fast_weights" in state and "momentum" in state


def test_causality_no_future_leak():
    torch.manual_seed(0)
    mem = TitansFastWeightsMemory(**_tiny()).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 8, 32)
    base, _ = mem(x)
    for fut in (5, 6, 7):
        x2 = x.clone()
        x2[0, fut] += 10.0
        out2, _ = mem(x2)
        delta = (out2 - base).abs().amax(-1)[0]
        assert delta[:5].max().item() == 0.0, f"future token {fut} leaked into past"
    assert (base[:, 1:] - base[:, :-1]).abs().max().item() > 0.0


def test_momentum_and_decay_change_output():
    torch.manual_seed(0)
    cfg = _tiny()
    torch.manual_seed(1)
    x = torch.randn(1, 6, 32)
    m_base, _ = TitansFastWeightsMemory(**cfg).eval()(x)
    m_nomom, _ = TitansFastWeightsMemory(**{**cfg, "use_momentum": False}).eval()(x)
    assert (m_base - m_nomom).abs().max().item() > 1e-6
    m_nodecay, _ = TitansFastWeightsMemory(**{**cfg, "use_weight_decay": False}).eval()(x)
    assert (m_base - m_nodecay).abs().max().item() > 1e-6


def test_surprise_memorizes_repeated_association():
    torch.manual_seed(0)
    mem = TitansFastWeightsMemory(**_tiny(memory_depth=1, num_persistent_tokens=0, use_momentum=False, use_weight_decay=False, use_short_conv=False)).eval()
    torch.manual_seed(1)
    # Constant stream => identical (k, v) pair every step (Eq. 11-12); the
    # inner loop must fit it, so pre-update surprise strictly decreases.
    x = torch.randn(1, 1, 32).repeat(1, 8, 1)
    losses = mem.associative_losses(x)
    assert losses.shape == (1, 8)
    diffs = losses[0, 1:] - losses[0, :-1]
    assert bool((diffs < 0).all()), losses
    assert losses[0, -1].item() < 0.9 * losses[0, 0].item()


def test_chunked_equals_sequential():
    torch.manual_seed(0)
    mem = TitansFastWeightsMemory(**_tiny()).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 9, 32)
    full, _ = mem(x, chunk_size=None)
    chunked, _ = mem(x, chunk_size=4)
    torch.testing.assert_close(full, chunked, atol=1e-5, rtol=1e-5)


def test_persistent_tokens_prepended_and_retrieval_uses_them():
    torch.manual_seed(0)
    mem = TitansFastWeightsMemory(**_tiny(num_persistent_tokens=3)).eval()
    assert tuple(mem.persistent.shape) == (3, 32)
    x = torch.randn(1, 4, 32)
    out, _ = mem(x)
    assert tuple(out.shape) == (1, 4, 32)
    mem0 = TitansFastWeightsMemory(**_tiny(num_persistent_tokens=0)).eval()
    out0, _ = mem0(x)
    assert (out - out0).abs().max().item() > 0.0


def test_qk_l2norm_and_silu_gate():
    torch.manual_seed(0)
    mem = TitansFastWeightsMemory(**_tiny()).eval()
    x = torch.randn(2, 5, 32)
    q, k, _, _ = mem.project_qkv(x)
    assert q.shape == k.shape == (2, 5, 4, 8)
    norms = (k * k).sum(-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-3, rtol=1e-3)


def test_mag_block_gated_fusion_shapes():
    torch.manual_seed(0)
    blk = TitansMAGBlock(**_tiny()).eval()
    x = torch.randn(1, 6, 32)
    out = blk(x)
    assert tuple(out.shape) == (1, 6, 32)
    # short-term branch bypass changes output (gate is open at init)
    out2 = blk(x, short_term=torch.zeros_like(x))
    assert (out - out2).abs().max().item() > 1e-6


def test_state_continuation_matches_full_forward():
    torch.manual_seed(0)
    mem = TitansFastWeightsMemory(**_tiny()).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 7, 32)
    with torch.no_grad():
        full, _ = mem(x)
        _, state = mem(x[:, :4], output_final_state=True)
        part, _ = mem(x[:, 4:], state=state)
    torch.testing.assert_close(full[:, 4:], part, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("backend", ["chunk", "tnt", "wy_fast"], ids=["chunk", "tnt", "wy_fast"])
def test_tnt_backend_matches_naive_at_chunk_size_one(backend: str) -> None:
    """TNT (arXiv:2511.07343): one-token chunks evaluate gradients at the current state, so they equal the sequential path."""
    torch.manual_seed(0)
    naive = TitansFastWeightsMemory(**_tnt(), backend="naive").eval()
    torch.manual_seed(0)
    tnt = TitansFastWeightsMemory(**_tnt(), backend=backend, chunk_size=1).eval()
    tnt.load_state_dict(naive.state_dict())
    torch.manual_seed(1)
    x = torch.randn(1, 6, 16)
    with torch.no_grad():
        out_naive, state_naive = naive(x, output_final_state=True)
        out_tnt, state_tnt = tnt(x, output_final_state=True)
    assert state_naive is not None and state_tnt is not None
    torch.testing.assert_close(out_tnt, out_naive, atol=1e-6, rtol=1e-5)
    for got, ref in zip(state_tnt["fast_weights"], state_naive["fast_weights"]):
        torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-5)
    for got, ref in zip(state_tnt["momentum"], state_naive["momentum"]):
        torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-5)


def test_tnt_backend_train_backward_reaches_params() -> None:
    """Training through the chunk/TNT branch keeps the BPTT graph over every slow parameter."""
    torch.manual_seed(0)
    mem = TitansFastWeightsMemory(**_tnt(num_persistent_tokens=0), backend="tnt", chunk_size=1).train()
    torch.manual_seed(1)
    x = torch.randn(1, 4, 16)
    out, _ = mem(x)
    out.sum().backward()
    for name, p in mem.named_parameters():
        assert p.grad is not None, name
        assert bool(torch.isfinite(p.grad).all()), name


@pytest.mark.parametrize("bad", [(32,), (1, 3, 31)], ids=["rank_2", "hidden_mismatch"])
def test_project_qkv_rejects_bad_shape(bad: tuple[int, ...]) -> None:
    mem = TitansFastWeightsMemory(**_tiny(memory_depth=1)).eval()
    with pytest.raises(ValueError, match=r"\[B, T, D="):
        mem.project_qkv(torch.randn(*bad))


def test_mag_block_rejects_bad_input_shape() -> None:
    blk = TitansMAGBlock(**_tiny()).eval()
    with pytest.raises(ValueError, match=r"\[B, T, D="):
        blk(torch.randn(1, 3, 31))


@pytest.mark.parametrize(("num_persistent_tokens", "expected"), [(2, (2, 32)), (0, None)], ids=["configured", "disabled"])
def test_mag_block_persistent_property(num_persistent_tokens: int, expected: tuple[int, ...] | None) -> None:
    blk = TitansMAGBlock(**_tiny(num_persistent_tokens=num_persistent_tokens))
    assert blk.persistent is blk.memory.persistent
    if expected is None:
        assert blk.persistent is None
    else:
        assert blk.persistent is not None
        assert tuple(blk.persistent.shape) == expected
