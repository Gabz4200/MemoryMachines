"""CED + auxiliary memory coverage tests (prefill/decode, engram, mHC, dspark)."""

import math
from dataclasses import replace

import torch

from memory_machines.models.blocks import (
    CausalEncoderDecoderStack,
    CEDGlobalKVProjection,
    CompressedSparseAttention2,
    DSparkDrafter,
    EngramMemory,
    SinglePassMHC,
)
from tests.test_blocks_shapes import tiny_cfg


def _ced_cfg():
    return replace(
        tiny_cfg(),
        swa_bounded_replay=True,
        swa_window_size=4,
        ced_num_encoder_layers=1,
        ced_num_decoder_layers=2,
        csa_encoder_compression_ratio=2,
        csa_decoder_compression_ratio=1,
    )


def test_ced_prefill_equals_decode_of_tail():
    torch.manual_seed(0)
    cfg = _ced_cfg()
    model = CausalEncoderDecoderStack(cfg).eval()
    torch.manual_seed(0)
    H = torch.randn(1, 10, 32)
    E = torch.randn(1, 8, 32)

    pos2d = torch.arange(10).unsqueeze(0)
    out_prefill = model.prefill(H, E, pos2d)
    out_decode = model.decode_full_context(H[:, -4:, :], E, pos2d[:, -4:])
    assert tuple(out_prefill.shape) == (1, 4, 32)
    torch.testing.assert_close(out_prefill, out_decode, atol=1e-5, rtol=1e-5)

    out_prefill_none = model.prefill(H, E, None)
    out_decode_none = model.decode_full_context(H[:, -4:, :], E, None)
    torch.testing.assert_close(out_prefill_none, out_decode_none, atol=1e-5, rtol=1e-5)

    pos1d = torch.arange(10)
    out_prefill_1d = model.prefill(H, E, pos1d)
    out_decode_1d = model.decode_full_context(H[:, -4:, :], E, pos1d[-4:])
    torch.testing.assert_close(out_prefill_1d, out_decode_1d, atol=1e-5, rtol=1e-5)


def test_ced_decode_is_globally_conditioned():
    torch.manual_seed(0)
    cfg = _ced_cfg()
    model = CausalEncoderDecoderStack(cfg).eval()
    # Sparse layers use a zero-start residual output; restore signal so the
    # global-conditioning comparison is meaningful.
    with torch.no_grad():
        for layer in list(model.encoder_layers) + list(model.decoder_layers):
            assert isinstance(layer, CompressedSparseAttention2)
            torch.nn.init.xavier_uniform_(layer.grouped_out.out_proj.weight)
    torch.manual_seed(1)
    H = torch.randn(1, 6, 32)
    E_A = torch.randn(1, 7, 32)
    torch.manual_seed(2)
    E_B = torch.randn(1, 7, 32)
    out_a = model.decode_full_context(H, E_A)
    out_b = model.decode_full_context(H, E_B)
    assert (out_a - out_b).abs().max().item() > 1e-4

    L = 7
    torch.manual_seed(3)
    H7 = torch.randn(1, L, 32)
    _, shared = model.encode(H7)
    # The stack seeds total_len once; layers do not advance it.
    assert shared.total_len == 0
    assert shared.compressed_kv is not None
    assert shared.compressed_kv.shape[1] == math.ceil(L / 2)

    torch.manual_seed(0)
    proj = CEDGlobalKVProjection(cfg).eval()
    torch.manual_seed(4)
    E7 = torch.randn(1, 7, 32)
    comp, ends = proj(E7)
    assert tuple(comp.shape) == (1, 7, 2, 8)
    assert ends.tolist() == [0, 1, 2, 3, 4, 5, 6]


def _engram_cfg():
    return replace(
        tiny_cfg(),
        engram_orders=(2, 3),
        engram_hash_heads=2,
        engram_dim_per_order=8,
        engram_num_entries=1024,
    )


def test_engram_residual_and_last_token_sensitivity():
    torch.manual_seed(0)
    cfg = _engram_cfg()

    # Zero out_proj recovers the residual exactly.
    mem = EngramMemory(cfg).eval()
    torch.nn.init.zeros_(mem.out_proj.weight)
    torch.manual_seed(1)
    input_ids = torch.randint(0, 50, (1, 6))
    hidden = torch.randn(1, 6, 32)
    out = mem(input_ids, hidden)
    torch.testing.assert_close(out, hidden, atol=1e-5, rtol=1e-5)

    torch.manual_seed(0)
    mem2 = EngramMemory(cfg).eval()
    # Zero-start out_proj would hide the memory path entirely; restore
    # signal so last-token sensitivity is observable.
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(mem2.out_proj.weight)
    torch.manual_seed(7)
    base_ids = torch.randint(0, 50, (1, 6))
    alt_ids = base_ids.clone()
    alt_ids[0, -1] = (int(base_ids[0, -1].item()) + 1) % 50
    assert alt_ids[0, -1].item() != base_ids[0, -1].item()
    h = torch.randn(1, 6, 32)
    o1 = mem2(base_ids, h)
    o2 = mem2(alt_ids, h)
    assert (o1[:, -1, :] - o2[:, -1, :]).abs().max().item() > 1e-6
    o1_repeat = mem2(base_ids, h)
    torch.testing.assert_close(o1, o1_repeat, atol=1e-5, rtol=1e-5)

    window = torch.tensor([[5, 7, 9], [1, 2, 3]])
    id0 = mem2._ngram_id(window, head=0)
    id1 = mem2._ngram_id(window, head=1)
    assert not torch.equal(id0, id1)


def test_mhc_routing():
    torch.manual_seed(0)
    cfg = replace(tiny_cfg(), mhc_num_streams=2)
    mhc = SinglePassMHC(cfg).eval()
    torch.manual_seed(0)
    streams = torch.randn(2, 5, 2, 32)

    a, b, _ = mhc.coefficients(streams)
    assert tuple(a.shape) == (2, 5, 1, 2)
    assert tuple(b.shape) == (2, 5, 2, 2)
    torch.testing.assert_close(b.sum(dim=-1), torch.ones_like(b.sum(dim=-1)), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(b.sum(dim=-2), torch.ones_like(b.sum(dim=-2)), atol=1e-4, rtol=1e-4)

    prev_A = torch.zeros(2, 5, 1, 2)
    prev_A[..., 0] = 1.0
    prev_B = torch.zeros(2, 5, 1, 2)
    prev_B[..., 1] = 1.0

    out_step_A, _ = mhc.step(streams, branch_fn=lambda h: h, prev_a=prev_A)
    out_step_B, _ = mhc.step(streams, branch_fn=lambda h: h, prev_a=prev_B)
    assert (out_step_A - out_step_B).abs().max().item() > 0.0

    branch_out = torch.randn(2, 5, 32)
    out_fwd_A, _ = mhc.forward(streams, branch_out, prev_a=prev_A)
    out_fwd_B, _ = mhc.forward(streams, branch_out, prev_a=prev_B)
    assert torch.equal(out_fwd_A, out_fwd_B)


def test_dspark_tail_only():
    torch.manual_seed(0)
    cfg = replace(tiny_cfg(), dspark_num_drafts=3, dspark_window=4)
    drafter = DSparkDrafter(cfg).eval()

    torch.manual_seed(0)
    base = torch.randn(1, 10, 32)
    torch.manual_seed(1)
    first6 = torch.randn(1, 6, 32)
    variant = torch.cat([first6, base[:, -4:, :]], dim=1)

    drafts0, conf0 = drafter(base)
    drafts1, conf1 = drafter(variant)
    assert tuple(drafts0.shape) == (1, 3, 32)
    assert tuple(conf0.shape) == (1, 3)
    torch.testing.assert_close(drafts0, drafts1, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(conf0, conf1, atol=1e-5, rtol=1e-5)
    assert bool((conf0 > 0).all()) and bool((conf0 < 1).all())
