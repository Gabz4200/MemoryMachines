"""CED + auxiliary memory coverage tests (prefill/decode, engram, mHC, dspark)."""

import math
from dataclasses import replace

import pytest
import torch

from memory_machines.models.blocks import (
    CausalDecoder,
    CausalEncoderDecoderStack,
    CEDGlobalKVProjection,
    CompressedSparseAttention2,
    DSparkDrafter,
    EngramMemory,
    MHCAttentionGDN2,
    SinglePassMHC,
)
from tests.test_blocks_shapes import tiny_cfg


def _csa(layer):
    """Reach the CSA2 attention branch of a CED layer.

    CED layers are parallel attention ∥ GDN-2 blocks by default
    (``ced_attention_gdn2_parallel=True``); the CSA2-specific attributes
    live on the block's ``attention`` branch.
    """
    return layer.attention if isinstance(layer, MHCAttentionGDN2) else layer


def _ced_cfg():
    return replace(
        tiny_cfg(),
        swa_bounded_replay=True,
        window_size=4,
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
    # Bounded replay decodes the last ``window_size`` tokens of the
    # prompt, anchored at their true positions ``[Le - n, Le)``.
    # The encoder and decoder prompts must cover the same tokens.
    L = 10
    H = torch.randn(1, L, 32)
    E = torch.randn(1, L, 32)

    pos2d = torch.arange(L).unsqueeze(0)
    out_prefill = model.prefill(H, E, pos2d)
    out_decode = model.decode_full_context(H[:, -4:, :], E, pos2d[:, -4:], start_pos=L - 4, seeded_store=True)
    assert tuple(out_prefill.shape) == (1, 4, 32)
    torch.testing.assert_close(out_prefill, out_decode, atol=1e-5, rtol=1e-5)

    out_prefill_none = model.prefill(H, E, None)
    out_decode_none = model.decode_full_context(H[:, -4:, :], E, None, start_pos=L - 4, seeded_store=True)
    torch.testing.assert_close(out_prefill_none, out_decode_none, atol=1e-5, rtol=1e-5)

    pos1d = torch.arange(L)
    out_prefill_1d = model.prefill(H, E, pos1d)
    out_decode_1d = model.decode_full_context(H[:, -4:, :], E, pos1d[-4:], start_pos=L - 4, seeded_store=True)
    torch.testing.assert_close(out_prefill_1d, out_decode_1d, atol=1e-5, rtol=1e-5)


def test_ced_bounded_replay_does_not_leak_future_tokens():
    """SWA Bounded Replay must not let future tokens influence past queries."""
    torch.manual_seed(0)
    cfg = _ced_cfg()
    model = CausalEncoderDecoderStack(cfg).eval()
    # Restore signal: the depth-scaled residual init
    # (std = initializer_range / sqrt(2N)) otherwise
    # leaves perturbation deltas below 1e-4 (see
    # test_ced_decode_is_globally_conditioned). The
    # query/key/value and output projections are all
    # restored so attention is selective (not uniform)
    # and the self-visibility floor is observable.
    with torch.no_grad():
        for layer in list(model.encoder_layers) + list(model.decoder_layers):
            csa = _csa(layer)
            assert isinstance(csa, CompressedSparseAttention2)
            for w in (
                csa.q_down.weight,
                csa.q_up_main.weight,
                csa.compressor.kv_proj.weight,
                csa.grouped_out.out_proj.weight,
            ):
                torch.nn.init.trunc_normal_(w, mean=0.0, std=0.5, a=-1.0, b=1.0)

    L = 10
    torch.manual_seed(1)
    H = torch.randn(1, L, 32)
    E = torch.randn(1, L, 32)
    pos = torch.arange(L).unsqueeze(0)
    base = model.prefill(H, E, pos)  # rows = tail queries at positions 6,7,8,9

    def _per_query(perturb_pos: int) -> torch.Tensor:
        H2 = H.clone()
        H2[0, perturb_pos] += 10.0
        out = model.prefill(H2, E, pos)
        return (out - base).abs().amax(-1)[0]

    # Perturbing a future tail token must not change earlier queries.
    for future in (7, 8, 9):
        assert _per_query(future)[0].item() == 0.0, f"future token {future} leaked into query at position 6"
    # A query must still see its own token (no self-invisibility).
    assert _per_query(6)[0].item() > 1e-4


def test_ced_decode_is_globally_conditioned():
    torch.manual_seed(0)
    cfg = _ced_cfg()
    model = CausalEncoderDecoderStack(cfg).eval()
    # Sparse layers use a depth-scaled residual output;
    # restore signal so the global-conditioning comparison
    # is meaningful.
    with torch.no_grad():
        for layer in list(model.encoder_layers) + list(model.decoder_layers):
            csa = _csa(layer)
            assert isinstance(csa, CompressedSparseAttention2)
            torch.nn.init.xavier_uniform_(csa.grouped_out.out_proj.weight)
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

    # Zero value_proj (and conv weights) recovers the residual exactly.
    mem = EngramMemory(cfg).eval()
    torch.nn.init.zeros_(mem.value_proj.weight)
    if mem.conv is not None:
        torch.nn.init.zeros_(mem.conv.conv.weight)
    torch.manual_seed(1)
    input_ids = torch.randint(0, 50, (1, 6))
    hidden = torch.randn(1, 6, 32)
    out = mem(input_ids, hidden)
    torch.testing.assert_close(out, hidden, atol=1e-5, rtol=1e-5)

    torch.manual_seed(0)
    mem2 = EngramMemory(cfg).eval()
    # Zero-start value_proj would hide the memory path entirely;
    # restore signal so last-token sensitivity is observable.
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(mem2.value_proj.weight)
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

    # Distinct hash heads map the same n-grams to different table ids.
    ids = torch.tensor([[5, 7, 9, 1, 2, 3]])
    hashes = mem2._ngram_hashes(ids, layer_id=0)
    # First two heads (order 2) must differ.
    assert not torch.equal(hashes[..., 0], hashes[..., 1])


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

    # Single-pass (Eq. 6): input mixing uses the previous block's A,
    # so different prev_a must produce different outputs through both
    # the canonical forward path and its ``step`` alias.
    out_fwd_A, _ = mhc(streams, branch_fn=lambda h: h, prev_a=prev_A)
    out_fwd_B, _ = mhc(streams, branch_fn=lambda h: h, prev_a=prev_B)
    assert (out_fwd_A - out_fwd_B).abs().max().item() > 0.0

    out_step_A, _ = mhc.step(streams, branch_fn=lambda h: h, prev_a=prev_A)
    out_step_B, _ = mhc.step(streams, branch_fn=lambda h: h, prev_a=prev_B)
    assert (out_step_A - out_step_B).abs().max().item() > 0.0
    # ``step`` is an alias of the single-pass forward path.
    torch.testing.assert_close(out_step_A, out_fwd_A, atol=1e-6, rtol=1e-6)

    # First block (prev_a=None) falls back to the current block's own a.
    out_first, a_first = mhc(streams, branch_fn=lambda h: h)
    out_bootstrap, _ = mhc(streams, branch_fn=lambda h: h, prev_a=a_first)
    torch.testing.assert_close(out_first, out_bootstrap, atol=1e-6, rtol=1e-6)


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


def _ced_decoder_cfg():
    return replace(
        tiny_cfg(),
        swa_bounded_replay=True,
        window_size=4,
        ced_num_decoder_layers=2,
        csa_encoder_compression_ratio=2,
        csa_decoder_compression_ratio=1,
    )


def test_causal_decoder_matches_stack_decoder():
    """The standalone CausalDecoder reproduces the stack's decoder path."""
    torch.manual_seed(0)
    cfg = _ced_decoder_cfg()
    stack = CausalEncoderDecoderStack(cfg).eval()
    dec = CausalDecoder(
        cfg,
        layers=list(stack.decoder_layers),
        global_projs=list(stack.global_projs),
    ).eval()

    L = 10
    torch.manual_seed(1)
    H = torch.randn(1, L, 32)
    E = torch.randn(1, L, 32)
    pos = torch.arange(L).unsqueeze(0)
    out_stack = stack.prefill(H, E, pos)
    out_dec = dec.prefill(H, E, pos)
    torch.testing.assert_close(out_stack, out_dec, atol=1e-5, rtol=1e-5)

    # forward() == prefill()
    torch.testing.assert_close(dec(H, E, pos), out_dec, atol=1e-5, rtol=1e-5)


def test_causal_decoder_global_kv_projected_from_encoder():
    """Decoder global KV is projected from encoder-final states (Eq. 1),
    not derived from the decoder's own hidden states."""
    torch.manual_seed(0)
    cfg = _ced_decoder_cfg()
    dec = CausalDecoder(cfg).eval()
    # Sparse layers use a depth-scaled residual output; restore
    # signal so the global-conditioning comparison is meaningful.
    with torch.no_grad():
        for layer in dec.layers:
            csa = _csa(layer)
            assert isinstance(csa, CompressedSparseAttention2)
            torch.nn.init.trunc_normal_(
                csa.grouped_out.out_proj.weight,
                mean=0.0,
                std=0.5,
                a=-1.0,
                b=1.0,
            )

    L = 8
    torch.manual_seed(1)
    H = torch.randn(1, L, 32)
    E = torch.randn(1, L, 32)
    # Same decoder inputs, different encoder-final states ->
    # different outputs (globally conditioned).
    out_a = dec(H, E)
    E_b = torch.randn(1, L, 32)
    out_b = dec(H, E_b)
    assert (out_a - out_b).abs().max().item() > 1e-4

    # Decoder layer modes cycle full/reindex/reuse.
    assert [layer.mode for layer in dec.layers] == ["full", "reindex"]


def test_causal_decoder_bounded_replay_no_leakage():
    """CausalDecoder SWA Bounded Replay must not leak future tokens."""
    torch.manual_seed(0)
    cfg = _ced_decoder_cfg()
    dec = CausalDecoder(cfg).eval()
    with torch.no_grad():
        for layer in dec.layers:
            csa = _csa(layer)
            assert isinstance(csa, CompressedSparseAttention2)
            for w in (
                csa.q_down.weight,
                csa.q_up_main.weight,
                csa.compressor.kv_proj.weight,
                csa.grouped_out.out_proj.weight,
            ):
                torch.nn.init.trunc_normal_(w, mean=0.0, std=0.5, a=-1.0, b=1.0)

    L = 10
    torch.manual_seed(1)
    H = torch.randn(1, L, 32)
    E = torch.randn(1, L, 32)
    pos = torch.arange(L).unsqueeze(0)
    base = dec.prefill(H, E, pos)

    def _per_query(perturb_pos: int) -> torch.Tensor:
        H2 = H.clone()
        H2[0, perturb_pos] += 10.0
        out = dec.prefill(H2, E, pos)
        return (out - base).abs().amax(-1)[0]

    # Perturbing a future tail token must not change earlier queries.
    for future in (7, 8, 9):
        delta = _per_query(future)
        assert delta[0].item() == 0.0, f"future token {future} leaked into query at position 6"
    # A query must still see its own token.
    assert _per_query(6)[0].item() > 1e-4


def test_causal_decoder_rejects_mismatched_layers_and_projs():
    """Explicit layers/global_projs must have equal length (zip guard)."""
    cfg = _ced_decoder_cfg()
    layer = CompressedSparseAttention2(cfg, cfg.csa_decoder_compression_ratio, "full")
    proj = CEDGlobalKVProjection(cfg)
    with pytest.raises(ValueError, match="must have equal length"):
        CausalDecoder(cfg, layers=[layer, layer], global_projs=[proj])


def test_causal_decoder_rejects_short_position_ids():
    """position_ids must cover the replay window."""
    cfg = _ced_decoder_cfg()
    dec = CausalDecoder(cfg).eval()
    L = 10
    torch.manual_seed(1)
    H = torch.randn(1, L, 32)
    E = torch.randn(1, L, 32)
    # 2D positions shorter than the replay window (window_size=4).
    short = torch.arange(L).unsqueeze(0)[:, :2]
    with pytest.raises(ValueError, match="must cover the replay"):
        dec.prefill(H, E, short)
