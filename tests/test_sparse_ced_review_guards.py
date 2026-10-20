"""Review-guard tests for sparse.py / ced.py (dead projs, corrupt caches,
position validation, dtype casts, seeded-store ablation, bounded replay,
zip guards, SWA positions, replay anchoring, selection parity)."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from memory_machines.models.blocks import (
    CausalDecoder,
    CausalEncoderDecoderStack,
    CEDGlobalKVProjection,
    CompressedSparseAttention,
    CompressedSparseAttention2,
    ParallelAttentionGDN2,
)
from memory_machines.models.blocks.shared import CompressedCache, CSA2SharedState
from tests.test_blocks_shapes import tiny_cfg

H = 32
HKV = 2
DH = 8


def _csa_branch(layer):
    """Reach the CSA2 attention branch of a CED layer (parallel hybrid or plain)."""
    return layer.attention if isinstance(layer, ParallelAttentionGDN2) else layer


def _csa(**kw) -> CompressedSparseAttention:
    torch.manual_seed(0)
    kw.setdefault("compression_ratio", 2)
    return CompressedSparseAttention(tiny_cfg(), **kw).eval()


def _csa2(mode="full", **kw) -> CompressedSparseAttention2:
    torch.manual_seed(0)
    kw.setdefault("compression_ratio", 2)
    kw.setdefault("top_k", 2)
    return CompressedSparseAttention2(tiny_cfg(), mode=mode, **kw).eval()


def _ced_cfg(**kw):
    base = {
        "swa_bounded_replay": True,
        "window_size": 4,
        "ced_num_encoder_layers": 1,
        "ced_num_decoder_layers": 2,
        "csa_encoder_compression_ratio": 2,
        "csa_decoder_compression_ratio": 1,
    }
    base.update(kw)
    return replace(tiny_cfg(), **base)


# 1. Dead projections -------------------------------------------------------
def test_dead_projs_identity_and_project_qkv_raises():
    layer = _csa()
    for name in ("q_proj", "k_proj", "o_proj"):
        assert isinstance(getattr(layer, name), nn.Identity), name
    params = dict(layer.named_parameters())
    for dead in ("q_proj.weight", "k_proj.weight", "o_proj.weight"):
        assert dead not in params, f"dead param {dead} still present"
    # The SWA branch keeps its own live projections.
    assert "swa.q_proj.weight" in params
    with pytest.raises(NotImplementedError, match="drops the parent dense"):
        layer._project_qkv(torch.randn(1, 2, H))


def test_legacy_qkv_keys_load_strict_false():
    layer = _csa()
    sd = layer.state_dict()
    sd["q_proj.weight"] = torch.randn(H, H)
    sd["k_proj.weight"] = torch.randn(HKV * DH, H)
    sd["o_proj.weight"] = torch.randn(H, H)
    out = layer.load_state_dict(sd, strict=False)
    assert "q_proj.weight" in out.unexpected_keys


# 2. Corrupt caches ----------------------------------------------------------
def _corrupt_call(path: str):
    x = torch.randn(1, 4, H)
    if path == "csa":
        layer = _csa()
        past = CompressedCache(
            compressed_kv=torch.randn(1, 2, HKV, DH),
            comp_end_pos=None,
            total_len=4,
        )
        return lambda: layer(x, past=past)
    if path == "full":
        layer = _csa2("full")
        shared = CSA2SharedState(
            compressed_kv=torch.randn(1, 2, HKV, DH),
            comp_end_pos=None,
        )
        return lambda: layer.forward_shared(x, shared)
    if path == "reindex":
        layer = _csa2("reindex")
        shared = CSA2SharedState(
            compressed_kv=torch.randn(1, 2, HKV, DH),
            comp_end_pos=None,
            indexer_k=torch.randn(1, 2, 8),
        )
        return lambda: layer.forward_shared(x, shared)
    assert path == "reuse"
    layer = _csa2("reuse")
    shared = CSA2SharedState(
        compressed_kv=torch.randn(1, 2, HKV, DH),
        comp_end_pos=None,
        topk_indices=torch.zeros(1, 4, 2, dtype=torch.long),
    )
    return lambda: layer.forward_shared(x, shared)


@pytest.mark.parametrize("path", ["csa", "full", "reindex", "reuse"])
def test_corrupt_cache_raises(path):
    with pytest.raises(ValueError, match="corrupt"), torch.no_grad():
        _corrupt_call(path)()


# 3. comp_end_pos rank/batch validation ---------------------------------------
def _bad_ends_cases_csa():
    kv = torch.randn(1, 2, HKV, DH)
    return [
        (torch.zeros(1, 2, 2, dtype=torch.long), "must be"),
        (torch.zeros(2, 2, dtype=torch.long), "does not match input batch"),
    ], kv


def test_csa_comp_end_pos_rank_and_batch():
    bad_cases, kv = _bad_ends_cases_csa()
    for ends, match in bad_cases:
        layer = _csa()
        past = CompressedCache(compressed_kv=kv.clone(), comp_end_pos=ends, total_len=0)
        with pytest.raises(ValueError, match=match), torch.no_grad():
            layer(torch.randn(1, 4, H), past=past)


def test_csa2_comp_end_pos_rank_and_batch():
    for ends, match in [
        (torch.zeros(1, 2, 2, dtype=torch.long), "must be"),
        (torch.zeros(2, 2, dtype=torch.long), "does not match input batch"),
    ]:
        layer = _csa2("full")
        shared = CSA2SharedState(
            compressed_kv=torch.randn(1, 2, HKV, DH),
            comp_end_pos=ends,
        )
        with pytest.raises(ValueError, match=match), torch.no_grad():
            layer.forward_shared(torch.randn(1, 4, H), shared)


def test_1d_comp_end_pos_broadcast_control():
    torch.manual_seed(1)
    x = torch.randn(2, 4, H)
    with torch.no_grad():
        layer = _csa()
        # Real past carries valid SWA caches; its shared (Lc,) ends take
        # the 1D broadcast path for a batched continuation.
        _, _, past = layer(x, use_cache=True)
        assert past is not None and past.comp_end_pos is not None
        assert past.comp_end_pos.dim() == 1
        torch.manual_seed(2)
        out, _, present = layer(torch.randn(2, 4, H), past=past, use_cache=True)
        assert bool(torch.isfinite(out).all())
        assert present is not None

        layer2 = _csa2("full")
        shared = CSA2SharedState(
            compressed_kv=torch.randn(1, 2, HKV, DH),
            comp_end_pos=torch.tensor([1, 3]),
        )
        torch.manual_seed(3)
        out2, _ = layer2.forward_shared(torch.randn(1, 4, H), shared)
        assert bool(torch.isfinite(out2).all())


# 4. Cache concat dtype/device cast + chunked-vs-full --------------------------
def test_cache_concat_cast_and_chunked_equivalence():
    torch.manual_seed(0)
    layer = _csa()
    torch.manual_seed(1)
    x = torch.randn(1, 8, H)
    with torch.no_grad():
        out_full, _, _ = layer(x)
        out1, _, past = layer(x[:, :4, :], use_cache=True)
        assert past is not None
        past.compressed_kv = past.compressed_kv.to(torch.float64)
        out2, _, past2 = layer(x[:, 4:, :], past=past, use_cache=True)
    assert bool(torch.isfinite(out2).all())
    assert past2 is not None
    assert past2.compressed_kv is not None
    assert past2.compressed_kv.dtype == torch.float32
    assert past2.compressed_kv.device == x.device
    torch.testing.assert_close(torch.cat([out1, out2], dim=1), out_full, atol=1e-5, rtol=1e-5)


# 5. seeded_store ablation -----------------------------------------------------
def _upscaled_decoder(cfg) -> CausalDecoder:
    torch.manual_seed(0)
    dec = CausalDecoder(cfg).eval()
    with torch.no_grad():
        for layer in dec.layers:
            csa = _csa_branch(layer)
            assert isinstance(csa, CompressedSparseAttention2)
            for w in (
                csa.q_down.weight,
                csa.q_up_main.weight,
                csa.compressor.kv_proj.weight,
                csa.grouped_out.out_proj.weight,
            ):
                torch.nn.init.trunc_normal_(w, mean=0.0, std=0.5, a=-1.0, b=1.0)
    return dec


def test_seeded_store_false_ablation_differs():
    cfg = _ced_cfg()
    dec = _upscaled_decoder(cfg)
    torch.manual_seed(1)
    Hd = torch.randn(1, 4, H)
    E = torch.randn(1, 6, H)
    with torch.no_grad():
        a = dec.decode_full_context(Hd, E)
        b = dec.decode_full_context(Hd, E, seeded_store=False)
    assert (a - b).abs().max().item() > 1e-4


def test_seeded_store_true_requires_store():
    layer = _csa2("full")
    with pytest.raises(ValueError, match="seed the store"), torch.no_grad():
        layer.forward_shared(torch.randn(1, 4, H), CSA2SharedState(), seeded_store=True)
    shared = CSA2SharedState(
        compressed_kv=torch.randn(1, 2, HKV, DH),
        comp_end_pos=None,
    )
    with pytest.raises(ValueError, match="corrupt"), torch.no_grad():
        layer.forward_shared(torch.randn(1, 4, H), shared, seeded_store=True)


# 6. Bounded replay guard ------------------------------------------------------
def test_bounded_replay_needs_position_ids_when_lengths_differ():
    cfg = _ced_cfg()
    torch.manual_seed(0)
    dec = CausalDecoder(cfg).eval()
    torch.manual_seed(1)
    Hd = torch.randn(1, 6, H)
    E = torch.randn(1, 8, H)
    with pytest.raises(ValueError, match="explicit position_ids"):
        dec.prefill(Hd, E, None)
    pos = torch.arange(6).unsqueeze(0)
    out = dec.prefill(Hd, E, pos)
    assert tuple(out.shape) == (1, 4, H)


# 7. dec_n-length zip guards ----------------------------------------------------
def test_decoder_zip_guards_both_directions():
    cfg = _ced_cfg()  # dec_n == 2
    dec_n = cfg.ced_num_decoder_layers
    assert dec_n == 2
    layer = CompressedSparseAttention2(cfg, cfg.csa_decoder_compression_ratio, "full")
    with pytest.raises(ValueError, match="must have length dec_n"):
        CausalDecoder(cfg, layers=[layer] * (dec_n + 1))
    proj = CEDGlobalKVProjection(cfg)
    with pytest.raises(ValueError, match="must have length dec_n"):
        CausalDecoder(cfg, global_projs=[proj])


# 8. CSA2 SWA receives explicit q_pos -------------------------------------------
def _capture_swa_qpos(layer: CompressedSparseAttention2, x, shared):
    orig = layer.swa.forward
    captured = {}

    def spy(hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kw):
        captured["position_ids"] = position_ids.clone() if position_ids is not None else None
        return orig(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache=use_cache,
            **kw,
        )

    layer.swa.forward = spy  # type: ignore[method-assign]
    try:
        with torch.no_grad():
            out, shared = layer.forward_shared(x, shared)
    finally:
        layer.swa.forward = orig  # type: ignore[method-assign]
    return captured["position_ids"], out, shared


@pytest.mark.parametrize("mode", ["full", "reuse"])
def test_csa2_swa_receives_sparse_qpos(mode):
    torch.manual_seed(0)
    cfg = tiny_cfg()
    full = CompressedSparseAttention2(cfg, 2, "full", top_k=2).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 4, H)
    shared = CSA2SharedState(total_len=4)
    expected = torch.arange(4, 8).unsqueeze(0)
    with torch.no_grad():
        if mode == "full":
            captured, out, _ = _capture_swa_qpos(full, x, shared)
        else:
            _, shared = full.forward_shared(x, shared)
            assert shared.topk_indices is not None
            reuse = CompressedSparseAttention2(cfg, 2, "reuse", top_k=2).eval()
            reuse.load_state_dict(full.state_dict(), strict=True)
            captured, out, _ = _capture_swa_qpos(reuse, x, shared)
    assert captured is not None
    torch.testing.assert_close(captured, expected)
    assert int(captured[0, -1].item()) == 4 + 4 - 1
    assert not bool((captured == torch.arange(4).unsqueeze(0)).all())
    assert bool(torch.isfinite(out).all())


# 9. start_pos default + fallback ------------------------------------------------
def test_prefill_seq_eq_enc_matches_decode_tail():
    cfg = _ced_cfg()
    torch.manual_seed(0)
    stack = CausalEncoderDecoderStack(cfg).eval()
    torch.manual_seed(1)
    L = 8
    Hd = torch.randn(1, L, H)
    E = torch.randn(1, L, H)
    replay = 4
    with torch.no_grad():
        out_pre = stack.prefill(Hd, E, None)
        out_dec = stack.decode_full_context(
            Hd[:, -replay:, :],
            E,
            torch.arange(L - replay, L).unsqueeze(0),
            start_pos=L - replay,
            seeded_store=True,
        )
    assert tuple(out_pre.shape) == (1, replay, H)
    torch.testing.assert_close(out_pre, out_dec, atol=1e-5, rtol=1e-5)


def test_prefill_fallback_returns_full_rows():
    torch.manual_seed(1)
    Hd = torch.randn(1, 6, H)
    E = torch.randn(1, 8, H)
    # window <= 0 disables replay.
    cfg = _ced_cfg(window_size=0)
    torch.manual_seed(0)
    dec = CausalDecoder(cfg).eval()
    with torch.no_grad():
        out = dec.prefill(Hd, E, None)
    assert tuple(out.shape) == (1, 6, H)
    # seq > enc cannot reconstruct a tail: full decode.
    cfg2 = _ced_cfg()
    torch.manual_seed(0)
    dec2 = CausalDecoder(cfg2).eval()
    torch.manual_seed(2)
    Hd2 = torch.randn(1, 8, H)
    E2 = torch.randn(1, 6, H)
    with torch.no_grad():
        out2 = dec2.prefill(Hd2, E2, None)
    assert tuple(out2.shape) == (1, 8, H)


# 10. _selected_positions parity -------------------------------------------------
def test_selected_positions_1d_vs_2d_parity():
    topk = torch.tensor([[[0, 2], [1, 0]], [[2, 0], [1, 1]]])
    e1d = torch.tensor([1, 3, 5])
    e2d_same = torch.tensor([[1, 3, 5], [1, 3, 5]])
    e2d_diff = torch.tensor([[1, 3, 5], [10, 30, 50]])
    f = CompressedSparseAttention._selected_positions
    torch.testing.assert_close(f(e1d, topk), f(e2d_same, topk))
    out_diff = f(e2d_diff, topk)
    torch.testing.assert_close(out_diff[0], f(e1d, topk)[0])
    assert not bool((out_diff[0] == out_diff[1]).all())


# NICE-TO-HAVE ------------------------------------------------------------------
def test_prefill_1d_position_ids_slicing():
    cfg = _ced_cfg()
    torch.manual_seed(0)
    dec = CausalDecoder(cfg).eval()
    torch.manual_seed(1)
    L = 8
    Hd = torch.randn(1, L, H)
    E = torch.randn(1, L, H)
    pos1d = torch.arange(L)
    pos2d = torch.arange(L).unsqueeze(0)
    with torch.no_grad():
        out1d = dec.prefill(Hd, E, pos1d)
        out2d = dec.prefill(Hd, E, pos2d)
    assert tuple(out1d.shape) == (1, 4, H)
    torch.testing.assert_close(out1d, out2d, atol=1e-5, rtol=1e-5)


def test_compress_from_start_pos_path():
    layer = _csa2("full")
    torch.manual_seed(1)
    x = torch.randn(1, 4, H)
    enc = torch.randn(1, 8, H)
    with torch.no_grad():
        out, shared = layer.forward_shared(x, CSA2SharedState(), compress_from=enc, compress_start_pos=0)
    assert shared.compressed_kv is not None
    assert shared.compressed_kv.shape[1] == 4
    assert shared.comp_end_pos is not None
    assert shared.comp_end_pos.tolist() == [1, 3, 5, 7]
    assert tuple(out.shape) == (1, 4, H)
    assert bool(torch.isfinite(out).all())


def test_reuse_q_len_2_expands_last_row():
    layer = _csa2()
    torch.manual_seed(0)
    x = torch.randn(1, 8, H)
    with torch.no_grad():
        _, shared = layer.forward_shared(x, CSA2SharedState(), use_cache=True)
    assert shared.topk_indices is not None
    full_topk = shared.topk_indices.clone()
    layer.mode = "reuse"
    captured = {}
    orig = type(layer)._gather_selected

    def spy(full_comp, topk):
        captured["topk"] = topk.clone()
        return orig(full_comp, topk)

    layer._gather_selected = spy  # type: ignore[method-assign]
    try:
        with torch.no_grad():
            out, _ = layer.forward_shared(x[:, -2:, :], shared, use_cache=True)
    finally:
        del layer._gather_selected  # type: ignore[attr-defined]
    expected = full_topk[:, -1:, :].expand(1, 2, -1)
    torch.testing.assert_close(captured["topk"], expected, atol=0, rtol=0)
    assert tuple(out.shape) == (1, 2, H)


def test_encoder_reuse_and_empty_layers():
    cfg = _ced_cfg(ced_num_encoder_layers=3)
    torch.manual_seed(0)
    stack = CausalEncoderDecoderStack(cfg).eval()
    # Layer 0 is plain SWA, then the first CSA2 layer seeds the store.
    assert [layer.mode for layer in stack.encoder_layers] == [None, "full", "reuse"]
    torch.manual_seed(1)
    with torch.no_grad():
        out, shared = stack.encode(torch.randn(1, 8, H))
    assert tuple(out.shape) == (1, 8, H)
    assert shared.compressed_kv is not None
    empty = CausalEncoderDecoderStack(cfg, encoder_layers=[])
    with pytest.raises(ValueError, match="empty"), torch.no_grad():
        empty.encode(torch.randn(1, 8, H))


def test_explicit_encoder_layers_bypass_config_building():
    cfg = _ced_cfg(ced_num_encoder_layers=2)
    explicit: list[nn.Module] = [CompressedSparseAttention2(cfg, cfg.csa_encoder_compression_ratio, "full")]
    stack = CausalEncoderDecoderStack(cfg, encoder_layers=explicit).eval()
    assert len(stack.encoder_layers) == len(stack.encoder_ffns) == 1
    assert type(stack.encoder_layers[0]) is CompressedSparseAttention2
    with torch.no_grad():
        out, shared = stack.encode(torch.randn(1, 8, H))
    assert tuple(out.shape) == (1, 8, H)
    assert shared.compressed_kv is not None


def test_stack_encode_rejects_a_wrong_typed_encoder_layer():
    cfg = _ced_cfg()
    stack = CausalEncoderDecoderStack(cfg, encoder_layers=[nn.Linear(H, H)])
    with pytest.raises(TypeError, match="CompressedSparseAttention2 or ParallelAttentionGDN2"), torch.no_grad():
        stack.encode(torch.randn(1, 4, H))


def test_explicit_encoder_layers_intervals_must_match_the_explicit_count():
    cfg = _ced_cfg(ced_num_encoder_layers=2, ced_encoder_ffn_update_intervals=(1, 2))
    explicit: list[nn.Module] = [CompressedSparseAttention2(cfg, cfg.csa_encoder_compression_ratio, "full")]
    with pytest.raises(ValueError, match="must equal number of encoder layers"):
        CausalEncoderDecoderStack(cfg, encoder_layers=explicit)
