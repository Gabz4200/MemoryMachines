"""Regression tests pinning the code-review fixes (10/10 gate).

Each test fails on the pre-fix code and passes after. Deterministic:
global seed + deterministic algorithms where applicable.
"""

import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.backends.cuda.matmul.allow_tf32 = False
except Exception:
    pass


def test_ops_package_reexports():
    import memory_machines.ops as ops

    for name in ("dispatch_dw_scale", "dispatch_causal_conv1d", "dispatch_causal_conv1d_update", "dw_scale"):
        assert hasattr(ops, name), f"ops.{name} missing from re-exports"


def test_dw_scale_validation():
    from memory_machines.ops.dispatch import dispatch_dw_scale

    x = torch.randn(2, 4)
    w = torch.randn(2, 5)
    try:
        dispatch_dw_scale(x, w, backend="naive")
    except ValueError as e:
        assert "shape mismatch" in str(e)
    else:
        raise AssertionError("expected shape mismatch")
    # Case-insensitive backend.
    y = dispatch_dw_scale(torch.ones(2, 3), torch.ones(2, 3) * 2, backend="Naive")
    torch.testing.assert_close(y, torch.ones(2, 3) * 2)
    # Non-2D raises (compiled path would OOB without this).
    from memory_machines.ops.dw_scale import _check_dw_scale_inputs

    try:
        _check_dw_scale_inputs(torch.randn(4), torch.randn(4))
    except ValueError as e:
        assert "2D" in str(e)
    else:
        raise AssertionError("expected 2D validation")


def test_attention_bidirectional_window():
    from tests.test_blocks_shapes import tiny_cfg

    from memory_machines.models.blocks import GroupedQueryAttention

    torch.manual_seed(0)
    import dataclasses

    cfg = dataclasses.replace(tiny_cfg(), cvpe_enabled=False, frvm_enabled=False, window_size=4)
    m = GroupedQueryAttention(cfg).eval()
    x = torch.randn(1, 6, 32)
    with torch.no_grad():
        out_causal, w_c, _ = m(x, is_causal=True, output_attentions=True)
        out_bi, w_b, _ = m(x, is_causal=False, output_attentions=True)
    # Bidirectional sees future within window -> outputs must differ.
    assert (out_causal - out_bi).abs().max().item() > 1e-6
    assert w_c is not None and w_b is not None
    # Causal row 0 attends only to key 0; bidirectional row 0 attends to 0..3.
    assert w_c[0, 0, 0, 1:].abs().max().item() == 0.0
    assert w_b[0, 0, 0, 1] > 0.0


def test_compressor_clamps_partial_block():
    from memory_machines.models.blocks import BlockCompressor

    comp = BlockCompressor(16, 2, 8, compression_ratio=4).eval()
    x = torch.randn(1, 6, 16)
    _, ends = comp(x)
    # Real last token is 5; unclamped would point into padding at 7.
    assert ends.tolist() == [3, 5]


def test_attention_sink_fp32_no_nan():
    from memory_machines.models.blocks.compression import AttentionSink

    sink = AttentionSink(2).eval()
    logits = torch.full((1, 2, 1, 4), 50.0)  # would overflow in fp16/bf16 exp
    with torch.no_grad():
        w = sink.weights(logits.float())
    assert torch.isfinite(w).all()
    assert (w.sum(-1) <= 1.0 + 1e-5).all()


def test_sinkhorn_large_logits_no_nan():
    from memory_machines.models.blocks.shared import _sinkhorn

    M = torch.full((2, 2), 50.0)
    P = _sinkhorn(M, 5)
    assert torch.isfinite(P).all()
    torch.testing.assert_close(P.sum(-1), torch.ones(2), atol=1e-4, rtol=1e-4)


def test_causal_conv_packed_rank_preserved_and_t0():
    from memory_machines.ops.references.causal_conv1d_reference import reference_causal_conv1d_fwd

    torch.manual_seed(0)
    D, W = 4, 3
    weight = torch.randn(D, W, dtype=torch.float64)
    packed_3d = torch.randn(1, 8, D, dtype=torch.float64)
    packed_2d = packed_3d.squeeze(0)
    cu = torch.tensor([0, 3, 8])
    y3 = reference_causal_conv1d_fwd(packed_3d, weight, cu_seqlens=cu)
    y2 = reference_causal_conv1d_fwd(packed_2d, weight, cu_seqlens=cu)
    assert tuple(y3.shape) == (1, 8, D)
    assert tuple(y2.shape) == (8, D)
    torch.testing.assert_close(y3.squeeze(0), y2, atol=1e-12, rtol=1e-12)
    # Empty leading segment (T=0) returns empty, not the whole output.
    cu_empty = torch.tensor([0, 0, 4])
    ye = reference_causal_conv1d_fwd(torch.randn(1, 4, D, dtype=torch.float64), weight, cu_seqlens=cu_empty)
    assert tuple(ye.shape) == (1, 4, D)


def test_causal_conv_bad_cu_seqlens_raise():
    from memory_machines.ops.references.causal_conv1d_reference import reference_causal_conv1d_fwd

    x = torch.randn(1, 4, 4)
    w = torch.randn(4, 3)
    for bad in (torch.tensor([1, 4]), torch.tensor([0, 3]), torch.tensor([0, 4, 2])):
        try:
            reference_causal_conv1d_fwd(x, w, cu_seqlens=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for cu={bad.tolist()}")


def test_ced_seeded_default_and_zip_guard():
    from memory_machines.models.blocks import CausalDecoder, CEDGlobalKVProjection, CompressedSparseAttention2
    from tests.test_blocks_shapes import tiny_cfg

    cfg = tiny_cfg()
    layer = CompressedSparseAttention2(cfg, cfg.csa_decoder_compression_ratio, "full")
    proj = CEDGlobalKVProjection(cfg)
    try:
        CausalDecoder(cfg, layers=[layer, layer], global_projs=[proj])
    except ValueError as e:
        assert "equal length" in str(e)
    else:
        raise AssertionError("expected zip guard")
    # Default decode equals explicit seeded_store=True (Eq. 1, no double-count).
    import dataclasses

    cfg2 = dataclasses.replace(tiny_cfg(), swa_bounded_replay=True, window_size=4, ced_num_decoder_layers=1, csa_decoder_compression_ratio=1)
    torch.manual_seed(0)
    dec = CausalDecoder(cfg2).eval()
    torch.manual_seed(1)
    H = torch.randn(1, 4, 32)
    E = torch.randn(1, 6, 32)
    with torch.no_grad():
        a = dec.decode_full_context(H, E)
        b = dec.decode_full_context(H, E, seeded_store=True)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_sparse_batched_comp_ends():
    from tests.test_blocks_shapes import tiny_cfg

    from memory_machines.models.blocks import CompressedSparseAttention

    torch.manual_seed(0)
    layer = CompressedSparseAttention(tiny_cfg()).eval()
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        out, _, cache = layer(x, use_cache=True)
        assert cache is not None
        # Simulate a batched (B, Lc) store and continue decoding.
        cache.comp_end_pos = cache.comp_end_pos.unsqueeze(0).expand(2, -1).contiguous()
        x2 = torch.randn(2, 2, 32)
        out2, _, cache2 = layer(x2, past=cache, use_cache=True)
    assert tuple(out.shape) == (2, 6, 32)
    assert tuple(out2.shape) == (2, 2, 32)
    assert cache2 is not None and cache2.comp_end_pos.dim() == 2


def test_positioning_bf16_no_dtype_crash():
    from memory_machines.models.blocks.positioning import CVPEContextBias

    mod = CVPEContextBias(head_dim=8).to(torch.bfloat16).eval()
    q = torch.randn(1, 2, 3, 8, dtype=torch.bfloat16)
    with torch.no_grad():
        out = mod(q, torch.tensor([[2, 3, 4]]), torch.tensor([[0, 1, 2, 3, 4]]))
    assert out.shape == (1, 2, 3, 5)
    assert torch.isfinite(out.float()).all()


def test_gdn2_cu_validation_and_masked_contract_and_chunk_passthrough():
    from memory_machines.models.blocks import gdn2
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer
    from memory_machines.utils.config_schema import MemoryMachineConfig

    torch.manual_seed(0)
    q = torch.randn(1, 4, 1, 4)
    k = torch.randn(1, 4, 1, 4)
    v = torch.randn(1, 4, 1, 4)
    g = -torch.rand(1, 4, 1, 4)
    b = torch.sigmoid(torch.randn(1, 4, 1, 4))
    w = torch.sigmoid(torch.randn(1, 4, 1, 4))
    for bad in (torch.tensor([1, 4]), torch.tensor([0, 3])):
        try:
            gdn2.gated_delta_rule_2(q, k, v, g, b, w, cu_seqlens=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected cu validation for {bad.tolist()}")

    blk = gdn2.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, use_short_conv=False).eval()
    x = torch.randn(1, 4, 16)
    with torch.no_grad():
        o, _, _ = blk(x, attention_mask=torch.zeros(1, 4))
    assert tuple(o.shape) == (1, 4, 16)
    assert o.abs().amax().item() == 0.0

    cfg = MemoryMachineConfig(hidden_size=16, num_attention_heads=2, num_key_value_heads=2, csa_num_output_groups=2)
    mixer = GatedDeltaNet2Mixer(cfg, use_short_conv=False).eval()
    before = mixer.layer.chunk_size
    with torch.no_grad():
        mixer(torch.randn(1, 4, 16), chunk_size=2)
    assert mixer.layer.chunk_size == before, "mixer must not mutate layer.chunk_size"


def test_shortconv_2d_no_grad_does_not_take_decode_path():
    from memory_machines.models.blocks.causal_conv1d import ShortConvolution

    torch.manual_seed(0)
    conv = ShortConvolution(hidden_size=4, kernel_size=3).eval()
    x = torch.randn(6, 4, requires_grad=True)  # 2D [L, D], grad enabled path
    with torch.no_grad():
        # 2D input without cu_seqlens must raise (dense requires 3D), not route to step().
        try:
            conv(x)
        except ValueError as e:
            assert "dense" in str(e)
        else:
            raise AssertionError("expected dense-rank validation")


def test_ssd_table_validation_and_engram_neg_ids():
    import tempfile
    from pathlib import Path

    import pytest

    from memory_machines.models.blocks.memory import _SsdTable
    from memory_machines.utils.config_schema import MemoryMachineConfig
    from memory_machines.models.blocks import EngramMemory

    with tempfile.TemporaryDirectory() as d:
        t = _SsdTable(Path(d) / "t.bin", 6, 3)
        with pytest.raises(ValueError, match="non-negative|out of range"):
            t(torch.tensor([-1, 0]))
        with pytest.raises(ValueError, match="out of range"):
            t(torch.tensor([0, 99]))
    mem = EngramMemory(MemoryMachineConfig(hidden_size=16, engram_num_entries=64)).eval()
    with pytest.raises(ValueError, match="non-negative"):
        mem._ngram_hashes(torch.tensor([[-1, 2]]), 0)
