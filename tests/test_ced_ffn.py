"""Dense feed-forward sub-layers on the CED encoder and decoder.

Both halves own a dense (non-MoE) FFN, selected independently through
``config.ced_encoder_ffn`` / ``config.ced_decoder_ffn`` (both default
``"swiglu"``) and with independent widths. Covers the FFN factory, the
identity/pointwise/locality behaviour, and the encoder/decoder independence.
"""

from dataclasses import replace
from typing import cast

import pytest
import torch

from memory_machines.models.blocks import (
    CausalEncoderDecoderStack,
    CompressedSparseAttention2,
    DenseFFN,
    ParallelAttentionGDN2,
    SwiGLU,
    build_ffn,
    to_streams,
)
from memory_machines.utils.config_schema import MemoryMachineConfig
from tests.test_blocks_shapes import tiny_cfg

H = 32
N = 2


def _ced_cfg(**kw) -> MemoryMachineConfig:
    return replace(
        tiny_cfg(),
        mhc_num_streams=N,
        swa_bounded_replay=True,
        window_size=4,
        ced_num_encoder_layers=2,
        ced_num_decoder_layers=3,
        csa_encoder_compression_ratio=2,
        csa_decoder_compression_ratio=1,
        **kw,
    )


def _encoder_ffns(stack: CausalEncoderDecoderStack) -> list[DenseFFN]:
    return cast("list[DenseFFN]", list(stack.encoder_ffns))


def _decoder_ffns(stack: CausalEncoderDecoderStack) -> list[DenseFFN]:
    return cast("list[DenseFFN]", list(stack.decoder_ffns))


def _inner(ffn: DenseFFN) -> SwiGLU:
    """The wrapped SwiGLU (cast for the type checker)."""
    return cast("SwiGLU", ffn.ffn)


# 1. factory + density: one SwiGLU, never a mixture of experts -----------------
def test_build_ffn_returns_swiglu_and_rejects_unknown_kind():
    assert type(build_ffn(tiny_cfg(), "swiglu")) is SwiGLU
    with pytest.raises(ValueError, match="unknown FFN kind"):
        build_ffn(tiny_cfg(), "bogus")


def test_dense_ffn_holds_exactly_one_swiglu_and_no_experts():
    ffn = DenseFFN(tiny_cfg())
    assert ffn.kind == "swiglu"
    assert type(ffn.ffn) is SwiGLU
    assert not hasattr(ffn, "router")
    assert not any("expert" in name for name, _ in ffn.named_modules())
    assert _inner(ffn).gate_proj.out_features == tiny_cfg().intermediate_size
    assert _inner(ffn).down_proj.out_features == tiny_cfg().hidden_size


def test_each_encoder_layer_carries_a_dense_swiglu_ffn():
    stack = CausalEncoderDecoderStack(_ced_cfg()).eval()
    assert len(stack.encoder_ffns) == len(stack.encoder_layers) == 2
    assert all(isinstance(ffn, DenseFFN) and type(ffn.ffn) is SwiGLU for ffn in _encoder_ffns(stack))


def test_each_decoder_layer_carries_a_dense_swiglu_ffn():
    stack = CausalEncoderDecoderStack(_ced_cfg()).eval()
    assert len(stack.decoder_ffns) == len(stack.decoder_layers) == 3
    assert all(isinstance(ffn, DenseFFN) and type(ffn.ffn) is SwiGLU for ffn in _decoder_ffns(stack))


# 2. encoder and decoder FFNs are configured independently ---------------------
def test_encoder_and_decoder_ffn_config_is_independent():
    cfg = _ced_cfg(ced_encoder_ffn_intermediate_size=64, ced_decoder_ffn_intermediate_size=128)
    stack = CausalEncoderDecoderStack(cfg).eval()
    assert {_inner(ffn).intermediate_size for ffn in _encoder_ffns(stack)} == {64}
    assert {_inner(ffn).intermediate_size for ffn in _decoder_ffns(stack)} == {128}


def test_ffn_defaults_to_swiglu_on_both_halves():
    cfg = _ced_cfg()
    assert cfg.ced_encoder_ffn == "swiglu"
    assert cfg.ced_decoder_ffn == "swiglu"
    stack = CausalEncoderDecoderStack(cfg).eval()
    assert {ffn.kind for ffn in _encoder_ffns(stack)} == {"swiglu"}
    assert {ffn.kind for ffn in _decoder_ffns(stack)} == {"swiglu"}


def test_config_rejects_bad_ffn_fields():
    with pytest.raises(ValueError, match="ced_encoder_ffn"):
        replace(_ced_cfg(), ced_encoder_ffn="bogus")
    with pytest.raises(ValueError, match="ced_decoder_ffn"):
        replace(_ced_cfg(), ced_decoder_ffn="bogus")
    with pytest.raises(ValueError, match="ced_encoder_ffn_intermediate_size"):
        replace(_ced_cfg(), ced_encoder_ffn_intermediate_size=-2)
    with pytest.raises(ValueError, match="ced_decoder_ffn_intermediate_size"):
        replace(_ced_cfg(), ced_decoder_ffn_intermediate_size=-2)


# 3. residual behaviour --------------------------------------------------------
def test_dense_ffn_is_an_identity_residual_when_down_proj_is_zero():
    cfg = replace(tiny_cfg(), residual_init_scale=False)  # zero-start down_proj
    ffn = DenseFFN(cfg).eval()
    torch.testing.assert_close(_inner(ffn).down_proj.weight, torch.zeros_like(_inner(ffn).down_proj.weight))
    x = torch.randn(2, 5, H)
    torch.testing.assert_close(ffn(x), x)


def test_dense_ffn_reset_restores_the_configured_start():
    cfg = replace(tiny_cfg(), residual_init_scale=False)
    ffn = DenseFFN(cfg)
    with torch.no_grad():
        _inner(ffn).down_proj.weight.normal_(std=1.0)
    ffn.reset_parameters()
    torch.testing.assert_close(_inner(ffn).down_proj.weight, torch.zeros_like(_inner(ffn).down_proj.weight))


# 4. pointwise / locality ------------------------------------------------------
def test_dense_ffn_is_pointwise_across_positions():
    torch.manual_seed(0)
    ffn = DenseFFN(tiny_cfg()).eval()
    with torch.no_grad():
        _inner(ffn).down_proj.weight.normal_(std=0.5)
    x = torch.randn(1, 6, H)
    x2 = x.clone()
    x2[0, 2] += 5.0
    out, out2 = ffn(x), ffn(x2)
    assert (out[0, :2] - out2[0, :2]).abs().max().item() == 0.0
    assert (out[0, 3:] - out2[0, 3:]).abs().max().item() == 0.0
    assert (out[0, 2] - out2[0, 2]).abs().max().item() > 1e-6


def test_dense_ffn_acts_on_the_trailing_hidden_axis_of_streams():
    torch.manual_seed(0)
    ffn = DenseFFN(tiny_cfg()).eval()
    with torch.no_grad():
        _inner(ffn).down_proj.weight.normal_(std=0.5)
    x = torch.randn(2, 4, H)
    streams = to_streams(x, N)
    out = ffn(streams)
    assert tuple(out.shape) == (2, 4, N, H)
    for s in range(N):
        torch.testing.assert_close(out[:, :, s], ffn(x))


# 5. end-to-end wiring ---------------------------------------------------------
def test_encoder_ffn_changes_the_encoded_context():
    torch.manual_seed(0)
    stack = CausalEncoderDecoderStack(_ced_cfg()).eval()
    with torch.no_grad():
        for ffn in _encoder_ffns(stack):
            _inner(ffn).down_proj.weight.normal_(std=0.5)
    enc = torch.randn(1, 8, H)
    base, _ = stack.encode(enc)
    with torch.no_grad():
        for ffn in _encoder_ffns(stack):
            _inner(ffn).down_proj.weight.zero_()
    ablated, _ = stack.encode(enc)
    assert (base - ablated).abs().max().item() > 1e-5


def test_decoder_ffn_changes_the_decoded_output():
    torch.manual_seed(0)
    stack = CausalEncoderDecoderStack(_ced_cfg()).eval()
    # CSA2 decoder outputs are depth-scaled and carry no residual, so restore
    # signal (as other decoder tests do) and amplify the FFN before comparing.
    with torch.no_grad():
        for layer in cast("list[CompressedSparseAttention2]", list(stack.decoder_layers)):
            layer.grouped_out.out_proj.weight.copy_(torch.randn_like(layer.grouped_out.out_proj.weight) * 0.5)
        for ffn in _decoder_ffns(stack):
            for w in (_inner(ffn).gate_proj.weight, _inner(ffn).up_proj.weight, _inner(ffn).down_proj.weight):
                w.normal_(std=0.5)
    enc = torch.randn(1, 6, H)
    ctxt, _ = stack.encode(enc)
    h = torch.randn(1, 6, H)
    base = stack.decode_full_context(h, ctxt)
    with torch.no_grad():
        for ffn in _decoder_ffns(stack):
            _inner(ffn).down_proj.weight.zero_()
    ablated = stack.decode_full_context(h, ctxt)
    assert (base - ablated).abs().max().item() > 1e-5


def test_plain_csa2_encoder_path_also_gets_dense_ffns():
    stack = CausalEncoderDecoderStack(replace(_ced_cfg(), ced_attention_gdn2_parallel=False)).eval()
    assert all(type(layer) is CompressedSparseAttention2 for layer in stack.encoder_layers)
    assert len(stack.encoder_ffns) == len(stack.encoder_layers) == 2
    with torch.no_grad():
        for ffn in _encoder_ffns(stack):
            _inner(ffn).down_proj.weight.normal_(std=0.5)
    out, shared = stack.encode(torch.randn(1, 8, H))
    assert tuple(out.shape) == (1, 8, H)
    assert shared.compressed_kv is not None


def test_ffns_do_not_leak_future_tokens():
    torch.manual_seed(0)
    stack = CausalEncoderDecoderStack(_ced_cfg()).eval()
    with torch.no_grad():
        for ffn in _encoder_ffns(stack) + _decoder_ffns(stack):
            _inner(ffn).down_proj.weight.normal_(std=0.5)
    x = torch.randn(1, 10, H)
    enc = torch.randn(1, 10, H)
    ctxt, _ = stack.encode(enc)
    pos = torch.arange(10).unsqueeze(0)
    base = stack.decode_full_context(x, ctxt, pos)

    def perturb(j: int) -> torch.Tensor:
        x2 = x.clone()
        x2[0, j] += 10.0
        out = stack.decode_full_context(x2, ctxt, pos)
        return (out - base).abs().amax(-1)[0]

    for future in (7, 8, 9):
        assert perturb(future)[0].item() == 0.0, f"future token {future} leaked"


def test_parallel_encoder_path_builds_hybrid_blocks():
    stack = CausalEncoderDecoderStack(_ced_cfg()).eval()
    assert all(isinstance(layer, ParallelAttentionGDN2) for layer in stack.encoder_layers)
    # Layer 0 is plain SWA, so the first CSA2 layer seeds the store.
    assert [layer.mode for layer in stack.encoder_layers] == [None, "full"]


def test_encoder_ffn_intermediate_size_zero_uses_the_shared_width():
    cfg = _ced_cfg(ced_encoder_ffn_intermediate_size=0)
    stack = CausalEncoderDecoderStack(cfg).eval()
    assert {_inner(ffn).intermediate_size for ffn in _encoder_ffns(stack)} == {cfg.intermediate_size}
