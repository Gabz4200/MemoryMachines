"""Asymmetric CED dimensions: wide encoder, narrow looped decoder.

The encoder keeps ``config.hidden_size`` while the decoder half runs at
``config.ced_decoder_hidden_size``. The decoder's global-KV projection doubles
as the cross-attention downsample (``W_K``/``W_V``: encoder width -> decoder
head space in one matmul), and :class:`VocabularyUpProjection` restores the
encoder width before the vocabulary head (``W_up``: decoder -> encoder width).
"""

from dataclasses import replace
from typing import cast

import pytest
import torch

from memory_machines.models.backbone import BLTBackbone, CheapLLM
from memory_machines.models.blocks import (
    CausalDecoder,
    CausalEncoderDecoderStack,
    CEDGlobalKVProjection,
    CompressedSparseAttention2,
    DenseFFN,
    VocabularyUpProjection,
)
from memory_machines.models.io import ModelInput
from memory_machines.utils.config_schema import MemoryMachineConfig
from tests.test_blocks_shapes import tiny_cfg

ENC = 32
DEC = 16
HEADS = 4


def _asym_cfg(**kw) -> MemoryMachineConfig:
    return replace(
        tiny_cfg(),
        swa_bounded_replay=False,
        ced_num_encoder_layers=1,
        ced_num_decoder_layers=2,
        csa_encoder_compression_ratio=2,
        csa_decoder_compression_ratio=1,
        ced_decoder_hidden_size=DEC,
        **kw,
    )


# 1. config invariant ----------------------------------------------------------
def test_config_rejects_bad_decoder_hidden_size():
    with pytest.raises(ValueError, match="ced_decoder_hidden_size"):
        replace(tiny_cfg(), ced_decoder_hidden_size=-8)
    with pytest.raises(ValueError, match="ced_decoder_hidden_size"):
        replace(tiny_cfg(), ced_decoder_hidden_size=DEC + 1)  # not % num_attention_heads


def test_zero_decoder_hidden_size_is_symmetric():
    cfg = replace(tiny_cfg(), ced_decoder_hidden_size=0)
    dec = CausalDecoder(cfg).eval()
    layer = cast("CompressedSparseAttention2", dec.layers[0])
    assert layer.q_up_main.out_features == cfg.hidden_size


# 2. the decoder half runs narrow ---------------------------------------------
def test_asymmetric_decoder_layers_run_at_the_narrow_width():
    stack = CausalEncoderDecoderStack(_asym_cfg()).eval()
    layer = cast("CompressedSparseAttention2", stack.decoder_layers[0])
    assert layer.q_down.in_features == DEC
    assert layer.q_up_main.out_features == DEC
    assert layer.grouped_out.out_proj.out_features == DEC
    ffn = cast("DenseFFN", stack.decoder_ffns[0])
    assert ffn.ffn.gate_proj.in_features == DEC  # type: ignore[union-attr]
    # The encoder keeps the wide width.
    enc_ffn = cast("DenseFFN", stack.encoder_ffns[0])
    assert enc_ffn.ffn.gate_proj.in_features == ENC  # type: ignore[union-attr]


def test_asymmetric_gate_reads_the_narrow_width():
    dec = CausalDecoder(_asym_cfg(ced_decoder_looping=True)).eval()
    gate = dec.gate
    assert gate is not None
    assert gate.proj.in_features == DEC
    assert gate(torch.randn(1, 4, DEC)).shape == (1, 4)


# 3. the global-KV projection is the cross-attention downsample ----------------
def test_global_kv_projection_downsamples_encoder_states():
    proj = CEDGlobalKVProjection(_asym_cfg(), source_size=ENC, decoder_hidden_size=DEC).eval()
    kv, ends = proj(torch.randn(1, 6, ENC))
    assert tuple(kv.shape) == (1, 6, tiny_cfg().num_key_value_heads, DEC // HEADS)
    assert ends.tolist() == [0, 1, 2, 3, 4, 5]


def test_asymmetric_decode_maps_wide_context_into_narrow_output():
    stack = CausalEncoderDecoderStack(_asym_cfg()).eval()
    out = stack.decode_full_context(torch.randn(1, 5, DEC), torch.randn(1, 6, ENC), exit_threshold=1.0)
    assert tuple(out.shape) == (1, 5, DEC)


def test_asymmetric_decoder_is_conditioned_on_the_wide_encoder():
    dec = CausalDecoder(_asym_cfg()).eval()
    with torch.no_grad():
        for layer in cast("list[CompressedSparseAttention2]", list(dec.layers)):
            torch.nn.init.trunc_normal_(layer.grouped_out.out_proj.weight, 0.0, 0.5, -1.0, 1.0)
    h = torch.randn(1, 5, DEC)
    out_a = dec.decode_full_context(h, torch.randn(1, 6, ENC))
    out_b = dec.decode_full_context(h, torch.randn(1, 6, ENC))
    assert (out_a - out_b).abs().max().item() > 1e-4


# 4. vocabulary up-projection (avoid the softmax bottleneck) -------------------
def test_vocab_up_projection_restores_the_encoder_width():
    up = VocabularyUpProjection(_asym_cfg()).eval()
    x = torch.randn(2, 5, DEC)
    assert tuple(up(x).shape) == (2, 5, ENC)
    assert up.proj.in_features == DEC and up.proj.out_features == ENC


def test_vocab_up_projection_is_square_when_widths_match():
    up = VocabularyUpProjection(replace(tiny_cfg(), ced_decoder_hidden_size=0))
    assert up.proj.in_features == up.proj.out_features == tiny_cfg().hidden_size


def test_asymmetric_looped_decoder_runs_end_to_end():
    stack = CausalEncoderDecoderStack(_asym_cfg(ced_decoder_looping=True, ced_decoder_loop_steps=3)).eval()
    assert len(stack.decoder_layers) == 1
    out = stack.decode_full_context(torch.randn(1, 5, DEC), torch.randn(1, 6, ENC), exit_threshold=1.0)
    assert tuple(out.shape) == (1, 5, DEC)
    assert tuple(VocabularyUpProjection(_asym_cfg())(out).shape) == (1, 5, ENC)


# 5. backbones downsample into the narrow decoder -------------------------------
def test_cheapllm_runs_with_a_narrow_decoder():
    model = CheapLLM(_asym_cfg(vocab_size=32)).eval()
    out = model(ModelInput(input_ids=torch.randint(0, 32, (1, 5))))
    assert tuple(out.logits.shape) == (1, 5, 32)


def test_cheapllm_masked_generation_runs_with_a_narrow_decoder():
    model = CheapLLM(_asym_cfg(vocab_size=32, ced_masked_query=True, ced_masked_token_count=4)).eval()
    out = model(ModelInput(input_ids=torch.randint(0, 32, (1, 5)), num_masked_tokens=2))
    assert tuple(out.logits.shape) == (1, 2, 32)


def test_blt_backbone_runs_with_a_narrow_decoder():
    cfg = _asym_cfg(vocab_size=32, blt_local_layers=1, blt_local_hidden_size=16, blt_local_window_size=8, blt_max_patch_bytes=2)
    model = BLTBackbone(cfg).eval()
    out = model(ModelInput(input_ids=torch.randint(0, 32, (1, 6))))
    assert tuple(out.logits.shape) == (1, 6, 32)


def test_blt_masked_generation_runs_with_a_narrow_decoder():
    cfg = _asym_cfg(vocab_size=32, blt_local_layers=1, blt_local_hidden_size=16, blt_local_window_size=8, blt_max_patch_bytes=2, ced_masked_query=True, ced_masked_token_count=4)
    model = BLTBackbone(cfg).eval()
    out = model(ModelInput(input_ids=torch.randint(0, 32, (1, 6)), num_masked_tokens=1))
    assert tuple(out.logits.shape) == (1, 2, 32)
