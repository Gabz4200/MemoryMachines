"""Alternating GDN-2 / SelfReferentialTitans encoder layers.

Covers:
- ``CausalEncoderDecoderStack`` builds alternating
  ``ParallelAttentionGDN2`` / ``ParallelAttentionTitan`` layers when
  ``ced_encoder_alternating=True`` (even -> GDN-2, odd -> Titan).
- ``ced_encoder_alternating=False`` recovers the all-GDN-2 stack.
- Output shapes are unchanged and the residual stream is threaded normally.
- Titan chunkwise / recurrent backends both run end-to-end through the stack.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch

from memory_machines.models.blocks import CompressedSparseAttention2, ParallelAttentionTitan
from memory_machines.models.blocks.attention import GroupedQueryAttention
from memory_machines.models.blocks.ced import CausalEncoderDecoderStack
from memory_machines.models.blocks.parallel_hybrid import ParallelAttentionGDN2
from memory_machines.utils.config_schema import MemoryMachineConfig

_BASE_KWARGS = {
    "hidden_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "ced_num_decoder_layers": 1,
    "csa_top_k": 4,
    "csa_num_output_groups": 4,
    "csa_encoder_compression_ratio": 2,
    "csa_decoder_compression_ratio": 2,
    "window_size": 4,
}


def _base_kwargs(num_encoder_layers: int = 4) -> dict:
    return dict(_BASE_KWARGS, ced_num_encoder_layers=num_encoder_layers)


def test_alternating_layers_default() -> None:
    cfg = MemoryMachineConfig(**_base_kwargs())
    stack = CausalEncoderDecoderStack(cfg)
    assert isinstance(stack.encoder_layers[0], ParallelAttentionGDN2)
    assert isinstance(stack.encoder_layers[1], ParallelAttentionTitan)
    assert isinstance(stack.encoder_layers[2], ParallelAttentionGDN2)
    assert isinstance(stack.encoder_layers[3], ParallelAttentionTitan)


def test_alternating_disabled_all_gdn2() -> None:
    cfg = MemoryMachineConfig(**_base_kwargs(), ced_encoder_alternating=False)
    stack = CausalEncoderDecoderStack(cfg)
    assert all(isinstance(layer, ParallelAttentionGDN2) for layer in stack.encoder_layers)
    assert not any(isinstance(layer, ParallelAttentionTitan) for layer in stack.encoder_layers)


def test_encode_alternating_runs() -> None:
    cfg = MemoryMachineConfig(**_base_kwargs())
    stack = CausalEncoderDecoderStack(cfg)
    x = torch.randn(1, 8, 32)
    out, shared = stack.encode(x)
    assert out.shape == x.shape
    assert shared is not None
    assert shared.compressed_kv is not None
    assert shared.compressed_kv.shape[1] == 4  # compression_ratio=2


@pytest.mark.parametrize(
    "backend,chunk_size",
    [
        ("naive", None),
        ("recurrent", None),
        ("chunk", 4),
        ("tnt", 4),
        ("wy_fast", 4),
    ],
    ids=["naive", "recurrent", "chunk", "tnt", "wy_fast"],
)
def test_encode_alternating_titan_backend(backend: str, chunk_size: int | None) -> None:
    """Every Titan backend runs end-to-end through the alternating encoder stack."""
    kw: dict = {"titan_backend": backend}
    if chunk_size is not None:
        kw["titan_chunk_size"] = chunk_size
    cfg = MemoryMachineConfig(**_base_kwargs(), **kw)
    stack = CausalEncoderDecoderStack(cfg)
    x = torch.randn(1, 8, 32)
    out, _ = stack.encode(x)
    assert out.shape == x.shape


def test_encode_alternating_titan_qk_projection() -> None:
    cfg = MemoryMachineConfig(**_base_kwargs(), titan_backend="chunk", titan_chunk_size=4, titan_use_qk_projection=True)
    stack = CausalEncoderDecoderStack(cfg)
    x = torch.randn(1, 8, 32)
    out, _ = stack.encode(x)
    assert out.shape == x.shape


def test_titan_is_exported_from_blocks_package() -> None:
    """ParallelAttentionTitan is part of the public memory_machines.models.blocks API."""
    from memory_machines.models import blocks

    assert blocks.ParallelAttentionTitan is ParallelAttentionTitan
    assert "ParallelAttentionTitan" in blocks.__all__


def test_full_stack_alternating_decode() -> None:
    cfg = MemoryMachineConfig(**_base_kwargs())
    stack = CausalEncoderDecoderStack(cfg)
    enc = torch.randn(1, 8, 32)
    enc_out, _ = stack.encode(enc)
    dec = stack.decode_full_context(torch.randn(1, 4, 32), enc_out)
    assert dec.shape == (1, 4, 32)


def test_single_layer_alternating() -> None:
    cfg = MemoryMachineConfig(**_base_kwargs(num_encoder_layers=1))
    stack = CausalEncoderDecoderStack(cfg)
    # Even index 0 is always GDN-2.
    assert isinstance(stack.encoder_layers[0], ParallelAttentionGDN2)
    enc = torch.randn(1, 6, 32)
    out, _ = stack.encode(enc)
    assert out.shape == enc.shape


def test_two_layers_alternating() -> None:
    cfg = MemoryMachineConfig(**_base_kwargs(num_encoder_layers=2))
    stack = CausalEncoderDecoderStack(cfg)
    assert isinstance(stack.encoder_layers[0], ParallelAttentionGDN2)
    assert isinstance(stack.encoder_layers[1], ParallelAttentionTitan)
    enc = torch.randn(1, 6, 32)
    out, _ = stack.encode(enc)
    assert out.shape == enc.shape


def test_titan_isinstance_of_gdn2_block() -> None:
    """Existing isinstance(layer, ParallelAttentionGDN2) checks still hold."""
    cfg = MemoryMachineConfig(**_base_kwargs())
    stack = CausalEncoderDecoderStack(cfg)
    titan = stack.encoder_layers[1]
    assert isinstance(titan, ParallelAttentionGDN2)
    assert isinstance(titan, ParallelAttentionTitan)


def test_gradient_flow_through_alternating_stack() -> None:
    cfg = MemoryMachineConfig(**_base_kwargs(), titan_backend="recurrent")
    stack = CausalEncoderDecoderStack(cfg)
    x = torch.randn(1, 8, 32, requires_grad=True)
    out, _ = stack.encode(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def _attention_branch(layer):
    """Unwrap the shared-state adapter to reach the attention variant."""
    attention = getattr(layer, "attention", layer)
    return getattr(attention, "inner", attention)


def test_alternating_layers_keep_the_dense_first_window_and_the_csa2_variant() -> None:
    cfg = MemoryMachineConfig(**_base_kwargs(num_encoder_layers=4), ced_encoder_first_window_size=3)
    stack = CausalEncoderDecoderStack(cfg)
    assert [layer.mode for layer in stack.encoder_layers] == [None, "full", "reuse", "reuse"]
    first = cast("GroupedQueryAttention", _attention_branch(stack.encoder_layers[0]))
    assert type(first) is GroupedQueryAttention
    assert first.window_size == 3
    titan = stack.encoder_layers[1]
    assert isinstance(titan, ParallelAttentionTitan)
    titan_branch = cast("CompressedSparseAttention2", _attention_branch(titan))
    assert isinstance(titan_branch, CompressedSparseAttention2)
    assert titan_branch.window_size == cfg.window_size_sparse
