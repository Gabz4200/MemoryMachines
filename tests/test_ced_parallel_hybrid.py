"""Parallel attention ∥ memory mixer CED encoder layers.

Covers ``ced_attention_gdn2_parallel``: the pre-norm + concat + ``2d -> d``
down-projection fusion in :class:`ParallelAttentionGDN2`, its MultiHead
attention variants, the GDN-2 / Titans alternate branch, and integration into
the CED *encoder* (the decoder stays plain CSA2).
"""

from dataclasses import replace
from typing import cast

import pytest
import torch
from torch import nn

from memory_machines.models.blocks import (
    CausalDecoder,
    CausalEncoderDecoderStack,
    CompressedSparseAttention2,
    DenseFFN,
    GatedDeltaNet2Mixer,
    ParallelAttentionGDN2,
    ParallelAttentionTitan,
    SelfReferentialTitanMixer,
)
from memory_machines.models.blocks.attention import GroupedQueryAttention
from memory_machines.models.blocks.dyt import DyT
from memory_machines.models.blocks.shared import CSA2SharedState
from memory_machines.utils.config_schema import MemoryMachineConfig
from tests.test_blocks_shapes import tiny_cfg

H = 32


def _cfg(**kw) -> MemoryMachineConfig:
    return replace(tiny_cfg(), **kw)


def _ced_cfg(**kw) -> MemoryMachineConfig:
    return replace(
        tiny_cfg(),
        swa_bounded_replay=True,
        window_size=4,
        ced_num_encoder_layers=2,
        ced_num_decoder_layers=3,
        csa_encoder_compression_ratio=2,
        csa_decoder_compression_ratio=1,
        **kw,
    )


# 1. config defaults -----------------------------------------------------------
def test_config_defaults_enable_parallel_blocks_and_dyt_norms() -> None:
    cfg = MemoryMachineConfig()
    assert cfg.ced_attention_gdn2_parallel is True
    assert cfg.ced_encoder_first_window_size == 0
    assert cfg.ced_encoder_norm == "dyT"
    assert cfg.ced_encoder_ffn == "swiglu"
    assert cfg.ced_encoder_ffn_norm == "dyT"
    assert cfg.ced_decoder_ffn == "swiglu"
    assert cfg.ced_decoder_ffn_norm == "dyT"


def test_config_rejects_unknown_norm_kinds() -> None:
    with pytest.raises(ValueError, match="ced_encoder_norm"):
        replace(_cfg(), ced_encoder_norm="bogus")
    with pytest.raises(ValueError, match="ced_encoder_ffn_norm"):
        replace(_cfg(), ced_encoder_ffn_norm="bogus")
    with pytest.raises(ValueError, match="ced_decoder_ffn_norm"):
        replace(_cfg(), ced_decoder_ffn_norm="bogus")


def test_config_rejects_negative_first_encoder_window() -> None:
    with pytest.raises(ValueError, match="ced_encoder_first_window_size"):
        replace(_cfg(), ced_encoder_first_window_size=-1)


# 2. block wiring --------------------------------------------------------------
def test_block_wires_norm_attention_memory_and_fusion() -> None:
    block = ParallelAttentionGDN2(_cfg(), mode="full")
    assert isinstance(block.norm, DyT)
    assert isinstance(block.attention, CompressedSparseAttention2)
    assert isinstance(block.gdn2, GatedDeltaNet2Mixer)
    assert block.out_proj.in_features == 2 * H
    assert block.out_proj.out_features == H
    assert block.out_proj.bias is None
    assert block.mode == "full"


def test_titan_block_swaps_only_the_memory_branch() -> None:
    block = ParallelAttentionTitan(_cfg())
    assert isinstance(block, ParallelAttentionGDN2)
    assert isinstance(block.gdn2, SelfReferentialTitanMixer)
    assert isinstance(block.attention, CompressedSparseAttention2)


def test_block_forward_shared_shapes_and_shared_state() -> None:
    torch.manual_seed(0)
    block = ParallelAttentionGDN2(_cfg(), mode="full").eval()
    out, shared = block.forward_shared(torch.randn(1, 6, H), use_cache=True)
    assert tuple(out.shape) == (1, 6, H)
    assert isinstance(shared, CSA2SharedState)
    assert shared.compressed_kv is not None
    assert tuple(block(torch.randn(1, 6, H)).shape) == (1, 6, H)


def test_block_rejects_bad_input_width() -> None:
    block = ParallelAttentionGDN2(_cfg())
    with pytest.raises(ValueError, match="hidden_states must be"):
        block.forward_shared(torch.randn(1, 6, H + 1))


def test_block_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        ParallelAttentionGDN2(_cfg(), mode="bogus")


# 3. fusion math ---------------------------------------------------------------
def test_output_is_residual_plus_concat_down_projection() -> None:
    torch.manual_seed(0)
    block = ParallelAttentionGDN2(_cfg(), mode="full").eval()
    x = torch.randn(1, 6, H)
    out, _ = block.forward_shared(x, CSA2SharedState(total_len=0))
    with torch.no_grad():
        h = block.norm(x)
        attn_out, _ = block.attention.forward_shared(h, CSA2SharedState(total_len=0))
        memory_out, _, _ = block.gdn2(h)
        expected = x + block.out_proj(torch.cat((attn_out, memory_out), dim=-1))
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_both_branches_read_the_same_pre_normed_input() -> None:
    torch.manual_seed(0)
    block = ParallelAttentionGDN2(_cfg()).eval()
    x = torch.randn(1, 5, H)
    seen: dict[str, torch.Tensor] = {}
    real_attention = block.attention.forward_shared
    real_memory = block.gdn2.forward

    def spy_attention(hidden_states, *args, **kwargs):
        seen["attention"] = hidden_states.detach().clone()
        return real_attention(hidden_states, *args, **kwargs)

    def spy_memory(hidden_states, *args, **kwargs):
        seen["memory"] = hidden_states.detach().clone()
        return real_memory(hidden_states, *args, **kwargs)

    block.attention.forward_shared = spy_attention  # type: ignore[method-assign]
    block.gdn2.forward = spy_memory  # type: ignore[method-assign]
    with torch.no_grad():
        block.forward_shared(x)
        expected = block.norm(x)
    torch.testing.assert_close(seen["attention"], expected)
    torch.testing.assert_close(seen["memory"], expected)


def test_zero_fusion_projection_is_exact_residual() -> None:
    block = ParallelAttentionGDN2(_cfg()).eval()
    with torch.no_grad():
        block.out_proj.weight.zero_()
    x = torch.randn(1, 4, H)
    with torch.no_grad():
        out, _ = block.forward_shared(x)
    torch.testing.assert_close(out, x)


def test_fusion_init_starts_attention_only() -> None:
    torch.manual_seed(0)
    block = ParallelAttentionGDN2(_cfg())
    weight = block.out_proj.weight
    attention_half = weight[:, :H]
    memory_half = weight[:, H:]
    assert attention_half.abs().max().item() > 1e-3
    assert memory_half.abs().max().item() < 5e-3
    assert memory_half.std().item() < attention_half.std().item() / 5


def test_reset_parameters_reapplies_the_split_init() -> None:
    torch.manual_seed(0)
    block = ParallelAttentionGDN2(_cfg())
    with torch.no_grad():
        block.out_proj.weight.fill_(1.0)
    block.reset_parameters()
    assert block.out_proj.weight[:, H:].abs().max().item() < 5e-3


# 4. pre-norm configuration ----------------------------------------------------
@pytest.mark.parametrize("kind,cls", [("dyT", DyT), ("rmsnorm", nn.RMSNorm), ("layernorm", nn.LayerNorm)])
def test_encoder_norm_kind_is_configurable(kind: str, cls: type[nn.Module]) -> None:
    block = ParallelAttentionGDN2(replace(_cfg(), ced_encoder_norm=kind))
    assert isinstance(block.norm, cls)


# 5. attention variant / per-type SWA window ----------------------------------
_VARIANT_WINDOW_ATTR = {
    "dense": "window_size",
    "compressed": "window_size_compressed",
    "compressed_sparse": "window_size_sparse",
    "heavily_compressed": "window_size_shared",
}


def _attention_branch(layer: nn.Module):
    """Unwrap the shared-state adapter to reach the attention variant."""
    attention = getattr(layer, "attention", layer)
    return getattr(attention, "inner", attention)


@pytest.mark.parametrize("variant", sorted(_VARIANT_WINDOW_ATTR))
def test_block_variant_is_multihead_with_its_type_window(variant: str) -> None:
    cfg = replace(_cfg(), ced_attention_variant=variant)
    block = ParallelAttentionGDN2(cfg)
    assert block.attention_variant == variant
    branch = _attention_branch(block)
    assert branch.num_heads == cfg.num_attention_heads
    assert branch.window_size == getattr(cfg, _VARIANT_WINDOW_ATTR[variant])


def test_block_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="ced_attention_variant"):
        ParallelAttentionGDN2(_cfg(), variant="bogus")


def test_block_explicit_window_overrides_variant_window() -> None:
    block = ParallelAttentionGDN2(_cfg(), variant="compressed_sparse", window_size=3)
    assert block.attention.window_size == 3


@pytest.mark.parametrize("variant", sorted(_VARIANT_WINDOW_ATTR))
def test_variants_run_forward_shared(variant: str) -> None:
    torch.manual_seed(0)
    block = ParallelAttentionGDN2(replace(_cfg(), ced_attention_variant=variant)).eval()
    out, _ = block.forward_shared(torch.randn(1, 6, H), use_cache=True)
    assert tuple(out.shape) == (1, 6, H)


# 6. CED integration -----------------------------------------------------------
def test_ced_default_builds_parallel_encoder_and_plain_decoder_with_mode_cycle() -> None:
    stack = CausalEncoderDecoderStack(_ced_cfg()).eval()
    assert all(isinstance(layer, ParallelAttentionGDN2) for layer in stack.encoder_layers)
    assert all(type(layer) is CompressedSparseAttention2 for layer in stack.decoder_layers)
    # Layer 0 is plain SWA (no cross-layer store), so the first CSA2 layer seeds it.
    assert [layer.mode for layer in stack.encoder_layers] == [None, "full"]
    assert [layer.mode for layer in stack.decoder_layers] == ["full", "reindex", "reuse"]


def test_ced_flag_off_builds_plain_csa2_layers() -> None:
    stack = CausalEncoderDecoderStack(replace(_ced_cfg(), ced_attention_gdn2_parallel=False)).eval()
    assert all(type(layer) is CompressedSparseAttention2 for layer in stack.encoder_layers)
    assert all(type(layer) is CompressedSparseAttention2 for layer in stack.decoder_layers)


def test_ced_encoder_memory_branch_contributes_to_the_encoded_context() -> None:
    torch.manual_seed(0)
    stack = CausalEncoderDecoderStack(_ced_cfg()).eval()
    enc = torch.randn(1, 8, H)
    base, _ = stack.encode(enc)
    with torch.no_grad():
        for layer in cast("list[ParallelAttentionGDN2]", list(stack.encoder_layers)):
            layer.gdn2.layer.o_proj.weight.zero_()
        zeroed, _ = stack.encode(enc)
    assert (base - zeroed).abs().max().item() > 1e-5


def test_ced_stack_and_standalone_decoder_agree_with_blocks() -> None:
    torch.manual_seed(0)
    cfg = _ced_cfg()
    stack = CausalEncoderDecoderStack(cfg).eval()
    dec = CausalDecoder(cfg, layers=list(stack.decoder_layers), global_projs=list(stack.global_projs), ffns=list(stack.decoder_ffns)).eval()
    L = 8
    torch.manual_seed(1)
    h = torch.randn(1, L, H)
    e = torch.randn(1, L, H)
    pos = torch.arange(L).unsqueeze(0)
    torch.testing.assert_close(stack.prefill(h, e, pos), dec.prefill(h, e, pos), atol=1e-5, rtol=1e-5)


def test_ced_parallel_blocks_do_not_leak_future_tokens() -> None:
    torch.manual_seed(0)
    stack = CausalEncoderDecoderStack(_ced_cfg()).eval()
    L = 10
    torch.manual_seed(1)
    h = torch.randn(1, L, H)
    pos = torch.arange(L).unsqueeze(0)
    enc = torch.randn(1, L, H)
    x, _ = stack.encode(enc)
    base = stack.decode_full_context(h, x, pos)

    def perturb(j: int) -> torch.Tensor:
        h2 = h.clone()
        h2[0, j] += 10.0
        return (stack.decode_full_context(h2, x, pos) - base).abs().amax(-1)[0]

    for future in (7, 8, 9):
        # Every position before the perturbed token must be untouched.
        assert perturb(future)[:future].abs().max().item() == 0.0, f"future token {future} leaked"


def test_ced_variant_configures_attention_branch_window() -> None:
    cfg = replace(_ced_cfg(), ced_attention_variant="compressed")
    stack = CausalEncoderDecoderStack(cfg).eval()
    x, _ = stack.encode(torch.randn(1, 8, H))
    assert tuple(x.shape) == (1, 8, H)
    # Layer 0 is the plain-SWA exception; every later layer takes the variant.
    for layer in stack.encoder_layers[1:]:
        assert _attention_branch(layer).window_size == cfg.window_size_compressed


def test_ced_first_encoder_layer_is_plain_swa_in_parallel_with_the_memory_branch() -> None:
    cfg = _ced_cfg(ced_encoder_first_window_size=3)
    stack = CausalEncoderDecoderStack(cfg).eval()
    layer = cast("ParallelAttentionGDN2", stack.encoder_layers[0])
    branch = cast("GroupedQueryAttention", _attention_branch(layer))
    assert type(branch) is GroupedQueryAttention  # uncompressed, non-sparse
    assert branch.window_size == 3
    assert isinstance(layer.gdn2, GatedDeltaNet2Mixer)


def test_ced_first_encoder_window_defaults_to_the_plain_swa_window() -> None:
    cfg = _ced_cfg()
    stack = CausalEncoderDecoderStack(cfg).eval()
    layer = cast("ParallelAttentionGDN2", stack.encoder_layers[0])
    assert cast("GroupedQueryAttention", _attention_branch(layer)).window_size == cfg.window_size


def test_ced_first_encoder_window_is_smaller_than_the_csa2_windows() -> None:
    cfg = _ced_cfg()
    stack = CausalEncoderDecoderStack(cfg).eval()
    first = cast("GroupedQueryAttention", _attention_branch(cast("ParallelAttentionGDN2", stack.encoder_layers[0])))
    later = cast("CompressedSparseAttention2", _attention_branch(cast("ParallelAttentionGDN2", stack.encoder_layers[1])))
    assert first.window_size < later.token_window_size


def test_ced_first_layer_global_reach_comes_from_the_memory_branch() -> None:
    """A 2-token SWA window still carries token 0 to position 7 via GDN-2."""
    torch.manual_seed(0)
    stack = CausalEncoderDecoderStack(_ced_cfg(ced_encoder_first_window_size=2)).eval()
    layer = cast("ParallelAttentionGDN2", stack.encoder_layers[0])
    with torch.no_grad():
        layer.out_proj.weight[:, :H].zero_()  # keep the memory half only
        layer.out_proj.weight[:, H:].normal_(std=0.1)
    enc = torch.randn(1, 8, H)
    with torch.no_grad():
        base = layer(enc)
        enc2 = enc.clone()
        enc2[0, 0] += 5.0
        perturbed = layer(enc2)
    assert (perturbed[0, -1] - base[0, -1]).abs().max().item() > 1e-4


# 7. memory branch is global ---------------------------------------------------
def test_memory_branch_mixes_the_whole_sequence_not_a_window() -> None:
    """GDN-2 is global: the earliest token still moves the final-position output."""
    torch.manual_seed(0)
    block = ParallelAttentionGDN2(replace(_cfg(), window_size_sparse=2)).eval()
    x = torch.randn(1, 8, H)
    with torch.no_grad():
        g_base, _, _ = block.gdn2(x)
        x2 = x.clone()
        x2[0, 0] += 5.0
        g_pert, _, _ = block.gdn2(x2)
    assert (g_pert[0, -1] - g_base[0, -1]).abs().max().item() > 1e-4


def test_windowed_attention_alone_cannot_see_the_first_token() -> None:
    """With the memory half ablated, a 2-token dense window keeps token 0 out of token 7."""
    torch.manual_seed(0)
    block = ParallelAttentionGDN2(replace(_cfg(), ced_attention_variant="dense"), window_size=2).eval()
    with torch.no_grad():
        block.out_proj.weight[:, H:].zero_()  # attention half only
    x = torch.randn(1, 8, H)
    with torch.no_grad():
        base, _ = block.forward_shared(x)
        x2 = x.clone()
        x2[0, 0] += 5.0
        pert, _ = block.forward_shared(x2)
    assert (pert[0, -1] - base[0, -1]).abs().max().item() == 0.0


# 8. GDN-2 form / knobs --------------------------------------------------------
def test_gdn2_backend_config_defaults_to_recurrent() -> None:
    assert MemoryMachineConfig().ced_gdn2_backend == "naive"
    assert MemoryMachineConfig().ced_gdn2_chunk_size == 64


def test_ced_decoder_carries_no_memory_branch() -> None:
    stack = CausalEncoderDecoderStack(_ced_cfg()).eval()
    assert all(not hasattr(layer, "gdn2") for layer in stack.decoder_layers)
    assert all(isinstance(layer, ParallelAttentionGDN2) for layer in stack.encoder_layers)


@pytest.mark.parametrize("backend", ["naive", "recurrent", "chunk", "wy_fast"])
def test_ced_gdn2_form_is_configurable_at_block_level(backend: str) -> None:
    torch.manual_seed(0)
    reference = ParallelAttentionGDN2(replace(_cfg(), ced_gdn2_backend="naive", ced_gdn2_chunk_size=4))
    block = ParallelAttentionGDN2(replace(_cfg(), ced_gdn2_backend=backend, ced_gdn2_chunk_size=4))
    block.load_state_dict(reference.state_dict())
    reference.eval()
    block.eval()
    x = torch.randn(1, 9, H)
    with torch.no_grad():
        out_ref, _ = reference.forward_shared(x, CSA2SharedState(total_len=0))
        out_block, _ = block.forward_shared(x, CSA2SharedState(total_len=0))
    # Both forms mix the full sequence, so they agree on the same weights.
    torch.testing.assert_close(out_block, out_ref, atol=1e-4, rtol=1e-4)


def test_ced_encoder_gdn2_form_is_configurable_end_to_end() -> None:
    cfg = replace(_ced_cfg(), ced_gdn2_backend="chunk", ced_gdn2_chunk_size=4)
    stack = CausalEncoderDecoderStack(cfg).eval()
    x, _ = stack.encode(torch.randn(1, 8, H))
    assert tuple(x.shape) == (1, 8, H)
    assert tuple(stack.decode_full_context(torch.randn(1, 6, H), x).shape) == (1, 6, H)


def test_gdn2_knobs_default_to_mixer_defaults() -> None:
    layer = ParallelAttentionGDN2(_cfg()).gdn2.layer
    assert layer.num_v_heads == 4  # num_attention_heads (tiny cfg)
    assert layer.use_short_conv is True
    assert layer.use_qk_l2norm is True
    assert layer.allow_neg_eigval is False
    assert layer.beta_guard is False
    assert layer.use_resting_state is False
    assert layer.q_conv1d.bias is None


def test_gdn2_knobs_are_config_controlled() -> None:
    cfg = replace(
        _cfg(),
        ced_gdn2_num_v_heads=8,
        ced_gdn2_expand_v=2.0,
        ced_gdn2_use_short_conv=False,
        ced_gdn2_allow_neg_eigval=True,
        ced_gdn2_beta_guard=True,
        ced_gdn2_use_resting_state=True,
    )
    layer = ParallelAttentionGDN2(cfg).gdn2.layer
    assert layer.num_v_heads == 8
    assert layer.head_v_dim == 2 * layer.head_k_dim
    assert layer.use_short_conv is False
    assert layer.allow_neg_eigval is True
    assert layer.beta_guard is True
    assert layer.use_resting_state is True


def test_ffn_norms_are_config_controlled_on_both_halves() -> None:
    cfg = replace(_ced_cfg(), ced_encoder_ffn_norm="rmsnorm", ced_decoder_ffn_norm="layernorm")
    stack = CausalEncoderDecoderStack(cfg)
    assert all(isinstance(cast(DenseFFN, ffn).norm, nn.RMSNorm) for ffn in stack.encoder_ffns)
    assert all(isinstance(cast(DenseFFN, ffn).norm, nn.LayerNorm) for ffn in stack.decoder_ffns)


def test_ced_ffn_kind_is_config_controlled() -> None:
    stack = CausalEncoderDecoderStack(replace(_ced_cfg(), ced_encoder_ffn="classic_mlp", ced_decoder_ffn="classic_mlp"))
    assert all(cast(DenseFFN, ffn).kind == "classic_mlp" for ffn in stack.encoder_ffns)
    assert all(cast(DenseFFN, ffn).kind == "classic_mlp" for ffn in stack.decoder_ffns)


# 9. argument forwarding through the wrapper -----------------------------------
def test_forward_shared_forwards_the_seeded_store_guard() -> None:
    block = ParallelAttentionGDN2(_cfg()).eval()
    with pytest.raises(ValueError, match="seed the store"), torch.no_grad():
        block.forward_shared(torch.randn(1, 4, H), CSA2SharedState(), seeded_store=True)


def test_forward_shared_forwards_compress_from_and_start_pos() -> None:
    torch.manual_seed(0)
    block = ParallelAttentionGDN2(_cfg()).eval()
    x = torch.randn(1, 4, H)
    enc = torch.randn(1, 8, H)
    with torch.no_grad():
        out, shared = block.forward_shared(x, CSA2SharedState(), compress_from=enc, compress_start_pos=10)
    assert isinstance(shared, CSA2SharedState)
    assert shared.comp_end_pos is not None
    assert shared.comp_end_pos.tolist() == [11, 13, 15, 17]
    assert isinstance(shared.compressed_kv, torch.Tensor)
    assert shared.compressed_kv.shape[1] == 4
    assert tuple(out.shape) == (1, 4, H)


def test_forward_wrapper_forwards_kwargs_and_drops_the_shared_state() -> None:
    torch.manual_seed(0)
    block = ParallelAttentionGDN2(_cfg()).eval()
    x = torch.randn(1, 6, H)
    pos = torch.arange(6).unsqueeze(0)
    shared = CSA2SharedState(total_len=0)
    calls: list[tuple[object, dict[str, object]]] = []
    original = block.forward_shared

    def spy(hidden_states, passed_shared=None, **kwargs):
        calls.append((passed_shared, kwargs))
        return original(hidden_states, passed_shared, **kwargs)

    block.forward_shared = spy  # type: ignore[method-assign]
    with torch.no_grad():
        out = block(x, shared, position_ids=pos)
    assert isinstance(out, torch.Tensor)
    assert len(calls) == 1
    passed_shared, kwargs = calls[0]
    assert passed_shared is shared
    assert kwargs["position_ids"] is pos


# 10. branch training / init / adapter delegation ------------------------------
def test_backward_reaches_both_branches_and_both_fusion_halves() -> None:
    torch.manual_seed(0)
    block = ParallelAttentionGDN2(_cfg())
    x = torch.randn(1, 6, H, requires_grad=True)
    out, _ = block.forward_shared(x)
    out.sum().backward()
    attention_grads = [p.grad for p in block.attention.parameters() if p.grad is not None]
    memory_grads = [p.grad for p in block.gdn2.parameters() if p.grad is not None]
    assert attention_grads and any(g.abs().sum().item() > 0 for g in attention_grads)
    assert memory_grads and any(g.abs().sum().item() > 0 for g in memory_grads)
    weight_grad = block.out_proj.weight.grad
    assert weight_grad is not None
    assert weight_grad[:, :H].abs().max().item() > 0
    assert weight_grad[:, H:].abs().max().item() > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_memory_init_scale_controls_the_memory_half() -> None:
    cfg = _cfg()
    block = ParallelAttentionGDN2(cfg, memory_init_scale=0.05)
    assert 0.03 < block.out_proj.weight[:, H:].std().item() < 0.08
    assert 0.003 < block.out_proj.weight[:, :H].std().item() < 0.009


def test_reset_parameters_restores_the_attention_half_to_standard_scale() -> None:
    block = ParallelAttentionGDN2(_cfg())
    with torch.no_grad():
        block.out_proj.weight.fill_(1.0)
    block.reset_parameters()
    attention_half = block.out_proj.weight[:, :H]
    assert attention_half.abs().max().item() <= 2 * block.initializer_range + 1e-6
    assert 0.003 < attention_half.std().item() < 0.009


def test_reset_parameters_restores_an_adapter_wrapped_branch() -> None:
    block = ParallelAttentionGDN2(_cfg(), variant="dense")
    inner = cast("GroupedQueryAttention", _attention_branch(block))
    with torch.no_grad():
        inner.q_proj.weight.zero_()
    block.reset_parameters()
    assert inner.q_proj.weight.abs().max().item() > 0


def test_titan_reset_parameters_restores_the_split_init() -> None:
    torch.manual_seed(0)
    block = ParallelAttentionTitan(_cfg())
    with torch.no_grad():
        block.out_proj.weight.fill_(1.0)
    block.reset_parameters()
    memory_std = block.out_proj.weight[:, H:].std().item()
    assert block.out_proj.weight[:, :H].abs().max().item() <= 2 * block.initializer_range + 1e-6
    assert 0.5e-3 < memory_std < 2e-3


# 11. adapter semantics --------------------------------------------------------
@pytest.mark.parametrize("variant", ["dense", "compressed", "heavily_compressed"])
def test_adapter_variants_pass_the_shared_state_through_unchanged(variant: str) -> None:
    block = ParallelAttentionGDN2(replace(_cfg(), ced_attention_variant=variant)).eval()
    shared = CSA2SharedState(total_len=3)
    with torch.no_grad():
        out, returned = block.forward_shared(torch.randn(1, 5, H), shared)
    assert returned is shared
    assert tuple(out.shape) == (1, 5, H)


@pytest.mark.parametrize("variant", ["dense", "compressed", "heavily_compressed"])
def test_mode_is_none_for_adapter_wrapped_variants(variant: str) -> None:
    assert ParallelAttentionGDN2(replace(_cfg(), ced_attention_variant=variant)).mode is None


@pytest.mark.parametrize("mode", ["full", "reindex", "reuse"])
def test_mode_reports_the_csa2_sharing_mode(mode: str) -> None:
    assert ParallelAttentionGDN2(_cfg(), mode=mode).mode == mode


@pytest.mark.parametrize("shape", [(1, H), (1, 1, 4, H)], ids=["2d", "4d"])
def test_block_rejects_non_3d_input(shape: tuple[int, ...]) -> None:
    block = ParallelAttentionGDN2(_cfg())
    with pytest.raises(ValueError, match="hidden_states must be"):
        block.forward_shared(torch.randn(*shape))


# 12. encoder integration boundaries -------------------------------------------
def test_first_encoder_window_does_not_shrink_later_layers() -> None:
    cfg = _ced_cfg(ced_encoder_first_window_size=3)
    stack = CausalEncoderDecoderStack(cfg).eval()
    first = cast("GroupedQueryAttention", _attention_branch(cast("ParallelAttentionGDN2", stack.encoder_layers[0])))
    later = cast("CompressedSparseAttention2", _attention_branch(cast("ParallelAttentionGDN2", stack.encoder_layers[1])))
    assert first.window_size == 3
    assert later.window_size == cfg.window_size_sparse
    assert later.token_window_size == cfg.window_size_sparse * cfg.csa_encoder_compression_ratio


def test_hybrid_encoder_without_a_csa2_layer_leaves_an_empty_store_and_still_decodes() -> None:
    torch.manual_seed(0)
    stack = CausalEncoderDecoderStack(replace(_ced_cfg(), ced_num_encoder_layers=1)).eval()
    with torch.no_grad():
        enc, shared = stack.encode(torch.randn(1, 8, H))
        out = stack.decode_full_context(torch.randn(1, 4, H), enc)
    assert shared.compressed_kv is None
    assert shared.total_len == 0
    assert tuple(out.shape) == (1, 4, H)


# 13. GDN-2 config guards ------------------------------------------------------
def test_config_rejects_invalid_gdn2_knobs() -> None:
    with pytest.raises(ValueError, match="ced_gdn2_backend"):
        replace(_cfg(), ced_gdn2_backend="bogus")
    with pytest.raises(ValueError, match="ced_gdn2_chunk_size"):
        replace(_cfg(), ced_gdn2_chunk_size=0)
    with pytest.raises(ValueError, match="ced_gdn2_expand_v"):
        replace(_cfg(), ced_gdn2_expand_v=0.0)
    with pytest.raises(ValueError, match="ced_gdn2_num_v_heads"):
        replace(_cfg(), ced_gdn2_num_v_heads=-1)
    with pytest.raises(ValueError, match="ced_gdn2_num_v_heads"):
        replace(_cfg(), ced_gdn2_num_v_heads=6)
