"""Parallel attention ∥ GDN-2 CED layers routed by Single-Pass mHC.

Covers the ``ced_attention_gdn2_parallel`` feature: two-branch mHC routing in
:class:`SinglePassMHC`, the :class:`MHCAttentionGDN2` block (streams, shared
state, single-pass ``prev_a`` threading), and its default-on integration into
the CED *encoder* (the decoder stays plain CSA2).
"""

from dataclasses import replace
from typing import cast

import pytest
import torch

from memory_machines.models.blocks import (
    CausalDecoder,
    CausalEncoderDecoderStack,
    CompressedSparseAttention2,
    EngramMemory,
    GatedDeltaNet2Mixer,
    MHCAttentionGDN2,
    SinglePassMHC,
    from_streams,
    to_streams,
)
from memory_machines.models.blocks.ced import _run_stream_layers
from memory_machines.models.blocks.shared import CSA2SharedState
from memory_machines.utils.config_schema import MemoryMachineConfig
from tests.test_blocks_shapes import tiny_cfg

H = 32
N = 2


def _cfg(**kw) -> MemoryMachineConfig:
    return replace(tiny_cfg(), mhc_num_streams=N, **kw)


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


# 1. config default -----------------------------------------------------------
def test_config_default_enables_parallel_blocks():
    """The feature is opt-out: default builds parallel attention ∥ GDN-2."""
    assert MemoryMachineConfig().ced_attention_gdn2_parallel is True


# 2. stream helpers -----------------------------------------------------------
def test_to_from_streams_roundtrip_and_validation():
    x = torch.randn(2, 5, H)
    streams = to_streams(x, N)
    assert tuple(streams.shape) == (2, 5, N, H)
    torch.testing.assert_close(streams[:, :, 0], streams[:, :, 1])  # replication
    torch.testing.assert_close(from_streams(streams), x)  # mean of identical streams
    with pytest.raises(ValueError, match="must be"):
        to_streams(torch.randn(5, H), N)
    with pytest.raises(ValueError, match="must be"):
        from_streams(torch.randn(2, 5, H))


# 3. two-branch mHC coefficients ---------------------------------------------
def test_parallel_coefficients_shapes_and_constraints():
    torch.manual_seed(0)
    mhc = SinglePassMHC(_cfg(), parallel_branches=True)
    streams = torch.randn(1, 6, N, H)
    a1, a2, b, c1, c2 = mhc.parallel_coefficients(streams)
    assert tuple(a1.shape) == (1, 6, 1, N)
    assert tuple(a2.shape) == (1, 6, 1, N)
    assert tuple(b.shape) == (1, 6, N, N)
    assert tuple(c1.shape) == (1, 6, N, 1)
    assert tuple(c2.shape) == (1, 6, N, 1)
    for a in (a1, a2):
        assert bool(((a > 0) & (a < 1)).all())
    for c in (c1, c2):
        assert bool(((c > 0) & (c < 2)).all())
    # Sinkhorn residual is doubly stochastic with non-negative entries.
    ones = torch.ones(1, 6, N)
    torch.testing.assert_close(b.sum(-1), ones, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(b.sum(-2), ones, atol=1e-5, rtol=1e-5)
    assert bool((b >= 0).all())


def test_parallel_coefficients_requires_parallel_branches_flag():
    mhc = SinglePassMHC(_cfg())  # single-branch
    with pytest.raises(RuntimeError, match="parallel_branches"):
        mhc.parallel_coefficients(torch.randn(1, 4, N, H))


def test_parallel_routing_is_branch_specific():
    torch.manual_seed(0)
    mhc = SinglePassMHC(_cfg(), parallel_branches=True)
    with torch.no_grad():
        mhc.alpha_pre.fill_(1.0)
        mhc.alpha_pre2.fill_(1.0)
        mhc.a_dyn.weight.normal_(std=0.5)
        mhc.a2_dyn.weight.normal_(std=0.5)
    a1, a2, _, _, _ = mhc.parallel_coefficients(torch.randn(1, 6, N, H))
    assert not torch.allclose(a1, a2), "the two branches must learn different routings"


# 4. forward_parallel / combine_parallel -------------------------------------
def test_forward_parallel_routes_branches_differently():
    torch.manual_seed(0)
    mhc = SinglePassMHC(_cfg(), parallel_branches=True)
    with torch.no_grad():
        mhc.alpha_pre.fill_(1.0)
        mhc.alpha_pre2.fill_(1.0)
        mhc.a_dyn.weight.normal_(std=0.5)
        mhc.a2_dyn.weight.normal_(std=0.5)
    streams = torch.randn(1, 5, N, H)
    seen: dict[str, torch.Tensor] = {}

    def branch_a(x):
        seen["a"] = x.detach().clone()
        return x

    def branch_g(x):
        seen["g"] = x.detach().clone()
        return x

    out, a1, a2 = mhc.forward_parallel(streams, branch_a, branch_g)
    assert tuple(out.shape) == (1, 5, N, H)
    assert tuple(a1.shape) == (1, 5, 1, N)
    assert tuple(a2.shape) == (1, 5, 1, N)
    # The routed views differ because A^{a} != A^{g}.
    assert not torch.allclose(seen["a"], seen["g"])
    # Output is the two-branch write-in of the branch outputs.
    _, _, b, c1, c2 = mhc.parallel_coefficients(streams)
    expected = mhc.combine_parallel(streams, b, c1, seen["a"], c2, seen["g"])
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_combine_parallel_residual_term_is_identity_when_branches_are_zero():
    streams = torch.randn(1, 4, N, H)
    b = torch.eye(N).expand(1, 4, N, N).contiguous()
    zeros = torch.zeros(1, 4, H)
    ones = torch.ones(1, 4, N, 1)
    out = SinglePassMHC.combine_parallel(streams, b, ones, zeros, ones, zeros)
    torch.testing.assert_close(out, streams)


# 5. MHCAttentionGDN2 block --------------------------------------------------
def test_mhc_block_wires_attention_gdn2_and_parallel_mhc():
    block = MHCAttentionGDN2(_cfg(), mode="full")
    assert isinstance(block.attention, CompressedSparseAttention2)
    assert isinstance(block.gdn2, GatedDeltaNet2Mixer)
    assert isinstance(block.mhc, SinglePassMHC)
    assert block.mhc.parallel_branches is True
    assert block.mode == "full"


def test_mhc_block_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        MHCAttentionGDN2(_cfg(), mode="bogus")


def test_mhc_block_forward_shared_shapes_and_shared_state():
    torch.manual_seed(0)
    block = MHCAttentionGDN2(_cfg(), mode="full").eval()
    streams = to_streams(torch.randn(1, 6, H), N)
    out, shared, a = block.forward_shared(streams, use_cache=True)
    assert tuple(out.shape) == (1, 6, N, H)
    assert tuple(a.shape) == (1, 6, 2, N)  # (attention, gdn2) routing
    assert isinstance(shared, CSA2SharedState)
    assert shared.compressed_kv is not None
    # Thin streams-only wrapper.
    assert tuple(block.forward(streams).shape) == (1, 6, N, H)


def test_mhc_block_prev_a_threading_changes_the_update():
    torch.manual_seed(0)
    block = MHCAttentionGDN2(_cfg()).eval()
    streams = to_streams(torch.randn(1, 6, H), N)
    out_own, _, a = block.forward_shared(streams, use_cache=True)
    prev_a = torch.rand_like(a)
    out_prev, _, _ = block.forward_shared(streams, use_cache=True, prev_a=prev_a)
    assert not torch.allclose(out_own, out_prev)


def test_mhc_block_validates_stream_and_prev_a_shapes():
    block = MHCAttentionGDN2(_cfg())
    streams = to_streams(torch.randn(1, 6, H), N)
    with pytest.raises(ValueError, match="streams must be"):
        block.forward_shared(torch.randn(1, 6, N + 1, H))
    with pytest.raises(ValueError, match="prev_a must be"):
        block.forward_shared(streams, prev_a=torch.randn(1, 6, 3, N))


def test_mhc_block_reset_parameters_restores_neutral_routing():
    torch.manual_seed(0)
    block = MHCAttentionGDN2(_cfg())
    with torch.no_grad():
        block.mhc.a2_dyn.weight.fill_(5.0)
        block.mhc.alpha_pre2.fill_(1.0)
    block.reset_parameters()
    _, _, _, c1, _ = block.mhc.parallel_coefficients(to_streams(torch.randn(1, 4, H), N))
    assert c1.mean().item() == pytest.approx(1.0, abs=1e-3)


# 6. CED integration ---------------------------------------------------------
def test_run_stream_layers_threads_prev_a_and_collapses():
    torch.manual_seed(0)
    cfg = _cfg()
    layers = [MHCAttentionGDN2(cfg, 2, "full"), MHCAttentionGDN2(cfg, 2, "reuse")]
    x = torch.randn(1, 6, H)
    out, _ = _run_stream_layers(layers, x, CSA2SharedState(total_len=0), N)
    assert tuple(out.shape) == (1, 6, H)

    # Manual persistent-stream + single-pass prev_a execution must match.
    streams = to_streams(x, N)
    shared: CSA2SharedState | None = CSA2SharedState(total_len=0)
    prev_a = None
    for layer in layers:
        streams, shared, prev_a = layer.forward_shared(streams, shared, prev_a=prev_a, use_cache=True)
    assert prev_a is not None
    torch.testing.assert_close(out, from_streams(streams))


def test_ced_default_builds_parallel_encoder_and_plain_decoder_with_mode_cycle():
    stack = CausalEncoderDecoderStack(_ced_cfg()).eval()
    assert all(isinstance(layer, MHCAttentionGDN2) for layer in stack.encoder_layers)
    assert all(type(layer) is CompressedSparseAttention2 for layer in stack.decoder_layers)
    assert [layer.mode for layer in stack.encoder_layers] == ["full", "reuse"]
    assert [layer.mode for layer in stack.decoder_layers] == ["full", "reindex", "reuse"]


def test_ced_flag_off_builds_plain_csa2_layers():
    stack = CausalEncoderDecoderStack(replace(_ced_cfg(), ced_attention_gdn2_parallel=False)).eval()
    assert all(type(layer) is CompressedSparseAttention2 for layer in stack.encoder_layers)
    assert all(type(layer) is CompressedSparseAttention2 for layer in stack.decoder_layers)


def test_ced_encoder_gdn2_branch_contributes_to_the_encoded_context():
    torch.manual_seed(0)
    stack = CausalEncoderDecoderStack(_ced_cfg()).eval()
    enc = torch.randn(1, 8, H)
    base, _ = stack.encode(enc)
    with torch.no_grad():
        for layer in cast("list[MHCAttentionGDN2]", list(stack.encoder_layers)):
            layer.gdn2.layer.o_proj.weight.zero_()
        zeroed, _ = stack.encode(enc)
    assert (base - zeroed).abs().max().item() > 1e-5


def test_ced_stack_and_standalone_decoder_agree_with_blocks():
    torch.manual_seed(0)
    cfg = _ced_cfg()
    stack = CausalEncoderDecoderStack(cfg).eval()
    dec = CausalDecoder(cfg, layers=list(stack.decoder_layers), global_projs=list(stack.global_projs)).eval()
    L = 8
    torch.manual_seed(1)
    h = torch.randn(1, L, H)
    e = torch.randn(1, L, H)
    pos = torch.arange(L).unsqueeze(0)
    out_stack = stack.prefill(h, e, pos)
    out_dec = dec.prefill(h, e, pos)
    torch.testing.assert_close(out_stack, out_dec, atol=1e-5, rtol=1e-5)


def test_ced_parallel_blocks_do_not_leak_future_tokens():
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
        assert perturb(future)[0].item() == 0.0, f"future token {future} leaked"


# 7. attention variant / per-type SWA window ---------------------------------
_VARIANT_WINDOW_ATTR = {
    "dense": "window_size",
    "compressed": "window_size_compressed",
    "compressed_sparse": "window_size_sparse",
    "heavily_compressed": "window_size_shared",
}


def _attention_branch(layer):
    """Unwrap the shared-state adapter to reach the attention variant."""
    return getattr(layer.attention, "inner", layer.attention)


@pytest.mark.parametrize("variant", sorted(_VARIANT_WINDOW_ATTR))
def test_mhc_block_variant_is_multihead_with_its_type_window(variant):
    cfg = replace(_cfg(), ced_attention_variant=variant)
    block = MHCAttentionGDN2(cfg)
    assert block.attention_variant == variant
    branch = _attention_branch(block)
    assert branch.num_heads == cfg.num_attention_heads
    assert branch.window_size == getattr(cfg, _VARIANT_WINDOW_ATTR[variant])


def test_mhc_block_rejects_unknown_variant():
    with pytest.raises(ValueError, match="ced_attention_variant"):
        MHCAttentionGDN2(_cfg(), variant="bogus")


def test_mhc_block_explicit_window_overrides_variant_window():
    block = MHCAttentionGDN2(_cfg(), variant="compressed_sparse", window_size=3)
    assert block.attention.window_size == 3


@pytest.mark.parametrize("variant", sorted(_VARIANT_WINDOW_ATTR))
def test_mhc_block_variants_run_forward_shared(variant):
    torch.manual_seed(0)
    block = MHCAttentionGDN2(replace(_cfg(), ced_attention_variant=variant)).eval()
    streams = to_streams(torch.randn(1, 6, H), N)
    out, _, a = block.forward_shared(streams, use_cache=True)
    assert tuple(out.shape) == (1, 6, N, H)
    assert tuple(a.shape) == (1, 6, 2, N)


def test_ced_variant_configures_attention_branch_window():
    cfg = replace(_ced_cfg(), ced_attention_variant="compressed")
    stack = CausalEncoderDecoderStack(cfg).eval()
    x, _ = stack.encode(torch.randn(1, 8, H))
    assert tuple(x.shape) == (1, 8, H)
    for layer in stack.encoder_layers:
        assert _attention_branch(layer).window_size == cfg.window_size_compressed


# 8. GDN-2 is the encoder's global mixer -------------------------------------
def test_gdn2_backend_config_defaults_to_recurrent():
    assert MemoryMachineConfig().ced_gdn2_backend == "naive"
    assert MemoryMachineConfig().ced_gdn2_chunk_size == 64


def test_ced_decoder_carries_no_gdn2_branch():
    stack = CausalEncoderDecoderStack(_ced_cfg()).eval()
    assert all(not hasattr(layer, "gdn2") for layer in stack.decoder_layers)
    assert all(isinstance(layer, MHCAttentionGDN2) for layer in stack.encoder_layers)


def test_gdn2_branch_mixes_the_whole_sequence_not_a_window():
    """GDN-2 is global: the earliest token still moves the final-position output."""
    torch.manual_seed(0)
    block = MHCAttentionGDN2(replace(_cfg(), window_size_sparse=2)).eval()
    x = torch.randn(1, 8, H)
    with torch.no_grad():
        g_base, _, _ = block.gdn2(x)
        x2 = x.clone()
        x2[0, 0] += 5.0
        g_pert, _, _ = block.gdn2(x2)
    assert (g_pert[0, -1] - g_base[0, -1]).abs().max().item() > 1e-4


@pytest.mark.parametrize("backend", ["naive", "recurrent", "chunk", "wy_fast"])
def test_ced_gdn2_form_is_configurable_at_block_level(backend):
    torch.manual_seed(0)
    reference = MHCAttentionGDN2(replace(_cfg(), ced_gdn2_backend="naive", ced_gdn2_chunk_size=4))
    block = MHCAttentionGDN2(replace(_cfg(), ced_gdn2_backend=backend, ced_gdn2_chunk_size=4))
    block.load_state_dict(reference.state_dict())
    reference.eval()
    block.eval()
    streams = to_streams(torch.randn(1, 9, H), N)
    with torch.no_grad():
        out_ref, _, _ = reference.forward_shared(streams, use_cache=True)
        out_block, _, _ = block.forward_shared(streams, use_cache=True)
    # Both forms mix the full sequence, so they agree on the same weights.
    torch.testing.assert_close(out_block, out_ref, atol=1e-4, rtol=1e-4)


def test_ced_encoder_gdn2_form_is_configurable_end_to_end():
    cfg = replace(_ced_cfg(), ced_gdn2_backend="chunk", ced_gdn2_chunk_size=4)
    stack = CausalEncoderDecoderStack(cfg).eval()
    x, _ = stack.encode(torch.randn(1, 8, H))
    assert tuple(x.shape) == (1, 8, H)
    dec = stack.decode_full_context(torch.randn(1, 6, H), x)
    assert tuple(dec.shape) == (1, 6, H)


# 9. GDN-2 branch knobs are config-controlled --------------------------------
def test_gdn2_knobs_default_to_mixer_defaults():
    layer = MHCAttentionGDN2(_cfg()).gdn2.layer
    assert layer.num_v_heads == 4  # num_attention_heads (tiny cfg)
    assert layer.use_short_conv is True
    assert layer.use_qk_l2norm is True
    assert layer.allow_neg_eigval is False
    assert layer.beta_guard is False
    assert layer.use_resting_state is False
    assert layer.q_conv1d.bias is None


def test_gdn2_knobs_are_config_controlled():
    cfg = replace(
        _cfg(),
        ced_gdn2_num_v_heads=8,
        ced_gdn2_expand_v=2.0,
        ced_gdn2_use_short_conv=False,
        ced_gdn2_allow_neg_eigval=True,
        ced_gdn2_beta_guard=True,
        ced_gdn2_use_resting_state=True,
    )
    layer = MHCAttentionGDN2(cfg).gdn2.layer
    assert layer.num_v_heads == 8
    assert layer.head_v_dim == 2 * layer.head_k_dim
    assert layer.use_short_conv is False
    assert layer.allow_neg_eigval is True
    assert layer.beta_guard is True
    assert layer.use_resting_state is True


def test_gdn2_conv_bias_is_config_controlled():
    layer = MHCAttentionGDN2(replace(_cfg(), ced_gdn2_conv_bias=True)).gdn2.layer
    assert layer.q_conv1d.bias is not None


def test_engram_conv_kernel_size_is_config_controlled():
    engram = EngramMemory(replace(_cfg(), engram_use_conv=True, engram_conv_kernel_size=6))
    assert engram.conv is not None
    assert engram.conv.conv.kernel_size == (6,)


def test_config_rejects_bad_gdn2_and_engram_fields():
    with pytest.raises(ValueError, match="ced_gdn2_backend"):
        replace(_cfg(), ced_gdn2_backend="bogus")
    with pytest.raises(ValueError, match="ced_gdn2_chunk_size"):
        replace(_cfg(), ced_gdn2_chunk_size=0)
    with pytest.raises(ValueError, match="ced_gdn2_expand_v"):
        replace(_cfg(), ced_gdn2_expand_v=0.0)
    with pytest.raises(ValueError, match="ced_gdn2_num_v_heads"):
        replace(_cfg(), ced_gdn2_num_v_heads=3)
    with pytest.raises(ValueError, match="engram_conv_kernel_size"):
        replace(_cfg(), engram_conv_kernel_size=0)
