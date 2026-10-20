"""Config-driven options of the GatedDeltaNet2Mixer wrapper.

Covers constructor knobs not exercised by tests/test_gdn2_multihead_mixer.py:
explicit head_dim / num_v_heads / expand_v, conv_size + norm_eps defaults and
overrides, backend aliases and error paths, allow_neg_eigval, beta_guard,
use_resting_state, per-call chunk_size restore (incl. on error), call-level
validation, use_cache state shapes, and hidden%heads rejection.
"""

import copy

import pytest
import torch

from memory_machines.utils.config_schema import MemoryMachineConfig


def _cfg(**kw):
    base = {
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "csa_num_output_groups": 2,
    }
    base.update(kw)
    return MemoryMachineConfig(**base)


def test_head_dim_override_and_default_width():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(0)
    mixer = GatedDeltaNet2Mixer(_cfg(), head_dim=4, use_short_conv=False).eval()
    assert mixer.layer.head_k_dim == 4
    assert mixer.head_dim == 4
    assert mixer.head_dim_k == 4
    assert mixer.head_dim_v == 4

    torch.manual_seed(0)
    wide = GatedDeltaNet2Mixer(
        _cfg(hidden_size=24, num_attention_heads=3, num_key_value_heads=3, csa_num_output_groups=1),
        use_short_conv=False,
    ).eval()
    assert wide.head_dim == 8
    assert wide.head_dim_k == 8
    assert wide.head_dim_v == 8


def test_num_v_heads_gva_build_and_forward():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(0)
    mixer = GatedDeltaNet2Mixer(_cfg(), num_v_heads=4, use_short_conv=False).eval()
    assert mixer.layer.num_v_heads == 4
    torch.manual_seed(1)
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        out, weights, cache = mixer(x)
    assert tuple(out.shape) == (1, 5, 16)
    assert weights is None and cache is None


def test_num_v_heads_invalid_raises():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    with pytest.raises(ValueError, match="multiple"):
        GatedDeltaNet2Mixer(_cfg(), num_v_heads=3, use_short_conv=False)


def test_expand_v_doubles_value_dim_and_runs():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(0)
    mixer = GatedDeltaNet2Mixer(_cfg(), expand_v=2.0, use_short_conv=False).eval()
    assert mixer.head_dim_v == 16
    torch.manual_seed(1)
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        out, _, _ = mixer(x)
    assert tuple(out.shape) == (1, 5, 16)


def test_expand_v_fractional_raises():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    with pytest.raises(ValueError, match="expand_v"):
        GatedDeltaNet2Mixer(_cfg(), expand_v=1.1, use_short_conv=False)


def test_conv_size_default_and_override():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    cfg = _cfg()
    torch.manual_seed(0)
    defaulted = GatedDeltaNet2Mixer(cfg)
    assert defaulted.layer.conv_size == cfg.conv_kernel_size
    torch.manual_seed(0)
    overridden = GatedDeltaNet2Mixer(cfg, conv_size=5)
    assert overridden.layer.conv_size == 5


def test_conv_enabled_forward_and_cache_continuation():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(0)
    mixer = GatedDeltaNet2Mixer(_cfg()).eval()  # use_short_conv=True default
    torch.manual_seed(1)
    x = torch.randn(1, 7, 16)
    with torch.no_grad():
        full, _, _ = mixer(x)
        _, _, state = mixer(x[:, :4], use_cache=True)
        assert state is not None and state["conv_state"] is not None
        part, _, _ = mixer(x[:, 4:], past_key_values=state)
    torch.testing.assert_close(full[:, 4:], part, atol=1e-5, rtol=1e-5)


def test_norm_eps_default_and_override():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    cfg = _cfg()
    torch.manual_seed(0)
    defaulted = GatedDeltaNet2Mixer(cfg, use_short_conv=False)
    assert defaulted.layer.o_norm.eps == cfg.rms_norm_eps
    torch.manual_seed(0)
    overridden = GatedDeltaNet2Mixer(cfg, use_short_conv=False, norm_eps=1e-3)
    assert overridden.layer.o_norm.eps == 1e-3


def test_backend_recurrent_matches_naive():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(0)
    recurrent = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False, backend="recurrent").eval()
    naive = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False, backend="naive").eval()
    naive.layer.load_state_dict(recurrent.layer.state_dict())
    torch.manual_seed(1)
    x = torch.randn(1, 6, 16)
    with torch.no_grad():
        out_rec, _, _ = recurrent(x)
        out_naive, _, _ = naive(x)
    torch.testing.assert_close(out_rec, out_naive, atol=1e-5, rtol=1e-5)


def test_backend_wy_fast_matches_chunk():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(0)
    chunk = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False, backend="chunk", chunk_size=4).eval()
    wy = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False, backend="wy_fast", chunk_size=4).eval()
    wy.layer.load_state_dict(chunk.layer.state_dict())
    torch.manual_seed(1)
    x = torch.randn(1, 9, 16)
    with torch.no_grad():
        out_chunk, _, _ = chunk(x)
        out_wy, _, _ = wy(x)
    torch.testing.assert_close(out_chunk, out_wy, atol=1e-5, rtol=1e-5)


def test_backend_error_paths():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    with pytest.raises(NotImplementedError):
        GatedDeltaNet2Mixer(_cfg(), use_short_conv=False, backend="triton")
    with pytest.raises(ValueError, match="Unknown backend"):
        GatedDeltaNet2Mixer(_cfg(), use_short_conv=False, backend="bogus")


def test_allow_neg_eigval_changes_output():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(0)
    neg = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False, allow_neg_eigval=True).eval()
    torch.manual_seed(0)
    pos = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False, allow_neg_eigval=False).eval()
    assert neg.layer.allow_neg_eigval is True
    torch.manual_seed(1)
    x = torch.randn(1, 6, 16)
    with torch.no_grad():
        out_neg, _, _ = neg(x)
        out_pos, _, _ = pos(x)
    assert (out_neg - out_pos).abs().amax().item() > 1e-6


def test_beta_guard_propagates_and_runs():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(0)
    mixer = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False, beta_guard=True).eval()
    assert mixer.layer.beta_guard is True
    torch.manual_seed(1)
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        out, _, _ = mixer(x)
    assert tuple(out.shape) == (1, 5, 16)


def test_beta_guard_without_l2norm_raises_on_forward():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(0)
    mixer = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False, beta_guard=True, use_qk_l2norm=False).eval()
    with pytest.raises(ValueError, match="beta_guard"):
        mixer(torch.randn(1, 4, 16))


def test_use_resting_state_param_and_effect():
    from torch import nn

    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(0)
    with_rest = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False, use_resting_state=True).eval()
    assert isinstance(with_rest.layer.resting_state, nn.Parameter)
    assert tuple(with_rest.layer.resting_state.shape) == (2, 8, 8)
    torch.manual_seed(0)
    without = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False, use_resting_state=False).eval()
    assert without.layer.resting_state is None
    torch.manual_seed(1)
    x = torch.randn(1, 6, 16)
    with torch.no_grad():
        out_rest, _, _ = with_rest(x)
        out_plain, _, _ = without(x)
    # Resting prior starts at zero so outputs match at init; nudge it to prove wiring.
    torch.testing.assert_close(out_rest, out_plain, atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        with_rest.layer.resting_state.fill_(0.5)
        out_nudged, _, _ = with_rest(x)
    assert (out_nudged - out_plain).abs().amax().item() > 1e-6


def test_per_call_chunk_size_override_restores():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(0)
    mixer = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False, backend="chunk", chunk_size=2).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 6, 16)
    with torch.no_grad():
        mixer(x, chunk_size=6)
    assert mixer.layer.chunk_size == 2
    assert mixer.chunk_size == 2


def test_chunk_size_restored_when_call_raises():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(0)
    mixer = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False, backend="chunk", chunk_size=2).eval()
    bad_x = torch.randn(2, 4, 16)
    with pytest.raises(ValueError, match="batch 1"):
        mixer(bad_x, cu_seqlens=torch.tensor([0, 4, 8]), chunk_size=6)
    assert mixer.layer.chunk_size == 2


def test_call_validation_errors():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(0)
    mixer = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False).eval()
    with pytest.raises(ValueError, match=r"\[B, T, D="):
        mixer(torch.randn(1, 4))
    torch.manual_seed(1)
    x = torch.randn(1, 4, 16)
    with torch.no_grad():
        _, _, state = mixer(x, use_cache=True)
    with pytest.raises(ValueError, match="past_key_values"):
        mixer(x, attention_mask=torch.ones(1, 4), past_key_values=state)


def test_use_cache_state_shape():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(0)
    mixer = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        _, _, cache = mixer(x, use_cache=True)
    assert cache is not None
    assert tuple(cache["recurrent_state"].shape) == (1, 2, 8, 8)


def test_hidden_not_divisible_by_heads_raises():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    bad = copy.copy(_cfg())
    bad.num_attention_heads = 3
    with pytest.raises(ValueError, match="divisible"):
        GatedDeltaNet2Mixer(bad, use_short_conv=False)


def test_liv_conv_is_default_and_per_stream():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer, LivConv1d

    layer = GatedDeltaNet2Mixer(_cfg()).layer
    convs = (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d)
    assert all(isinstance(c, LivConv1d) for c in convs)
    assert len({id(c) for c in convs}) == 3  # unique weights per stream
    assert convs[0].hidden_size == layer.key_dim
    assert convs[2].hidden_size == layer.value_dim
    assert convs[0].kernel_size == (layer.conv_size,)


def test_liv_conv_rejects_non_positive_dims():
    from memory_machines.models.blocks.gdn2 import LivConv1d

    with pytest.raises(ValueError, match="hidden_size/kernel_size"):
        LivConv1d(hidden_size=0, kernel_size=3)
    with pytest.raises(ValueError, match="hidden_size/kernel_size"):
        LivConv1d(hidden_size=4, kernel_size=0)


def test_plain_short_conv_when_liv_disabled():
    from memory_machines.models.blocks.causal_conv1d import ShortConvolution
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    mixer = GatedDeltaNet2Mixer(_cfg(), liv_conv=False).eval()
    layer = mixer.layer
    assert isinstance(layer.q_conv1d, ShortConvolution)
    assert isinstance(layer.v_conv1d, ShortConvolution)
    with torch.no_grad():
        out, _, _ = mixer(torch.randn(1, 4, 16))
    assert tuple(out.shape) == (1, 4, 16)


def test_liv_conv_starts_as_identity():
    from memory_machines.models.blocks.gdn2 import LivConv1d

    conv = LivConv1d(hidden_size=6, kernel_size=3).eval()
    x = torch.randn(2, 5, 6)
    with torch.no_grad():
        y, _ = conv(x)
    torch.testing.assert_close(y, x)


def _randomize_liv_conv(conv) -> None:
    with torch.no_grad():
        for p in (conv.input_projection.weight, conv.input_projection.bias, conv.output_projection.weight, conv.conv.weight):
            torch.nn.init.normal_(p, std=0.3)


def test_liv_conv_cache_decode_matches_full_forward():
    from memory_machines.models.blocks.gdn2 import LivConv1d

    torch.manual_seed(3)
    conv = LivConv1d(hidden_size=6, kernel_size=3).double().eval()
    _randomize_liv_conv(conv)
    x = torch.randn(1, 7, 6, dtype=torch.float64)
    with torch.no_grad():
        full, _ = conv(x)
        head, state = conv(x[:, :4], output_final_state=True)
        tail, _ = conv(x[:, 4:], cache=state, output_final_state=True)
    torch.testing.assert_close(torch.cat([head, tail], dim=1), full, atol=1e-10, rtol=1e-10)


def test_reset_parameters_restores_liv_conv_identity():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(4)
    mixer = GatedDeltaNet2Mixer(_cfg()).eval()
    layer = mixer.layer
    for conv in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d):
        _randomize_liv_conv(conv)
    mixer.reset_parameters()
    for conv in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d):
        x = torch.randn(2, 5, conv.hidden_size)
        with torch.no_grad():
            y, _ = conv(x)
        torch.testing.assert_close(y, x)


def test_liv_conv_block_cache_decode_matches_full_forward():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(5)
    mixer = GatedDeltaNet2Mixer(_cfg()).eval()
    layer = mixer.layer
    for conv in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d):
        _randomize_liv_conv(conv)
    x = torch.randn(1, 7, 16)
    with torch.no_grad():
        full, _, _ = mixer(x)
        head, _, state = mixer(x[:, :4], use_cache=True)
        assert state is not None
        tail, _, _ = mixer(x[:, 4:], past_key_values=state)
    torch.testing.assert_close(torch.cat([head, tail], dim=1), full, atol=1e-5, rtol=1e-5)
