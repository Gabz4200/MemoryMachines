"""Parity, continuity, decode, and error-path tests for depthwise causal Conv1d.

Covers the reference oracle, the ``ShortConvolution`` module, the naive
facade, and the dispatch entry points. Float64 + tight tolerances for parity
checks, float32 default elsewhere. All shapes kept tiny (B<=2, T<=8, D<=8,
W<=4).
"""

import warnings

import pytest
import torch
import torch.nn.functional as F

from memory_machines.models.blocks.causal_conv1d import ShortConvolution
from memory_machines.ops.causal_conv1d import (
    _causal_conv1d_fake,
    causal_conv1d_naive,
    causal_conv1d_op,
)
from memory_machines.ops.dispatch import (
    dispatch_causal_conv1d,
    dispatch_causal_conv1d_update,
)
from memory_machines.ops.references.causal_conv1d_reference import (
    reference_causal_conv1d_fwd,
    reference_causal_conv1d_update,
    reference_causal_conv1d_update_states,
)
from tests.test_blocks_shapes import tiny_cfg


def _d64(*shape):
    return torch.randn(*shape, dtype=torch.float64)


def test_ref_fwd_matches_grouped_conv1d():
    torch.manual_seed(0)
    B, T, D, W = 2, 6, 4, 3
    x = _d64(B, T, D)
    weight = _d64(D, W)
    y = reference_causal_conv1d_fwd(x, weight)
    expected = F.conv1d(
        F.pad(x.transpose(1, 2), (W - 1, 0)),
        weight.unsqueeze(1),
        groups=D,
    ).transpose(1, 2)
    torch.testing.assert_close(y, expected, atol=1e-12, rtol=1e-12)


def test_ref_fwd_last_tap_current_token_orientation():
    a, b, c = 2.0, -1.0, 0.5
    weight = torch.tensor([[a, b, c]], dtype=torch.float64)
    x = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]], dtype=torch.float64)
    y = reference_causal_conv1d_fwd(x, weight)
    xs = [1.0, 2.0, 3.0, 4.0]
    pad = [0.0, 0.0] + xs
    expected = torch.tensor(
        [[[a * pad[t] + b * pad[t + 1] + c * pad[t + 2]] for t in range(4)]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(y, expected, atol=1e-12, rtol=1e-12)


def test_ref_fwd_bias_silu_residual_manual_parity():
    torch.manual_seed(2)
    B, T, D, W = 1, 5, 3, 3
    x = _d64(B, T, D)
    weight = _d64(D, W)
    bias = _d64(D)
    residual = _d64(B, T, D)
    y = reference_causal_conv1d_fwd(x, weight, bias=bias, residual=residual, activation="silu")
    base = reference_causal_conv1d_fwd(x, weight)
    manual = (base + bias) * torch.sigmoid(base + bias) + residual
    torch.testing.assert_close(y, manual, atol=1e-12, rtol=1e-12)


def test_ref_fwd_swish_equals_silu_exactly():
    torch.manual_seed(3)
    x = _d64(2, 5, 4)
    weight = _d64(4, 3)
    bias = _d64(4)
    y_silu = reference_causal_conv1d_fwd(x, weight, bias=bias, activation="silu")
    y_swish = reference_causal_conv1d_fwd(x, weight, bias=bias, activation="swish")
    assert torch.equal(y_silu, y_swish)


def test_ref_fwd_decode_loop_equals_fwd():
    torch.manual_seed(4)
    B, T, D, W = 2, 6, 4, 3
    x = _d64(B, T, D)
    weight = _d64(D, W)
    y_dense = reference_causal_conv1d_fwd(x, weight)
    cache = torch.zeros(B, D, W, dtype=torch.float64)
    outs = []
    for t in range(T):
        y_tok, cache = reference_causal_conv1d_update(x[:, t, :], cache, weight)
        outs.append(y_tok)
    y_loop = torch.stack(outs, dim=1)
    torch.testing.assert_close(y_loop, y_dense, atol=1e-12, rtol=1e-12)


def test_ref_fwd_varlen_packed_matches_dense_no_leakage():
    torch.manual_seed(5)
    D, W = 4, 3
    # First sequence ends with zeros; second is nonzero throughout, so any
    # leakage across the boundary would visibly corrupt the comparison.
    s1 = _d64(3, D)
    s1[-2:] = 0.0
    s2 = torch.arange(1.0, 1.0 + 5 * D, dtype=torch.float64).reshape(5, D)
    packed = torch.cat([s1, s2], dim=0).unsqueeze(0)  # [1, L, D]
    cu = torch.tensor([0, 3, 8])
    weight = _d64(D, W)
    for packed_x in (packed, packed.squeeze(0)):  # [1, L, D] and [L, D]
        y = reference_causal_conv1d_fwd(packed_x, weight, cu_seqlens=cu)
        # Rank is preserved: 3D packed -> [1, L, D], 2D packed -> [L, D].
        expected_shape = (1, 8, D) if packed_x.dim() == 3 else (8, D)
        assert tuple(y.shape) == expected_shape
        y1 = reference_causal_conv1d_fwd(s1.unsqueeze(0), weight)
        y2 = reference_causal_conv1d_fwd(s2.unsqueeze(0), weight)
        y3d = y.unsqueeze(0) if y.dim() == 2 else y
        torch.testing.assert_close(y3d[:, :3], y1, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(y3d[:, 3:], y2, atol=1e-12, rtol=1e-12)


def test_ref_fwd_prefill_continue():
    torch.manual_seed(6)
    B, T1, T2, D, W = 2, 3, 4, 4, 4
    x1 = _d64(B, T1, D)
    x2 = _d64(B, T2, D)
    weight = _d64(D, W)
    init = _d64(B, D, W)
    init[:, :, 0] = 0.0  # slot 0 is padding
    s1 = reference_causal_conv1d_update_states(x1, state_len=W, initial_state=init)
    y_cont = reference_causal_conv1d_fwd(x2, weight, initial_state=s1)
    y_full = reference_causal_conv1d_fwd(torch.cat([x1, x2], dim=1), weight, initial_state=init)
    torch.testing.assert_close(y_cont, y_full[:, T1:], atol=1e-12, rtol=1e-12)


def test_ref_fwd_grad_flow():
    torch.manual_seed(7)
    x = torch.randn(2, 5, 4, requires_grad=True)
    weight = torch.randn(4, 3, requires_grad=True)
    bias = torch.randn(4, requires_grad=True)
    y = reference_causal_conv1d_fwd(x, weight, bias=bias, activation="silu")
    y.sum().backward()
    for g in (x.grad, weight.grad, bias.grad):
        assert g is not None
        assert torch.isfinite(g).all()


def test_ref_update_hand_computable():
    # y = 2*cache[1] + 3*cache[2] + 4*token after the shift.
    weight = torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float64)
    cache = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64)
    x = torch.tensor([[5.0]], dtype=torch.float64)
    y, out_cache = reference_causal_conv1d_update(x, cache, weight)
    assert y.item() == pytest.approx(2 * 2.0 + 3 * 3.0 + 4 * 5.0)
    torch.testing.assert_close(
        out_cache,
        torch.tensor([[[2.0, 3.0, 5.0]]], dtype=torch.float64),
        atol=0.0,
        rtol=0.0,
    )


def test_ref_update_mutates_cache_in_place():
    torch.manual_seed(9)
    N, D, W = 2, 3, 3
    cache = _d64(N, D, W)
    before_tail = cache[:, :, 1:].clone()
    tok = _d64(N, D)
    weight = _d64(D, W)
    y, out = reference_causal_conv1d_update(tok, cache, weight)
    assert out is cache  # same tensor object, updated in place
    torch.testing.assert_close(cache[:, :, :-1], before_tail, atol=0.0, rtol=0.0)
    torch.testing.assert_close(cache[:, :, -1], tok, atol=1e-12, rtol=1e-12)
    assert tuple(y.shape) == (N, D)


def test_ref_update_accepts_batched_singleton_shapes():
    torch.manual_seed(10)
    N, D, W = 2, 3, 2
    weight = _d64(D, W)
    tok = _d64(N, D)
    expected, _ = reference_causal_conv1d_update(tok.clone(), torch.zeros(N, D, W, dtype=torch.float64), weight)
    for shape in ((N, D), (1, N, D), (N, 1, D)):
        x = tok.reshape(shape)
        y, cache = reference_causal_conv1d_update(x, torch.zeros(N, D, W, dtype=torch.float64), weight)
        assert tuple(y.shape) == shape
        assert tuple(cache.shape) == (N, D, W)
        torch.testing.assert_close(y.reshape(N, D), expected, atol=1e-12, rtol=1e-12)


def test_ref_update_states_short_sequence_inherits_initial():
    torch.manual_seed(11)
    B, T, D, W = 1, 2, 3, 4  # T < W
    x = _d64(B, T, D)
    init = _d64(B, D, W)
    init[:, :, 0] = 0.0
    final = reference_causal_conv1d_update_states(x, state_len=W, initial_state=init)
    assert tuple(final.shape) == (B, D, W)
    torch.testing.assert_close(final[:, :, : W - T], init[:, :, : W - T], atol=0.0, rtol=0.0)
    torch.testing.assert_close(final[:, :, W - T :], x.transpose(1, 2), atol=0.0, rtol=0.0)


def test_ref_update_states_long_sequence_overwrites_initial():
    torch.manual_seed(12)
    B, T, D, W = 2, 6, 3, 4  # T >= W
    x = _d64(B, T, D)
    init = _d64(B, D, W)
    init[:, :, 0] = 0.0
    final = reference_causal_conv1d_update_states(x, state_len=W, initial_state=init)
    torch.testing.assert_close(final, x[:, -W:, :].transpose(1, 2), atol=0.0, rtol=0.0)


def test_ref_update_states_packed_matches_dense():
    torch.manual_seed(13)
    D, W = 3, 3
    s1 = _d64(2, D)
    s2 = _d64(5, D)
    packed = torch.cat([s1, s2], dim=0).unsqueeze(0)
    cu = torch.tensor([0, 2, 7])
    init = _d64(2, D, W)
    got = reference_causal_conv1d_update_states(packed, state_len=W, initial_state=init, cu_seqlens=cu)
    e1 = reference_causal_conv1d_update_states(s1.unsqueeze(0), state_len=W, initial_state=init[:1])
    e2 = reference_causal_conv1d_update_states(s2.unsqueeze(0), state_len=W, initial_state=init[1:])
    torch.testing.assert_close(got, torch.cat([e1, e2], dim=0), atol=0.0, rtol=0.0)


def test_ref_fwd_rejects_3d_weight():
    torch.manual_seed(20)
    with pytest.raises(ValueError):
        reference_causal_conv1d_fwd(_d64(1, 4, 2), _d64(2, 1, 3))


def test_ref_fwd_rejects_channel_mismatch():
    torch.manual_seed(21)
    with pytest.raises(ValueError):
        reference_causal_conv1d_fwd(_d64(1, 4, 3), _d64(2, 3))


def test_ref_fwd_rejects_bad_bias_shape():
    torch.manual_seed(22)
    with pytest.raises(ValueError):
        reference_causal_conv1d_fwd(_d64(1, 4, 2), _d64(2, 3), bias=torch.zeros(3, dtype=torch.float64))


def test_ref_update_rejects_bad_bias_shape():
    torch.manual_seed(23)
    with pytest.raises(ValueError):
        reference_causal_conv1d_update(
            _d64(1, 2),
            torch.zeros(1, 2, 3, dtype=torch.float64),
            _d64(2, 3),
            bias=torch.zeros(3, dtype=torch.float64),
        )


def test_ref_fwd_rejects_bad_activation():
    torch.manual_seed(24)
    with pytest.raises(ValueError, match="activation"):
        reference_causal_conv1d_fwd(_d64(1, 4, 2), _d64(2, 3), activation="relu")


def test_ref_update_rejects_bad_activation():
    torch.manual_seed(25)
    with pytest.raises(ValueError, match="activation"):
        reference_causal_conv1d_update(_d64(1, 2), torch.zeros(1, 2, 3, dtype=torch.float64), _d64(2, 3), activation="relu")


def test_ref_packed_rejects_batch_dim_not_one():
    torch.manual_seed(26)
    with pytest.raises(ValueError):
        reference_causal_conv1d_fwd(_d64(2, 7, 2), _d64(2, 3), cu_seqlens=torch.tensor([0, 3, 7]))


def test_ref_update_rejects_bad_cache_shape():
    torch.manual_seed(27)
    with pytest.raises(ValueError):
        reference_causal_conv1d_update(_d64(2, 3), torch.zeros(2, 3, 5, dtype=torch.float64), _d64(3, 4))


def test_ref_update_rejects_bad_x_shape():
    torch.manual_seed(28)
    with pytest.raises(ValueError):
        reference_causal_conv1d_update(_d64(2, 2, 3), torch.zeros(2, 3, 3, dtype=torch.float64), _d64(3, 3))


def test_ref_update_states_rejects_nonpositive_state_len():
    torch.manual_seed(29)
    with pytest.raises(ValueError):
        reference_causal_conv1d_update_states(_d64(1, 4, 2), state_len=0)


def test_ref_fwd_rejects_residual_mismatch():
    torch.manual_seed(30)
    with pytest.raises(ValueError):
        reference_causal_conv1d_fwd(_d64(1, 4, 2), _d64(2, 3), residual=_d64(1, 3, 2))


def test_ref_update_rejects_residual_mismatch():
    torch.manual_seed(31)
    with pytest.raises(ValueError):
        reference_causal_conv1d_update(
            _d64(2, 3),
            torch.zeros(2, 3, 3, dtype=torch.float64),
            _d64(3, 3),
            residual=torch.zeros(2, 2, 3, dtype=torch.float64),
        )


def test_dispatch_and_naive_reject_bad_activation():
    torch.manual_seed(32)
    x, w = _d64(1, 4, 2), _d64(2, 3)
    cache = torch.zeros(1, 2, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="activation"):
        dispatch_causal_conv1d(x, w, activation="relu")
    with pytest.raises(ValueError, match="activation"):
        dispatch_causal_conv1d_update(x[:, 0, :], cache, w, activation="relu")
    with pytest.raises(ValueError, match="activation"):
        causal_conv1d_naive(x, w, activation="relu")


def test_module_ctor_rejects_bad_activation_and_backend():
    with pytest.raises(ValueError, match="activation"):
        ShortConvolution(hidden_size=tiny_cfg().hidden_size, kernel_size=3, activation="relu")
    with pytest.raises(ValueError, match="naive"):
        ShortConvolution(hidden_size=4, kernel_size=3, backend="cuda")


def test_module_wiring_groups_weight_padding():
    torch.manual_seed(41)
    D, W = 4, 3
    conv = ShortConvolution(hidden_size=D, kernel_size=W)
    assert conv.groups == D
    assert tuple(conv.weight.shape) == (D, 1, W)
    assert tuple(conv.padding) == (0,)
    assert conv.in_channels == D and conv.out_channels == D


def test_module_dense_forward_matches_reference():
    torch.manual_seed(42)
    B, T, D, W = 2, 6, 4, 3
    for activation in (None, "silu", "swish"):
        conv = ShortConvolution(hidden_size=D, kernel_size=W, bias=True, activation=activation).double()
        assert conv.bias is not None
        x = _d64(B, T, D)
        residual = _d64(B, T, D)
        y, final = conv(x, residual=residual)
        expected = reference_causal_conv1d_fwd(
            x,
            conv.weight.reshape(D, W).detach(),
            bias=conv.bias.detach(),
            residual=residual,
            activation=activation,
        )
        torch.testing.assert_close(y, expected, atol=1e-12, rtol=1e-12)
        assert final is None


def test_module_mask_zeroes_masked_contribution():
    torch.manual_seed(43)
    B, T, D, W = 1, 6, 3, 3
    conv = ShortConvolution(hidden_size=D, kernel_size=W, activation=None).eval()
    x = torch.randn(B, T, D)
    mask = torch.ones(B, T)
    mask[:, 2:4] = 0.0
    y_masked, _ = conv(x, mask=mask)
    # Masked input positions contribute nothing: changing them changes nothing.
    x_pert = x.clone()
    x_pert[:, 2:4, :] = torch.randn(1, 2, D) * 10.0
    y_pert, _ = conv(x_pert, mask=mask)
    torch.testing.assert_close(y_masked, y_pert, atol=1e-6, rtol=1e-6)
    # And masking changes the output vs unmasked (history carried zeros).
    y_plain, _ = conv(x)
    assert not torch.allclose(y_masked, y_plain, atol=1e-6, rtol=1e-6)


def test_module_mask_with_cu_seqlens_raises():
    torch.manual_seed(44)
    conv = ShortConvolution(hidden_size=2, kernel_size=2).eval()
    with pytest.raises(ValueError, match="cu_seqlens"):
        conv(
            torch.randn(1, 4, 2),
            mask=torch.ones(1, 4),
            cu_seqlens=torch.tensor([0, 2, 4]),
        )


def test_module_output_final_state():
    torch.manual_seed(45)
    B, T, D, W = 2, 5, 3, 3
    conv = ShortConvolution(hidden_size=D, kernel_size=W, activation=None).double()
    x = _d64(B, T, D)
    y, final_none = conv(x)
    assert final_none is None
    y2, final = conv(x, output_final_state=True)
    torch.testing.assert_close(y, y2, atol=0.0, rtol=0.0)
    assert final is not None
    assert tuple(final.shape) == (B, D, W)
    expected = reference_causal_conv1d_update_states(x, state_len=W)
    torch.testing.assert_close(final, expected, atol=0.0, rtol=0.0)


def test_module_decode_routing_no_grad_vs_grad():
    torch.manual_seed(46)
    B, D, W = 2, 3, 3
    conv = ShortConvolution(hidden_size=D, kernel_size=W, activation=None).eval()
    x = torch.randn(B, 1, D)
    with torch.no_grad():
        y, cache = conv(x, cache=None)
    assert cache is not None
    assert tuple(cache.shape) == (B, D, W)
    assert tuple(y.shape) == (B, 1, D)
    with torch.enable_grad():
        conv.train()
        y2, final = conv(x)
    assert final is None
    assert tuple(y2.shape) == (B, 1, D)


def test_module_step_loop_equals_dense():
    torch.manual_seed(47)
    B, T, D, W = 2, 5, 4, 3
    for activation in (None, "silu"):
        conv = ShortConvolution(hidden_size=D, kernel_size=W, activation=activation).eval()
        x = torch.randn(B, T, D)
        with torch.no_grad():
            y_dense, _ = conv(x)
            cache = None
            outs = []
            for t in range(T):
                y_tok, cache = conv.step(x[:, t : t + 1, :], None, cache)
                outs.append(y_tok)
            y_loop = torch.cat(outs, dim=1)
        torch.testing.assert_close(y_loop, y_dense, atol=1e-6, rtol=1e-6)


def test_module_grad_flow_weight_and_bias():
    torch.manual_seed(48)
    for bias in (False, True):
        conv = ShortConvolution(hidden_size=4, kernel_size=3, bias=bias, activation="silu").train()
        x = torch.randn(2, 5, 4)
        conv.zero_grad()
        conv(x)[0].sum().backward()
        assert conv.weight.grad is not None
        assert torch.isfinite(conv.weight.grad).all()
        if bias:
            assert conv.bias is not None and conv.bias.grad is not None
            assert torch.isfinite(conv.bias.grad).all()
        else:
            assert conv.bias is None


def test_module_step_cache_aliasing_in_place():
    torch.manual_seed(49)
    B, D, W = 2, 3, 3
    conv = ShortConvolution(hidden_size=D, kernel_size=W, activation=None).eval()
    x = torch.randn(B, 1, D)
    cache = torch.zeros(B, D, W)
    with torch.no_grad():
        y, out = conv.step(x, None, cache)
    assert out is cache
    torch.testing.assert_close(cache[:, :, -1], x.squeeze(1), atol=1e-6, rtol=1e-6)
    assert tuple(y.shape) == (B, 1, D)


def test_module_prefill_then_step_continuity():
    torch.manual_seed(50)
    B, T1, T2, D, W = 1, 4, 3, 4, 3
    for activation in (None, "silu"):
        conv = ShortConvolution(hidden_size=D, kernel_size=W, activation=activation).eval()
        x = torch.randn(B, T1 + T2, D)
        with torch.no_grad():
            y_dense, _ = conv(x)
            _, cache = conv(x[:, :T1, :], output_final_state=True)
            outs = []
            for t in range(T1, T1 + T2):
                y_tok, cache = conv.step(x[:, t : t + 1, :], None, cache)
                outs.append(y_tok)
        torch.testing.assert_close(torch.cat(outs, dim=1), y_dense[:, T1:], atol=1e-6, rtol=1e-6)


def test_module_packed_forward_parity():
    torch.manual_seed(51)
    D, W = 3, 3
    conv = ShortConvolution(hidden_size=D, kernel_size=W, bias=True, activation="silu").double().eval()
    assert conv.bias is not None
    s1 = _d64(2, D)
    s2 = _d64(5, D)
    packed = torch.cat([s1, s2], dim=0).unsqueeze(0)
    cu = torch.tensor([0, 2, 7])
    y, _ = conv(packed, cu_seqlens=cu)
    expected = reference_causal_conv1d_fwd(
        packed,
        conv.weight.reshape(D, W).detach(),
        bias=conv.bias.detach(),
        activation="silu",
        cu_seqlens=cu,
    )
    torch.testing.assert_close(y, expected, atol=1e-12, rtol=1e-12)


def test_module_bias_state_size_extra_repr():
    torch.manual_seed(52)
    off = ShortConvolution(hidden_size=4, kernel_size=3, bias=False)
    on = ShortConvolution(hidden_size=4, kernel_size=3, bias=True)
    assert off.bias is None
    assert on.bias is not None and tuple(on.bias.shape) == (4,)
    x = torch.randn(1, 4, 4)
    with torch.no_grad():
        assert tuple(off(x)[0].shape) == (1, 4, 4)
        assert tuple(on(x)[0].shape) == (1, 4, 4)
    assert off.state_size == 4 * 3
    assert "activation=" in off.extra_repr() and "backend=" in off.extra_repr()
    assert "silu" in off.extra_repr()


def test_module_causality_future_perturbation_no_effect_on_past():
    torch.manual_seed(53)
    conv = ShortConvolution(hidden_size=3, kernel_size=3, activation=None).eval()
    x = torch.randn(1, 6, 3)
    x2 = x.clone()
    x2[:, 4:, :] += 5.0
    with torch.no_grad():
        y1, _ = conv(x)
        y2, _ = conv(x2)
    torch.testing.assert_close(y1[:, :3], y2[:, :3], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(y1[:, 4:], y2[:, 4:], atol=1e-6, rtol=1e-6)


def test_module_kernel_size_1_degenerate():
    torch.manual_seed(54)
    D = 3
    conv = ShortConvolution(hidden_size=D, kernel_size=1, bias=True, activation=None).double()
    assert conv.bias is not None
    x = _d64(2, 5, D)
    y, _ = conv(x)
    w = conv.weight.reshape(D).detach()
    expected = x * w + conv.bias.detach()
    torch.testing.assert_close(y, expected, atol=1e-12, rtol=1e-12)


def test_ref_fwd_width_1_ignores_history():
    torch.manual_seed(55)
    B, T, D, W = 2, 5, 3, 1
    x = _d64(B, T, D)
    weight = _d64(D, W)
    init = _d64(B, D, W) * 10.0
    y_plain = reference_causal_conv1d_fwd(x, weight)
    y_hist = reference_causal_conv1d_fwd(x, weight, initial_state=init)
    torch.testing.assert_close(y_hist, y_plain, atol=1e-12, rtol=1e-12)


def test_dispatch_naive_matches_reference_with_all_options():
    torch.manual_seed(60)
    B, T, D, W = 2, 5, 4, 3
    x = _d64(B, T, D)
    weight = _d64(D, W)
    bias = _d64(D)
    residual = _d64(B, T, D)
    init = _d64(B, D, W)
    init[:, :, 0] = 0.0
    y, final = dispatch_causal_conv1d(
        x,
        weight,
        bias=bias,
        residual=residual,
        initial_state=init,
        output_final_state=True,
        activation="silu",
    )
    expected = reference_causal_conv1d_fwd(
        x,
        weight,
        bias=bias,
        residual=residual,
        initial_state=init,
        activation="silu",
    )
    torch.testing.assert_close(y, expected, atol=1e-12, rtol=1e-12)
    assert final is not None
    torch.testing.assert_close(
        final,
        reference_causal_conv1d_update_states(x, state_len=W, initial_state=init),
        atol=0.0,
        rtol=0.0,
    )


def test_dispatch_update_naive_matches_reference():
    torch.manual_seed(61)
    N, D, W = 2, 4, 3
    x = _d64(N, D)
    cache = _d64(N, D, W)
    weight = _d64(D, W)
    bias = _d64(D)
    residual = _d64(N, D)
    y, out = dispatch_causal_conv1d_update(x, cache.clone(), weight, bias=bias, residual=residual, activation="silu")
    ey, ecache = reference_causal_conv1d_update(x, cache.clone(), weight, bias=bias, residual=residual, activation="silu")
    torch.testing.assert_close(y, ey, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(out, ecache, atol=1e-12, rtol=1e-12)


def test_dispatch_compiled_backends_raise_not_implemented():
    torch.manual_seed(62)
    x = torch.randn(1, 4, 2)
    w = torch.randn(2, 3)
    cache = torch.zeros(1, 2, 3)
    for backend in ("cpu", "gpu", "taichi", "compiled"):
        with pytest.raises(NotImplementedError, match="naive"):
            dispatch_causal_conv1d(x, w, backend=backend)
        with pytest.raises(NotImplementedError, match="naive"):
            dispatch_causal_conv1d_update(x[:, 0, :], cache.clone(), w, backend=backend)


def test_dispatch_unknown_backend_raises_value_error():
    torch.manual_seed(63)
    x = torch.randn(1, 4, 2)
    w = torch.randn(2, 3)
    cache = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="Unknown backend"):
        dispatch_causal_conv1d(x, w, backend="tpu")
    with pytest.raises(ValueError, match="Unknown backend"):
        dispatch_causal_conv1d_update(x[:, 0, :], cache, w, backend="tpu")


def test_causal_conv1d_naive_final_state_flag():
    torch.manual_seed(64)
    B, T, D, W = 2, 5, 4, 3
    x = _d64(B, T, D)
    weight = _d64(D, W)
    y, none_state = causal_conv1d_naive(x, weight)
    assert none_state is None
    torch.testing.assert_close(y, reference_causal_conv1d_fwd(x, weight), atol=1e-12, rtol=1e-12)
    y2, state = causal_conv1d_naive(x, weight, output_final_state=True)
    torch.testing.assert_close(y2, y, atol=0.0, rtol=0.0)
    assert state is not None
    torch.testing.assert_close(
        state,
        reference_causal_conv1d_update_states(x, state_len=W),
        atol=0.0,
        rtol=0.0,
    )


def test_causal_conv1d_op_raises_not_implemented():
    torch.manual_seed(65)
    with pytest.raises(NotImplementedError):
        causal_conv1d_op(torch.randn(1, 4, 2), torch.randn(2, 3), None)


def test_dispatch_auto_warns_and_falls_back_to_reference():
    # Ops dispatch regression: backend="auto" must try the compiled kernel
    # (Taichi -> CPU cpp) and, since neither is implemented yet, emit a
    # UserWarning and return the pure-PyTorch reference result. The warning
    # must name the op so callers can silence it with backend="naive".
    torch.manual_seed(66)
    x = _d64(1, 4, 2)
    w = _d64(2, 3)
    cache = _d64(1, 2, 3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y, _ = dispatch_causal_conv1d(x, w, backend="auto")
        yu, cu = dispatch_causal_conv1d_update(x[:, 0, :], cache.clone(), w, backend="auto")
    assert any(issubclass(w.category, UserWarning) for w in caught)
    assert any("causal_conv1d" in str(w.message) for w in caught)
    assert any("causal_conv1d_update" in str(w.message) for w in caught)
    # The fallback result must equal the reference exactly.
    ey = reference_causal_conv1d_fwd(x, w)
    torch.testing.assert_close(y, ey, atol=1e-12, rtol=1e-12)
    eyu, ecu = reference_causal_conv1d_update(x[:, 0, :], cache.clone(), w)
    torch.testing.assert_close(yu, eyu, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(cu, ecu, atol=1e-12, rtol=1e-12)


def test_dispatch_auto_matches_naive_for_all_options():
    # auto (warns + reference) and naive (reference directly) must agree
    # exactly for every option combination — the fallback is the same code.
    torch.manual_seed(67)
    B, T, D, W = 2, 5, 4, 3
    x = _d64(B, T, D)
    weight = _d64(D, W)
    bias = _d64(D)
    residual = _d64(B, T, D)
    init = _d64(B, D, W)
    init[:, :, 0] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_auto, f_auto = dispatch_causal_conv1d(
            x,
            weight,
            bias=bias,
            residual=residual,
            initial_state=init,
            output_final_state=True,
            activation="silu",
            backend="auto",
        )
    y_naive, f_naive = dispatch_causal_conv1d(
        x,
        weight,
        bias=bias,
        residual=residual,
        initial_state=init,
        output_final_state=True,
        activation="silu",
        backend="naive",
    )
    torch.testing.assert_close(y_auto, y_naive, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(f_auto, f_naive, atol=0.0, rtol=0.0)


def test_dispatch_auto_no_warning_when_explicit_naive():
    # backend="naive" must not emit the fallback warning — it is the reference
    # by construction, not a downgrade.
    torch.manual_seed(68)
    x = _d64(1, 4, 2)
    w = _d64(2, 3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dispatch_causal_conv1d(x, w, backend="naive")
        dispatch_causal_conv1d_update(x[:, 0, :], torch.zeros(1, 2, 3), w, backend="naive")
    assert not any(issubclass(w.category, UserWarning) for w in caught)


def test_dispatch_backend_normalization_is_case_insensitive_and_tolerant():
    # backend strings are normalized (case-insensitive, whitespace-tolerant)
    # so " AUTO ", "Naive", and "cpu" all resolve to the same backend.
    torch.manual_seed(69)
    x = _d64(1, 4, 2)
    w = _d64(2, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_auto = dispatch_causal_conv1d(x, w, backend=" AUTO ")
        y_naive = dispatch_causal_conv1d(x, w, backend="  naive  ")
    torch.testing.assert_close(y_auto, y_naive, atol=1e-12, rtol=1e-12)
    # An explicit compiled backend still raises (no silent downgrade).
    with pytest.raises(NotImplementedError):
        dispatch_causal_conv1d(x, w, backend="GPU")


def test_causal_conv1d_fake_valid_shapes_return_x_shape() -> None:
    torch.manual_seed(70)
    x = torch.randn(2, 5, 4)
    weight = torch.randn(4, 3)
    out = _causal_conv1d_fake(x, weight)
    assert tuple(out.shape) == tuple(x.shape)
    assert out.dtype == x.dtype
    # Optional args that are shape-correct must be accepted.
    out2 = _causal_conv1d_fake(x, weight, bias=torch.randn(4), residual=torch.randn(2, 5, 4))
    assert tuple(out2.shape) == tuple(x.shape)


@pytest.mark.parametrize(
    "x,weight,bias,residual,message",
    [
        (torch.randn(1, 4, 2), torch.randn(2, 1, 3), None, None, "weight must be"),
        (torch.randn(1, 4, 3), torch.randn(2, 3), None, None, "must match weight dim 0"),
        (torch.randn(1, 4, 2), torch.randn(2, 3), torch.randn(3), None, "bias must be"),
        (torch.randn(1, 4, 2), torch.randn(2, 3), None, torch.randn(1, 3, 2), "residual shape"),
    ],
    ids=["weight-not-2d", "x-channel-mismatch", "bias-not-d", "residual-mismatch"],
)
def test_causal_conv1d_fake_rejects_bad_shapes(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _causal_conv1d_fake(x, weight, bias=bias, residual=residual)


def test_causal_conv1d_custom_op_dispatches_to_registered_fake() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    torch.manual_seed(71)
    with FakeTensorMode():
        x = torch.randn(1, 4, 2)
        weight = torch.randn(2, 3)
        out = causal_conv1d_op(x, weight)
        assert tuple(out.shape) == (1, 4, 2)
        with pytest.raises(ValueError, match="x last dim"):
            causal_conv1d_op(torch.randn(1, 4, 5), weight)
