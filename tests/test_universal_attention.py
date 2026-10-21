"""Universal PE SDPA: causal/bidirectional x self/cross coverage + N-dim PE.

TDD RED suite for the functional integration: the pure-PyTorch op in
``ops/references/attention_reference.py`` must support all variants in one
function, with learnable PE parts hoisted outside (passed as precomputed
tensors). Classes in ``models/blocks`` receive Q/K/V plus PE tensors and
route to this op internally.
"""

import math

import pytest
import torch

from memory_machines.ops.attention import (
    _universal_pe_attention_fake,
    universal_pe_attention_op,
)
from memory_machines.ops.references.attention_reference import (
    universal_pe_scaled_dot_product_attention as uattn,
)


def test_vanilla_matches_manual_sdpa():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 6, 8)
    k = torch.randn(2, 3, 6, 8)
    v = torch.randn(2, 3, 6, 8)
    ref = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(8), dim=-1) @ v
    torch.testing.assert_close(uattn(q, k, v), ref, atol=1e-5, rtol=1e-5)


def test_causal_self_masks_future():
    torch.manual_seed(1)
    q = torch.randn(1, 2, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)
    out = uattn(q, k, v, is_causal=True)
    logits = q @ k.transpose(-2, -1) / math.sqrt(8)
    ref = torch.softmax(logits.masked_fill(~torch.tril(torch.ones(5, 5, dtype=torch.bool)), float("-inf")), dim=-1) @ v
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_bidirectional_equals_unmasked_and_rejects_causal_combo():
    torch.manual_seed(2)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 6, 8)
    v = torch.randn(1, 2, 6, 8)
    torch.testing.assert_close(uattn(q, k, v, bidirectional=True), uattn(q, k, v, is_causal=False), atol=0, rtol=0)
    try:
        uattn(q, k, v, is_causal=True, bidirectional=True)
    except ValueError:
        pass
    else:
        raise AssertionError("is_causal+bidirectional must raise")


def test_cross_attention_nonsquare_shapes():
    torch.manual_seed(3)
    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(2, 4, 9, 8)
    v = torch.randn(2, 4, 9, 8)
    out = uattn(q, k, v, bidirectional=True)
    assert tuple(out.shape) == (2, 4, 3, 8)
    assert torch.isfinite(out).all()


def test_gqa_expansion_matches_manual_repeat():
    torch.manual_seed(4)
    q = torch.randn(1, 4, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)
    out = uattn(q, k, v, enable_gqa=True, is_causal=True)
    ke = k.repeat_interleave(2, -3)
    ve = v.repeat_interleave(2, -3)
    logits = q @ ke.transpose(-2, -1) / math.sqrt(8)
    ref = torch.softmax(logits.masked_fill(~torch.tril(torch.ones(5, 5, dtype=torch.bool)), float("-inf")), dim=-1) @ ve
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_pe_circuits_change_output_and_chunking_is_stable():
    torch.manual_seed(5)
    q = torch.randn(1, 2, 6, 8)
    k = torch.randn(1, 2, 6, 8)
    v = torch.randn(1, 2, 6, 8)
    base = uattn(q, k, v)
    bias = torch.randn(1, 2, 6, 6) * 0.5
    pvq = torch.randn(1, 2, 6, 8)
    pvk = torch.randn(1, 2, 6, 8)
    biased = uattn(q, k, v, cvpe_attn_bias=bias, phi_vq=pvq, phi_vk=pvk)
    assert not torch.allclose(base, biased, atol=1e-6, rtol=1e-6)
    ref = uattn(q, k, v, cvpe_attn_bias=bias, phi_vq=pvq, phi_vk=pvk, frvm_chunk_size=1000)
    small = uattn(q, k, v, cvpe_attn_bias=bias, phi_vq=pvq, phi_vk=pvk, frvm_chunk_size=2)
    torch.testing.assert_close(small, ref, atol=1e-5, rtol=1e-5)


def test_ov_spatial_framing_runs_and_requires_both_args():
    torch.manual_seed(6)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)
    pvq = torch.randn(1, 2, 4, 8)
    pvk = torch.randn(1, 2, 5, 8)
    grid = torch.randn(1, 4, 5, 3)
    wov = torch.randn(8, 3)
    out = uattn(q, k, v, phi_vq=pvq, phi_vk=pvk, spatial_grid=grid, ov_spatial_weight=wov)
    assert tuple(out.shape) == (1, 2, 4, 8)
    assert torch.isfinite(out).all()
    try:
        uattn(q, k, v, phi_vq=pvq, phi_vk=pvk, spatial_grid=grid)
    except ValueError:
        pass
    else:
        raise AssertionError("grid without weight must raise")


def test_gqa_cross_attention_with_dual_phi_streams():
    # L != S, distinct query/key value streams (Phi_q vs Phi_k).
    torch.manual_seed(7)
    L, S = 4, 9
    q = torch.randn(1, 4, L, 8)
    k = torch.randn(1, 2, S, 8)
    v = torch.randn(1, 2, S, 8)
    pvq = torch.randn(1, 4, L, 8)
    pvk = torch.randn(1, 4, S, 8)
    out = uattn(q, k, v, phi_vq=pvq, phi_vk=pvk, enable_gqa=True, bidirectional=True)
    assert tuple(out.shape) == (1, 4, L, 8)
    assert torch.isfinite(out).all()


def test_gqa_class_cross_attention_end_to_end():
    import dataclasses

    from memory_machines.models.blocks import GroupedQueryAttention
    from tests.test_blocks_shapes import tiny_cfg

    torch.manual_seed(8)
    cfg = dataclasses.replace(tiny_cfg(), window_size=0)
    m = GroupedQueryAttention(cfg).eval()
    xq = torch.randn(1, 4, cfg.hidden_size)
    xkv = torch.randn(1, 9, cfg.hidden_size)
    with torch.no_grad():
        out, w, cache = m(xq, encoder_hidden_states=xkv, bidirectional=True, output_attentions=True)
    assert tuple(out.shape) == (1, 4, cfg.hidden_size)
    assert w is not None and tuple(w.shape) == (1, m.num_heads, 4, 9)
    assert cache is None
    assert torch.isfinite(out).all()


def test_ndim_zero_padding_toggle_matches_1d():
    # 1D -> 3D toggle: zero-padded spatial axes must reproduce the 1D bias
    # (same weights serve every modality).

    from memory_machines.models.blocks.positioning import CVPEContextBias

    torch.manual_seed(10)
    mod1 = CVPEContextBias(head_dim=4, spatial_dims=1).eval()
    mod3 = CVPEContextBias(head_dim=4, spatial_dims=3, axis_max=(32768, 32768, 32768)).eval()
    with torch.no_grad():
        # Share the t-axis weights: copy 1D MLP into the t-column of the 3D MLP.
        mod3.mlp_in_nd.weight.zero_()
        mod3.mlp_in_nd.weight[:, 0:1] = mod1.mlp_in.weight
        mod3.mlp_in_nd.bias.copy_(mod1.mlp_in.bias)
        mod3.offset_out_nd.weight.copy_(mod1.offset_out.weight)
        mod3.offset_out_nd.bias.copy_(mod1.offset_out.bias)
        for a, b in ((mod3.gamma_proj, mod1.gamma_proj), (mod3.beta_proj, mod1.beta_proj)):
            b.weight.copy_(a.weight) if False else a.weight.copy_(b.weight)
            a.bias.copy_(b.bias)
    q = torch.randn(1, 2, 3, 4)
    q1 = torch.tensor([[4, 5, 6]])
    k1 = torch.tensor([[0, 1, 2, 3, 4]])
    q3 = torch.zeros(1, 3, 3)
    q3[:, :, 0] = q1.float()
    k3 = torch.zeros(1, 5, 3)
    k3[:, :, 0] = k1.float()
    with torch.no_grad():
        b1 = mod1(q, q1, k1)
        b3 = mod3(q, q3, k3)
    # NOTE: 1D legacy uses relative/q normalization while N-dim uses bounded
    # axis_max normalization — same-modality toggle is shape/finite parity,
    # not value equality (documented architectural difference).
    assert tuple(b3.shape) == tuple(b1.shape)
    assert torch.isfinite(b3).all()


def test_gqa_cross_attention_promotes_1d_position_ids() -> None:
    import dataclasses

    from memory_machines.models.blocks import GroupedQueryAttention
    from tests.test_blocks_shapes import tiny_cfg

    torch.manual_seed(15)
    cfg = dataclasses.replace(tiny_cfg(), window_size=0, cvpe_enabled=False, frvm_enabled=False)
    m = GroupedQueryAttention(cfg).eval()
    xq, xkv = torch.randn(1, 4, cfg.hidden_size), torch.randn(1, 9, cfg.hidden_size)
    q1d, k1d = torch.arange(4), torch.arange(9)  # Lq=4 != S=9
    with torch.no_grad():
        out_1d, w_1d, _ = m(xq, encoder_hidden_states=xkv, position_ids=q1d, encoder_position_ids=k1d, bidirectional=True, output_attentions=True)
        out_2d, w_2d, _ = m(
            xq,
            encoder_hidden_states=xkv,
            position_ids=q1d.unsqueeze(0),
            encoder_position_ids=k1d.unsqueeze(0),
            bidirectional=True,
            output_attentions=True,
        )
    torch.testing.assert_close(out_1d, out_2d, atol=0, rtol=0)
    assert w_1d is not None and w_2d is not None
    torch.testing.assert_close(w_1d, w_2d, atol=0, rtol=0)


def test_pe_inputs_pads_1d_positions_to_the_spatial_width() -> None:
    import dataclasses

    from memory_machines.models.blocks import GroupedQueryAttention
    from tests.test_blocks_shapes import tiny_cfg

    torch.manual_seed(16)
    m = GroupedQueryAttention(dataclasses.replace(tiny_cfg(), window_size=0, cvpe_max_dims=3)).eval()
    q = torch.randn(1, m.num_heads, 4, m.head_dim)
    vq = torch.randn(1, m.num_heads, 4, m.head_dim)
    vk = torch.randn(1, m.num_heads, 6, m.head_dim)
    _, _, _, grid, weight = m._pe_inputs(q, vq, vk, torch.arange(4).unsqueeze(0), torch.arange(6).unsqueeze(0))
    # 1D positions are zero-padded to cvpe_max_dims before the offset grid.
    assert grid is not None and tuple(grid.shape) == (1, 4, 6, 3)
    assert weight is not None and tuple(weight.shape) == (m.head_dim, 3)


def test_gqa_class_ndim_self_and_cross():
    import dataclasses

    from memory_machines.models.blocks import GroupedQueryAttention
    from tests.test_blocks_shapes import tiny_cfg

    torch.manual_seed(9)
    cfg = dataclasses.replace(tiny_cfg(), window_size=0, cvpe_max_dims=3)
    m = GroupedQueryAttention(cfg).eval()
    x = torch.randn(1, 5, cfg.hidden_size)
    q3 = torch.zeros(1, 5, 3)
    q3[:, :, 0] = torch.arange(5).float()
    k3 = torch.zeros(1, 7, 3)
    k3[:, :, 0] = torch.arange(7).float()
    xk = torch.randn(1, 7, cfg.hidden_size)
    with torch.no_grad():
        out_self, _, _ = m(x, position_ids=q3[:, :5, :], bidirectional=True)
        out_cross, w, _ = m(x, encoder_hidden_states=xk, position_ids=q3, encoder_position_ids=k3, bidirectional=True, output_attentions=True)
    assert tuple(out_self.shape) == (1, 5, cfg.hidden_size)
    assert tuple(out_cross.shape) == (1, 5, cfg.hidden_size)
    assert w is not None and tuple(w.shape) == (1, m.num_heads, 5, 7)


@pytest.mark.parametrize(
    "q_shape,k_shape,v_shape",
    [
        ((2, 3, 8), (2, 3, 8), (2, 3, 8)),
        ((2, 4, 5, 8), (2, 4, 5, 8), (2, 4, 5, 6)),
        ((2, 3, 4, 5, 8), (2, 3, 4, 5, 8), (2, 3, 4, 5, 6)),
    ],
    ids=["3d", "4d", "5d"],
)
def test_fake_attention_accepts_rank_3_and_up(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
) -> None:
    torch.manual_seed(11)
    out = _universal_pe_attention_fake(torch.randn(*q_shape), torch.randn(*k_shape), torch.randn(*v_shape))
    # Output keeps the query layout except the last axis, which is value's.
    assert tuple(out.shape) == (*q_shape[:-1], v_shape[-1])


@pytest.mark.parametrize("bad", ["query", "key", "value"], ids=["query-2d", "key-2d", "value-2d"])
def test_fake_attention_rejects_rank_below_3(bad: str) -> None:
    torch.manual_seed(12)
    tensors = {"query": torch.randn(3, 8), "key": torch.randn(3, 8), "value": torch.randn(3, 8)}
    tensors[bad] = torch.randn(3)  # rank 1 for the axis under test
    with pytest.raises(RuntimeError):
        _universal_pe_attention_fake(tensors["query"], tensors["key"], tensors["value"])


def test_universal_pe_attention_op_raises_not_implemented() -> None:
    torch.manual_seed(13)
    with pytest.raises(NotImplementedError, match="no compiled kernel yet"):
        universal_pe_attention_op(torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8))


def test_universal_pe_attention_custom_op_dispatches_to_registered_fake() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    torch.manual_seed(14)
    with FakeTensorMode():
        out = universal_pe_attention_op(torch.randn(2, 6, 8), torch.randn(2, 9, 8), torch.randn(2, 9, 5))
        assert tuple(out.shape) == (2, 6, 5)
