"""Validation/error-branch tests for the RAW gdn2 module (non-mixer).

Covers untested ValueError branches of ``gated_delta_rule_2``,
``gated_delta_rule_2_recurrent_forward``, ``gated_delta_rule_2_chunk_forward``,
and ``GatedDeltaNet2``. Small shapes (B=1, T=4, H=1, K=4/HV=1/V=4).
"""

import pytest
import torch

from memory_machines.models.blocks import gdn2 as G


def _inputs(B=1, T=4, H=1, HV=1, K=4, V=4, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, HV, V)
    g = -torch.rand(B, T, H, K)
    b = torch.rand(B, T, HV, K)
    w = torch.rand(B, T, HV, V)
    return q, k, v, g, b, w


def test_recurrent_3d_q_raises():
    q, k, v, g, b, w = _inputs()
    with pytest.raises(ValueError, match="must be 4D"):
        G.gated_delta_rule_2_recurrent_forward(q[0], k, v, g, b, w)


def test_recurrent_3d_v_raises():
    q, k, v, g, b, w = _inputs()
    with pytest.raises(ValueError, match="must be 4D"):
        G.gated_delta_rule_2_recurrent_forward(q, k, v[0], g, b, w)


def test_vw_batch_mismatch_raises():
    q, k, _, g, b, _ = _inputs()
    torch.manual_seed(1)
    v2 = torch.randn(2, 4, 1, 4)
    w2 = torch.randn(2, 4, 1, 4)
    with pytest.raises(ValueError, match="v/w must share"):
        G.gated_delta_rule_2_recurrent_forward(q, k, v2, g, b, w2)


def test_vw_t_mismatch_raises():
    q, k, _, g, b, _ = _inputs()
    torch.manual_seed(1)
    v2 = torch.randn(1, 2, 1, 4)
    w2 = torch.randn(1, 2, 1, 4)
    with pytest.raises(ValueError, match="v/w must share"):
        G.gated_delta_rule_2_recurrent_forward(q, k, v2, g, b, w2)


def test_cu_seqlens_not_ending_at_T_raises():
    q, k, v, g, b, w = _inputs()
    cu = torch.tensor([0, 2])
    with pytest.raises(ValueError, match="must end at T"):
        G.gated_delta_rule_2(q, k, v, g, b, w, cu_seqlens=cu)


def test_ctor_hidden_size_zero_raises():
    with pytest.raises(ValueError, match="must be positive"):
        G.GatedDeltaNet2(hidden_size=0, num_heads=1, head_dim_k=4, use_short_conv=False)


def test_ctor_num_heads_zero_raises():
    with pytest.raises(ValueError, match="must be positive"):
        G.GatedDeltaNet2(hidden_size=16, num_heads=0, head_dim_k=4, use_short_conv=False)


def test_ctor_head_dim_k_negative_raises():
    with pytest.raises(ValueError, match="must be positive"):
        G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=-1, use_short_conv=False)


def test_ctor_num_v_heads_not_multiple_raises():
    with pytest.raises(ValueError, match="must be a multiple"):
        G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, num_v_heads=3, use_short_conv=False)


def test_ctor_expand_v_nonintegral_raises():
    with pytest.raises(ValueError, match="inconsistent with head dims"):
        G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, expand_v=0.35, use_short_conv=False)


def test_forward_mask_with_cache_raises():
    torch.manual_seed(0)
    blk = G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, use_short_conv=False)
    x = torch.randn(1, 4, 16)
    _, _, cache = blk(x, use_cache=True)
    mask = torch.ones(1, 4)
    with pytest.raises(ValueError, match="with cached past_key_values"):
        blk.forward(x, attention_mask=mask, past_key_values=cache)


def test_all_zero_mask_row_outputs_zeros():
    # Canonical contract: fully-masked input returns full-width zeros (B, T, D),
    # matching the mixed-batch convention. Fixed T always.
    torch.manual_seed(0)
    blk = G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, use_short_conv=False)
    x = torch.randn(1, 4, 16)
    mask = torch.zeros(1, 4)
    o, _, _ = blk(x, attention_mask=mask)
    assert tuple(o.shape) == (1, 4, 16)
    assert o.abs().amax().item() == 0.0


def test_allow_neg_eigval_changes_output():
    torch.manual_seed(0)
    blk_false = G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, use_short_conv=False, allow_neg_eigval=False)
    torch.manual_seed(0)
    blk_true = G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, use_short_conv=False, allow_neg_eigval=True)
    assert blk_true.allow_neg_eigval is True
    x = torch.randn(1, 4, 16)
    o_false, _, _ = blk_false(x)
    o_true, _, _ = blk_true(x)
    assert (o_false - o_true).abs().max().item() > 0


def test_ctor_chunk_size_zero_raises():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, use_short_conv=False, chunk_size=0)


def test_ctor_chunk_size_negative_raises():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        G.GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=4, use_short_conv=False, chunk_size=-1)


def _blk(**kw):
    args = {"hidden_size": 16, "num_heads": 2, "head_dim_k": 4, "use_short_conv": False}
    args.update(kw)
    torch.manual_seed(0)
    return G.GatedDeltaNet2(**args).eval()


def test_masked_mixed_batch_empty_row_is_zeros():
    blk = _blk()
    torch.manual_seed(1)
    x = torch.randn(2, 4, 16)
    mask = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])
    with torch.no_grad():
        out, weights, cache = blk(x, attention_mask=mask)
        ref, _, _ = blk(x[1:2])
    assert tuple(out.shape) == (2, 4, 16)
    assert weights is None and cache is None
    assert out[0].abs().amax().item() == 0.0
    torch.testing.assert_close(out[1:2], ref, atol=1e-6, rtol=1e-5)


def test_masked_mixed_batch_partial_and_empty_rows():
    blk = _blk()
    torch.manual_seed(1)
    x = torch.randn(2, 6, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    with torch.no_grad():
        out, _, _ = blk(x, attention_mask=mask)
        ref, _, _ = blk(x[0:1, :3])
    assert tuple(out.shape) == (2, 6, 16)
    assert out[0, 3:].abs().amax().item() == 0.0
    assert out[1].abs().amax().item() == 0.0
    torch.testing.assert_close(out[0:1, :3], ref, atol=1e-6, rtol=1e-5)


def test_masked_all_empty_batch_collapses():
    blk = _blk()
    torch.manual_seed(1)
    x = torch.randn(2, 4, 16)
    with torch.no_grad():
        out, _, _ = blk(x, attention_mask=torch.zeros(2, 4))
    # Canonical contract: full-width zeros, not a (B, 0, D) collapse.
    assert tuple(out.shape) == (2, 4, 16)
    assert out.abs().amax().item() == 0.0


def test_masked_wrong_shape_raises():
    blk = _blk()
    with pytest.raises(ValueError, match=r"\[B, T\]"):
        blk(torch.randn(2, 4, 16), attention_mask=torch.ones(2, 3))


def test_masked_with_use_cache_raises():
    blk = _blk()
    with pytest.raises(ValueError, match="use_cache"):
        blk(torch.randn(1, 4, 16), attention_mask=torch.ones(1, 4), use_cache=True)
