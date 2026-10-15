"""Guard tests for recurrent-block review gaps (gdn2 / Titan mixer / ShortConv).

Locks in validation and parity behavior NOT covered by
``test_gdn2*.py``, ``test_titans_*.py``, ``test_hope_*.py``,
``test_titan_masked_mixed.py``, ``test_causal_conv1d.py``,
``test_liv_causal_conv.py``, or ``test_regression_fixes.py``:

- ``cu_seqlens`` shape/dtype/monotonicity guards at the ``_check_shapes``,
  chunk-forward, and ``SelfReferentialTitanMixer._forward_packed`` levels.
- Zero-length packed segments (``cu=[0, 0, 4]``) stay finite and match a
  fresh-segment run.
- Per-call ``chunk_size`` overrides validate, agree across widths, and never
  mutate module state.
- ``ShortConvolution`` single-token decode gate: ``no_grad`` routes to
  ``step()`` with dense-identical numerics; the grad path stays
  differentiable (``weight.grad`` accumulates).
- Fully-masked ``B=1`` Titan mixer rows return full-width zeros in the input
  dtype.
"""

import pytest
import torch

from memory_machines.models.blocks import gdn2 as G
from memory_machines.models.blocks.causal_conv1d import ShortConvolution
from memory_machines.models.blocks.fast_weights import SelfReferentialTitanMixer
from memory_machines.utils.config_schema import MemoryMachineConfig


def _gdn2_inputs(B=1, T=4, H=1, HV=1, K=4, V=4, seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, dtype=dtype)
    k = torch.randn(B, T, H, K, dtype=dtype)
    v = torch.randn(B, T, HV, V, dtype=dtype)
    g = -torch.rand(B, T, H, K, dtype=dtype)
    b = torch.rand(B, T, HV, K, dtype=dtype)
    w = torch.rand(B, T, HV, V, dtype=dtype)
    return q, k, v, g, b, w


def _titan_mixer(**kw):
    base = {"hidden_size": 16, "num_attention_heads": 2, "num_key_value_heads": 2, "csa_num_output_groups": 2}
    args = {"memory_depth": 1, "use_short_conv": False}
    args.update(kw)
    torch.manual_seed(0)
    return SelfReferentialTitanMixer(MemoryMachineConfig(**base), **args).eval()


def _gdn2_block(**kw):
    args = {"hidden_size": 16, "num_heads": 2, "head_dim_k": 4, "use_short_conv": False}
    args.update(kw)
    torch.manual_seed(0)
    return G.GatedDeltaNet2(**args).eval()


# --- 1. gdn2 _check_shapes cu gaps -------------------------------------------


def test_check_shapes_2d_cu_raises():
    q, k, v, g, b, w = _gdn2_inputs()
    with pytest.raises(ValueError, match="must be 1D"):
        G._check_shapes(q, k, v, g, b, w, torch.tensor([[0, 4]]), None)


def test_check_shapes_float_cu_raises():
    q, k, v, g, b, w = _gdn2_inputs()
    with pytest.raises(ValueError, match="integer dtype"):
        G._check_shapes(q, k, v, g, b, w, torch.tensor([0.0, 4.0]), None)


def test_check_shapes_decreasing_cu_raises():
    q, k, v, g, b, w = _gdn2_inputs()
    with pytest.raises(ValueError, match="non-decreasing"):
        G._check_shapes(q, k, v, g, b, w, torch.tensor([0, 3, 2, 4]), None)


def test_check_shapes_empty_cu_raises():
    q, k, v, g, b, w = _gdn2_inputs()
    with pytest.raises(ValueError, match="must start at 0"):
        G._check_shapes(q, k, v, g, b, w, torch.tensor([], dtype=torch.long), None)


# --- 1b. same 4 gaps at gated_delta_rule_2_chunk_forward level ----------------


def test_chunk_forward_2d_cu_raises():
    q, k, v, g, b, w = _gdn2_inputs()
    with pytest.raises(ValueError, match="must be 1D"):
        G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, cu_seqlens=torch.tensor([[0, 4]]))


def test_chunk_forward_float_cu_raises():
    q, k, v, g, b, w = _gdn2_inputs()
    with pytest.raises(ValueError, match="integer dtype"):
        G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, cu_seqlens=torch.tensor([0.0, 4.0]))


def test_chunk_forward_decreasing_cu_raises():
    q, k, v, g, b, w = _gdn2_inputs()
    with pytest.raises(ValueError, match="non-decreasing"):
        G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, cu_seqlens=torch.tensor([0, 3, 2, 4]))


def test_chunk_forward_empty_cu_raises():
    q, k, v, g, b, w = _gdn2_inputs()
    with pytest.raises(ValueError, match="must start at 0"):
        G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, cu_seqlens=torch.tensor([], dtype=torch.long))


# --- 2. TitanMixer _forward_packed same 4 gaps --------------------------------


def test_titan_packed_2d_cu_raises():
    mixer = _titan_mixer()
    x = torch.randn(1, 4, 16)
    with pytest.raises(ValueError, match="1D starting at 0"):
        mixer(x, cu_seqlens=torch.tensor([[0, 4]]))


def test_titan_packed_float_cu_raises():
    # NOTE: unlike gdn2._check_shapes, the mixer has no integer-dtype guard,
    # so a float cu surfaces as a TypeError at segment slicing. Locked in so
    # a future dtype guard (ValueError) shows up as an intentional change.
    mixer = _titan_mixer()
    x = torch.randn(1, 4, 16)
    with pytest.raises(TypeError):
        mixer(x, cu_seqlens=torch.tensor([0.0, 4.0]))


def test_titan_packed_decreasing_cu_raises():
    mixer = _titan_mixer()
    x = torch.randn(1, 4, 16)
    with pytest.raises(ValueError, match="non-decreasing"):
        mixer(x, cu_seqlens=torch.tensor([0, 3, 2, 4]))


def test_titan_packed_empty_cu_raises():
    mixer = _titan_mixer()
    x = torch.randn(1, 4, 16)
    with pytest.raises(ValueError, match="1D starting at 0"):
        mixer(x, cu_seqlens=torch.tensor([], dtype=torch.long))


# --- 3. gdn2 zero-length segment ----------------------------------------------


def test_gdn2_zero_length_segment_matches_fresh_tail():
    q, k, v, g, b, w = _gdn2_inputs()
    cu = torch.tensor([0, 0, 4])
    o, _ = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, cu_seqlens=cu, chunk_size=2)
    assert tuple(o.shape) == (1, 4, 1, 4)
    assert torch.isfinite(o).all()
    ref, _ = G.gated_delta_rule_2_chunk_forward(q, k, v, g, b, w, chunk_size=2)
    torch.testing.assert_close(o, ref, atol=1e-10, rtol=0)


# --- 4. GatedDeltaNet2.forward chunk_size override -----------------------------


def test_gdn2_forward_chunk_size_zero_raises():
    blk = _gdn2_block()
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        blk(torch.randn(1, 5, 16), chunk_size=0)


def test_gdn2_forward_chunk_size_negative_raises():
    blk = _gdn2_block()
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        blk(torch.randn(1, 5, 16), chunk_size=-1)


def test_gdn2_forward_chunk_size_parity_and_no_mutation():
    blk = _gdn2_block()
    torch.manual_seed(1)
    x = torch.randn(1, 5, 16)
    before = blk.chunk_size
    with torch.no_grad():
        narrow, _, _ = blk(x, chunk_size=2)
        wide, _, _ = blk(x, chunk_size=64)
    torch.testing.assert_close(narrow, wide, atol=1e-5, rtol=1e-4)
    assert blk.chunk_size == before == 64


# --- 5. ShortConv decode gate --------------------------------------------------


def test_shortconv_decode_gate_matches_step_and_dense():
    torch.manual_seed(0)
    conv = ShortConvolution(hidden_size=4, kernel_size=3).eval()
    torch.manual_seed(1)
    x = torch.randn(2, 1, 4)
    with torch.no_grad():
        routed, _ = conv(x)
        stepped, _ = conv.step(x, None, None)
    torch.testing.assert_close(routed, stepped, atol=1e-10, rtol=0)
    with torch.enable_grad():
        dense, _ = conv(x)
    torch.testing.assert_close(routed, dense, atol=1e-6, rtol=1e-5)


def test_shortconv_single_token_grad_path_accumulates_weight_grad():
    torch.manual_seed(0)
    conv = ShortConvolution(hidden_size=4, kernel_size=3).eval()
    x = torch.randn(2, 1, 4, requires_grad=True)
    y, _ = conv(x)
    assert y.requires_grad
    y.sum().backward()
    assert conv.weight.grad is not None
    assert bool((conv.weight.grad != 0).any())


# --- 6. fast_weights B=1 all-masked dtype/shape --------------------------------


def test_titan_masked_single_row_zeros_and_dtype():
    mixer = _titan_mixer()
    torch.manual_seed(1)
    x = torch.randn(1, 4, 16)
    with torch.no_grad():
        out, _, _ = mixer(x, attention_mask=torch.zeros(1, 4))
    assert tuple(out.shape) == (1, 4, 16)
    assert out.dtype == x.dtype
    assert out.abs().amax().item() == 0.0


# --- NICE_TO_HAVE --------------------------------------------------------------


def test_chunk_forward_packed_initial_state_n_mismatch_raises():
    q, k, v, g, b, w = _gdn2_inputs()
    with pytest.raises(ValueError, match="initial_state must be"):
        G.gated_delta_rule_2_chunk_forward(
            q,
            k,
            v,
            g,
            b,
            w,
            cu_seqlens=torch.tensor([0, 2, 4]),
            initial_state=torch.zeros(1, 1, 4, 4),
            chunk_size=2,
        )


def test_titan_packed_chunk_size_1_parity():
    mixer = _titan_mixer()
    torch.manual_seed(1)
    x = torch.randn(1, 5, 16)
    cu = torch.tensor([0, 2, 5])
    with torch.no_grad():
        one, _, _ = mixer(x, cu_seqlens=cu, chunk_size=1)
        full, _, _ = mixer(x, cu_seqlens=cu)
    torch.testing.assert_close(one, full, atol=1e-5, rtol=1e-4)


def test_titan_failed_chunk_size_zero_leaves_state_untouched():
    mixer = _titan_mixer()
    before = mixer.chunk_size
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        mixer(torch.randn(1, 4, 16), chunk_size=0)
    assert mixer.chunk_size == before
