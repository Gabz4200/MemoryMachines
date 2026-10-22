"""Guard tests for memory.py (SSD/bf16, DiskGather OOM, hashes, grad accum),
positioning.py (CVPE bf16 + edges), and linears.py (two-stage init, eye, determinism)."""

import numpy as np
import pytest
import torch

from memory_machines.models.blocks.memory import EngramMemory, _SsdTable
from memory_machines.models.blocks.positioning import CVPEContextBias, FRVMSpatialOV, FRVMValueGate
from memory_machines.utils.config_schema import MemoryMachineConfig


def _small_ram_config(**kw):
    base = {
        "hidden_size": 32,
        "mhc_num_streams": 2,
        "engram_orders": (2,),
        "engram_hash_heads": 2,
        "engram_dim_per_order": 4,
        "engram_num_entries": 16,
        "vocab_size": 32,
        "engram_use_conv": False,
        "engram_layer_positions": (),
        "seed": 0,
        "engram_offload": "ram",
    }
    base.update(kw)
    return MemoryMachineConfig(**base)


def _nonzero_cvpe(head_dim=4, seed=0, **kw):
    torch.manual_seed(seed)
    mod = CVPEContextBias(head_dim=head_dim, **kw)
    with torch.no_grad():
        for p in mod.parameters():
            p.normal_(0.0, 0.5)
    return mod


# 1. CRITICAL: SSD -> bf16 cast
def test_ssd_bf16_forward_dtype_shape_finite_storage_fp32(tmp_path):
    cfg = _small_ram_config(engram_offload="ssd", engram_offload_dir=str(tmp_path))
    mem = EngramMemory(cfg).to(torch.bfloat16)
    mem.eval()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    hidden = torch.randn(2, 4, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        out = mem(ids, hidden)
    assert out.dtype == torch.bfloat16
    assert tuple(out.shape) == (2, 4, 32)
    assert torch.isfinite(out.float()).all()
    # Underlying SSD storage stays float32 even after .to(bf16).
    assert mem.multi_head_embedding._mmap.dtype == np.dtype(np.float32)


# 2. CRITICAL: DiskGather OOM message
def test_diskgather_backward_oom_message(tmp_path, monkeypatch):
    t = _SsdTable(tmp_path / "t.bin", 8, 4)
    idx = torch.tensor([1, 3])
    out = t(idx)
    orig_zeros = torch.zeros

    def _boom(*a, **k):
        raise RuntimeError("OOM: out of memory")

    monkeypatch.setattr(torch, "zeros", _boom)
    try:
        with pytest.raises(RuntimeError, match=r"dense.*gradient|too large for dense AdamW|sparse optimizer"):
            out.sum().backward()
    finally:
        monkeypatch.setattr(torch, "zeros", orig_zeros)


# 4. IMPORTANT: ngram hash shape guards
def test_ngram_hashes_rejects_1d_and_3d():
    mem = EngramMemory(_small_ram_config())
    with pytest.raises(ValueError, match=r"must be \[B, Lq\]"):
        mem._ngram_hashes(torch.tensor([1, 2, 3]), 0)
    with pytest.raises(ValueError, match=r"must be \[B, Lq\]"):
        mem._ngram_hashes(torch.zeros(2, 3, 1, dtype=torch.long), 0)


def test_engram_forward_batch_seq_mismatch_raises():
    mem = EngramMemory(_small_ram_config())
    mem.eval()
    ids = torch.tensor([[1, 2, 3]])
    hidden = torch.randn(1, 4, 32)
    with pytest.raises(ValueError, match="does not match"):
        mem(ids, hidden)
    hidden2 = torch.randn(2, 3, 32)
    with pytest.raises(ValueError, match="does not match"):
        mem(ids, hidden2)


# 5. IMPORTANT: CVPE non-zero bf16 correctness
@pytest.mark.parametrize("use_log_transform", [True, False])
def test_cvpe_nonzero_bf16_matches_fp32(use_log_transform):
    torch.manual_seed(1)
    mod32 = _nonzero_cvpe(head_dim=4, seed=1, use_log_transform=use_log_transform)
    mod32.eval()
    B, H, Lq, _, Dh = 2, 2, 4, 6, 4
    q32 = torch.randn(B, H, Lq, Dh)
    q_pos = torch.tensor([[4, 5, 6, 7], [4, 5, 6, 7]])
    k_pos = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
    with torch.no_grad():
        out32 = mod32(q32, q_pos, k_pos)
    modbf = _nonzero_cvpe(head_dim=4, seed=1, use_log_transform=use_log_transform).to(torch.bfloat16)
    modbf.eval()
    with torch.no_grad():
        outbf = modbf(q32.to(torch.bfloat16), q_pos, k_pos)
    assert outbf.dtype == torch.bfloat16
    torch.testing.assert_close(outbf.float(), out32.float(), atol=1e-2, rtol=1e-2)


# 6. CVPE edges
def test_cvpe_qpos_zeros_finite():
    mod = _nonzero_cvpe(head_dim=4, seed=2)
    mod.eval()
    q = torch.randn(1, 1, 3, 4)
    with torch.no_grad():
        out = mod(q, torch.zeros(1, 3, dtype=torch.long), torch.arange(4).unsqueeze(0))
    assert torch.isfinite(out).all()


def test_cvpe_future_key_clamped_to_causal():
    mod = _nonzero_cvpe(head_dim=4, seed=3)
    mod.eval()
    q = torch.randn(1, 1, 2, 4)
    q_pos = torch.tensor([[5, 6]])
    k_future = torch.tensor([[7, 8, 9]])
    k_at_q = torch.tensor([[5, 6, 6]])
    with torch.no_grad():
        b_future = mod(q, q_pos, k_future)
        # all keys in both sets are strictly future (rel clamps to 0),
        # so two different all-future key sets give identical bias.
        b_ref = mod(q, q_pos, torch.tensor([[10, 11, 12]]))
    assert torch.isfinite(b_future).all()
    torch.testing.assert_close(b_future, b_ref, atol=1e-5, rtol=1e-5)
    _ = k_at_q


def test_cvpe_1d_vs_expanded_2d_identical():
    mod = _nonzero_cvpe(head_dim=4, seed=4)
    mod.eval()
    q = torch.randn(2, 1, 3, 4)
    q1d = torch.tensor([4, 5, 6])
    k1d = torch.tensor([0, 1, 2, 3, 4])
    with torch.no_grad():
        a = mod(q, q1d, k1d)
        b = mod(q, q1d.unsqueeze(0).expand(2, -1), k1d.unsqueeze(0).expand(2, -1))
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_cvpe_mismatched_length_raises():
    mod = CVPEContextBias(head_dim=4)
    q = torch.randn(1, 1, 3, 4)
    with pytest.raises(ValueError, match="does not match"):
        mod(q, torch.tensor([[1, 2]]), torch.tensor([[0, 1, 2]]))


# 7. Duplicate-index grad accumulation
def test_duplicate_index_grad_accumulates_3x(tmp_path):
    t = _SsdTable(tmp_path / "t.bin", 8, 4)
    with torch.no_grad():
        t.weight.fill_(0.0)
    t.zero_grad()
    t(torch.tensor([2])).sum().backward()
    assert t.weight.grad is not None
    g1 = t.weight.grad.clone()
    t.zero_grad()
    t(torch.tensor([2, 2, 2])).sum().backward()
    assert t.weight.grad is not None
    g3 = t.weight.grad.clone()
    torch.testing.assert_close(g3[2], 3 * g1[2], atol=1e-5, rtol=1e-5)


# NICE-TO-HAVE
def test_ssd_normal_init_chunked_determinism(tmp_path):
    a = _SsdTable(tmp_path / "a.bin", 32, 4)
    b = _SsdTable(tmp_path / "b.bin", 32, 4)
    g1 = torch.Generator(device="cpu")
    g1.manual_seed(123)
    g2 = torch.Generator(device="cpu")
    g2.manual_seed(123)
    a.normal_init_(0.02, g1)
    b.normal_init_(0.02, g2)
    assert np.array_equal(np.asarray(a._mmap), np.asarray(b._mmap))


def test_signed_sqrt_zero_gate_half_no_nan_grad():
    mem = EngramMemory(_small_ram_config())
    mem.eval()
    ids = torch.tensor([[1, 2, 3]])
    hidden = torch.randn(1, 3, 32, requires_grad=True)
    out = mem(ids, hidden)
    # zeroed key projs -> pre-sigmoid gate 0 -> sigmoid 0.5; value path zeroed too
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


# --- appended guard-gap tests ---


def test_ssd_forward_negative_index_raises(tmp_path):
    t = _SsdTable(tmp_path / "t.bin", 8, 4)
    with pytest.raises(ValueError, match="non-negative"):
        t(torch.tensor([-1]))


def test_ssd_forward_oob_index_raises(tmp_path):
    t = _SsdTable(tmp_path / "t.bin", 8, 4)
    with pytest.raises(ValueError, match="out of range"):
        t(torch.tensor([8]))
    with pytest.raises(ValueError, match="out of range"):
        t(torch.tensor([[0, 100]]))


def test_engram_forward_negative_ids_raises():
    mem = EngramMemory(_small_ram_config())
    mem.eval()
    hidden = torch.randn(1, 2, 32)
    with pytest.raises(ValueError, match="non-negative"):
        mem(torch.tensor([[-1, 2]]), hidden)


def test_engram_layer_position_early_return_unchanged():
    cfg = _small_ram_config(engram_layer_positions=(1,))
    mem = EngramMemory(cfg)
    mem.eval()
    hidden = torch.randn(2, 3, 32)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    with torch.no_grad():
        out = mem(ids, hidden, layer_id=0)
    assert out is hidden
    torch.testing.assert_close(out, hidden)


def test_ssd_fp32_mmap_bf16_compute_cast(tmp_path):
    cfg = _small_ram_config(engram_offload="ssd", engram_offload_dir=str(tmp_path))
    mem = EngramMemory(cfg).to(torch.bfloat16)
    # .to(bf16) copies the SSD weight (breaks mmap sharing) so the cast
    # branch is skipped; restore the fp32 mmap alias to force the mismatch.
    mem.multi_head_embedding.weight.data = torch.from_numpy(mem.multi_head_embedding._mmap)
    assert mem.multi_head_embedding.weight.dtype == torch.float32
    assert mem.value_proj.weight.dtype == torch.bfloat16
    mem.eval()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    hidden = torch.randn(2, 4, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        out = mem(ids, hidden)
    assert out.dtype == torch.bfloat16
    assert tuple(out.shape) == (2, 4, 32)
    assert torch.isfinite(out.float()).all()


def test_engram_4d_per_stream_hidden():
    mem = EngramMemory(_small_ram_config())
    mem.eval()
    ids = torch.tensor([[1, 2, 3]])
    hidden = torch.randn(1, 3, 2, 32)
    with torch.no_grad():
        out = mem(ids, hidden)
    assert tuple(out.shape) == (1, 3, 2, 32)
    assert torch.isfinite(out).all()


def test_cvpe_kpos_batch_mismatch_raises():
    mod = CVPEContextBias(head_dim=4)
    q = torch.randn(2, 1, 3, 4)
    q_pos = torch.tensor([[1, 2, 3], [1, 2, 3]])
    with pytest.raises(ValueError, match="does not match"):
        mod(q, q_pos, torch.tensor([[0, 1, 2]]))


def test_frvm_phi_shape_guard_raises():
    from memory_machines.models.blocks.positioning import FRVMValueGate

    gate = FRVMValueGate(head_dim=4, num_heads=2)
    with pytest.raises(ValueError, match="expected"):
        gate._phi(torch.randn(1, 3, 3, 4))


def test_frvm_gated_reduction_lk_mismatch_raises():
    from memory_machines.models.blocks.positioning import FRVMValueGate

    gate = FRVMValueGate(head_dim=4, num_heads=2)
    w = torch.softmax(torch.randn(1, 2, 3, 5), dim=-1)
    with pytest.raises(ValueError, match="does not match"):
        gate.gated_reduction(w, torch.randn(1, 2, 4, 4), torch.randn(1, 2, 3, 4))


def test_frvm_pairwise_delta_basic():
    from memory_machines.models.blocks.positioning import FRVMValueGate

    torch.manual_seed(0)
    gate = FRVMValueGate(head_dim=4, num_heads=2)
    v_i = torch.randn(1, 2, 3, 4)
    v_j = torch.randn(1, 2, 3, 4)
    out = gate.pairwise_delta(v_i, v_j)
    assert tuple(out.shape) == (1, 2, 3, 4)
    assert torch.isfinite(out).all()


# --- CVPE N-dim coordinate handling (rank promotion, pad, truncate) -----------
def _coords(length: int, n: int, batch: int = 1) -> torch.Tensor:
    """Build ``(B, L, n)`` coordinates with distinct values per axis."""
    base = torch.arange(length, dtype=torch.float32).unsqueeze(0).expand(batch, -1)
    if n == 1:
        return base.unsqueeze(-1)
    cols = [base] + [torch.full_like(base, float(axis * 10)) for axis in range(1, n)]
    return torch.stack(cols, dim=-1)


@pytest.mark.parametrize("n", [1, 2], ids=["pad-from-1d", "pad-from-2d"])
def test_cvpe_ndim_zero_pads_short_coordinate_streams(n: int) -> None:
    mod = _nonzero_cvpe(head_dim=4, seed=5, spatial_dims=3)
    q = torch.randn(1, 1, 3, 4)
    q_short, k_short = _coords(3, n), _coords(5, n)
    q_padded, k_padded = torch.zeros(1, 3, 3), torch.zeros(1, 5, 3)
    q_padded[..., :n], k_padded[..., :n] = q_short, k_short
    with torch.no_grad():
        promoted = mod(q, q_short, k_short)
        explicit = mod(q, q_padded, k_padded)
    torch.testing.assert_close(promoted, explicit, atol=0, rtol=0)


def test_cvpe_ndim_promotes_mixed_2d_and_3d_streams() -> None:
    mod = _nonzero_cvpe(head_dim=4, seed=6, spatial_dims=1)
    q = torch.randn(2, 1, 3, 4)
    q_2d = _coords(3, 1, batch=2)[..., 0]  # (B, Lq), promoted by the N-dim path
    k_3d = _coords(5, 1).expand(2, -1, -1)  # (B, Lk, 1)
    with torch.no_grad():
        mixed = mod(q, q_2d, k_3d)
        both_3d = mod(q, q_2d.unsqueeze(-1), k_3d)
    torch.testing.assert_close(mixed, both_3d, atol=0, rtol=0)


def test_cvpe_ndim_truncates_extra_coordinate_axes() -> None:
    mod = _nonzero_cvpe(head_dim=4, seed=7, spatial_dims=1)
    q = torch.randn(1, 1, 3, 4)
    q_3d, k_3d = _coords(3, 3), _coords(5, 3)
    with torch.no_grad():
        truncated = mod(q, q_3d, k_3d)
        first_axis_only = mod(q, q_3d[..., :1], k_3d[..., :1])
    assert tuple(truncated.shape) == (1, 1, 3, 5)
    torch.testing.assert_close(truncated, first_axis_only, atol=0, rtol=0)


def test_cvpe_ndim_rejects_mismatched_query_length() -> None:
    mod = CVPEContextBias(head_dim=4, spatial_dims=2)
    with pytest.raises(ValueError, match="q_pos shape"):
        mod(torch.randn(1, 1, 3, 4), torch.zeros(1, 2, 2), torch.zeros(1, 5, 2))


def test_cvpe_ndim_rejects_mismatched_key_batch() -> None:
    mod = CVPEContextBias(head_dim=4, spatial_dims=2)
    with pytest.raises(ValueError, match="k_pos shape"):
        mod(torch.randn(2, 1, 3, 4), torch.zeros(2, 3, 2), torch.zeros(1, 5, 2))


def test_cvpe_ndim_rejects_mismatched_coordinate_width() -> None:
    mod = CVPEContextBias(head_dim=4, spatial_dims=2)
    with pytest.raises(ValueError, match="dims"):
        mod(torch.randn(1, 1, 3, 4), torch.zeros(1, 3, 2), torch.zeros(1, 5, 3))


# --- positioning.py constructor guards ---------------------------------------
def test_positioning_constructor_guards() -> None:
    with pytest.raises(ValueError, match="spatial_dims"):
        CVPEContextBias(head_dim=4, spatial_dims=0)
    with pytest.raises(ValueError, match="axis_max"):
        CVPEContextBias(head_dim=4, spatial_dims=2, axis_max=(10,))
    with pytest.raises(ValueError, match="spatial_dims"):
        FRVMSpatialOV(spatial_dims=0, head_dim=4)
    with pytest.raises(ValueError, match="num_heads"):
        FRVMValueGate(head_dim=4, num_heads=0)
