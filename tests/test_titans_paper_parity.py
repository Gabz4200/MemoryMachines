"""Titans (arXiv:2501.00663) paper-parity tests: exact update math + mechanisms.

Each test ties to a paper equation. All tests are deterministic (fixed seeds,
no sampling noise) and use tiny configs so the sequential reference runs fast.
"""

import math

import pytest
import torch

from memory_machines.models.blocks.fast_weights import (
    TitansFastWeightsMemory,
    TitansMAGBlock,
)


def _tiny(**kw):
    base = {"hidden_size": 32, "num_heads": 4, "memory_depth": 1, "num_persistent_tokens": 0, "use_short_conv": False}
    base.update(kw)
    return base


def _fix_gates(mem, eta=0.5, theta=0.5, alpha=0.5):
    """Pin data-dependent gates to constants; return the exact gate values used."""
    actual = []
    with torch.no_grad():
        for proj, val in ((mem.eta_proj, eta), (mem.theta_proj, theta), (mem.alpha_proj, alpha)):
            clipped = min(max(val, 1e-6), 1.0 - 1e-6)
            proj.weight.zero_()
            proj.bias.fill_(math.log(clipped / (1.0 - clipped)))
            actual.append(torch.sigmoid(proj.bias.detach()).mean().item())
    return tuple(actual)  # (eta, theta, alpha)


def _hand_roll(mem, x, eta, theta, alpha, use_momentum):
    """Hand-derived Eq. 9/10/13/14 rollout with plain torch ops (no module internals)."""
    heads = mem.num_heads
    qf, kf, vf, _ = mem.project_qkv(x)
    w = mem.fast_init[0].detach().expand(x.shape[0], heads, *mem.fast_init[0].shape).contiguous()
    s = torch.zeros_like(w)
    ys, losses = [], []
    for t in range(x.shape[1]):
        leaf = w.detach().clone().requires_grad_(True)
        pred = torch.einsum("...hd,...hde->...he", kf[:, t], leaf)
        loss = ((pred - vf[:, t]) ** 2).mean()
        (g,) = torch.autograd.grad(loss, leaf)
        s = eta * s - theta * g if use_momentum else -theta * g
        w = (1.0 - alpha) * w + s
        ys.append(torch.einsum("...hd,...hde->...he", qf[:, t], w))
        losses.append(loss.detach().reshape(x.shape[0], 1))
    return torch.stack(ys, dim=1), torch.cat(losses, dim=1), w, s


@pytest.mark.parametrize("use_momentum", [True, False])
def test_inner_update_matches_hand_derived_equations(use_momentum):
    """Eq. 9/10/13/14: module trajectory == hand-rolled surprise/momentum/decay."""
    torch.manual_seed(0)
    mem = TitansFastWeightsMemory(**_tiny(use_momentum=use_momentum)).eval()
    e, t, a = _fix_gates(mem)
    torch.manual_seed(1)
    x = torch.randn(1, 5, 32)
    out, state = mem(x, output_final_state=True)
    assert state is not None
    y_hand, loss_hand, w_hand, s_hand = _hand_roll(mem, x, e, t, a, use_momentum)
    # Pre-update associative surprise (Eq. 12) matches per position.
    torch.testing.assert_close(mem.associative_losses(x), loss_hand, atol=1e-6, rtol=1e-5)
    # Final fast weights and momentum buffer match the hand rollout.
    torch.testing.assert_close(state["fast_weights"][0], w_hand, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(state["momentum"][0], s_hand, atol=1e-6, rtol=1e-5)
    # Full output pipeline (retrieval Eq. 15 + gate/norm/proj + residual).
    mem_out = y_hand.reshape(1, 5, 32)
    expected = mem.o_proj(mem.out_norm(mem_out) * torch.nn.functional.silu(mem.gate_proj(x))) + x
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)


def test_eta_one_accumulates_past_surprise():
    """Eq. 10 with eta -> 1: momentum keeps past surprise (vs eta = 0, Gated-DeltaNet-like)."""
    torch.manual_seed(0)
    mem = TitansFastWeightsMemory(**_tiny()).eval()
    e, t, a = _fix_gates(mem, eta=0.9999, theta=0.5, alpha=0.0)
    torch.manual_seed(1)
    x = torch.randn(1, 1, 32).repeat(1, 3, 1)
    _, state = mem(x, output_final_state=True)
    assert state is not None
    s_final = state["momentum"][0]
    # With eta ~ 1 and near-constant gradients, surprise accumulates over steps.
    _, _, _, s_hand = _hand_roll(mem, x, e, t, a, True)
    torch.testing.assert_close(s_final, s_hand, atol=1e-3, rtol=1e-3)
    _, state1 = mem(x[:, :1], output_final_state=True)
    assert state1 is not None
    assert s_final.abs().amax().item() > state1["momentum"][0].abs().amax().item()


def test_forget_gate_clears_distant_past():
    """Eq. 13: alpha -> 1 erases memory; alpha -> 0 keeps it. Gates stay in [0, 1]."""
    torch.manual_seed(0)
    keep = TitansFastWeightsMemory(**_tiny(use_momentum=False)).eval()
    torch.manual_seed(0)
    clear = TitansFastWeightsMemory(**_tiny(use_momentum=False)).eval()
    _fix_gates(keep, alpha=0.0)
    _fix_gates(clear, alpha=1.0 - 1e-3)
    torch.manual_seed(1)
    x = torch.randn(1, 6, 32)
    xp = x.clone()
    xp[0, 0] += 10.0
    with torch.no_grad():
        for m in (keep, clear):
            gates = torch.sigmoid(m.alpha_proj(x))
            assert bool(((gates > 0) & (gates < 1)).all())  # paper: alpha_t in [0, 1]
        d_keep = (keep(xp)[0] - keep(x)[0]).abs().amax(-1)[0, -1].item()
        d_clear = (clear(xp)[0] - clear(x)[0]).abs().amax(-1)[0, -1].item()
    assert d_clear < 1e-6, d_clear
    assert d_keep > 1e-3, d_keep


def test_bptt_through_decay_matches_finite_differences():
    """Outer loop (§3.1) differentiates the full Eq. 13 trajectory, decay path included."""
    torch.manual_seed(0)
    mem = TitansFastWeightsMemory(**_tiny(memory_depth=1)).double().train()
    torch.manual_seed(1)
    x = torch.randn(1, 2, 32, dtype=torch.float64)
    out, _ = mem(x)
    analytic = torch.autograd.grad(out.sum(), mem.fast_init[0])[0]
    eps, idxs = 1e-6, [(1, 3), (0, 0), (5, 7)]
    for i, j in idxs:
        with torch.no_grad():
            mem.fast_init[0][i, j] += eps
            fp, _ = mem(x)
            mem.fast_init[0][i, j] -= 2 * eps
            fm, _ = mem(x)
            mem.fast_init[0][i, j] += eps
        numeric = (fp.sum() - fm.sum()) / (2 * eps)
        torch.testing.assert_close(analytic[i, j], numeric, atol=1e-6, rtol=1e-4)


def test_train_mode_all_params_get_grads():
    """Meta setup (§3.1): inner loop optimizes M, outer loop reaches every parameter."""
    torch.manual_seed(0)
    mem = TitansFastWeightsMemory(**_tiny(num_persistent_tokens=2, memory_depth=2)).train()
    x = torch.randn(1, 4, 32)
    mem(x)[0].sum().backward()
    for name, p in mem.named_parameters():
        assert p.grad is not None, name
        assert bool(torch.isfinite(p.grad).all()), name


def test_prior_state_untouched_by_continuation():
    """Eq. 15 (M*): retrieval/update returns new state; cached tensors are never mutated."""
    torch.manual_seed(0)
    mem = TitansFastWeightsMemory(hidden_size=32, num_heads=4, memory_depth=2, num_persistent_tokens=2).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 6, 32)
    with torch.no_grad():
        assert torch.equal(mem(x)[0], mem(x)[0])  # stateless wrt module
        _, state = mem(x[:, :3], output_final_state=True)
        frozen = [[w.clone() for w in state["fast_weights"]], [w.clone() for w in state["momentum"]]]
        mem(x[:, 3:], state=state)
        for old, new in zip(frozen[0] + frozen[1], state["fast_weights"] + state["momentum"]):
            assert torch.equal(old, new)


def test_zero_output_projection_recovers_residual():
    """§4.4: every block carries a residual connection."""
    torch.manual_seed(0)
    mem = TitansFastWeightsMemory(**_tiny()).eval()
    with torch.no_grad():
        mem.o_proj.weight.zero_()
    x = torch.randn(1, 5, 32)
    torch.testing.assert_close(mem(x)[0], x, atol=0.0, rtol=0.0)


def test_mag_gate_bounds_and_residual():
    """Eq. 28: MAG fuses normalized branches through a learned gate + residual."""
    torch.manual_seed(0)
    blk = TitansMAGBlock(hidden_size=32, num_heads=4, memory_depth=1, num_persistent_tokens=2).eval()
    x = torch.randn(1, 5, 32)
    gate = torch.sigmoid(blk.fuse(x))
    assert bool(((gate > 0) & (gate < 1)).all())
    with torch.no_grad():
        blk.o_proj.weight.zero_()
    torch.testing.assert_close(blk(x), x, atol=0.0, rtol=0.0)


def test_deep_memory_fits_nonlinear_map_better():
    """§5.5/Hornik: L_M = 2 fast MLP reaches lower surprise than linear memory."""
    torch.manual_seed(0)
    lin = TitansFastWeightsMemory(**_tiny(memory_depth=1, use_momentum=False, use_weight_decay=False)).eval()
    torch.manual_seed(0)
    deep = TitansFastWeightsMemory(**_tiny(memory_depth=2, use_momentum=False, use_weight_decay=False)).eval()
    with torch.no_grad():  # isolate memory depth: share every outer projection
        for attr in ("q_proj", "k_proj", "v_proj", "eta_proj", "theta_proj", "alpha_proj"):
            getattr(deep, attr).load_state_dict(getattr(lin, attr).state_dict())
    torch.manual_seed(1)
    x = torch.randn(1, 16, 32)

    def tail(m):
        return m.associative_losses(x)[0, -4:].mean().item()

    assert tail(deep) < tail(lin)


def test_error_paths():
    """Invalid configs and shapes fail loudly."""
    with pytest.raises(ValueError, match="divisible"):
        TitansFastWeightsMemory(hidden_size=30, num_heads=4)
    with pytest.raises(ValueError, match=">= 1"):
        TitansFastWeightsMemory(hidden_size=32, num_heads=4, memory_depth=0)
    with pytest.raises(ValueError, match="non-negative"):
        TitansFastWeightsMemory(hidden_size=32, num_heads=4, num_persistent_tokens=-1)
    mem = TitansFastWeightsMemory(**_tiny()).eval()
    with pytest.raises(ValueError, match="chunk_size"):
        mem(torch.randn(1, 3, 32), chunk_size=0)
    with pytest.raises(ValueError, match=r"\[B, T, D="):
        mem(torch.randn(1, 3, 31))
    blk = TitansMAGBlock(hidden_size=32, num_heads=4).eval()
    with pytest.raises(ValueError, match="must match"):
        blk(torch.randn(1, 3, 32), short_term=torch.randn(1, 4, 32))
    noconv = TitansFastWeightsMemory(**_tiny(use_short_conv=False)).eval()
    with pytest.raises(ValueError, match="conv_state"):
        noconv.project_qkv(torch.randn(1, 2, 32), conv_state=(torch.zeros(1, 2, 3),) * 3)
