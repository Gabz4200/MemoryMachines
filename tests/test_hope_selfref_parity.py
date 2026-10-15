"""HOPE self-referential memory (arXiv:2512.24695 Sec. 8.1) paper-parity tests.

Each test ties to a paper equation. Deterministic (fixed seeds) and tiny
(D=8, H=2) with `use_short_conv=False`. References are hand-rolled with plain
torch ops (einsum/matmul), never the module's update internals.
"""

import pytest
import torch
import torch.nn.functional as F

from memory_machines.models.blocks.fast_weights import (
    HOPE_BOXES,
    SelfReferentialTitanMemory,
    _sr_mlp,
)

D, H = 8, 2
KD = D // H


def _mk(**kw):
    base = {"hidden_size": D, "num_heads": H, "memory_depth": 1, "objective": "l2", "use_short_conv": False}
    base.update(kw)
    torch.manual_seed(0)
    return SelfReferentialTitanMemory(**base).eval()


def _l2(x, eps=1e-6):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def _q_hand(mem, x):
    """Static query path: x @ Wq, reshaped per head, L2-normalized."""
    return _l2((x @ mem.q_proj.weight.T).view(x.shape[0], x.shape[1], H, KD), mem.norm_eps)


def _head_out(mem, mem_out, x):
    """Fixed output head with plain ops: RMSNorm + SiLU gate + proj + residual."""
    g = F.silu(x @ mem.gate_proj.weight.T + mem.gate_proj.bias)
    n = F.rms_norm(mem_out, (D,), mem.out_norm.weight, mem.out_norm.eps)
    return (n * g) @ mem.o_proj.weight.T + x


def _lin_step(mem, x):
    """Plain-op one-step refs (depth 1): {box: (W_new, vhat, dgd, grad)} + (eta, alpha)."""
    init = {b: mem.fast_init[b][0].detach() for b in HOPE_BOXES}
    xh = x.view(x.shape[0], x.shape[1], H, KD)[:, 0]
    kh = _l2(xh @ init["k"], mem.norm_eps)
    vh = xh @ init["v"]
    eta = torch.sigmoid((xh @ init["eta"]).mean(-1))
    alp = torch.sigmoid((xh @ init["alpha"]).mean(-1))
    out = {}
    for b in HOPE_BOXES:
        vhat = vh @ init[b]
        pred = kh @ init[b]
        kk = torch.einsum("...d,...e->...de", kh, kh)
        dgd = torch.einsum("...de,...ev->...dv", kk, init[b])
        if mem.objective == "l2":
            grad = torch.einsum("...d,...v->...dv", kh, pred - vhat)
        else:
            grad = -torch.einsum("...d,...v->...dv", kh, vhat)
        sh = (eta.shape[0], eta.shape[1], 1, 1)
        out[b] = (alp.view(*sh) * init[b] - eta.view(*sh) * dgd - eta.view(*sh) * grad, vhat, dgd, grad)
    return eta, alp, out


def test_pre_update_retrieval_eq86():
    """Eq. 86: retrieval uses the PRE-update memory; output matches the hand path."""
    mem = _mk()
    torch.manual_seed(1)
    x = torch.randn(1, 1, D)
    with torch.no_grad():
        mem.fast_init["memory"][0].copy_(2.0 * torch.eye(KD))
        out, _ = mem(x)
    # Arrange: known pre-update weights; Act above; Assert hand path.
    w0 = (2.0 * torch.eye(KD)).expand(1, H, KD, KD)
    y = torch.einsum("bthd,bhde->bhte", _q_hand(mem, x), w0)
    torch.testing.assert_close(out, _head_out(mem, y.reshape(1, 1, D), x), atol=1e-6, rtol=1e-5)


def test_linear_l2_matches_eq93():
    """Eq. 93: depth-1 L2 closed form (alpha*W - eta*dgd - eta*grad) matches state."""
    mem = _mk(objective="l2")
    torch.manual_seed(1)
    x = torch.randn(1, 1, D)
    with torch.no_grad():
        _, state = mem(x, output_final_state=True)
    _, _, ref = _lin_step(mem, x)
    torch.testing.assert_close(state["fast_weights"]["memory"][0], ref["memory"][0], atol=1e-6, rtol=1e-5)


def test_linear_dot_matches_eq92():
    """Eq. 92: dot objective uses grad=-outer(k, vhat); the dgd term is load-bearing."""
    mem = _mk(objective="dot")
    torch.manual_seed(1)
    x = torch.randn(1, 1, D)
    with torch.no_grad():
        _, state = mem(x, output_final_state=True)
    eta, alp, ref = _lin_step(mem, x)
    w_new, _, _dgd, grad = ref["memory"]
    torch.testing.assert_close(state["fast_weights"]["memory"][0], w_new, atol=1e-6, rtol=1e-5)
    # Dropping the decay-gradient term must NOT match.
    sh = (eta.shape[0], eta.shape[1], 1, 1)
    nodgd = alp.view(*sh) * mem.fast_init["memory"][0].detach() - eta.view(*sh) * grad
    assert (state["fast_weights"]["memory"][0] - nodgd).abs().amax().item() > 1e-4


def test_vhat_generated_from_v_eq84():
    """Eq. 84/87: each box's target flows through v (vhat=Mbox(vh)), verified via its update."""
    mem = _mk(objective="l2")
    torch.manual_seed(1)
    x = torch.randn(1, 1, D)
    with torch.no_grad():
        _, state = mem(x, output_final_state=True)
    _, _, ref = _lin_step(mem, x)
    w_new, vhat, _, _ = ref["memory"]
    torch.testing.assert_close(state["fast_weights"]["memory"][0], w_new, atol=1e-6, rtol=1e-5)
    # A target bypassing the v-box (raw xh) would give a different update.
    init = {b: mem.fast_init[b][0].detach() for b in HOPE_BOXES}
    xh = x.view(1, 1, H, KD)[:, 0]
    kh = _l2(xh @ init["k"], mem.norm_eps)
    wrong = xh @ init["memory"]
    assert (vhat - wrong).abs().amax().item() > 1e-5
    pred = kh @ init["memory"]
    grad_wrong = torch.einsum("...d,...v->...dv", kh, pred - wrong)
    assert (ref["memory"][3] - grad_wrong).abs().amax().item() > 1e-6


def test_q_static_kv_adaptive():
    """Sec. 8.1: q is a static projection (suffix-identical under prefix shift); k/v adapt."""
    mem = _mk()
    torch.manual_seed(1)
    x1 = torch.randn(1, 4, D)
    x2 = x1.clone()
    x2[:, 0] += 5.0
    with torch.no_grad():
        q1, _, _, _ = mem.project_qkv(x1)
        q2, _, _, _ = mem.project_qkv(x2)
        o1, _ = mem(x1)
        o2, _ = mem(x2)
    assert torch.equal(q1[:, 1:], q2[:, 1:])
    assert (o1[:, 1:] - o2[:, 1:]).abs().amax().item() > 1e-6


def test_deep_update_is_alpha_decayed_grad():
    """Eq. 88 (deep): W <- alpha*W - eta*autograd-grad of the box loss."""
    mem = _mk(memory_depth=2)
    torch.manual_seed(1)
    x = torch.randn(1, 1, D)
    with torch.no_grad():
        _, state = mem(x, output_final_state=True)

    def rmlp(xh, Ws):
        h = xh
        for i, w in enumerate(Ws):
            h = torch.einsum("...hd,...hde->...he", h, w)
            if i < len(Ws) - 1:
                h = F.silu(h)
        return xh + h

    leaves = {b: [w.detach().expand(1, H, *w.shape).clone() for w in mem.fast_init[b]] for b in HOPE_BOXES}
    xh = x.view(1, 1, H, KD)[:, 0]
    # The module builds kh/vh/scalars/vhat under no-grad (detached constants);
    # only pred -> loss -> grad runs in the inner autograd graph.
    with torch.no_grad():
        kh = _l2(rmlp(xh, leaves["k"]), mem.norm_eps)
        vh = rmlp(xh, leaves["v"])
        eta = torch.sigmoid(rmlp(xh, leaves["eta"]).mean(-1))
        alp = torch.sigmoid(rmlp(xh, leaves["alpha"]).mean(-1))
        vhat = {b: rmlp(vh, leaves[b]) for b in HOPE_BOXES}
    req = {b: [w.clone().requires_grad_(True) for w in leaves[b]] for b in HOPE_BOXES}
    for box in HOPE_BOXES:
        pred_b = rmlp(kh, req[box])
        (g0, g1) = torch.autograd.grad(((pred_b - vhat[box]) ** 2).mean(), req[box])
        for i, (w, g) in enumerate(zip(req[box], (g0, g1))):
            exp = alp[..., None, None] * w.detach() - eta[..., None, None] * g
            torch.testing.assert_close(state["fast_weights"][box][i], exp, atol=1e-5, rtol=1e-4)


def test_eta_alpha_freeze_and_range():
    """Saturated gates (alpha~1, eta~0) freeze fast weights; scalars stay in (0, 1)."""
    mem = _mk()
    with torch.no_grad():
        for w in mem.fast_init["alpha"]:
            w.fill_(10.0)
        for w in mem.fast_init["eta"]:
            w.fill_(-10.0)
    torch.manual_seed(1)
    x = torch.rand(1, 3, D)  # positive inputs saturate the constant-sign projections
    with torch.no_grad():
        _, state = mem(x, output_final_state=True)
    assert state is not None
    for box in HOPE_BOXES:
        torch.testing.assert_close(
            state["fast_weights"][box][0],
            mem.fast_init[box][0].detach().expand(1, H, KD, KD),
            atol=1e-4,
            rtol=1e-4,
        )
    s = mem._box_scalars(torch.randn(2, 3, 5))
    assert bool(((s > 0) & (s < 1)).all())


def test_residual_mlp_identity_at_zero_init():
    """Eq. 89: zero-init residual MLP is the identity; the linear path has no residual."""
    mem = _mk(memory_depth=2)
    with torch.no_grad():
        for plist in mem.fast_init.values():
            for w in plist:
                w.zero_()
    xh = torch.randn(2, H, KD)
    Ws = [w.expand(2, H, *w.shape) for w in mem.fast_init["memory"]]
    torch.testing.assert_close(_sr_mlp(xh, Ws), xh, atol=0.0, rtol=0.0)
    lin = _mk(memory_depth=1)
    with torch.no_grad():
        for plist in lin.fast_init.values():
            for w in plist:
                w.zero_()
    Wl = [w.expand(2, H, *w.shape) for w in lin.fast_init["memory"]]
    got = _sr_mlp(xh, Wl)
    assert bool((got == 0).all()) and not bool(torch.equal(got, xh))


def test_hope_boxes_contract():
    """Six boxes exist; the q box never feeds retrieval (out[:,0] immune to its init)."""
    mem = _mk()
    assert tuple(mem.fast_init.keys()) == ("k", "v", "q", "eta", "alpha", "memory")
    torch.manual_seed(1)
    x = torch.randn(1, 3, D)
    with torch.no_grad():
        o1, s1 = mem(x, output_final_state=True)
        assert s1 is not None
        mem.fast_init["q"][0].add_(torch.randn_like(mem.fast_init["q"][0]))
        o2, s2 = mem(x, output_final_state=True)
        assert s2 is not None
    assert torch.equal(o1[:, 0], o2[:, 0])
    assert not bool(torch.equal(s1["fast_weights"]["q"][0], s2["fast_weights"]["q"][0]))
    assert bool(torch.equal(s1["fast_weights"]["memory"][0], s2["fast_weights"]["memory"][0]))


def test_selfref_error_paths():
    """Invalid configs and shapes fail loudly."""
    with pytest.raises(ValueError, match="objective"):
        SelfReferentialTitanMemory(hidden_size=D, num_heads=H, objective="huber")
    with pytest.raises(ValueError, match=">= 1"):
        SelfReferentialTitanMemory(hidden_size=D, num_heads=H, memory_depth=0)
    with pytest.raises(ValueError, match="divisible"):
        SelfReferentialTitanMemory(hidden_size=7, num_heads=2)
    mem = _mk()
    with pytest.raises(ValueError, match=r"\[B, T, D="):
        mem(torch.randn(1, 3, D + 1))
    with pytest.raises(ValueError, match="chunk_size"):
        mem(torch.randn(1, 3, D), chunk_size=0)
    with pytest.raises(ValueError, match="conv_state"):
        mem.project_qkv(torch.randn(1, 2, D), conv_state=torch.zeros(1))


def test_state_not_mutated_by_continuation():
    """Continuation must not mutate the caller's saved state tensors."""
    mem = _mk()
    torch.manual_seed(1)
    x1 = torch.randn(1, 4, D)
    x2 = torch.randn(1, 3, D)
    with torch.no_grad():
        _, state = mem(x1, output_final_state=True)
        snap = {b: [w.clone() for w in state["fast_weights"][b]] for b in HOPE_BOXES}
        mem(x2, state=state)
    # Arrange/Act above; Assert the saved state is bitwise unchanged.
    for b in HOPE_BOXES:
        for w, c in zip(state["fast_weights"][b], snap[b]):
            assert torch.equal(w, c)


def test_train_backward_reaches_all_boxes():
    """Training backward reaches every slow parameter (depth 2, T=3, fp64)."""
    torch.manual_seed(0)
    mem = SelfReferentialTitanMemory(hidden_size=D, num_heads=H, memory_depth=2, use_short_conv=True).train().double()
    torch.manual_seed(1)
    x = torch.randn(1, 3, D, dtype=torch.float64)
    out, state = mem(x, output_final_state=True)
    # The q box never feeds retrieval, so out alone leaves its init grad-less;
    # include its live state (still a differentiable function of the init).
    (out.sum() + sum(w.sum() for w in state["fast_weights"]["q"])).backward()
    # Arrange/Act above; Assert every slow parameter got a nonzero grad.
    params = [mem.q_proj.weight, mem.gate_proj.weight, mem.gate_proj.bias, mem.o_proj.weight]
    params += list(mem.input_conv.parameters())
    for b in HOPE_BOXES:
        params += list(mem.fast_init[b])
    for p in params:
        assert p.grad is not None and bool(p.grad.norm().item() > 0)


def test_seq_len_accumulates():
    """seq_len counts streamed tokens; continuation matches the one-shot suffix."""
    mem = _mk()
    torch.manual_seed(1)
    x1 = torch.randn(1, 4, D)
    x2 = torch.randn(1, 3, D)
    with torch.no_grad():
        _, s1 = mem(x1, output_final_state=True)
        assert s1["seq_len"] == 4
        o2, s2 = mem(x2, state=s1, output_final_state=True)
        assert s2["seq_len"] == 7
        full, _ = mem(torch.cat([x1, x2], dim=1))
    # Arrange/Act above; Assert suffix parity.
    torch.testing.assert_close(o2, full[:, 4:], atol=1e-6, rtol=1e-5)
