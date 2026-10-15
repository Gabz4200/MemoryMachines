"""Pure PyTorch reference oracle for depthwise causal Conv1d.

Ground truth for parity tests against the future compiled kernel.
Reached ONLY via ``dispatch_causal_conv1d(backend="naive")`` or explicitly
by tests — never as a silent fallback from a compiled backend.

Semantics mirror ``fla.modules.causal_conv1d`` / Dao-AILab ``causal-conv1d``:

- ``x``: ``[B, T, D]`` dense, or packed ``[1, L, D]`` / ``[L, D]`` with ``cu_seqlens``.
- ``weight``: ``[D, W]`` (``rearrange(conv.weight, "d 1 w -> d w")``).
- Orientation matches ``nn.Conv1d(padding=W-1)`` cross-correlation: the
  *last* weight tap multiplies the current token,
  ``y[t] = sum_j x[t-j] * w[W-1-j]`` (zero-pad on the left).
- ``initial_state``: ``[N, D, W]`` raw-input history, oldest-first. Slot 0 is
  padding (never read, matching FLA); the effective history is slots
  ``1..W-1`` where slot ``W-1`` is the most recent token.
- ``activation``: ``None`` or ``"silu"``/``"swish"`` applied as ``y * sigmoid(y)``.
- ``residual`` is added after bias/activation,
  matching FLA's Canon ordering (residual added to the activated conv output).
"""

import torch
import torch.nn.functional as F

_ACTIVATIONS = (None, "silu", "swish")


def _check_activation(activation: str | None) -> None:
    if activation not in _ACTIVATIONS:
        raise ValueError(f"activation must be one of {_ACTIVATIONS}, got {activation!r}")


def _apply_activation(y: torch.Tensor, activation: str | None) -> torch.Tensor:
    if activation in ("silu", "swish"):
        return y * torch.sigmoid(y)
    return y


def _split_packed(x: torch.Tensor, cu_seqlens: torch.Tensor) -> tuple[list[torch.Tensor], int, int]:
    """Split packed input into per-sequence ``[T_n, D]`` views."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() != 3 or x.shape[0] != 1:
        raise ValueError(f"packed x must be [1, L, D] or [L, D], got {tuple(x.shape)}")
    seqs = x.squeeze(0)
    lens = cu_seqlens.diff().tolist()
    out, pos = [], 0
    for ln in lens:
        out.append(seqs[pos : pos + ln])
        pos += ln
    return out, len(lens), x.shape[-1]


def reference_causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Depthwise causal conv forward (differentiable w.r.t. x/weight/bias)."""
    _check_activation(activation)
    if weight.dim() != 2:
        raise ValueError(f"weight must be [D, W], got {tuple(weight.shape)}")
    d, w = weight.shape
    if x.shape[-1] != d:
        raise ValueError(f"x last dim ({x.shape[-1]}) must match weight dim 0 ({d})")
    if bias is not None and tuple(bias.shape) != (d,):
        raise ValueError(f"bias must be [D], got {tuple(bias.shape)}")

    w_conv = weight.to(x.dtype).unsqueeze(1)  # [D, 1, W]

    def _conv_seq(seq: torch.Tensor, hist: torch.Tensor | None) -> torch.Tensor:
        # seq: [T, D]; hist: [W-1, D] effective history (most-recent-last) or None.
        t = seq.shape[0]
        if hist is not None and hist.shape[0] != w - 1:
            raise ValueError(f"history must have W-1={w - 1} tokens, got {hist.shape[0]}")
        inp = seq if hist is None else torch.cat([hist.to(seq.dtype), seq], dim=0)
        pad = w - 1 if hist is None else 0
        # groups=D depthwise; left-pad W-1 zeros when no history supplied.
        xt = inp.transpose(0, 1).unsqueeze(0)  # [1, D, T(+hist)]
        if pad:
            xt = F.pad(xt, (pad, 0))
        y = F.conv1d(xt, w_conv, bias=None, stride=1, padding=0, groups=d)
        y = y.squeeze(0).transpose(0, 1)  # [T(+hist), D]
        return y[-t:]

    if cu_seqlens is None:
        if x.dim() != 3:
            raise ValueError(f"dense x must be [B, T, D], got {tuple(x.shape)}")
        b = x.shape[0]
        if initial_state is not None:
            if initial_state.shape != (b, d, w):
                raise ValueError(f"initial_state must be [N, D, W]={(b, d, w)}, got {tuple(initial_state.shape)}")
            hists = [initial_state[i, :, 1:].transpose(0, 1) for i in range(b)]  # [W-1, D]
        else:
            hists = [None] * b
        ys = [_conv_seq(x[i], h) for i, h in enumerate(hists)]
        y = torch.stack(ys, dim=0)
    else:
        seqs, n, _ = _split_packed(x, cu_seqlens)
        if initial_state is not None:
            if initial_state.shape != (n, d, w):
                raise ValueError(f"initial_state must be [N, D, W]={(n, d, w)}, got {tuple(initial_state.shape)}")
            hists = [initial_state[i, :, 1:].transpose(0, 1) for i in range(n)]
        else:
            hists = [None] * n
        ys = [_conv_seq(s, h) for s, h in zip(seqs, hists)]
        y = torch.cat(ys, dim=0).unsqueeze(0)

    if bias is not None:
        y = y + bias.to(y.dtype)
    y = _apply_activation(y, activation)
    if residual is not None:
        if residual.shape != y.shape:
            raise ValueError(f"residual shape {tuple(residual.shape)} != output shape {tuple(y.shape)}")
        y = y + residual
    return y


def reference_causal_conv1d_update_states(
    x: torch.Tensor,
    state_len: int,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the ``[N, D, W]`` final state (last W raw inputs per sequence).

    Short sequences inherit the tail of ``initial_state``: the state is a copy
    of ``initial_state`` with the last ``T_n`` slots overwritten by ``x``.
    """
    if state_len <= 0:
        raise ValueError(f"state_len must be positive, got {state_len}")
    w = state_len
    if cu_seqlens is None:
        if x.dim() != 3:
            raise ValueError(f"dense x must be [B, T, D], got {tuple(x.shape)}")
        b, t, d_ = x.shape
        final = x.new_zeros(b, d_, w)
        for i in range(b):
            seq = x[i]  # [T, D]
            take = min(t, w)
            if initial_state is not None:
                if initial_state.shape != (b, d_, w):
                    raise ValueError(f"initial_state must be {(b, d_, w)}, got {tuple(initial_state.shape)}")
                final[i] = initial_state[i]
            if take:
                final[i, :, w - take :] = seq[t - take :].transpose(0, 1)
        return final
    seqs, n, d_ = _split_packed(x, cu_seqlens)
    final = x.new_zeros(n, d_, w)
    for i, seq in enumerate(seqs):
        take = min(seq.shape[0], w)
        if initial_state is not None:
            if initial_state.shape != (n, d_, w):
                raise ValueError(f"initial_state must be {(n, d_, w)}, got {tuple(initial_state.shape)}")
            final[i] = initial_state[i]
        if take:
            final[i, :, w - take :] = seq[seq.shape[0] - take :].transpose(0, 1)
    return final


def reference_causal_conv1d_update(
    x: torch.Tensor,
    cache: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    activation: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-token decode step: ``y = sum(cache' * weight)`` + bias/act/residual.

    ``x`` holds exactly one token per sequence: ``[N, D]``, ``[1, N, D]`` or
    ``[N, 1, D]``. ``cache`` is ``[N, D, W]`` and is updated **in place**
    (left-shift, new token at the end), mirroring the kernel. Returns
    ``(y, cache)`` with ``y`` shaped like ``x``.
    """
    _check_activation(activation)
    shape = x.shape
    if weight.dim() != 2:
        raise ValueError(f"weight must be [D, W], got {tuple(weight.shape)}")
    d, w = weight.shape
    if x.dim() == 2:
        tokens = x  # [N, D]
    elif x.dim() == 3 and x.shape[0] == 1:
        tokens = x.squeeze(0)  # [N, D]
    elif x.dim() == 3 and x.shape[1] == 1:
        tokens = x.squeeze(1)  # [N, D]
    else:
        raise ValueError(f"update x must be [N, D], [1, N, D] or [N, 1, D], got {tuple(x.shape)}")
    n = tokens.shape[0]
    if tokens.shape[-1] != d:
        raise ValueError(f"token dim ({tokens.shape[-1]}) must match weight dim 0 ({d})")
    if cache.shape != (n, d, w):
        raise ValueError(f"cache must be [N, D, W]={(n, d, w)}, got {tuple(cache.shape)}")

    with torch.no_grad():
        # copy_() with overlapping slices is undefined; shift via a temp.
        shifted = torch.cat([cache[:, :, 1:], tokens.to(cache.dtype).unsqueeze(-1)], dim=-1)
        cache.copy_(shifted)
    y_tok = (cache.to(tokens.dtype) * weight.to(tokens.dtype).unsqueeze(0)).sum(dim=-1)  # [N, D]
    if bias is not None:
        if tuple(bias.shape) != (d,):
            raise ValueError(f"bias must be [D], got {tuple(bias.shape)}")
        y_tok = y_tok + bias.to(y_tok.dtype)
    y_tok = _apply_activation(y_tok, activation)
    if residual is not None:
        if residual.shape != shape:
            raise ValueError(f"residual shape {tuple(residual.shape)} != x shape {tuple(shape)}")
        y_tok = y_tok + residual.reshape(n, d).to(y_tok.dtype)
    return y_tok.view(shape), cache
