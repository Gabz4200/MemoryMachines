r"""Titans long-term memory (arXiv:2501.00663) fast-weights reference.

Paper-accurate map (Behrouz et al., "Titans: Learning to Memorize at Test Time"):

- ``TitansFastWeightsMemory`` is the neural long-term memory (LMM, Sec. 3):
  associative-memory objective ``l = ||M(k) - v||^2`` (Eq. 12) with
  ``k = xW_K``, ``v = xW_V`` (Eq. 11); retrieval ``y = M*(q)`` with
  ``q = xW_Q`` (Eq. 15).
- Surprise with momentum (Eq. 9-10) plus adaptive forgetting / weight decay
  (Eq. 13-14)::

      S_t = eta_t * S_{t-1} - theta_t * grad l(M_{t-1}; x_t)
      M_t = (1 - alpha_t) * M_{t-1} + S_t

  with data-dependent ``eta_t`` (surprise decay), ``theta_t`` (momentary
  surprise rate) and ``alpha_t`` (forget gate), each a sigmoid projection of
  ``x_t``. Setting ``eta_t = 0`` recovers the Gated DeltaNet update, so this
  module generalizes it (App. C) via momentum, deep (``L_M >= 1``) MLP memory,
  and nonlinear recurrence.
- Architectural details (Sec. 4.4): SiLU q/k/v projections, L2-normalized
  queries and keys, and a depthwise causal convolution after each projection.
- Persistent memory (Sec. 3.3, Eq. 19): learnable input-independent tokens
  prepended to the sequence.
- ``TitansMAGBlock`` is the Memory-as-a-Gate variant (Sec. 4.2, Eq. 26-28):
  persistent-memory prefix, sliding-window-style short-term branch fused with
  the long-term memory branch through a learned gate.

The token loop is exact sequential test-time training (inner-loop gradients via
autograd, second-order when the outer graph is enabled). ``chunk_size`` only
bounds peak activation lifetime while carrying the state across chunks, so
chunked and sequential forwards agree exactly; the matmul/associative-scan
parallelization of Sec. 3.2 applies to the linear-memory special case and is
not used for the deep MLP reference path.

Examples::

    >>> import torch
    >>> from memory_machines.models.blocks.fast_weights import TitansFastWeightsMemory
    >>> mem = TitansFastWeightsMemory(hidden_size=16, num_heads=2, memory_depth=2)
    >>> out, state = mem(torch.randn(1, 5, 16), output_final_state=True)
    >>> out.shape
    torch.Size([1, 5, 16])
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .causal_conv1d import ShortConvolution

__all__ = ["TitansFastWeightsMemory", "TitansMAGBlock"]


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)


class TitansFastWeightsMemory(nn.Module):
    r"""TitansFastWeightsMemory(hidden_size, ...) — neural long-term memory.

    Deep MLP fast weights trained at test time with surprise momentum and
    forgetting (paper Sec. 3.1, Eq. 8-15). Per-head fast MLPs memorize the
    ``k -> v`` association stream and answer ``q`` queries.

    Args:
        hidden_size: model width ``D``.
        num_heads: memory heads ``H`` (``D`` must be divisible by ``H``).
        memory_depth: fast-MLP layers ``L_M`` (``1`` = linear matrix memory).
        fast_expansion: hidden width multiplier of the fast MLP (``L_M >= 2``).
        num_persistent_tokens: ``N_p`` learnable prefix tokens (Eq. 19, ``0`` disables).
        use_momentum: use past surprise ``eta_t * S_{t-1}`` (Eq. 10).
        use_weight_decay: use forget gate ``(1 - alpha_t)`` (Eq. 13).
        use_short_conv: depthwise causal conv after q/k/v projections (Sec. 4.4).
        conv_kernel_size: conv width.
        norm_eps: epsilon for L2/RMS norms.
        init_std: init scale of fast-weight priors and projections.

    Examples::

        >>> import torch
        >>> from memory_machines.models.blocks.fast_weights import TitansFastWeightsMemory
        >>> m = TitansFastWeightsMemory(hidden_size=16, num_heads=2)
        >>> m(torch.randn(1, 4, 16))[0].shape
        torch.Size([1, 4, 16])
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        memory_depth: int = 2,
        fast_expansion: float = 1.0,
        num_persistent_tokens: int = 4,
        use_momentum: bool = True,
        use_weight_decay: bool = True,
        use_short_conv: bool = True,
        conv_kernel_size: int = 3,
        norm_eps: float = 1e-6,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0:
            raise ValueError("hidden_size/num_heads must be positive")
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        if memory_depth < 1:
            raise ValueError(f"memory_depth must be >= 1, got {memory_depth}")
        if num_persistent_tokens < 0:
            raise ValueError("num_persistent_tokens must be non-negative")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.memory_depth = memory_depth
        self.fast_dim = max(1, round(self.head_dim * fast_expansion))
        self.num_persistent_tokens = num_persistent_tokens
        self.use_momentum = use_momentum
        self.use_weight_decay = use_weight_decay
        self.use_short_conv = use_short_conv
        self.norm_eps = norm_eps
        self.init_std = init_std

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        if use_short_conv:
            self.q_conv = ShortConvolution(hidden_size=hidden_size, kernel_size=conv_kernel_size, bias=False)
            self.k_conv = ShortConvolution(hidden_size=hidden_size, kernel_size=conv_kernel_size, bias=False)
            self.v_conv = ShortConvolution(hidden_size=hidden_size, kernel_size=conv_kernel_size, bias=False)
        else:
            self.register_module("q_conv", None)
            self.register_module("k_conv", None)
            self.register_module("v_conv", None)
        # Data-dependent gates (functions of x_t, Sec. 3.1): per-head scalars.
        self.eta_proj = nn.Linear(hidden_size, num_heads)
        self.theta_proj = nn.Linear(hidden_size, num_heads)
        self.alpha_proj = nn.Linear(hidden_size, num_heads)
        if num_persistent_tokens > 0:
            self.persistent = nn.Parameter(torch.empty(num_persistent_tokens, hidden_size))
        else:
            self.register_parameter("persistent", None)
        # Fast-weight priors (outer-loop parameters shared across sequences).
        self.fast_init: nn.ParameterList = nn.ParameterList([nn.Parameter(torch.empty(*shape)) for shape in self._fast_shapes()])
        self.gate_proj = nn.Linear(hidden_size, hidden_size)
        self.out_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.reset_parameters()

    def _fast_shapes(self) -> list[tuple[int, ...]]:
        kd, fd, vd, depth = self.head_dim, self.fast_dim, self.head_dim, self.memory_depth
        if depth == 1:
            return [(kd, vd)]
        shapes = [(kd, fd)]
        shapes += [(fd, fd)] * (depth - 2)
        shapes.append((fd, vd))
        return shapes

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None — init priors, gates, and projections."""
        for w in self.fast_init:
            nn.init.normal_(w, std=self.init_std)
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.gate_proj):
            nn.init.xavier_uniform_(proj.weight, gain=2**-2.5)
        for proj in (self.eta_proj, self.theta_proj, self.alpha_proj):
            nn.init.normal_(proj.weight, std=self.init_std)
            nn.init.zeros_(proj.bias)
        if self.persistent is not None:
            nn.init.normal_(self.persistent, std=self.init_std)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=2**-2.5)

    def project_qkv(
        self,
        x: torch.Tensor,
        conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        output_conv_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
        r"""project_qkv(x, ...) -> (q, k, v, conv_state) — SiLU projections, L2-normed q/k per head.

        Args:
            x: ``[B, T, D]`` hidden states.
            conv_state: cached short-conv histories for continuing a stream.
            output_conv_state: whether to return updated conv histories.

        Returns:
            Per-head ``q, k, v`` of shape ``[B, T, H, Kd]`` plus the conv
            state (or ``None``).
        """
        if x.dim() != 3 or x.shape[-1] != self.hidden_size:
            raise ValueError(f"x must be [B, T, D={self.hidden_size}], got {tuple(x.shape)}")
        if self.use_short_conv:
            assert self.q_conv is not None and self.k_conv is not None and self.v_conv is not None
            csq, csk, csv = (None, None, None) if conv_state is None else conv_state
            q, csq = self.q_conv(self.q_proj(x), cache=csq, output_final_state=output_conv_state)
            k, csk = self.k_conv(self.k_proj(x), cache=csk, output_final_state=output_conv_state)
            v, csv = self.v_conv(self.v_proj(x), cache=csv, output_final_state=output_conv_state)
            new_conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = (csq, csk, csv) if output_conv_state else None
        else:
            if conv_state is not None:
                raise ValueError("conv_state given but use_short_conv=False")
            q = F.silu(self.q_proj(x))
            k = F.silu(self.k_proj(x))
            v = F.silu(self.v_proj(x))
            new_conv_state = None
        b, t, _ = x.shape
        q = _l2norm(q.view(b, t, self.num_heads, self.head_dim), self.norm_eps)
        k = _l2norm(k.view(b, t, self.num_heads, self.head_dim), self.norm_eps)
        v = v.view(b, t, self.num_heads, self.head_dim)
        return q, k, v, new_conv_state

    def _mlp(self, x: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
        h = x
        for i, w in enumerate(weights):
            h = torch.einsum("...hd,...hde->...he", h, w)
            if i < len(weights) - 1:
                h = F.silu(h)
        return h

    def _init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        fast = [w.to(device=device, dtype=dtype).expand(batch, self.num_heads, *w.shape).contiguous() for w in self.fast_init]
        mom = [torch.zeros_like(w) for w in fast]
        return fast, mom

    def _step(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        eta: torch.Tensor,
        theta: torch.Tensor,
        alpha: torch.Tensor,
        fast: list[torch.Tensor],
        mom: list[torch.Tensor],
        create_graph: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        r"""_step(k, v, q, ...) -> (y, fast', mom', loss) — one Eq. 13-15 update."""
        for w in fast:
            w.requires_grad_(True)
        with torch.enable_grad():
            pred = self._mlp(k, fast)
            loss = ((pred - v) ** 2).mean()
            grads = torch.autograd.grad(loss, fast, create_graph=create_graph, allow_unused=False)
        new_mom, new_fast = [], []
        for s_prev, g, w_prev in zip(mom, grads, fast):
            s = eta * s_prev - theta * g.to(s_prev.dtype) if self.use_momentum else -theta * g.to(s_prev.dtype)
            # Train mode keeps the full BPTT graph through the Eq. 13
            # trajectory (outer loop optimizes priors/projections/gates);
            # eval/no-grad detaches to bound memory.
            w = w_prev if create_graph else w_prev.detach()
            if self.use_weight_decay:
                w = (1.0 - alpha) * w
            new_mom.append(s.detach() if not create_graph else s)
            w_new = w + s
            new_fast.append(w_new.detach() if not create_graph else w_new)
        y = self._mlp(q, [w.detach() if not create_graph else w for w in new_fast])
        return y, new_fast, new_mom, loss.detach()

    def associative_losses(self, x: torch.Tensor) -> torch.Tensor:
        r"""associative_losses(x) -> Tensor — pre-update Eq. 12 loss per position.

        Args:
            x: ``[B, T, D]`` hidden states.

        Returns:
            Tensor: ``[B, T]`` mean-squared ``||M_{t-1}(k_t) - v_t||^2``.
        """
        gate_in = torch.cat([self.persistent.expand(x.shape[0], -1, -1), x], dim=1) if self.persistent is not None else x
        eta_all = torch.sigmoid(self.eta_proj(gate_in))
        theta_all = torch.sigmoid(self.theta_proj(gate_in))
        alpha_all = torch.sigmoid(self.alpha_proj(gate_in))
        # q/k/v over the prefixed sequence so persistent tokens are memorized too.
        qf, kf, vf, _ = self.project_qkv(gate_in)
        fast, mom = self._init_state(x.shape[0], x.device, qf.dtype)
        np_ = self.num_persistent_tokens
        losses = []
        with torch.no_grad():
            for t in range(gate_in.shape[1]):
                _, fast, mom, loss = self._step(
                    kf[:, t],
                    vf[:, t],
                    qf[:, t],
                    eta_all[:, t].view(-1, self.num_heads, *([1] * (fast[0].dim() - 2))),
                    theta_all[:, t].view(-1, self.num_heads, *([1] * (fast[0].dim() - 2))),
                    alpha_all[:, t].view(-1, self.num_heads, *([1] * (fast[0].dim() - 2))),
                    fast,
                    mom,
                    create_graph=False,
                )
                if t >= np_:
                    losses.append(loss.reshape(x.shape[0], 1))
        return torch.cat(losses, dim=1) if losses else x.new_zeros(x.shape[0], 0)

    def forward(
        self,
        x: torch.Tensor,
        state: dict | None = None,
        output_final_state: bool = False,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        r"""forward(x, ...) -> (out, cache) — memorize the stream, retrieve per query.

        Persistent tokens are prepended (Eq. 19) and memorized first; each
        position retrieves from its post-update memory (Eq. 15) before the
        output gate, RMSNorm, projection, and residual (Sec. 4.2/4.4).

        Args:
            x: ``[B, T, D]`` hidden states (only the *new* tokens when
                ``state`` continues a stream).
            state: optional cache from a previous call (continued memorization).
            output_final_state: whether to return ``fast_weights``/``momentum``.
            chunk_size: positions per state-carry chunk (exact; bounds lifetime).

        Returns:
            Tensor: ``[B, T, D]`` outputs and the cache (or ``None``).
        """
        if x.dim() != 3 or x.shape[-1] != self.hidden_size:
            raise ValueError(f"x must be [B, T, D={self.hidden_size}], got {tuple(x.shape)}")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        bsz = x.shape[0]
        create_graph = torch.is_grad_enabled()
        if state is None:
            full = torch.cat([self.persistent.expand(bsz, -1, -1), x], dim=1) if self.persistent is not None else x
            start = 0
            conv_state = None
        else:
            # Continuing a stream: x holds only new tokens; the persistent
            # prefix was memorized by the first call, so prepend nothing.
            full, start = x, int(state.get("seq_len", 0))
            conv_state = state.get("conv_state")
        np_ = 0 if state is not None else self.num_persistent_tokens
        qf, kf, vf, new_conv_state = self.project_qkv(full, conv_state=conv_state, output_conv_state=output_final_state)
        eta_all = torch.sigmoid(self.eta_proj(full))
        theta_all = torch.sigmoid(self.theta_proj(full))
        alpha_all = torch.sigmoid(self.alpha_proj(full))
        if state is None:
            fast, mom = self._init_state(bsz, x.device, qf.dtype)
        else:
            fast = [w.to(device=x.device, dtype=qf.dtype) for w in state["fast_weights"]]
            mom = [w.to(device=x.device, dtype=qf.dtype) for w in state["momentum"]]
        total = qf.shape[1]
        width = total if chunk_size is None else chunk_size
        outs: list[torch.Tensor] = []
        for pos in range(0, total, width):
            for t in range(pos, min(pos + width, total)):
                y, fast, mom, _ = self._step(
                    kf[:, t],
                    vf[:, t],
                    qf[:, t],
                    eta_all[:, t].view(bsz, self.num_heads, *([1] * (fast[0].dim() - 2))),
                    theta_all[:, t].view(bsz, self.num_heads, *([1] * (fast[0].dim() - 2))),
                    alpha_all[:, t].view(bsz, self.num_heads, *([1] * (fast[0].dim() - 2))),
                    fast,
                    mom,
                    create_graph=create_graph,
                )
                if t >= np_:
                    outs.append(y)
        mem_out = torch.stack(outs, dim=1) if outs else x.new_zeros(bsz, 0, self.num_heads, self.head_dim)
        mem_out = mem_out.reshape(bsz, mem_out.shape[1], self.hidden_size).to(x.dtype)
        gated = self.out_norm(mem_out) * F.silu(self.gate_proj(x).to(mem_out.dtype))
        out = self.o_proj(gated.to(x.dtype)) + x
        cache = None
        if output_final_state:
            cache = {
                "fast_weights": [(w.detach() if not create_graph else w) for w in fast],
                "momentum": [(w.detach() if not create_graph else w) for w in mom],
                "conv_state": new_conv_state,
                "seq_len": start + total,
            }
        return out, cache


class TitansMAGBlock(nn.Module):
    r"""TitansMAGBlock(...) — Memory-as-a-Gate fusion (paper Sec. 4.2, Eq. 26-28).

    Prepends persistent memory, reads the long-term branch
    (:class:`TitansFastWeightsMemory`) and fuses it with a short-term branch
    (caller-provided sliding-window attention output, defaulting to the input)
    via a learned per-token gate over RMS-normalized branches, followed by an
    output projection and residual.

    Args:
        hidden_size: model width ``D``.
        num_heads: long-term memory heads.
        memory_depth: fast-MLP depth ``L_M``.
        num_persistent_tokens: ``N_p`` prefix tokens.
        norm_eps: RMSNorm epsilon.

    Examples::

        >>> import torch
        >>> from memory_machines.models.blocks.fast_weights import TitansMAGBlock
        >>> blk = TitansMAGBlock(hidden_size=16, num_heads=2)
        >>> blk(torch.randn(1, 4, 16)).shape
        torch.Size([1, 4, 16])
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        memory_depth: int = 2,
        num_persistent_tokens: int = 4,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.memory = TitansFastWeightsMemory(
            hidden_size=hidden_size,
            num_heads=num_heads,
            memory_depth=memory_depth,
            num_persistent_tokens=num_persistent_tokens,
            norm_eps=norm_eps,
        )
        self.norm_short = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.norm_long = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.fuse = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None — zero-start fusion, Xavier output."""
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=2**-2.5)

    def forward(self, x: torch.Tensor, short_term: torch.Tensor | None = None) -> torch.Tensor:
        r"""forward(x, short_term=None) -> Tensor — gated short/long fusion.

        Args:
            x: ``[B, T, D]`` hidden states.
            short_term: short-term branch output ``[B, T, D]`` (default: ``x``).

        Returns:
            Tensor: fused ``[B, T, D]`` output (with residual).
        """
        if x.dim() != 3 or x.shape[-1] != self.memory.hidden_size:
            raise ValueError(f"x must be [B, T, D={self.memory.hidden_size}], got {tuple(x.shape)}")
        st = x if short_term is None else short_term
        if st.shape != x.shape:
            raise ValueError(f"short_term {tuple(st.shape)} must match x {tuple(x.shape)}")
        long_out, _ = self.memory(x)
        # Remove the memory block's own residual to isolate the retrieved trace.
        long_trace = long_out - x
        gate = torch.sigmoid(self.fuse(x))
        fused = gate * self.norm_short(st) + (1.0 - gate) * self.norm_long(long_trace)
        return self.o_proj(fused) + x

    @property
    def persistent(self) -> torch.Tensor | None:
        """Persistent prefix tokens owned by the long-term branch."""
        return self.memory.persistent
