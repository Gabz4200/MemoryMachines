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

import itertools

import torch
import torch.nn.functional as F
from torch import nn

from memory_machines.utils.config_schema import MemoryMachineConfig

from .causal_conv1d import ShortConvolution

__all__ = [
    "ContinuumMemorySystem",
    "HopeAttentionBlock",
    "HopeBlock",
    "SelfReferentialTitanMemory",
    "SelfReferentialTitanMixer",
    "TitansFastWeightsMemory",
    "TitansMAGBlock",
]

#: Boxes of the self-referential system (Eq. 88/96): the five adaptive
#: generators plus the main retrieval memory. ``"q"`` owns fast weights used
#: only to generate its own values (Eq. 84); the query itself stays a static
#: projection ``q = xW_q``.
HOPE_BOXES = ("k", "v", "q", "eta", "alpha", "memory")


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


def _sr_mlp(x: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
    """Residual fast MLP ``M(x) = x + W1 silu(W2 x)`` (Eq. 89/91)."""
    if len(weights) == 1:
        return torch.einsum("...hd,...hde->...he", x, weights[0])
    h = x
    for i, w in enumerate(weights):
        h = torch.einsum("...hd,...hde->...he", h, w)
        if i < len(weights) - 1:
            h = F.silu(h)
    return x + h


def _gate_shape(eta: torch.Tensor, w: torch.Tensor) -> list[int]:
    """Broadcast shape for per-head ``eta``/``alpha`` scalars against ``w``."""
    return [eta.shape[0], eta.shape[1], *([1] * (w.dim() - 2))]


class SelfReferentialTitanMemory(nn.Module):
    r"""SelfReferentialTitanMemory(...) — self-modifying Titans (Sec. 8.1, Eq. 86-89).

    Every projection adapts in context: ``k = Mk(x)``, ``v = Mv(x)``,
    ``eta = Meta(x)``, ``alpha = Malpha(x)`` with static ``q = xW_q`` only.
    Each box generates its own values ``vhat_box = Mbox(v)`` (Eq. 87) and is
    updated by DGD with weight decay (Eq. 88)::

        Mbox,t = Mbox,t-1 [alpha I - eta k kT] - eta grad L(Mbox,t-1; k, vhat)

    with ``L = ||M(k) - vhat||^2`` (``objective="l2"``) or ``L = -<Mk, vhat>``
    (``objective="dot"``). ``memory_depth == 1`` uses the exact closed forms
    of Eq. 92/93; deeper memories use inner-loop autograd with ``alpha``
    decay. Retrieval uses the pre-update state ``y = Mmemory,t-1(q)`` (Eq. 86).

    Like :class:`TitansFastWeightsMemory`, ``chunk_size`` only bounds peak
    activation lifetime while carrying the state across chunks, so chunked
    and sequential forwards agree exactly (the parallel chunk approximation
    of Eq. 90 is left to the linear dual form).

    Args:
        hidden_size: model width ``D``.
        num_heads: heads ``H`` (``D`` divisible by ``H``).
        memory_depth: fast-MLP layers per box (``1`` = linear matrix memory).
        fast_expansion: fast hidden width multiplier.
        objective: ``"l2"`` regression or ``"dot"`` similarity.
        use_short_conv: local depthwise causal conv on the input (window 4).
        conv_kernel_size: conv width (paper uses 4).
        norm_eps: epsilon for L2 norms.
        init_std: init scale of fast-weight priors.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        memory_depth: int = 2,
        fast_expansion: float = 1.0,
        objective: str = "l2",
        use_short_conv: bool = True,
        conv_kernel_size: int = 4,
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
        if objective not in ("l2", "dot"):
            raise ValueError(f"objective must be 'l2' or 'dot', got {objective!r}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.memory_depth = memory_depth
        self.fast_dim = max(1, round(self.head_dim * fast_expansion))
        self.objective = objective
        self.use_short_conv = use_short_conv
        self.norm_eps = norm_eps
        self.init_std = init_std
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        if use_short_conv:
            self.input_conv = ShortConvolution(hidden_size=hidden_size, kernel_size=conv_kernel_size, bias=False)
        else:
            self.register_module("input_conv", None)
        self.fast_init: nn.ModuleDict = nn.ModuleDict({box: nn.ParameterList([nn.Parameter(torch.empty(*s)) for s in self._fast_shapes()]) for box in HOPE_BOXES})
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
        """Init fast-weight priors and projections."""
        for plist in self.fast_init.values():
            for w in plist:
                nn.init.normal_(w, std=self.init_std)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=2**-2.5)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=2**-2.5)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim)

    def project_qkv(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        output_conv_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Stateless inspection probe: ``(q, k, v)`` at the meta-learned init.

        Runs the input conv (when enabled) and applies the *initial* fast
        weights, so ``q``/``k`` are L2-normalized per head. The stateful
        forward evolves these weights token by token.
        """
        if x.dim() != 3 or x.shape[-1] != self.hidden_size:
            raise ValueError(f"x must be [B, T, D={self.hidden_size}], got {tuple(x.shape)}")
        if self.use_short_conv:
            assert self.input_conv is not None
            xc, new_state = self.input_conv(x, cache=conv_state, output_final_state=output_conv_state)
            new_conv_state: torch.Tensor | None = new_state if output_conv_state else None
        else:
            if conv_state is not None:
                raise ValueError("conv_state given but use_short_conv=False")
            xc, new_conv_state = x, None
        h = self._heads(xc)
        q = _l2norm(self._heads(self.q_proj(xc)), self.norm_eps)
        bh = h.shape[0] * h.shape[1]
        hf = h.reshape(bh, self.num_heads, self.head_dim)
        k = _l2norm(_sr_mlp(hf, [w.expand(bh, self.num_heads, *w.shape) for w in self.fast_init["k"]]).view(*h.shape), self.norm_eps)
        v = _sr_mlp(hf, [w.expand(bh, self.num_heads, *w.shape) for w in self.fast_init["v"]]).view(*h.shape)
        return q, k, v, new_conv_state

    def _init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> dict[str, list[torch.Tensor]]:
        return {box: [w.to(device=device, dtype=dtype).expand(batch, self.num_heads, *w.shape).contiguous() for w in self.fast_init[box]] for box in HOPE_BOXES}

    def _box_scalars(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(raw.mean(dim=-1))

    def _update_linear_box(
        self,
        w: torch.Tensor,
        k: torch.Tensor,
        vhat: torch.Tensor,
        eta: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Exact Eq. 92/93 row-vector form for matrix memories."""
        kk = torch.einsum("...d,...e->...de", k, k)
        dgd = torch.einsum("...de,...ev->...dv", kk, w)
        pred = torch.einsum("...d,...dv->...v", k, w)
        if self.objective == "l2":
            grad = torch.einsum("...d,...v->...dv", k, pred - vhat)
        else:
            grad = -torch.einsum("...d,...v->...dv", k, vhat)
        shape = _gate_shape(eta, w)
        return alpha.view(*shape) * w - eta.view(*shape) * dgd - eta.view(*shape) * grad

    def forward(
        self,
        x: torch.Tensor,
        state: dict | None = None,
        output_final_state: bool = False,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        """forward(x, ...) -> (out, cache) — self-modifying retrieval (Eq. 86-88)."""
        if x.dim() != 3 or x.shape[-1] != self.hidden_size:
            raise ValueError(f"x must be [B, T, D={self.hidden_size}], got {tuple(x.shape)}")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        bsz = x.shape[0]
        create_graph = torch.is_grad_enabled()
        if state is None:
            full, start, conv_state = x, 0, None
        else:
            full, start = x, int(state.get("seq_len", 0))
            conv_state = state.get("conv_state")
        if self.use_short_conv:
            assert self.input_conv is not None
            xc, new_conv_state = self.input_conv(full, cache=conv_state, output_final_state=output_final_state)
        else:
            if conv_state is not None:
                raise ValueError("conv_state given but use_short_conv=False")
            xc, new_conv_state = full, None
        heads = self._heads(xc)
        q_all = _l2norm(self._heads(self.q_proj(xc)), self.norm_eps)
        if state is None:
            fast = self._init_state(bsz, x.device, q_all.dtype)
        else:
            fast = {box: [w.to(device=x.device, dtype=q_all.dtype) for w in state["fast_weights"][box]] for box in HOPE_BOXES}
        total = q_all.shape[1]
        width = total if chunk_size is None else chunk_size
        outs: list[torch.Tensor] = []
        for pos in range(0, total, width):
            for t in range(pos, min(pos + width, total)):
                xh = heads[:, t]
                kh = _l2norm(_sr_mlp(xh, fast["k"]), self.norm_eps)
                vh = _sr_mlp(xh, fast["v"])
                qh = q_all[:, t]
                eta = self._box_scalars(_sr_mlp(xh, fast["eta"]))
                alpha = self._box_scalars(_sr_mlp(xh, fast["alpha"]))
                vhat = {box: _sr_mlp(vh, fast[box]) for box in HOPE_BOXES}
                outs.append(_sr_mlp(qh, fast["memory"]))
                if self.memory_depth == 1:
                    for box in HOPE_BOXES:
                        w_old = fast[box][0]
                        fast[box][0] = self._update_linear_box(w_old, kh, vhat[box], eta, alpha)
                        if not create_graph:
                            fast[box][0] = fast[box][0].detach()
                else:
                    for box in HOPE_BOXES:
                        for w in fast[box]:
                            w.requires_grad_(True)
                    with torch.enable_grad():
                        grads: dict[str, list[torch.Tensor]] = {}
                        for box in HOPE_BOXES:
                            pred_b = _sr_mlp(kh, fast[box])
                            loss_b = ((pred_b - vhat[box]) ** 2).mean() if self.objective == "l2" else -(pred_b * vhat[box]).mean()
                            grads[box] = list(torch.autograd.grad(loss_b, fast[box], create_graph=create_graph, allow_unused=False))
                    for box in HOPE_BOXES:
                        for i, (w_prev, g) in enumerate(zip(fast[box], grads[box])):
                            w = w_prev if create_graph else w_prev.detach()
                            shape = _gate_shape(eta, w)
                            w_new = alpha.view(*shape) * w - eta.view(*shape) * g.to(w.dtype)
                            fast[box][i] = w_new if create_graph else w_new.detach()
        mem_out = torch.stack(outs, dim=1).reshape(bsz, total, self.hidden_size).to(x.dtype)
        gated = self.out_norm(mem_out) * F.silu(self.gate_proj(x).to(mem_out.dtype))
        out = self.o_proj(gated.to(x.dtype)) + x
        cache = None
        if output_final_state:
            cache = {
                "fast_weights": {box: [(w.detach() if not create_graph else w) for w in fast[box]] for box in HOPE_BOXES},
                "conv_state": new_conv_state,
                "seq_len": start + total,
            }
        return out, cache


class SelfReferentialTitanMixer(nn.Module):
    r"""SelfReferentialTitanMixer(config, ...) — config-driven MultiHead mixer.

    Thin config-driven wrapper around :class:`SelfReferentialTitanMemory`
    matching the other token-mixer blocks (:class:`GatedDeltaNet2`,
    :class:`GroupedQueryAttention`): ``hidden_size``/``num_heads`` come from
    ``config`` (``hidden_size`` / ``num_attention_heads``), heads stay batched
    as ``[B, T, H, Dh]`` inside the wrapped memory, and forward follows the
    ``(hidden_states, attention_mask, past_key_values, use_cache,
    output_attentions, cu_seqlens, chunk_size) -> (out, None, cache)`` triple.

    The math is exactly the wrapped memory's Eq. 86-88 path; this class only
    translates the mixer interface (padding mask, packed ``cu_seqlens``,
    incremental ``past_key_values``) onto it. The Continuum Memory System
    chain stays separate (see :class:`HopeBlock`).

    Args:
        config: provides ``hidden_size``, ``num_attention_heads``,
            ``rms_norm_eps``, and ``initializer_range``.
        memory_depth: fast-MLP layers per box (``1`` = linear matrix memory).
        fast_expansion: fast hidden width multiplier.
        objective: ``"l2"`` regression or ``"dot"`` similarity.
        use_short_conv: local depthwise causal conv on the input.
        conv_kernel_size: conv width (paper uses 4).
        chunk_size: default positions per state-carry chunk (exact; bounds
            peak activation lifetime). Per-call ``chunk_size`` overrides it.

    Examples::

        >>> import torch
        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> from memory_machines.models.blocks.fast_weights import SelfReferentialTitanMixer
        >>> cfg = MemoryMachineConfig(hidden_size=16, num_attention_heads=2,
        ...     num_key_value_heads=2, csa_num_output_groups=2)
        >>> m = SelfReferentialTitanMixer(cfg, memory_depth=1, use_short_conv=False)
        >>> m(torch.randn(1, 4, 16))[0].shape
        torch.Size([1, 4, 16])
    """

    def __init__(
        self,
        config: MemoryMachineConfig,
        memory_depth: int = 2,
        fast_expansion: float = 1.0,
        objective: str = "l2",
        use_short_conv: bool = True,
        conv_kernel_size: int = 4,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})")
        self.head_dim = self.hidden_size // self.num_heads
        self.chunk_size = chunk_size
        self.memory = SelfReferentialTitanMemory(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            memory_depth=memory_depth,
            fast_expansion=fast_expansion,
            objective=objective,
            use_short_conv=use_short_conv,
            conv_kernel_size=conv_kernel_size,
            norm_eps=config.rms_norm_eps,
            init_std=config.initializer_range,
        )

    def reset_parameters(self) -> None:
        """reset_parameters() -> None — re-init the wrapped memory."""
        self.memory.reset_parameters()

    @property
    def fast_init(self) -> nn.ModuleDict:
        """Fast-weight priors owned by the wrapped memory."""
        return self.memory.fast_init

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: dict | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        r"""forward(hidden_states, ...) -> (out, None, cache).

        Args:
            hidden_states: input ``[B, T, D]`` (only *new* tokens when
                ``past_key_values`` continues a stream).
            attention_mask: optional ``[B, T]`` 0/1 padding mask.
            past_key_values: cache from a previous call (continued
                memorization).
            use_cache: whether to return the updated cache.
            output_attentions: unused, kept for interface compat.
            cu_seqlens: optional ``[N+1]`` packed offsets (``B == 1``);
                each segment runs from a fresh memory.
            chunk_size: positions per state-carry chunk (exact); defaults to
                the constructor ``chunk_size`` (``None`` = whole sequence).

        Returns:
            ``[B, T, D]`` outputs, ``None`` (no attention map), and the
            updated cache (or ``None``).
        """
        if hidden_states.dim() != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden_states must be [B, T, D={self.hidden_size}], got {tuple(hidden_states.shape)}")
        if attention_mask is not None and cu_seqlens is not None:
            raise ValueError("attention_mask and cu_seqlens cannot be provided at the same time")
        width = chunk_size if chunk_size is not None else self.chunk_size
        if width is not None and width <= 0:
            raise ValueError(f"chunk_size must be positive, got {width}")
        if cu_seqlens is not None:
            return self._forward_packed(hidden_states, cu_seqlens, use_cache, width)
        if attention_mask is not None:
            return self._forward_masked(hidden_states, attention_mask, past_key_values, use_cache, width)
        out, cache = self.memory(hidden_states, state=past_key_values, output_final_state=use_cache, chunk_size=width)
        return out, None, cache

    def _forward_packed(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        use_cache: bool,
        chunk_size: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        if hidden_states.shape[0] != 1:
            raise ValueError(f"cu_seqlens packing requires batch 1, got {hidden_states.shape[0]}")
        bounds = cu_seqlens.tolist()
        total = hidden_states.shape[1]
        if int(cu_seqlens[-1].item()) != total:
            raise ValueError(f"cu_seqlens must end at T={total}, got {int(cu_seqlens[-1].item())}")
        if use_cache:
            raise ValueError("cu_seqlens with use_cache=True is not supported; decode packed segments without caching")
        outs = []
        for s, e in itertools.pairwise(bounds):
            seg = hidden_states[:, s:e, :]
            if seg.shape[1] == 0:
                outs.append(seg)
                continue
            o, _ = self.memory(seg, state=None, output_final_state=False, chunk_size=chunk_size)
            outs.append(o)
        return torch.cat(outs, dim=1), None, None

    def _forward_masked(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: dict | None,
        use_cache: bool,
        chunk_size: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        if past_key_values is not None:
            raise ValueError("attention_mask with cached past_key_values is not supported in the naive reference; decode without padding")
        if use_cache:
            raise ValueError("attention_mask with use_cache=True is not supported in the naive reference; pass explicit positions via cu_seqlens or decode without padding")
        bsz, tlen, _ = hidden_states.shape
        if tuple(attention_mask.shape) != (bsz, tlen):
            raise ValueError(f"attention_mask must be [B, T]=({bsz}, {tlen}), got {tuple(attention_mask.shape)}")
        outs = []
        for i in range(bsz):
            keep = attention_mask[i].to(torch.bool)
            xi = hidden_states[i : i + 1, keep]
            if xi.shape[1] == 0:
                outs.append(hidden_states[i : i + 1, :0].new_zeros(1, 0, self.hidden_size))
                continue
            oi, _ = self.memory(xi, state=None, output_final_state=False, chunk_size=chunk_size)
            full = oi.new_zeros(1, tlen, self.hidden_size)
            full[0, keep] = oi[0]
            outs.append(full)
        return torch.cat(outs, dim=0), None, None


class ContinuumMemorySystem(nn.Module):
    r"""ContinuumMemorySystem(...) — chain of MLP levels (Sec. 7.1, Eq. 70-71).

    ``y = MLP_fk(...MLP_f1(x))`` where level ``l`` owns update interval
    ``C^(l)`` via :meth:`should_update`. Levels are static parameters in a
    forward pass (outer-loop / scheduled updates own them); the schedule
    metadata is what makes the chain a continuum rather than one MLP.

    Args:
        hidden_size: model width ``D``.
        num_levels: chain length ``k``.
        update_intervals: per-level ``C^(l)`` (default powers of two).
        expansion: FFN hidden multiplier.
        norm_eps: RMSNorm epsilon.
    """

    def __init__(
        self,
        hidden_size: int,
        num_levels: int = 2,
        update_intervals: tuple[int, ...] | None = None,
        expansion: float = 4.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_levels < 1:
            raise ValueError("hidden_size must be positive and num_levels >= 1")
        if update_intervals is None:
            update_intervals = tuple(2**i for i in range(num_levels))
        if len(update_intervals) != num_levels or any(c < 1 for c in update_intervals):
            raise ValueError(f"update_intervals must hold {num_levels} positive ints, got {update_intervals}")
        self.hidden_size = hidden_size
        self.num_levels = num_levels
        self.update_intervals = tuple(update_intervals)
        hidden = max(1, round(hidden_size * expansion))
        self.norms = nn.ModuleList([nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(num_levels)])
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, hidden), nn.SiLU(), nn.Linear(hidden, hidden_size)) for _ in range(num_levels)])

    def should_update(self, level: int, step: int) -> bool:
        """should_update(level, step) -> bool — Eq. 71 schedule gate."""
        if not 0 <= level < self.num_levels:
            raise ValueError(f"level must be in [0, {self.num_levels}), got {level}")
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        return step % self.update_intervals[level] == 0

    def load_pretrained_into_levels(self, modules: list[nn.Module]) -> None:
        """Ad-hoc level stacking (Sec. 7.3): init levels from pretrained MLPs.

        Copies matching ``weight``/``bias`` tensors where shapes agree and
        leaves the rest at their random init.
        """
        if len(modules) != self.num_levels:
            raise ValueError(f"expected {self.num_levels} modules, got {len(modules)}")
        for dst_seq, src in zip(self.mlps, modules):
            src_params = {n: p for n, p in src.named_parameters()}
            for n, p in dst_seq.named_parameters():
                if n in src_params and src_params[n].shape == p.shape:
                    p.data.copy_(src_params[n].data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward(x) -> Tensor — Eq. 70 chain with per-level residuals."""
        if x.dim() != 3 or x.shape[-1] != self.hidden_size:
            raise ValueError(f"x must be [B, T, D={self.hidden_size}], got {tuple(x.shape)}")
        h = x
        for norm, mlp in zip(self.norms, self.mlps):
            h = h + mlp(norm(h))
        return h


class HopeBlock(nn.Module):
    r"""HopeBlock(...) — self-modifying Titans followed by CMS (Sec. 8.3, Eq. 94-97).

    ``o = SelfRefTitan(x)`` then ``y = MLP_fk(...MLP_f1(o))``. Queries and
    keys are L2-normalized and the input carries a local convolution of
    width 4, matching the paper's experimental setup.

    Args:
        hidden_size: model width ``D``.
        num_heads: self-referential heads.
        memory_depth: fast-MLP depth per box.
        num_levels: CMS chain length.
        update_intervals: CMS per-level intervals (default powers of two).
        use_short_conv: local conv window on the Titan input.
        norm_eps: epsilon for norms.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        memory_depth: int = 2,
        num_levels: int = 2,
        update_intervals: tuple[int, ...] | None = None,
        use_short_conv: bool = True,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.titan = SelfReferentialTitanMemory(hidden_size=hidden_size, num_heads=num_heads, memory_depth=memory_depth, use_short_conv=use_short_conv, conv_kernel_size=4, norm_eps=norm_eps)
        self.cms = ContinuumMemorySystem(hidden_size=hidden_size, num_levels=num_levels, update_intervals=update_intervals, norm_eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        state: dict | None = None,
        output_final_state: bool = False,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        """forward(x, ...) -> (y, cache) — Eq. 94-97 with residuals."""
        o, sr_state = self.titan(x, state=None if state is None else state.get("self_ref"), output_final_state=output_final_state, chunk_size=chunk_size)
        y = self.cms(o)
        cache = None
        if output_final_state:
            cache = {"self_ref": sr_state, "cms": {"num_levels": self.cms.num_levels, "update_intervals": self.cms.update_intervals}}
        return y, cache


class HopeAttentionBlock(nn.Module):
    r"""HopeAttentionBlock(...) — Hope variant with softmax attention (Sec. 8.3).

    Replaces the self-modifying Titans with causal global attention, keeping
    the Continuum Memory System chain. Used in the paper as the
    ``Hope-Attention`` baseline.

    Args:
        hidden_size: model width ``D``.
        num_heads: attention heads.
        num_levels: CMS chain length.
        update_intervals: CMS per-level intervals.
        norm_eps: epsilon for norms.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_levels: int = 2,
        update_intervals: tuple[int, ...] | None = None,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.cms = ContinuumMemorySystem(hidden_size=hidden_size, num_levels=num_levels, update_intervals=update_intervals, norm_eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward(x) -> Tensor — causal attention then CMS chain."""
        if x.dim() != 3 or x.shape[-1] != self.hidden_size:
            raise ValueError(f"x must be [B, T, D={self.hidden_size}], got {tuple(x.shape)}")
        b, t, _ = x.shape
        h = self.attn_norm(x)
        q = self.q_proj(h).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, self.hidden_size)
        return self.cms(self.o_proj(a) + x)
