r"""Auxiliary memory blocks: hyper-connections, engram memory, speculative drafter.

:class:`SinglePassMHC` expands the residual stream into parallel streams mixed
by learned coefficients, :class:`EngramMemory` adds hashed n-gram memory, and
:class:`DSparkDrafter` proposes speculative hidden-state drafts.
"""

from collections.abc import Callable

import torch
from torch import nn

from memory_machines.utils.config_schema import MemoryMachineConfig

from .shared import _sinkhorn


class SinglePassMHC(nn.Module):
    r"""SinglePassMHC(config)

    Single-pass manifold-constrained hyper-connections.

    The hidden state is viewed as ``n`` parallel streams ``(B, L, n, d)``.
    Input mixing ``a`` (sigmoid), residual mixing ``b`` (Sinkhorn
    doubly-stochastic), and output scaling ``c`` (``2 * sigmoid``) combine the
    streams with a branch output in one pass.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``mhc_num_streams`` (``n``), ``mhc_sinkhorn_iters``,
            ``hidden_size`` (``d``), and ``rms_norm_eps``.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> mhc = SinglePassMHC(MemoryMachineConfig(hidden_size=32, mhc_num_streams=2))
        >>> streams = torch.randn(1, 4, 2, 32)
        >>> out, a = mhc(streams, torch.randn(1, 4, 32))
        >>> out.shape
        torch.Size([1, 4, 2, 32])
    """

    def __init__(self, config: MemoryMachineConfig):
        super().__init__()
        self.num_streams = config.mhc_num_streams
        self.iters = config.mhc_sinkhorn_iters
        n, d = config.mhc_num_streams, config.hidden_size
        self.pre_norm = nn.RMSNorm(n * d, config.rms_norm_eps)
        self.a_dyn = nn.Linear(n * d, n)
        self.b_dyn = nn.Linear(n * d, n * n)
        self.c_dyn = nn.Linear(n * d, n)
        self.a_static = nn.Parameter(torch.zeros(n))
        self.b_static = nn.Parameter(torch.zeros(n, n))
        self.c_static = nn.Parameter(torch.zeros(n))
        self.alpha_pre = nn.Parameter(torch.tensor(1e-3))
        self.alpha_res = nn.Parameter(torch.tensor(1e-3))
        self.alpha_post = nn.Parameter(torch.tensor(1e-3))

    def coefficients(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""coefficients(x) -> (Tensor, Tensor, Tensor)

        Compute input, residual, and output mixing coefficients.

        Args:
            x (Tensor): streams of shape ``(B, L, n, d)``.

        Returns:
            Tensor: input weights ``a`` of shape ``(B, L, 1, n)`` in
            ``(0, 1)``, residual matrix ``b`` of shape ``(B, L, n, n)``
            (doubly-stochastic), and output scales ``c`` of shape
            ``(B, L, n, 1)`` in ``(0, 2)``.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> mhc = SinglePassMHC(MemoryMachineConfig(hidden_size=16, mhc_num_streams=2))
            >>> a, b, c = mhc.coefficients(torch.randn(1, 3, 2, 16))
            >>> a.shape, b.shape, c.shape
            (torch.Size([1, 3, 1, 2]), torch.Size([1, 3, 2, 2]), torch.Size([1, 3, 2, 1]))
        """
        if x.dim() != 4 or x.shape[2] != self.num_streams:
            raise ValueError(f"streams must be [B, L, n={self.num_streams}, d], got {tuple(x.shape)}")
        bsz, lq, n, _ = x.shape
        flat = self.pre_norm(x.reshape(bsz, lq, -1))
        a = torch.sigmoid(self.alpha_pre * self.a_dyn(flat) + self.a_static).unsqueeze(2)
        b = _sinkhorn(self.alpha_res * self.b_dyn(flat).view(bsz, lq, n, n) + self.b_static, self.iters)
        c = (2 * torch.sigmoid(self.alpha_post * self.c_dyn(flat) + self.c_static)).unsqueeze(-1)
        return a, b, c

    @staticmethod
    def mix_input(
        streams: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        r"""mix_input(streams, a) -> Tensor

        Mix streams into a single branch input.

        Args:
            streams (Tensor): streams of shape ``(B, L, n, d)``.
            a (Tensor): input weights of shape ``(B, L, 1, n)``.

        Returns:
            Tensor: mixed input of shape ``(B, L, 1, d)``.

        Examples::

            >>> import torch
            >>> s = torch.randn(1, 2, 3, 4)
            >>> a = torch.softmax(torch.randn(1, 2, 1, 3), dim=-1)
            >>> SinglePassMHC.mix_input(s, a).shape
            torch.Size([1, 2, 1, 4])
        """
        if a.shape[:2] != streams.shape[:2] or a.shape[2:] != (1, streams.shape[2]):
            raise ValueError(f"mixing coefficients shape {tuple(a.shape)} incompatible with streams {tuple(streams.shape)} (expect [B, L, 1, n])")
        return torch.matmul(a, streams)

    def forward(
        self,
        streams: torch.Tensor,
        branch_out: torch.Tensor,
        prev_a: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""forward(streams, branch_out, prev_a=None) -> (Tensor, Tensor)

        Combine streams with an already-computed branch output.

        Args:
            streams (Tensor): streams of shape ``(B, L, n, d)``.
            branch_out (Tensor): branch output of shape ``(B, L, d)``.
            prev_a (Tensor, optional): cached input weights of shape
                ``(B, L, 1, n)``; validated but not used for mixing here.
                Default: ``None``.

        Returns:
            Tensor: updated streams of shape ``(B, L, n, d)`` and current
            input weights ``a`` of shape ``(B, L, 1, n)``.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> mhc = SinglePassMHC(MemoryMachineConfig(hidden_size=16, mhc_num_streams=2))
            >>> mhc(torch.randn(1, 3, 2, 16), torch.randn(1, 3, 16))[0].shape
            torch.Size([1, 3, 2, 16])
        """
        a, b, c = self.coefficients(streams)
        if prev_a is not None and (prev_a.shape != a.shape):
            raise ValueError(f"prev_a shape {tuple(prev_a.shape)} does not match current coefficients {tuple(a.shape)}")
        out = torch.matmul(b, streams) + c * branch_out.unsqueeze(2)
        return out, a

    def step(
        self,
        streams: torch.Tensor,
        branch_fn: Callable[[torch.Tensor], torch.Tensor],
        prev_a: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""step(streams, branch_fn, prev_a=None) -> (Tensor, Tensor)

        Run one hyper-connection step around a branch function.

        Mixes streams with ``prev_a`` (or current ``a``), applies
        ``branch_fn``, then recombines with ``b`` and ``c``.

        Args:
            streams (Tensor): streams of shape ``(B, L, n, d)``.
            branch_fn (Callable): maps the mixed input ``(B, L, d)`` to a
                branch output of the same shape.
            prev_a (Tensor, optional): cached input weights of shape
                ``(B, L, 1, n)``. Default: ``None`` (use current ``a``).

        Returns:
            Tensor: updated streams ``(B, L, n, d)`` and current ``a``.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> mhc = SinglePassMHC(MemoryMachineConfig(hidden_size=16, mhc_num_streams=2))
            >>> out, _ = mhc.step(torch.randn(1, 3, 2, 16), lambda t: t)
            >>> out.shape
            torch.Size([1, 3, 2, 16])
        """
        a, b, c = self.coefficients(streams)
        mix_a = prev_a if prev_a is not None else a
        if mix_a.shape != a.shape:
            raise ValueError(f"prev_a shape {tuple(mix_a.shape)} does not match coefficients {tuple(a.shape)}")
        mixed = self.mix_input(streams, mix_a).squeeze(2)
        branch_out = branch_fn(mixed)
        if branch_out.shape != mixed.shape:
            raise ValueError(f"branch output shape {tuple(branch_out.shape)} does not match mixed input {tuple(mixed.shape)}")
        out = torch.matmul(b, streams) + c * branch_out.unsqueeze(2)
        return out, a


class EngramMemory(nn.Module):
    r"""EngramMemory(config)

    Hashed n-gram memory added residually to hidden states.

    For each n-gram order and hash head, token windows are hashed FNV-style
    into table ids (id ``0`` reserved for padding/short prefixes) and looked
    up in learned embedding tables. Lookups are gated per order/head and
    projected back to the model dimension.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``engram_orders``, ``engram_hash_heads``,
            ``engram_dim_per_order``, ``engram_num_entries`` (``>= 2``), and
            ``hidden_size``.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> mem = EngramMemory(MemoryMachineConfig(hidden_size=32, engram_num_entries=64))
        >>> mem(torch.tensor([[1, 2, 3, 4]]), torch.randn(1, 4, 32)).shape
        torch.Size([1, 4, 32])
    """

    _PAD_ID = 0

    def __init__(self, config: MemoryMachineConfig):
        super().__init__()
        self.orders = tuple(config.engram_orders)
        self.hash_heads = config.engram_hash_heads
        self.dim = config.engram_dim_per_order
        self.entries = config.engram_num_entries
        self.hidden_size = config.hidden_size
        if self.entries < 2:
            raise ValueError(f"engram_num_entries ({self.entries}) must be >= 2")
        self.tables = nn.ModuleList([nn.Embedding(self.entries, self.dim, padding_idx=self._PAD_ID) for _ in self.orders for _ in range(self.hash_heads)])
        self.gate_proj = nn.Linear(config.hidden_size, len(self.orders) * self.hash_heads)
        self.out_proj = nn.Linear(len(self.orders) * self.hash_heads * self.dim, config.hidden_size, bias=False)

    def _ngram_id(self, window: torch.Tensor, head: int) -> torch.Tensor:
        r"""_ngram_id(window, head) -> Tensor

        Hash n-gram windows to table ids with an FNV-style mix.

        The accumulator is reduced modulo ``entries - 1`` at each step to stay
        in int64 range, then shifted by one so ``0`` stays reserved for padding.

        Args:
            window (Tensor): token windows of shape ``(N, order)``.
            head (int): hash-head salt.

        Returns:
            Tensor: table ids of shape ``(N,)`` in ``[1, entries - 1]``.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> mem = EngramMemory(MemoryMachineConfig(hidden_size=16, engram_num_entries=64))
            >>> mem._ngram_id(torch.tensor([[1, 2], [3, 4]]), 0).shape
            torch.Size([2])
        """
        # FNV-style mix; mod each step keeps h in int64 range.
        h = torch.zeros_like(window[:, 0])
        for t in window.unbind(dim=1):
            h = (h * 9113823 + t + 1 + head * 2654435761) % (self.entries - 1) + 1
        return h

    def forward(self, input_ids: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""forward(input_ids, hidden_states) -> Tensor

        Add gated n-gram memory to hidden states.

        Args:
            input_ids (Tensor): token ids of shape ``(B, Lq)``; non-negative.
            hidden_states (Tensor): hidden states of shape ``(B, Lq, D)``.

        Returns:
            Tensor: memory-augmented states of shape ``(B, Lq, D)`` (residual).

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> mem = EngramMemory(MemoryMachineConfig(hidden_size=16, engram_num_entries=64))
            >>> mem(torch.tensor([[5, 6, 7]]), torch.randn(1, 3, 16)).shape
            torch.Size([1, 3, 16])
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, Lq], got {tuple(input_ids.shape)}")
        if hidden_states.shape[:2] != input_ids.shape:
            raise ValueError(f"hidden_states batch/seq {tuple(hidden_states.shape[:2])} does not match input_ids {tuple(input_ids.shape)}")
        if bool((input_ids < 0).any()):
            raise ValueError("input_ids must be non-negative token ids")
        bsz, lq = input_ids.shape
        feats: list[torch.Tensor] = []
        ti = 0
        for order in self.orders:
            for head in range(self.hash_heads):
                if lq >= order:
                    windows = input_ids.unfold(1, order, 1)
                    ids = self._ngram_id(windows.reshape(-1, order), head).view(bsz, -1)
                    pad = torch.full((bsz, lq - ids.shape[1]), self._PAD_ID, dtype=torch.long, device=ids.device)
                    ids = torch.cat([pad, ids], dim=1)
                else:
                    ids = torch.full((bsz, lq), self._PAD_ID, dtype=torch.long, device=input_ids.device)
                feats.append(self.tables[ti](ids))
                ti += 1
        mem = torch.cat(feats, dim=-1)
        gate = torch.sigmoid(self.gate_proj(hidden_states)).unsqueeze(-1)
        mem = mem.view(bsz, lq, -1, self.dim) * gate
        return self.out_proj(mem.reshape(bsz, lq, -1)) + hidden_states


class DSparkDrafter(nn.Module):
    r"""DSparkDrafter(config)

    Hidden-state speculative drafter.

    Pools the trailing ``dspark_window`` hidden states into one context vector,
    proposes ``num_drafts`` hidden-state drafts, refines them with a residual
    ``tanh`` Markov head, and scores each draft with a sigmoid confidence head.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``dspark_num_drafts``, ``dspark_window``, and
            ``hidden_size``.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> d = DSparkDrafter(MemoryMachineConfig(hidden_size=32))
        >>> drafts, conf = d(torch.randn(2, 10, 32))
        >>> drafts.shape, conf.shape
        (torch.Size([2, 5, 32]), torch.Size([2, 5]))
    """

    def __init__(self, config: MemoryMachineConfig):
        super().__init__()
        self.num_drafts = config.dspark_num_drafts
        self.window = config.dspark_window
        d = config.hidden_size
        self.draft_proj = nn.Linear(d, self.num_drafts * d, bias=False)
        self.markov_head = nn.Linear(d, d, bias=False)
        self.confidence_head = nn.Linear(d, self.num_drafts)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""forward(hidden_states) -> (Tensor, Tensor)

        Draft speculative continuations from recent hidden states.

        Args:
            hidden_states (Tensor): input of shape ``(B, L, D)``.

        Returns:
            Tensor: drafts of shape ``(B, num_drafts, D)`` and confidences of
            shape ``(B, num_drafts)`` in ``(0, 1)``.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> d = DSparkDrafter(MemoryMachineConfig(hidden_size=16, dspark_num_drafts=3))
            >>> drafts, conf = d(torch.randn(1, 8, 16))
            >>> drafts.shape, conf.shape
            (torch.Size([1, 3, 16]), torch.Size([1, 3]))
        """
        tail = hidden_states[:, -self.window :, :]
        ctx = tail.mean(dim=1, keepdim=True)
        drafts = self.draft_proj(ctx).view(hidden_states.shape[0], self.num_drafts, -1)
        drafts = drafts + torch.tanh(self.markov_head(drafts))
        conf = torch.sigmoid(self.confidence_head(ctx).squeeze(1))
        return drafts, conf
