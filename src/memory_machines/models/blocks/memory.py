r"""Auxiliary memory blocks: hyper-connections, engram memory, speculative drafter.

:class:`SinglePassMHC` expands the residual stream into parallel streams mixed
by learned coefficients, :class:`EngramMemory` adds hashed n-gram memory, and
:class:`DSparkDrafter` proposes speculative hidden-state drafts.
"""

from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from memory_machines.utils.config_schema import MemoryMachineConfig

from .shared import _sinkhorn


def _is_prime(n: int) -> bool:
    r"""Deterministic Miller-Rabin primality test for ``n < 2^64``."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _next_prime(start: int, seen: set[int]) -> int:
    r"""Return the smallest prime >= ``start`` that is not in ``seen``."""
    candidate = max(2, start)
    while candidate in seen or not _is_prime(candidate):
        candidate += 1
    return candidate


def _odd_multipliers(num: int, seed: int, upper: int) -> torch.Tensor:
    r"""Draw ``num`` odd multipliers in ``[1, 2*upper)``.

    Deterministic via a CPU generator seeded with ``seed``; the
    exact values differ from the reference's NumPy stream but the
    oddness and uniform distribution match.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    raw = torch.randint(0, upper, (num,), generator=g, dtype=torch.int64)
    return raw * 2 + 1


class _DiskGather(torch.autograd.Function):
    r"""Gather rows from an SSD-backed (numpy memmap) table with gradients.

    The forward reads only the gathered rows from the memory-mapped
    file (the OS pages the rest in lazily, so the full table never
    needs to fit in RAM) and moves them to the compute device. The
    backward scatters the upstream gradient into a full-sized zero
    tensor on the weight's (CPU) device, so ``weight`` receives a
    dense gradient exactly like a normal :class:`torch.nn.Embedding`.
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(index)
        ctx.weight_shape = tuple(weight.shape)
        ctx.weight_dtype = weight.dtype
        ctx.weight_device = weight.device
        idx = index.to(device=weight.device, dtype=torch.long)
        return weight.index_select(0, idx).to(index.device)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        (index,) = ctx.saved_tensors
        grad = torch.zeros(ctx.weight_shape, dtype=ctx.weight_dtype, device=ctx.weight_device)
        idx = index.to(device=ctx.weight_device, dtype=torch.long)
        grad.index_add_(0, idx, grad_output.to(device=ctx.weight_device, dtype=ctx.weight_dtype))
        return grad, None


class _SsdTable(nn.Module):
    r"""_SsdTable(path, num_embeddings, embedding_dim)

    Hash-table embedding whose weights live in a numpy memory-mapped
    file on SSD and are gathered on demand.

    The module holds no dedicated RAM or GPU copy of the table: a
    forward gathers only the rows named by ``index`` into the
    compute device via :class:`_DiskGather`, while ``weight`` (a
    plain CPU tensor aliasing the memmap) carries a dense gradient
    for the optimizer and writes through to the file. Automatic
    offloading: the whole table stays on SSD and the OS pages in
    only the touched rows, so a table far larger than RAM is
    usable. Storage is always float32.

    Args:
        path (str | Path): file backing the table; created (and
            parent directories made) if missing, reused otherwise.
        num_embeddings (int): number of rows.
        embedding_dim (int): width of each row.

    Examples::

        >>> import tempfile, torch
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     t = _SsdTable(Path(d) / "t.bin", 6, 3)
        ...     t(torch.tensor([0, 5, 2])).shape
        torch.Size([3, 3])
    """

    def __init__(self, path, num_embeddings: int, embedding_dim: int):
        super().__init__()
        import numpy as np

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nbytes = num_embeddings * embedding_dim * np.dtype(np.float32).itemsize
        if not path.exists() or path.stat().st_size != nbytes:
            # Preallocate via sparse extend, not a materialized
            # zero buffer, so huge tables never occupy RAM here.
            with open(path, "wb") as f:
                f.truncate(nbytes)
        self._mmap = np.memmap(str(path), dtype=np.float32, mode="r+", shape=(num_embeddings, embedding_dim))
        # Plain CPU tensor aliasing the memmap: RAM holds only the OS
        # page cache (never a dedicated allocation). Marking it as a
        # leaf that requires grad lets autograd accumulate the scatter
        # gradient into .grad, and in-place optimizer steps write
        # through to the file.
        self.weight = torch.from_numpy(self._mmap).requires_grad_(True)
        self.embedding_dim = embedding_dim

    def forward(self, index: torch.Tensor) -> torch.Tensor:
        r"""forward(index) -> Tensor

        Gather rows from the SSD-backed table.

        Args:
            index (Tensor): row indices of any shape, non-negative and
                below ``num_embeddings``.

        Returns:
            Tensor: gathered rows of shape ``(*index.shape, embedding_dim)``
            on the compute device.
        """
        flat = index.reshape(-1).to(torch.long)
        gathered = _DiskGather.apply(self.weight, flat)
        return gathered.reshape(*index.shape, self.embedding_dim)

    def normal_init_(self, std: float, generator: torch.Generator) -> None:
        r"""normal_init_(std, generator) -> None

        Fill the table with ``normal_(0, std)`` via a deterministic
        generator, writing in chunks so initialization never
        materializes the whole table in memory.
        """
        rows = self._mmap.shape[0]
        width = self._mmap.shape[1]
        chunk = max(1, 2**22 // max(1, width))
        with torch.no_grad():
            for start in range(0, rows, chunk):
                end = min(rows, start + chunk)
                block = torch.empty((end - start, width), dtype=torch.float32)
                torch.nn.init.normal_(block, std=std, generator=generator)
                self._mmap[start:end] = block.numpy()
        self._mmap.flush()


class _EngramShortConv(nn.Module):
    r"""_EngramShortConv(hidden_size, streams, kernel_size, dilation)

    Short depthwise causal convolution over the Engram value tensor.

    Each MHC stream is RMS-normed independently, the normalized streams
    are concatenated along channels, and a depthwise (grouped)
    :class:`torch.nn.Conv1d` with ``kernel_size`` and ``dilation``
    (set to the maximum n-gram order so the receptive field spans one
    full n-gram) mixes each channel with its causal context before a
    SiLU activation.

    Args:
        hidden_size (int): model dimension per stream.
        streams (int): number of MHC streams (conv groups per channel block).
        kernel_size (int): conv kernel width (``4`` in the reference).
        dilation (int): conv dilation (maximum n-gram order).

    Examples::

        >>> conv = _EngramShortConv(hidden_size=8, streams=2, kernel_size=4, dilation=3)
        >>> conv(torch.randn(1, 5, 2, 8)).shape
        torch.Size([1, 5, 2, 8])
    """

    def __init__(self, hidden_size: int, streams: int, kernel_size: int, dilation: int):
        super().__init__()
        total_channels = hidden_size * streams
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.norms = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(streams)])
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x) -> Tensor

        Apply the depthwise causal convolution.

        Args:
            x (Tensor): value tensor of shape ``(B, L, streams, hidden_size)``.

        Returns:
            Tensor: refined tensor of shape ``(B, L, streams, hidden_size)``.
        """
        bsz, length, streams, hidden = x.shape
        normed = torch.cat([self.norms[i](x[:, :, i, :]) for i in range(streams)], dim=-1)
        y = self.conv(normed.transpose(1, 2))[..., :length]
        y = self.act(y)
        return y.transpose(1, 2).reshape(bsz, length, streams, hidden)


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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        RMSNorm to unit gain; dynamic mixers Xavier-uniform but
        temperature-gated by ``alpha = 1e-3``, so coefficients start at
        the statics: uniform input/residual mixing and unit output scale
        (near-identity stream mixing). Static biases stay zero, i.e.
        ``sigmoid(0) = 0.5`` input weights, uniform Sinkhorn rows, and
        ``2 * sigmoid(0) = 1`` output scales.
        """
        self.pre_norm.reset_parameters()
        nn.init.xavier_uniform_(self.a_dyn.weight)
        nn.init.zeros_(self.a_dyn.bias)
        nn.init.xavier_uniform_(self.b_dyn.weight)
        nn.init.zeros_(self.b_dyn.bias)
        nn.init.xavier_uniform_(self.c_dyn.weight)
        nn.init.zeros_(self.c_dyn.bias)
        nn.init.zeros_(self.a_static)
        nn.init.zeros_(self.b_static)
        nn.init.zeros_(self.c_static)
        nn.init.constant_(self.alpha_pre, 1e-3)
        nn.init.constant_(self.alpha_res, 1e-3)
        nn.init.constant_(self.alpha_post, 1e-3)

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

    @staticmethod
    def _combine(
        streams: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        branch_out: torch.Tensor,
    ) -> torch.Tensor:
        r"""_combine(streams, b, c, branch_out) -> Tensor

        Residual recombination shared by :meth:`forward` and
        :meth:`step`.

        Args:
            streams (Tensor): streams of shape ``(B, L, n, d)``.
            b (Tensor): residual mixing weights of shape ``(B, L, n, n)``.
            c (Tensor): output scaling of shape ``(B, L, n, 1)``.
            branch_out (Tensor): branch output of shape ``(B, L, d)``.

        Returns:
            Tensor: updated streams of shape ``(B, L, n, d)``.
        """
        return torch.matmul(b, streams) + c * branch_out.unsqueeze(2)

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
        out = self._combine(streams, b, c, branch_out)
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
        out = self._combine(streams, b, c, branch_out)
        return out, a


class EngramMemory(nn.Module):
    r"""EngramMemory(config)

    Hashed n-gram memory from "Engrams: Decoupling Memorization from
    Attention in Language Models" (arXiv:2601.07372), following the
    official DeepSeek reference implementation.

    Token windows of each configured order are hashed with a
    multiplicative-XOR function into prime-sized tables (one flat
    :class:`torch.nn.Embedding` with per-head offsets), looked up,
    gated against the current hidden state per MHC stream, projected
    back to the model dimension, and refined by a short depthwise
    causal convolution, then added residually to the input.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``engram_orders``, ``engram_hash_heads``,
            ``engram_dim_per_order``, ``engram_num_entries``,
            ``hidden_size``, ``vocab_size``, ``mhc_num_streams``,
            ``pad_token_id``, ``engram_use_conv``,
            ``engram_layer_positions``, and ``seed``.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> mem = EngramMemory(MemoryMachineConfig(hidden_size=32, engram_num_entries=64))
        >>> mem(torch.tensor([[1, 2, 3, 4]]), torch.randn(1, 4, 32), layer_id=0).shape
        torch.Size([1, 4, 32])
    """

    def __init__(self, config: MemoryMachineConfig):
        super().__init__()
        self.orders = tuple(config.engram_orders)
        self.hash_heads = config.engram_hash_heads
        self.dim = config.engram_dim_per_order
        self.entries = config.engram_num_entries
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.streams = config.mhc_num_streams
        self.pad_id = config.pad_token_id
        self.use_conv = config.engram_use_conv
        self.engram_layer_positions = tuple(config.engram_layer_positions)
        self.engram_offload = config.engram_offload
        self.engram_offload_dir = config.engram_offload_dir
        self.engram_offload_ram_threshold = config.engram_offload_ram_threshold
        self.seed = config.seed
        self.init_std = config.initializer_range
        if self.entries < 2:
            raise ValueError(f"engram_num_entries ({self.entries}) must be >= 2")
        if self.hidden_size % self.streams != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by mhc_num_streams ({self.streams})")
        if any(order < 2 for order in self.orders):
            raise ValueError(f"engram_orders must be >= 2, got {self.orders}")
        max_order = max(self.orders)
        self.conv: _EngramShortConv | None = None
        if self.use_conv:
            self.conv = _EngramShortConv(
                hidden_size=self.hidden_size,
                streams=self.streams,
                kernel_size=4,
                dilation=max_order,
            )
        # Distinct primes >= entries - 1, shared across layers so
        # table geometry is layer-independent.
        seen: set[int] = set()
        primes: list[int] = []
        for _ in self.orders:
            for _ in range(self.hash_heads):
                prime = _next_prime(self.entries - 1, seen)
                seen.add(prime)
                primes.append(prime)
        self.primes = primes
        # Per-head offsets into the flat table: head ti owns
        # [offsets[ti], offsets[ti] + primes[ti]).
        offsets = [0]
        for p in primes[:-1]:
            offsets.append(offsets[-1] + p)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        # Storage mode: "ram" keeps the flat table in memory; "ssd"
        # and "auto" back it with a memmap, "auto" only above
        # engram_offload_ram_threshold bytes.
        self._use_ssd = self._resolve_offload()
        if self._use_ssd:
            self.multi_head_embedding = _SsdTable(
                Path(config.engram_offload_dir) / f"engram_layer_{config.seed}.bin",
                sum(primes),
                self.dim,
            )
        else:
            self.multi_head_embedding = nn.Embedding(sum(primes), self.dim)
        # Per-layer odd multipliers, seeded seed + 10007 * layer_id
        # as in the reference implementation.
        self._multiplier_cache: dict[int, torch.Tensor] = {}
        self.value_proj = nn.Linear(len(self.orders) * self.hash_heads * self.dim, self.hidden_size, bias=False)
        self.key_projs = nn.ModuleList([nn.Linear(len(self.orders) * self.hash_heads * self.dim, self.hidden_size, bias=False) for _ in range(self.streams)])
        self.norm1 = nn.ModuleList([nn.RMSNorm(self.hidden_size) for _ in range(self.streams)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(self.hidden_size) for _ in range(self.streams)])
        self.reset_parameters()

    def _resolve_offload(self) -> bool:
        r"""_resolve_offload() -> bool

        Whether the flat table is SSD-backed. ``"ssd"`` always
        offloads; ``"auto"`` offloads only above
        ``engram_offload_ram_threshold`` bytes of RAM; ``"ram"``
        never offloads.
        """
        if self.engram_offload == "ssd":
            return True
        if self.engram_offload == "auto":
            return sum(self.primes) * self.dim * 4 > self.engram_offload_ram_threshold
        return False

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Tables from a small normal (``std = initializer_range``,
        GPT-2 embedding style); value and key projections zeroed so
        the block contributes nothing at step 0 and learns to use
        n-gram memory gradually (cf. LoRA's zero-start ``B``). The
        convolution weights are also zero-initialized in the
        reference, making the conv an identity at start.
        SSD-backed tables are initialized in chunks directly into
        the memmap so initialization never materializes the table.
        """
        if isinstance(self.multi_head_embedding, _SsdTable):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed)
            self.multi_head_embedding.normal_init_(self.init_std, gen)
        else:
            nn.init.normal_(self.multi_head_embedding.weight, std=self.init_std)
        nn.init.zeros_(self.value_proj.weight)
        for proj in self.key_projs:
            assert isinstance(proj, nn.Linear)
            nn.init.zeros_(proj.weight)
        if self.conv is not None:
            nn.init.zeros_(self.conv.conv.weight)

    def _get_multipliers(self, layer_id: int) -> torch.Tensor:
        r"""Odd hash multipliers for ``layer_id``, seeded ``seed + 10007 * layer_id``.

        The multiplier bound mirrors the reference implementation's
        ``int64_max // vocab_size // 2`` so that ``token * multiplier``
        stays inside the int64 range.
        """
        if layer_id not in self._multiplier_cache:
            max_order = max(self.orders)
            half_bound = max(1, (torch.iinfo(torch.int64).max // max(self.vocab_size, 2)) // 2)
            self._multiplier_cache[layer_id] = _odd_multipliers(max_order, self.seed + 10007 * layer_id, half_bound)
        return self._multiplier_cache[layer_id]

    def _ngram_hashes(self, input_ids: torch.Tensor, layer_id: int) -> torch.Tensor:
        r"""_ngram_hashes(input_ids, layer_id) -> Tensor

        Multiplicative-XOR hash of suffix n-grams into per-head table ids.

        For order ``n`` the n-gram ending at position ``t`` is
        ``(x'_{t-n+1}, ..., x'_t)``; the hash is the XOR of each token
        scaled by its per-layer odd multiplier, reduced modulo the head's
        prime table size. Prefixes shorter than ``n`` are left-padded with
        ``pad_token_id``.

        Args:
            input_ids (Tensor): token ids of shape ``(B, Lq)``; non-negative.
            layer_id (int): layer index selecting the multiplier set.

        Returns:
            Tensor: table ids of shape ``(B, Lq, heads * orders)``.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> mem = EngramMemory(MemoryMachineConfig(hidden_size=16, engram_num_entries=64))
            >>> mem._ngram_hashes(torch.tensor([[1, 2], [3, 4]]), 0).shape
            torch.Size([2, 2, 12])
        """
        x = input_ids.to(torch.int64)
        bsz, lq = x.shape
        multipliers = self._get_multipliers(layer_id)
        max_order = max(self.orders)
        # shift_k left-pads x with exactly k pad-ids, so the n-gram
        # ending at t contributes x[t-k] and the pad id before that.
        pads = torch.full((bsz, max_order), self.pad_id, dtype=torch.int64, device=x.device)
        padded = torch.cat([pads, x], dim=1)
        shifts = [padded[:, max_order - k : max_order - k + lq] for k in range(max_order)]
        hashes: list[torch.Tensor] = []
        ti = 0
        for order in self.orders:
            mix = shifts[0] * multipliers[0]
            for k in range(1, order):
                mix = torch.bitwise_xor(mix, shifts[k] * multipliers[k])
            for head in range(self.hash_heads):
                hashes.append(mix % self.primes[ti] + self.offsets[ti])
                ti += 1
        return torch.stack(hashes, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        layer_id: int = 0,
    ) -> torch.Tensor:
        r"""forward(input_ids, hidden_states, layer_id) -> Tensor

        Add gated n-gram memory to hidden states.

        Args:
            input_ids (Tensor): token ids of shape ``(B, Lq)``; non-negative.
            hidden_states (Tensor): hidden states of shape ``(B, Lq, D)``.
            layer_id (int): layer index used to select the hash multipliers;
                the block returns ``hidden_states`` unchanged when the
                layer is not in ``engram_layer_positions``.

        Returns:
            Tensor: memory-augmented states of shape ``(B, Lq, D)`` (residual).

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> mem = EngramMemory(MemoryMachineConfig(hidden_size=16, engram_num_entries=64))
            >>> mem(torch.tensor([[5, 6, 7]]), torch.randn(1, 3, 16)).shape
            torch.Size([1, 3, 16])
        """
        if self.engram_layer_positions and layer_id not in self.engram_layer_positions:
            return hidden_states
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, Lq], got {tuple(input_ids.shape)}")
        if hidden_states.shape[:2] != input_ids.shape:
            raise ValueError(f"hidden_states batch/seq {tuple(hidden_states.shape[:2])} does not match input_ids {tuple(input_ids.shape)}")
        if bool((input_ids < 0).any()):
            raise ValueError("input_ids must be non-negative token ids")
        hashes = self._ngram_hashes(input_ids, layer_id)
        embeddings = self.multi_head_embedding(hashes).flatten(start_dim=-2)
        gates: list[torch.Tensor] = []
        for stream in range(self.streams):
            key = self.key_projs[stream](embeddings)
            normed_key = self.norm1[stream](key)
            # 4-D input carries one hidden state per MHC stream; 3-D
            # input is a single stream, the shared query for every key.
            query = hidden_states[:, :, stream] if hidden_states.dim() == 4 else hidden_states
            normed_query = self.norm2[stream](query)
            gate = (normed_key * normed_query).sum(dim=-1) / (self.hidden_size**0.5)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gates.append(torch.sigmoid(gate).unsqueeze(-1))
        gate = torch.stack(gates, dim=2)
        value = gate * self.value_proj(embeddings).unsqueeze(2)
        if self.conv is not None:
            value = value + self.conv(value)
        if hidden_states.dim() == 4:
            return value + hidden_states
        # Collapse the per-stream gated values into the single input stream:
        # summing matches the hyper-connection output mixing of M streams.
        return value.sum(dim=2) + hidden_states


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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Xavier-uniform draft proposals; the Markov refinement head
        zeroed so drafts start unrefined (``tanh(0) = 0`` residual) and
        the confidence head zeroed for maximal initial uncertainty
        (``sigmoid(0) = 0.5``). The drafter thus proposes from context
        statistics first and earns confidence through training.
        """
        nn.init.xavier_uniform_(self.draft_proj.weight)
        nn.init.zeros_(self.markov_head.weight)
        nn.init.zeros_(self.confidence_head.weight)
        nn.init.zeros_(self.confidence_head.bias)

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
