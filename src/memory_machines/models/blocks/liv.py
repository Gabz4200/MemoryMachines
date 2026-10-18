r"""LIV blocks: multihead double-gated conv and perceiver.

Heads stay batched via reshapes (grouped Conv1d, batched GEMMs); ``torch.vmap``
would forfeit that fusion for tracing overhead.
"""

import typing

import torch
from torch import nn

from memory_machines.utils.config_schema import MemoryMachineConfig

from .attention import GroupedQueryAttention
from .causal_conv1d import ShortConvolution
from .shared import CSA2SharedState, residual_init_scale
from .sparse import CompressedSparseAttention2


class LivDoubleGatedBase(nn.Module):
    r"""LivDoubleGatedBase(config)

    Double-gated LIV wrapper; subclasses define :meth:`mix_heads`.

    The input is projected to per-head ``(pre_gate, post_gate, x_proj)`` views
    of shape ``(B, L, H, Dh)``. The mixer transforms ``pre_gate * x_proj`` with
    head-factorized token mixing (no cross-head mixing), then the output is
    ``post_gate * mixed`` followed by the output projection. See
    :class:`LivConvBlock` and :class:`LivPerceiver` for concrete mixers.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``hidden_size``, ``num_attention_heads``, and
            ``hidden_dropout``.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> from memory_machines.models.blocks.liv import LivConvBlock
        >>> blk = LivConvBlock(MemoryMachineConfig(hidden_size=32, num_attention_heads=4, csa_num_output_groups=4))
        >>> blk(torch.randn(1, 8, 32)).shape
        torch.Size([1, 8, 32])
    """

    def __init__(self, config: MemoryMachineConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.initializer_range: float = config.initializer_range
        self.num_layers: int = config.num_layers
        self.residual_init_scale: bool = config.residual_init_scale
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})")
        self.head_dim = self.hidden_size // self.num_heads

        self.input_projection = nn.Linear(
            self.hidden_size,
            3 * self.hidden_size,
            bias=False,
        )

        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.hidden_dropout)
        LivDoubleGatedBase.reset_parameters(self)

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Truncated normal ``N(0, initializer_range^2)`` for
        the 3-way input projection; the output projection
        is either zero-initialized (identity residual start,
        Fixup/ReZero) or depth-scaled to
        ``initializer_range / sqrt(2N)`` (Megatron-LM/Qwen
        residual scaling) per ``residual_init_scale``, so
        the whole double-gated branch starts as a bounded
        residual despite its two multiplicative gatings.
        Subclass mixers define their own start (Dirac
        identity for conv, small-normal latents for
        perceiver).
        """
        std = self.initializer_range
        scale = residual_init_scale(std, self.num_layers, self.residual_init_scale)
        nn.init.trunc_normal_(self.input_projection.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        if scale is None:
            nn.init.zeros_(self.output_projection.weight)
        else:
            nn.init.trunc_normal_(self.output_projection.weight, mean=0.0, std=scale, a=-2 * scale, b=2 * scale)

    def split_gates_heads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""split_gates_heads(x) -> (Tensor, Tensor, Tensor, Tensor)

        Split the input into per-head gate/value/projected views and the
        gated product ``pre_gate * x_proj``.

        Gates are raw linear products (``pre * x``, ``post * mixed``) with no
        bounded activation by design (LIV): unbounded variance is contained by
        the zero-start/depth-scaled output projection, which keeps the whole
        branch a bounded residual at init. Do not add SiLU/sigmoid without
        re-tuning parity tests.

        Args:
            x (Tensor): input of shape ``(B, L, hidden)``.

        Returns:
            Tensor: ``pre_gate``, ``post_gate``, ``x_proj`` of shape
            ``(B, L, H, Dh)``, and the gated product ``pre_gate * x_proj``
            with the same shape.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> from memory_machines.models.blocks.liv import LivConvBlock
            >>> blk = LivConvBlock(MemoryMachineConfig(hidden_size=16, num_attention_heads=2, csa_num_output_groups=2, num_key_value_heads=1))
            >>> [t.shape for t in blk.split_gates_heads(torch.randn(1, 4, 16))]
            [torch.Size([1, 4, 2, 8]), torch.Size([1, 4, 2, 8]), torch.Size([1, 4, 2, 8]), torch.Size([1, 4, 2, 8])]
        """
        bsz, seqlen, _ = x.shape
        projected = self.input_projection(x).view(bsz, seqlen, 3, self.num_heads, self.head_dim)
        pre_gate, post_gate, x_proj = projected.unbind(dim=2)
        return pre_gate, post_gate, x_proj, pre_gate * x_proj

    def split_gates(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""split_gates(x) -> (Tensor, Tensor, Tensor, Tensor)

        Merged ``(B, L, hidden)`` view of :meth:`split_gates_heads`.

        Args:
            x (Tensor): input of shape ``(B, L, hidden)``.

        Returns:
            Tensor: ``pre``, ``post``, ``proj``, and ``gated`` views, each of
            shape ``(B, L, hidden)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> from memory_machines.models.blocks.liv import LivConvBlock
            >>> blk = LivConvBlock(MemoryMachineConfig(hidden_size=16, num_attention_heads=2, csa_num_output_groups=2, num_key_value_heads=1))
            >>> [t.shape for t in blk.split_gates(torch.randn(1, 4, 16))]
            [torch.Size([1, 4, 16]), torch.Size([1, 4, 16]), torch.Size([1, 4, 16]), torch.Size([1, 4, 16])]
        """
        bsz, seqlen, _ = x.shape
        pre, post, proj, gated = self.split_gates_heads(x)

        def _merge(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(bsz, seqlen, self.hidden_size)

        return _merge(pre), _merge(post), _merge(proj), _merge(gated)

    def mix_heads(self, x_gated: torch.Tensor, x_query: torch.Tensor | None = None, query_mask: torch.Tensor | None = None) -> torch.Tensor:
        r"""mix_heads(x_gated, x_query=None, query_mask=None) -> Tensor

        Per-head token mixing: ``(B, L, H, Dh)`` -> same shape.

        ``x_query`` is the ungated projected value stream
        (``x_proj`` from :meth:`split_gates_heads`); mixers that seed a
        cross-attention query stream from the input (e.g. the Perceiver's
        ``"cross"`` query mode) read it here. Mixers that use a learned latent
        query (``"learned"``) or a convolution ignore it.

        ``query_mask`` is a ``(B, L)`` boolean mask selecting which tokens
        produce queries in the Perceiver's ``"masked"`` query mode; all tokens
        still produce keys and values. Mixers that do not use a query mask
        ignore it.

        Args:
            x_gated (Tensor): gated inputs of shape ``(B, L, H, Dh)``.
            x_query (Tensor, optional): ungated projected inputs of shape
                ``(B, L, H, Dh)``. Default: ``None``.
            query_mask (Tensor, optional): ``(B, L)`` bool query-selector mask.
                Default: ``None``.

        Returns:
            Tensor: mixed outputs with the same shape as ``x_gated``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> from memory_machines.models.blocks.liv import LivConvBlock
            >>> blk = LivConvBlock(MemoryMachineConfig(hidden_size=16, num_attention_heads=2, csa_num_output_groups=2, num_key_value_heads=1))
            >>> blk.mix_heads(torch.randn(1, 4, 2, 8)).shape
            torch.Size([1, 4, 2, 8])
        """
        raise NotImplementedError

    def token_mix(self, x_gated: torch.Tensor, x_query: torch.Tensor | None = None, query_mask: torch.Tensor | None = None) -> torch.Tensor:
        r"""token_mix(x_gated, x_query=None, query_mask=None) -> Tensor

        Merged ``(B, L, hidden)`` adapter around :meth:`mix_heads`.

        Args:
            x_gated (Tensor): merged input of shape ``(B, L, hidden)``.
            x_query (Tensor, optional): ungated projected inputs of shape
                ``(B, L, hidden)``; forwarded to mixers that seed a
                cross-attention query stream from the input.
            query_mask (Tensor, optional): ``(B, L)`` bool query-selector mask;
                forwarded to mixers that use a query mask.

        Returns:
            Tensor: mixed output of shape ``(B, L, hidden)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> from memory_machines.models.blocks.liv import LivConvBlock
            >>> blk = LivConvBlock(MemoryMachineConfig(hidden_size=16, num_attention_heads=2, csa_num_output_groups=2, num_key_value_heads=1))
            >>> blk.token_mix(torch.randn(1, 4, 16)).shape
            torch.Size([1, 4, 16])
        """
        bsz, seqlen, _ = x_gated.shape
        heads = x_gated.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        q_heads = x_query.reshape(bsz, seqlen, self.num_heads, self.head_dim) if x_query is not None else None
        return self.mix_heads(heads, q_heads, query_mask).reshape(bsz, seqlen, self.hidden_size)

    def forward(self, x: torch.Tensor, query_mask: torch.Tensor | None = None) -> torch.Tensor:
        r"""forward(x, query_mask=None) -> Tensor

        Apply double-gated mixing to the input sequence.

        Args:
            x (Tensor): input of shape ``(B, L, hidden)``.
            query_mask (Tensor, optional): ``(B, L)`` bool query-selector mask
                forwarded to mixers that use one (e.g. the Perceiver's
                ``"masked"`` query mode). Mixers that do not use a query mask
                (the convolution, and the Perceiver's ``"learned"``/``"cross"``
                query modes) accept it for API parity and ignore it.
                Default: ``None``.

        Returns:
            Tensor: output of shape ``(B, L, hidden)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> from memory_machines.models.blocks.liv import LivConvBlock
            >>> LivConvBlock(MemoryMachineConfig(hidden_size=16, num_attention_heads=2, csa_num_output_groups=2, num_key_value_heads=1))(torch.randn(1, 4, 16)).shape
            torch.Size([1, 4, 16])
        """
        _, post_gate, x_proj, x_gated = self.split_gates_heads(x)
        return self.finish_heads(post_gate, self.mix_heads(x_gated, x_proj, query_mask))

    def finish_heads(self, post_gate: torch.Tensor, mixed: torch.Tensor) -> torch.Tensor:
        r"""finish_heads(post_gate, mixed) -> Tensor

        Merge head outputs and apply the output projection.

        Args:
            post_gate (Tensor): post gates of shape ``(B, L, H, Dh)``.
            mixed (Tensor): mixed values of shape ``(B, L, H, Dh)``.

        Returns:
            Tensor: projected output of shape ``(B, L, hidden)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> from memory_machines.models.blocks.liv import LivConvBlock
            >>> blk = LivConvBlock(MemoryMachineConfig(hidden_size=16, num_attention_heads=2, csa_num_output_groups=2, num_key_value_heads=1))
            >>> blk.finish_heads(torch.randn(1, 4, 2, 8), torch.randn(1, 4, 2, 8)).shape
            torch.Size([1, 4, 16])
        """
        bsz, seqlen, _, _ = post_gate.shape
        merged = (post_gate * mixed).reshape(bsz, seqlen, self.hidden_size)
        return self.output_projection(self.dropout(merged))

    def finish(self, post_gate: torch.Tensor, mixed: torch.Tensor) -> torch.Tensor:
        r"""finish(post_gate, mixed) -> Tensor

        Merge already-flattened gates and apply the output projection.

        Args:
            post_gate (Tensor): post gates of shape ``(B, L, hidden)``.
            mixed (Tensor): mixed values of shape ``(B, L, hidden)``.

        Returns:
            Tensor: projected output of shape ``(B, L, hidden)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> from memory_machines.models.blocks.liv import LivConvBlock
            >>> blk = LivConvBlock(MemoryMachineConfig(hidden_size=16, num_attention_heads=2, csa_num_output_groups=2, num_key_value_heads=1))
            >>> blk.finish(torch.randn(1, 4, 16), torch.randn(1, 4, 16)).shape
            torch.Size([1, 4, 16])
        """
        return self.output_projection(self.dropout(post_gate * mixed))


class LivConvBlock(LivDoubleGatedBase):
    r"""LivConvBlock(config)

    Short-conv LIV block: depthwise causal convolution over the merged hidden
    state via :class:`~memory_machines.models.blocks.causal_conv1d.ShortConvolution`.

    Token mixing is strictly per-channel (hence per-head independent), causal
    (no future leakage — position ``t`` only reads ``<= t``), and stateless
    here (no cache is threaded through; the mixer runs the full-sequence
    path). Kernel size comes from ``config.conv_kernel_size``. The mixer is
    linear (``activation=None``); non-linearity comes from the double gating.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``hidden_size``, ``num_attention_heads``,
            ``conv_kernel_size``, and ``hidden_dropout``.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> blk = LivConvBlock(MemoryMachineConfig(hidden_size=32, num_attention_heads=4, csa_num_output_groups=4))
        >>> blk(torch.randn(2, 8, 32)).shape
        torch.Size([2, 8, 32])
    """

    def __init__(self, config: MemoryMachineConfig):
        super().__init__(config)

        self.conv = ShortConvolution(
            hidden_size=self.hidden_size,
            kernel_size=config.conv_kernel_size,
            bias=False,
            activation=None,
            backend="naive",
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Causal-Dirac (identity) init: each channel reads only its own
        current token (last tap = 1, rest = 0), so the mixer preserves the
        token stream exactly at step 0 and learns local past context from
        an identity start (DiracNets, Zagoruyko & Komodakis, 2016; cf.
        Fixup identity-start).
        """
        LivDoubleGatedBase.reset_parameters(self)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, 0, -1] = 1.0

    def mix_heads(self, x_gated: torch.Tensor, x_query: torch.Tensor | None = None, query_mask: torch.Tensor | None = None) -> torch.Tensor:
        r"""mix_heads(x_gated, x_query=None, query_mask=None) -> Tensor

        Apply the per-channel causal short convolution. ``x_query`` and
        ``query_mask`` are accepted for API parity with :class:`LivPerceiver`
        and ignored: the conv mixer is stateless and reads only the gated
        stream.

        Args:
            x_gated (Tensor): gated inputs of shape ``(B, L, H, Dh)``.
            x_query (Tensor, optional): ungated projected inputs; ignored.
            query_mask (Tensor, optional): ``(B, L)`` bool query-selector mask;
                ignored.

        Returns:
            Tensor: convolved outputs with the same shape ``(B, L, H, Dh)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> blk = LivConvBlock(MemoryMachineConfig(hidden_size=16, num_attention_heads=2, csa_num_output_groups=2, num_key_value_heads=1))
            >>> blk.mix_heads(torch.randn(1, 6, 2, 8)).shape
            torch.Size([1, 6, 2, 8])
        """
        bsz, seqlen, _, _ = x_gated.shape
        x_merged = x_gated.reshape(bsz, seqlen, self.hidden_size)
        x_conv, _ = self.conv(x_merged)
        return x_conv.reshape(bsz, seqlen, self.num_heads, self.head_dim)


class LivPerceiver(LivDoubleGatedBase):
    r"""LivPerceiver(config, window_size=None, query_mode=None, num_iterations=None)

    Perceiver LIV block (Jaegle et al., 2021) routing head views through an
    iterative attention bottleneck.

    A small set of ``num_perceiver_query`` latents anchored uniformly over the
    sequence distills the (potentially very large) input stream through
    cross-attention, is refined by a deep latent self-attention core, and is
    then broadcast back to sequence length. Iterating the cross-attend lets
    the model channel limited capacity to the most relevant inputs, informed
    by previous passes.

    Three query-seeding variants are supported via ``query_mode`` (or
    ``config.perceiver_query_mode``):

    - ``"learned"``: the latent array is a free learned parameter
      (``latent_queries`` / ``latent_values``); keys and values come from the
      single gated input stream. This is the classic Perceiver.
    - ``"cross"``: the query stream is the ungated projected input
      (``x_proj``), gathered at the latent anchors; the KV stream is the
      gated input (``x_gated``). Two input streams, one per role.
    - ``"masked"``: a ``(B, L)`` boolean mask selects which tokens produce
      queries; every token (masked or not) produces keys and values.
      Attention is causal over the whole sequence (with the CVPE QK bias as
      the positional encoding), and only the masked tokens get their
      representation updated — they attend to the full causal past,
      including other masked tokens. Pass the mask to
      :meth:`forward` / :meth:`mix_heads` as ``query_mask``.

    The cross-attend operator is selected by ``perceiver_attention_mode``:
    ``"full"`` (attend latents to every token, ``windowed=False``),
    ``"window"`` (causal SWA bounded to ``window_size`` tokens), or
    ``"sparse"`` (route the cross-attend through the
    :class:`~memory_machines.models.blocks.sparse.CompressedSparseAttention2`
    core: compress the KV stream ``m:1`` with
    :class:`~memory_machines.models.blocks.compression.BlockCompressor`,
    retrieve the top-``k`` entries with
    :class:`~memory_machines.models.blocks.compression.SparseIndexer`, and
    fuse the sparse selection with a SWA latent self-attend via
    :class:`~memory_machines.models.blocks.compression.BranchGate` — no
    full-attention matrix over the input is ever materialized). The latent
    self-attend core always runs full attention over the (small) latent
    array.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``hidden_size``, ``num_attention_heads``,
            ``num_perceiver_query``, ``perceiver_*``, and the GQA
            projection hyperparameters.
        window_size (int, optional): window for the internal core.
            Default: ``None`` (falls back to ``config.window_size``).
        query_mode (str, optional): ``"learned"`` or ``"cross"``.
            Default: ``None`` (falls back to
            ``config.perceiver_query_mode``).
        num_iterations (int, optional): number of cross-attend + latent
            self-attend rounds. Default: ``None`` (falls back to
            ``config.perceiver_num_iterations``).

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> blk = LivPerceiver(MemoryMachineConfig(hidden_size=32, num_attention_heads=4,
        ...     num_key_value_heads=2, num_perceiver_query=4, csa_num_output_groups=4))
        >>> blk(torch.randn(1, 8, 32)).shape
        torch.Size([1, 8, 32])
    """

    def __init__(
        self,
        config: MemoryMachineConfig,
        window_size: int | None = None,
        query_mode: str | None = None,
        num_iterations: int | None = None,
    ):
        super().__init__(config)
        if config.num_perceiver_query <= 0:
            raise ValueError(f"num_perceiver_query ({config.num_perceiver_query}) must be positive")
        self.query_mode = query_mode if query_mode is not None else config.perceiver_query_mode
        if self.query_mode not in ("learned", "cross", "masked"):
            raise ValueError(f"query_mode ({self.query_mode!r}) must be 'learned', 'cross', or 'masked'")
        self.num_iterations = num_iterations if num_iterations is not None else config.perceiver_num_iterations
        self.attention_mode = config.perceiver_attention_mode
        if self.attention_mode not in ("full", "window", "sparse"):
            raise ValueError(f"attention_mode ({self.attention_mode!r}) must be one of 'full', 'window', 'sparse'")
        self.share_weights = config.perceiver_share_weights
        self.sparse_m = config.perceiver_sparse_compression_ratio
        self.causal = config.perceiver_causal
        self.num_latents = config.num_perceiver_query
        self.swa = GroupedQueryAttention(config, window_size)
        if self.share_weights:
            self.attns = nn.ModuleList([self.swa])
        else:
            self.attns = nn.ModuleList([self.swa] + [GroupedQueryAttention(config, window_size) for _ in range(max(0, self.num_iterations - 1))])
        heads, kv_heads, dh = self.swa.num_heads, self.swa.num_key_value_heads, self.swa.head_dim
        self.heads = heads
        self.kv_heads = kv_heads
        self.dh = dh
        if self.query_mode == "learned":
            self.latent_queries = nn.Parameter(torch.empty(heads, self.num_latents, dh))
            self.latent_values = nn.Parameter(torch.empty(kv_heads, self.num_latents, dh))
        # Refinement MLP for masked-mode latent-Transformer updates (Eq. 92-style).
        if self.query_mode == "masked":
            self.refine = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            )
        else:
            self.refine = None  # type: ignore[assignment]
        self.latent_std = config.initializer_range
        # Sparse cross-attend runs through the CSA2 core
        # (BlockCompressor + SparseIndexer top-k + BranchGate + SWA
        # self-attend fusion), which compresses the KV stream m:1 and
        # attends the latents to the top-k selected store instead of
        # materializing a full O(Lq x L) attention matrix over the input.
        # The CSA2 module owns its own BlockCompressor, so no separate
        # compressor is kept here.
        self.csa2 = (
            CompressedSparseAttention2(
                config,
                compression_ratio=self.sparse_m,
                mode="full",
                top_k=max(1, config.csa_top_k),
                window_size=0,
            )
            if self.attention_mode == "sparse"
            else None
        )
        self.reset_parameters()

    def _attn(self, i: int) -> GroupedQueryAttention:
        return typing.cast(GroupedQueryAttention, self.attns[i % len(self.attns)])

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Small-normal init (``std = config.initializer_range``, GPT-2
        style) for the learned latent queries/values, matching Perceiver
        practice (Jaegle et al., 2021) where latents start as weak,
        near-orthogonal seeds and differentiate through cross-attention.
        Every attention module in :attr:`attns` keeps its own attention init;
        the CSA2 core (if any) re-resets its own compressor/indexer/gate.
        In ``"cross"`` query mode there are no learned latents — the query
        stream is gathered from the input.
        """
        LivDoubleGatedBase.reset_parameters(self)
        for mod in self.attns:
            reset = getattr(mod, "reset_parameters", None)
            if callable(reset):
                reset()
        if self.csa2 is not None:
            self.csa2.reset_parameters()
        if self.query_mode == "learned":
            nn.init.normal_(self.latent_queries, std=self.latent_std)
            nn.init.normal_(self.latent_values, std=self.latent_std)
        if self.refine is not None:
            nn.init.trunc_normal_(self.refine[0].weight, mean=0.0, std=self.latent_std, a=-2 * self.latent_std, b=2 * self.latent_std)
            nn.init.zeros_(self.refine[0].bias)
            nn.init.trunc_normal_(self.refine[2].weight, mean=0.0, std=self.latent_std, a=-2 * self.latent_std, b=2 * self.latent_std)
            nn.init.zeros_(self.refine[2].bias)

    def _latent_anchors(self, seqlen: int, device: torch.device) -> torch.Tensor:
        r"""_latent_anchors(seqlen, device) -> Tensor

        Compute uniform anchor positions for the latents.

        Args:
            seqlen (int): current sequence length.
            device (torch.device): device for the anchor tensor.

        Returns:
            Tensor: anchors of shape ``(1, num_latents)`` linearly spaced over
            ``[0, seqlen - 1]``. When ``num_latents > seqlen`` anchors
            necessarily duplicate (multiple latents share a position);
            harmless but wastes latent budget — prefer ``num_latents <= seqlen``.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> blk = LivPerceiver(MemoryMachineConfig(hidden_size=16, num_attention_heads=2,
            ...     num_key_value_heads=1, num_perceiver_query=3, csa_num_output_groups=2))
            >>> blk._latent_anchors(6, torch.device("cpu"))
            tensor([[0, 2, 5]])
        """
        anchors = torch.linspace(0, max(seqlen - 1, 0), self.num_latents, device=device).round().long()
        return anchors.unsqueeze(0)

    def _project_latent_kv(self, attn: GroupedQueryAttention, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""_project_latent_kv(attn, latents) -> (Tensor, Tensor)

        Project merged latents to the given attention module's key/value heads.

        Args:
            attn (GroupedQueryAttention): the attention module owning the
                ``k_proj``/``v_proj`` used here (one of :attr:`attns`).
            latents (Tensor): latents of shape ``(B, num_latents, hidden)``.

        Returns:
            Tensor: keys of shape ``(B, Hkv, num_latents, Dh)`` and values
            with the same shape.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> blk = LivPerceiver(MemoryMachineConfig(hidden_size=16, num_attention_heads=2,
            ...     num_key_value_heads=1, num_perceiver_query=3, csa_num_output_groups=2))
            >>> k, v = blk._project_latent_kv(blk._attn(0), torch.randn(1, 3, 16))
            >>> k.shape, v.shape
            (torch.Size([1, 1, 3, 8]), torch.Size([1, 1, 3, 8]))
        """
        bsz, n_lat, _ = latents.shape
        kv_heads = attn.num_key_value_heads
        dh = attn.head_dim
        k = attn.k_proj(latents).view(bsz, n_lat, kv_heads, dh).transpose(1, 2)
        v = attn.v_proj(latents).view(bsz, n_lat, kv_heads, dh).transpose(1, 2)
        return k, v

    def mix_heads(self, x_gated: torch.Tensor, x_query: torch.Tensor | None = None, query_mask: torch.Tensor | None = None) -> torch.Tensor:
        r"""mix_heads(x_gated, x_query=None, query_mask=None) -> Tensor

        Run the iterative Perceiver core: cross-attend latents to the input,
        refine with latent self-attend, then broadcast back to sequence
        length. ``x_query`` is the ungated projected input stream; it seeds
        the latent queries in ``"cross"`` query mode and is ignored in
        ``"learned"`` mode. ``query_mask`` is a ``(B, L)`` boolean mask
        selecting which tokens produce queries in ``"masked"`` mode; all
        tokens produce keys and values, attention is causal, and only the
        masked tokens get their representation updated.

        Args:
            x_gated (Tensor): gated inputs of shape ``(B, L, H, Dh)`` — the
                KV stream for the cross-attend and the query stream for the
                broadcast-back.
            x_query (Tensor, optional): ungated projected inputs of shape
                ``(B, L, H, Dh)`` — the query stream in ``"cross"`` mode.
            query_mask (Tensor, optional): ``(B, L)`` bool query-selector mask
                — the query selector in ``"masked"`` mode. Default: ``None``.

        Returns:
            Tensor: sequence-length outputs of shape ``(B, L, H, Dh)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> blk = LivPerceiver(MemoryMachineConfig(hidden_size=16, num_attention_heads=2,
            ...     num_key_value_heads=1, num_perceiver_query=3, csa_num_output_groups=2))
            >>> blk.mix_heads(torch.randn(1, 6, 2, 8)).shape
            torch.Size([1, 6, 2, 8])
        """
        bsz, seqlen, _, _ = x_gated.shape
        x_merged = x_gated.reshape(bsz, seqlen, self.hidden_size)
        device = x_merged.device
        if seqlen > self.swa.max_position_embeddings:
            raise ValueError(f"Sequence length ({seqlen}) exceeds configured max_position_embeddings ({self.swa.max_position_embeddings})")
        anchors = self._latent_anchors(seqlen, device).expand(bsz, -1)
        k_pos = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)

        # KV stream: always the gated input. In sparse mode the CSA2 core
        # compresses the KV stream m:1 (BlockCompressor), retrieves the
        # top-k entries (SparseIndexer), and fuses the sparse selection with
        # a latent self-attend (BranchGate + SWA) — no O(Lq x L) attention
        # matrix over the input is ever materialized.
        if self.query_mode == "masked":
            # Third Perceiver mode: a (B, L) boolean mask selects which
            # tokens produce queries; every token (masked or not) produces
            # keys and values. Attention is causal over the whole sequence,
            # and only the masked tokens get their representation updated —
            # they attend to the full causal past, including other masked
            # tokens. Positional encoding is the standard CVPE QK bias.
            return self._masked_mix(x_gated, x_merged, x_query, query_mask, k_pos, anchors, device)
        if self.query_mode == "learned":
            latents = self.latent_queries.unsqueeze(0).expand(bsz, -1, -1, -1)
            v_ref = self.latent_values.unsqueeze(0).expand(bsz, -1, -1, -1)
        else:
            latents, v_ref = self._seed_cross_queries(x_query, anchors, device)
        latents_q = latents
        latents_hidden: torch.Tensor = latents_q.transpose(1, 2).reshape(bsz, self.num_latents, self.hidden_size)

        if self.attention_mode == "sparse":
            assert self.csa2 is not None
            # The cross-attend (latents <- input) is bidirectional in the
            # Perceiver: latents must see the whole input. CSA2 masks entries
            # by ``comp_end <= q_pos``, so position every latent at the last
            # input token to grant visibility of the entire compressed
            # store; the CSA2 SWA branch then self-attends the latents
            # bidirectionally (all share one position), which is the standard
            # Perceiver latent-Transformer behaviour.
            csa_q_pos = torch.full((bsz, self.num_latents), seqlen - 1, device=device, dtype=k_pos.dtype)
            csa_shared: CSA2SharedState | None = None
            for i in range(self.num_iterations):
                # Compress the input stream m:1 once; subsequent rounds reuse
                # the seeded store instead of re-compressing the identical
                # ``x_merged`` (which would grow the store with duplicate
                # entries). CSA2's SWA branch still self-attends the latents
                # each round and re-indexes the top-k, so the latent-Transformer
                # refinement is intact.
                latents_hidden, csa_shared = self.csa2.forward_shared(
                    latents_hidden,
                    shared=csa_shared,
                    compress_from=x_merged if i == 0 else None,
                    compress_start_pos=0,
                    position_ids=csa_q_pos,
                    mode="full",
                    seeded_store=i > 0,
                )
        else:
            kv_store, kv_pos = self._project_latent_kv(self._attn(0), x_merged), k_pos
            for i in range(self.num_iterations):
                attn = self._attn(i)
                # Cross-attend: latents <- input.
                heads_lat, _ = attn.attend(
                    latents_q,
                    kv_store[0],
                    kv_store[1],
                    v_ref,
                    anchors,
                    kv_pos,
                    windowed=self.attention_mode == "window",
                    is_causal=self.causal,
                )
                latents_hidden = attn.project_heads(heads_lat, bsz, self.num_latents)
                # Latent self-attend refine (the Perceiver's latent Transformer).
                q_self = attn.q_proj(latents_hidden).view(bsz, self.num_latents, attn.num_heads, attn.head_dim).transpose(1, 2)
                k_self, v_self = self._project_latent_kv(attn, latents_hidden)
                heads_self, _ = attn.attend(
                    q_self,
                    k_self,
                    v_self,
                    v_self,
                    anchors,
                    anchors,
                    windowed=False,
                    is_causal=self.causal,
                )
                latents_q = heads_self
                latents_hidden = attn.project_heads(heads_self, bsz, self.num_latents)

        # Broadcast back: sequence <- latents. The per-query reference
        # values come from the input's own values (the query stream), so the
        # reduction is anchored to the sequence rather than the latents.
        out_attn = self._attn(self.num_iterations - 1)
        q_in = out_attn.q_proj(x_merged).view(bsz, seqlen, out_attn.num_heads, out_attn.head_dim).transpose(1, 2)
        k_lat, v_lat = self._project_latent_kv(out_attn, latents_hidden)
        vq_in = out_attn.v_proj(x_merged).view(bsz, seqlen, out_attn.num_key_value_heads, out_attn.head_dim).transpose(1, 2)
        heads_out, _ = out_attn.attend(
            q_in,
            k_lat,
            v_lat,
            vq_in,
            k_pos,
            anchors,
            windowed=self.attention_mode == "window",
            is_causal=self.causal,
        )
        out_merged = out_attn.project_heads(heads_out, bsz, seqlen)
        return out_merged.view(bsz, seqlen, self.num_heads, self.head_dim)

    def _seed_cross_queries(
        self,
        x_query: torch.Tensor | None,
        anchors: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""_seed_cross_queries(x_query, anchors, device) -> (Tensor, Tensor)

        Gather the ungated query stream at the latent anchors to seed the
        latent queries and reference values.

        Args:
            x_query (Tensor, optional): ungated projected inputs of shape
                ``(B, L, H, Dh)``.
            anchors (Tensor): latent anchor positions of shape ``(B, N)``.
            device (torch.device): device for the output tensors.

        Returns:
            Tensor: latent queries ``(B, H, N, Dh)`` and reference values
            ``(B, Hkv, N, Dh)``.

        Raises:
            ValueError: if ``x_query`` is ``None`` in ``"cross"`` mode.
        """
        if x_query is None:
            raise ValueError("query_mode='cross' requires the ungated query stream (x_proj); pass it via mix_heads(x_gated, x_query)")
        bsz = x_query.shape[0]
        batch = torch.arange(bsz, device=device).unsqueeze(-1)
        gathered = x_query[batch, anchors, :, :]  # (B, N, H, Dh)
        q_lat = gathered.transpose(1, 2)  # (B, H, N, Dh)
        v_ref = q_lat[:, : self.kv_heads, :, :] if self.kv_heads < self.heads else q_lat
        return q_lat, v_ref

    def _masked_mix(
        self,
        x_gated: torch.Tensor,
        x_merged: torch.Tensor,
        _x_query: torch.Tensor | None,
        query_mask: torch.Tensor | None,
        k_pos: torch.Tensor,
        _anchors: torch.Tensor,
        _device: torch.device,
    ) -> torch.Tensor:
        r"""_masked_mix(x_gated, x_merged, _x_query, query_mask, k_pos, _anchors, _device) -> Tensor

        Masked Perceiver cross-attend: a ``(B, L)`` boolean mask selects which
        tokens produce queries; every token produces keys and values.
        Attention is causal over the whole sequence (with the CVPE QK bias as
        the positional encoding), and only the masked tokens get their
        representation updated — they attend to the full causal past,
        including other masked tokens. Optionally routes through the CSA2
        core for sparse key/value compression.

        Args:
            x_gated (Tensor): gated inputs of shape ``(B, L, H, Dh)``; the
                KV stream and the baseline for non-masked tokens.
            x_merged (Tensor): merged input of shape ``(B, L, hidden)``.
            _x_query (Tensor, optional): ungated projected inputs of shape
                ``(B, L, H, Dh)``; accepted for API parity with the other
                Perceiver helpers and intentionally ignored — in masked mode
                the query stream is derived from ``x_merged`` (every token
                produces a query), not from the ungated projection.
            query_mask (Tensor, optional): ``(B, L)`` bool query-selector mask.
            k_pos (Tensor): key positions of shape ``(B, L)``.
            _anchors (Tensor): latent anchor positions of shape ``(B, N)``;
                accepted for API parity with the other Perceiver helpers and
                intentionally ignored — masked mode has no latent array.
            _device (torch.device): device for constructed tensors; accepted
                for API parity and intentionally ignored (derived from
                ``x_merged``).

        Returns:
            Tensor: mixed outputs of shape ``(B, L, H, Dh)``; non-masked
            tokens keep their gated input unchanged.

        Raises:
            ValueError: if ``query_mask`` is ``None`` in ``"masked"`` mode.
        """
        if query_mask is None:
            raise ValueError("query_mode='masked' requires a (B, L) bool query_mask")
        bsz, seqlen, _ = x_merged.shape
        mask = query_mask.unsqueeze(-1).unsqueeze(-1)  # (B, L, 1, 1)

        # KV stream: every token (masked or not) produces keys and values,
        # derived once from the input. The query stream is the evolving
        # masked-token representation, refined across ``num_iterations``
        # rounds of causal attention — masked tokens attend to the full
        # causal past, including other masked tokens.
        kv_attn = self._attn(0)
        k_states, v_states = self._project_latent_kv(kv_attn, x_merged)

        assert self.refine is not None, "masked mode requires self.refine MLP"
        # The evolving query representation. Starts as the gated input;
        # only masked tokens are updated between rounds (non-masked tokens
        # keep their input unchanged, so their queries never change).
        cur = x_gated
        csa_shared: CSA2SharedState | None = None
        for i in range(self.num_iterations):
            attn = self._attn(i)
            cur_merged = cur.reshape(bsz, seqlen, self.hidden_size)
            q_in = attn.q_proj(cur_merged).view(bsz, seqlen, attn.num_heads, attn.head_dim).transpose(1, 2)
            vq_in = attn.v_proj(cur_merged).view(bsz, seqlen, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            if self.attention_mode == "sparse":
                assert self.csa2 is not None
                # CSA2 compresses the KV stream m:1 once and attends the
                # masked queries to the top-k selected store; the query
                # positions are the token positions themselves, so
                # causality is preserved. Subsequent rounds reuse the seeded
                # store instead of re-compressing (which would grow it with
                # duplicate entries). CSA2 returns fused model-dimension
                # outputs, reshaped directly to head space — no second query
                # projection.
                out_hidden, csa_shared = self.csa2.forward_shared(
                    cur_merged,
                    shared=csa_shared,
                    compress_from=x_merged if i == 0 else None,
                    compress_start_pos=0,
                    position_ids=k_pos,
                    mode="full",
                    seeded_store=i > 0,
                )
                heads_out = out_hidden.view(bsz, seqlen, attn.num_heads, attn.head_dim)
            else:
                heads_out, _ = attn.attend(
                    q_in,
                    k_states,
                    v_states,
                    vq_in,
                    k_pos,
                    k_pos,
                    windowed=self.attention_mode == "window",
                    is_causal=self.causal,
                )
            out_merged = attn.project_heads(heads_out, bsz, seqlen)
            cur = out_merged.view(bsz, seqlen, self.num_heads, self.head_dim)
            out_hidden = out_merged  # (B, L, hidden) from project_heads
            # Residual refinement on masked tokens: refined = out_hidden + DenseFFN(out_hidden)
            refined_hidden = out_hidden + self.refine(out_hidden)
            # Only masked tokens take the refined result; non-masked stay at original gated input (x_merged reshaped to heads = x_gated).
            masked_hidden = torch.where(mask.reshape(bsz, seqlen, 1), refined_hidden, x_merged.reshape(bsz, seqlen, self.hidden_size))
            cur = masked_hidden.reshape(bsz, seqlen, self.num_heads, self.head_dim)
            # Non-masked forced back to original gated stream (no drift).
            cur = torch.where(mask, cur, x_gated)

        # Only the masked tokens get their representation updated; the
        # non-masked tokens keep their gated input unchanged.
        return cur
