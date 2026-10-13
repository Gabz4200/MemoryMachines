r"""LIV blocks: multihead double-gated conv and perceiver.

Heads stay batched via reshapes (grouped Conv1d, batched GEMMs); ``torch.vmap``
would forfeit that fusion for tracing overhead.
"""

import torch
from torch import nn

from memory_machines.utils.config_schema import MemoryMachineConfig

from .attention import GroupedQueryAttention


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

        Xavier-uniform for the 3-way input projection; the output
        projection is zero-initialized so the whole double-gated branch
        starts as an identity residual despite its two multiplicative
        gatings (Fixup; ReZero). Subclass mixers define their own start
        (Dirac identity for conv, small-normal latents for perceiver).
        """
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def split_gates_heads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""split_gates_heads(x) -> (Tensor, Tensor, Tensor, Tensor)

        Split the input into per-head gate and value views.

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

    def mix_heads(self, x_gated: torch.Tensor) -> torch.Tensor:
        r"""mix_heads(x_gated) -> Tensor

        Per-head token mixing: ``(B, L, H, Dh)`` -> same shape.

        This base implementation performs no mixing across heads or tokens;
        subclasses override it with a convolution or perceiver core.

        Args:
            x_gated (Tensor): gated inputs of shape ``(B, L, H, Dh)``.

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

    def token_mix(self, x_gated: torch.Tensor) -> torch.Tensor:
        r"""token_mix(x_gated) -> Tensor

        Merged ``(B, L, hidden)`` adapter around :meth:`mix_heads`.

        Args:
            x_gated (Tensor): merged input of shape ``(B, L, hidden)``.

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
        return self.mix_heads(heads).reshape(bsz, seqlen, self.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x) -> Tensor

        Apply double-gated mixing to the input sequence.

        Args:
            x (Tensor): input of shape ``(B, L, hidden)``.

        Returns:
            Tensor: output of shape ``(B, L, hidden)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> from memory_machines.models.blocks.liv import LivConvBlock
            >>> LivConvBlock(MemoryMachineConfig(hidden_size=16, num_attention_heads=2, csa_num_output_groups=2, num_key_value_heads=1))(torch.randn(1, 4, 16)).shape
            torch.Size([1, 4, 16])
        """
        _, post_gate, _, x_gated = self.split_gates_heads(x)
        return self.finish_heads(post_gate, self.mix_heads(x_gated))

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

    Short-conv LIV block: grouped convolution with one group per head.

    Token mixing is a depthwise-over-heads ``Conv1d`` (``groups=num_heads``),
    so there is no cross-head mixing. Kernel size comes from
    ``config.conv_kernel_size`` with ``kernel // 2`` padding to preserve length.

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

        self.conv = nn.Conv1d(
            self.hidden_size,
            self.hidden_size,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
            groups=self.num_heads,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Grouped-Dirac (identity) init: output channel ``c`` reads its own
        group's channel ``c % head_dim`` at the center tap, so the mixer
        preserves the token stream exactly at step 0 and learns local
        context from an identity start (DiracNets, Zagoruyko & Komodakis,
        2016; cf. Fixup identity-start). ``nn.init.dirac_`` cannot be
        used here: it maps output channel ``c`` to input channel ``c``,
        which leaves all but the first group dead for grouped convolutions
        with ``out_channels > in_channels // groups``.
        """
        LivDoubleGatedBase.reset_parameters(self)
        with torch.no_grad():
            self.conv.weight.zero_()
            center = self.conv.kernel_size[0] // 2
            for c in range(self.conv.out_channels):
                self.conv.weight[c, c % self.head_dim, center] = 1.0

    def mix_heads(self, x_gated: torch.Tensor) -> torch.Tensor:
        r"""mix_heads(x_gated) -> Tensor

        Apply the per-head short convolution.

        Args:
            x_gated (Tensor): gated inputs of shape ``(B, L, H, Dh)``.

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
        x_merged = x_gated.reshape(bsz, seqlen, self.hidden_size).transpose(1, 2)
        x_conv = self.conv(x_merged)
        return x_conv.transpose(1, 2).reshape(bsz, seqlen, self.num_heads, self.head_dim)


class LivPerceiver(LivDoubleGatedBase):
    r"""LivPerceiver(config, window_size=None)

    Perceiver LIV block routing head views through the attention core.

    Gated head states are compressed into ``num_perceiver_query`` latents
    anchored uniformly over the sequence, then broadcast back: sequence ->
    latents -> sequence. Both cross-attentions run through
    :class:`~memory_machines.models.blocks.attention.GroupedQueryAttention`
    with ``windowed=False`` (pure causal, full visibility to the past).

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``hidden_size``, ``num_attention_heads``,
            ``num_perceiver_query`` (must be positive), ``hidden_dropout``, and
            the GQA projection hyperparameters.
        window_size (int, optional): window for the internal core.
            Default: ``None`` (falls back to ``config.window_size``).

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> blk = LivPerceiver(MemoryMachineConfig(hidden_size=32, num_attention_heads=4,
        ...     num_key_value_heads=2, num_perceiver_query=4, csa_num_output_groups=4))
        >>> blk(torch.randn(1, 8, 32)).shape
        torch.Size([1, 8, 32])
    """

    def __init__(self, config: MemoryMachineConfig, window_size: int | None = None):
        super().__init__(config)
        if config.num_perceiver_query <= 0:
            raise ValueError(f"num_perceiver_query ({config.num_perceiver_query}) must be positive")
        self.swa = GroupedQueryAttention(config, window_size)
        self.num_latents = config.num_perceiver_query
        heads, kv_heads, dh = self.swa.num_heads, self.swa.num_key_value_heads, self.swa.head_dim
        self.latent_queries = nn.Parameter(torch.empty(heads, self.num_latents, dh))
        self.latent_values = nn.Parameter(torch.empty(kv_heads, self.num_latents, dh))
        self.latent_std = config.initializer_range
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Small-normal init (``std = config.initializer_range``, GPT-2
        style) for the learned latent queries/values, matching Perceiver
        practice (Jaegle et al., 2021) where latents start as weak,
        near-orthogonal seeds and differentiate through cross-attention.
        The internal SWA core keeps its own attention init.
        """
        LivDoubleGatedBase.reset_parameters(self)
        self.swa.reset_parameters()
        nn.init.normal_(self.latent_queries, std=self.latent_std)
        nn.init.normal_(self.latent_values, std=self.latent_std)

    def _latent_anchors(self, seqlen: int, device: torch.device) -> torch.Tensor:
        r"""_latent_anchors(seqlen, device) -> Tensor

        Compute uniform anchor positions for the latents.

        Args:
            seqlen (int): current sequence length.
            device (torch.device): device for the anchor tensor.

        Returns:
            Tensor: anchors of shape ``(1, num_latents)`` linearly spaced over
            ``[0, seqlen - 1]``.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> blk = LivPerceiver(MemoryMachineConfig(hidden_size=16, num_attention_heads=2,
            ...     num_key_value_heads=1, num_perceiver_query=3, csa_num_output_groups=2))
            >>> blk._latent_anchors(6, torch.device("cpu"))
            tensor([[0, 2, 5]])
        """
        anchors = torch.linspace(0, max(seqlen - 1, 0), self.num_latents, device=device).round().long()
        return anchors.unsqueeze(0)

    def _project_latent_kv(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""_project_latent_kv(latents) -> (Tensor, Tensor)

        Project merged latents to SWA key/value heads.

        Args:
            latents (Tensor): latents of shape ``(B, num_latents, hidden)``.

        Returns:
            Tensor: keys of shape ``(B, Hkv, num_latents, Dh)`` and values
            with the same shape.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> blk = LivPerceiver(MemoryMachineConfig(hidden_size=16, num_attention_heads=2,
            ...     num_key_value_heads=1, num_perceiver_query=3, csa_num_output_groups=2))
            >>> k, v = blk._project_latent_kv(torch.randn(1, 3, 16))
            >>> k.shape, v.shape
            (torch.Size([1, 1, 3, 8]), torch.Size([1, 1, 3, 8]))
        """
        bsz, n_lat, _ = latents.shape
        kv_heads = self.swa.num_key_value_heads
        dh = self.swa.head_dim
        k = self.swa.k_proj(latents).view(bsz, n_lat, kv_heads, dh).transpose(1, 2)
        v = self.swa.v_proj(latents).view(bsz, n_lat, kv_heads, dh).transpose(1, 2)
        return k, v

    def mix_heads(self, x_gated: torch.Tensor) -> torch.Tensor:
        r"""mix_heads(x_gated) -> Tensor

        Compress the sequence to latents and broadcast back.

        Args:
            x_gated (Tensor): gated inputs of shape ``(B, L, H, Dh)``.

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
        swa = self.swa
        bsz, seqlen, _, _ = x_gated.shape
        x_merged = x_gated.reshape(bsz, seqlen, self.hidden_size)
        device = x_merged.device
        if seqlen > swa.max_position_embeddings:
            raise ValueError(f"Sequence length ({seqlen}) exceeds configured max_position_embeddings ({swa.max_position_embeddings})")
        anchors = self._latent_anchors(seqlen, device).expand(bsz, -1)
        k_pos = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)

        kv_in, vv_in = self._project_latent_kv(x_merged)
        q_lat = self.latent_queries.unsqueeze(0).expand(bsz, -1, -1, -1)
        v_ref = self.latent_values.unsqueeze(0).expand(bsz, -1, -1, -1)
        heads_lat, _ = swa.attend(q_lat, kv_in, vv_in, v_ref, anchors, k_pos, windowed=False)
        latents = swa.project_heads(heads_lat, bsz, self.num_latents)

        q_in = swa.q_proj(x_merged).view(bsz, seqlen, swa.num_heads, swa.head_dim).transpose(1, 2)
        k_lat, v_lat = self._project_latent_kv(latents)
        heads_out, _ = swa.attend(q_in, k_lat, v_lat, vv_in, k_pos, anchors, windowed=False)
        out_merged = swa.project_heads(heads_out, bsz, seqlen)
        return out_merged.view(bsz, seqlen, self.num_heads, self.head_dim)
