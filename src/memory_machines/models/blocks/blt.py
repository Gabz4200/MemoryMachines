r"""Byte Latent Transformer local models (arXiv:2412.09871).

The BLT architecture splits byte-level modelling into three modules: a
lightweight :class:`VocabularyEncoder` that turns bytes into dynamically sized
patch representations, the latent transformer over patches (here the CED
stack), and a lightweight :class:`VocabularyDecoder` that turns patch
representations back into bytes.

:class:`VocabularyEncoder` embeds bytes, optionally adds hash n-gram
embeddings (sizes 3-8 in the paper), runs a causal sliding-window byte
transformer, and after every layer pools the byte states into patch queries
with a cross-attention layer. Each patch query attends *only* to the bytes of
its own patch (Eqs. 5-8).

:class:`VocabularyDecoder` takes the encoder's final byte states as queries
and the latent patch representations as keys/values. It alternates a
patch cross-attention layer with a causal sliding-window byte transformer
layer (Eqs. 9-12) and projects to the byte vocabulary. Query ``i`` reads the
latent of the patch *preceding* the one that contains byte ``i``, so it never
sees its own next-byte target; the shift is part of the ``patch_ids`` contract
(``0`` selects the learned begin-of-bytes patch). The masked-patch decode path
(:meth:`VocabularyDecoder.decode_masked_patches`) instead conditions each byte
slot on its own patch's latent decoder output, which is future content during
generation rather than teacher-forced input.

Both modules reuse the repository's
:class:`~memory_machines.models.blocks.attention.GroupedQueryAttention` for
attention (including its CVPE positional bias and FRVM value path on the byte
transformer) and :class:`~memory_machines.models.blocks.ffn.DenseFFN` for the
feed-forward sub-layer.
"""

from dataclasses import replace
from typing import cast

import torch
from torch import nn

from memory_machines.utils.config_schema import MemoryMachineConfig

from .attention import GroupedQueryAttention
from .ffn import DenseFFN, build_norm
from .linear_init import init_linear

_HASH_PRIME = 1_000_000_007


def compute_patch_ids(byte_ids: torch.Tensor, patch_size: int) -> torch.Tensor:
    r"""compute_patch_ids(byte_ids, patch_size) -> Tensor

    Strided patching: every ``patch_size`` consecutive bytes form one patch.

    This is the paper's fixed-size patching baseline (§2.1). Entropy or space
    patching can be supplied instead through ``ModelInput.patch_ids``.

    Args:
        byte_ids (Tensor): byte ids of shape ``(B, N)``.
        patch_size (int): bytes per patch; must be positive.

    Returns:
        Tensor: int64 patch ids of shape ``(B, N)``, non-decreasing from ``0``.

    Examples::

        >>> import torch
        >>> compute_patch_ids(torch.zeros(1, 5, dtype=torch.long), 2).tolist()
        [[0, 0, 1, 1, 2]]
    """
    if patch_size < 1:
        raise ValueError(f"patch_size ({patch_size}) must be >= 1")
    n = byte_ids.shape[1]
    return torch.arange(n, device=byte_ids.device, dtype=torch.long).unsqueeze(0).expand(byte_ids.shape[0], -1) // patch_size


def _validate_patch_ids(patch_ids: torch.Tensor, num_bytes: int) -> int:
    r"""_validate_patch_ids(patch_ids, num_bytes) -> int

    Validate a contiguous byte-to-patch map and return the patch count.

    Ids must start at ``0`` and increase by at most one per byte, so every
    patch owns a non-empty contiguous byte span. Gaps would leave a patch
    without bytes and make max-pooling return ``-inf``.
    """
    if num_bytes < 1:
        raise ValueError("byte sequence must contain at least one byte")
    if patch_ids.dim() != 2 or patch_ids.shape[1] != num_bytes:
        raise ValueError(f"patch_ids must be (B, N={num_bytes}), got {tuple(patch_ids.shape)}")
    if patch_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"patch_ids must be an integer tensor, got {patch_ids.dtype}")
    if bool((patch_ids < 0).any()):
        raise ValueError("patch_ids must be non-negative")
    if bool((patch_ids[:, 0] != 0).any()):
        raise ValueError("patch_ids must start at 0 for every sequence")
    if patch_ids.shape[1] > 1:
        step = patch_ids[:, 1:] - patch_ids[:, :-1]
        if bool(((step < 0) | (step > 1)).any()):
            raise ValueError("patch_ids must be non-decreasing and contiguous (each patch owns a non-empty byte span)")
    row_max = patch_ids.max(dim=1).values
    if not bool((row_max == row_max[0]).all()):
        raise ValueError("all sequences in a batch must have the same number of patches; pack or pad patches before the local models")
    return int(row_max[0].item()) + 1


def _max_pool_patches(byte_states: torch.Tensor, patch_ids: torch.Tensor, num_patches: int) -> torch.Tensor:
    r"""_max_pool_patches(byte_states, patch_ids, num_patches) -> Tensor

    Max-pool byte states into patch states (the paper's query initialization).
    """
    bsz, _, dim = byte_states.shape
    pooled = byte_states.new_full((bsz, num_patches, dim), float("-inf"))
    index = patch_ids.unsqueeze(-1).expand(-1, -1, dim)
    pooled.scatter_reduce_(1, index, byte_states, reduce="amax", include_self=True)
    return pooled


def _ngram_hashes(byte_ids: torch.Tensor, n: int, table_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    r"""_ngram_hashes(byte_ids, n, table_size) -> (Tensor, Tensor)

    Rolling polynomial hash of every ``n``-gram ending at each byte position.

    The hash is ``sum(byte * prime**offset) % table_size`` (Appendix C),
    evaluated with a modular recurrence so it never overflows int64. The
    boolean mask marks positions ``i >= n - 1`` where a full n-gram exists.
    """
    bsz, seq_len = byte_ids.shape
    if n < 2:
        raise ValueError(f"n-gram size ({n}) must be >= 2")
    padded = torch.cat([byte_ids.new_zeros((bsz, n - 1)), byte_ids.long()], dim=1)
    windows = padded.unfold(1, n, 1)
    hashed = torch.zeros((bsz, seq_len), dtype=torch.long, device=byte_ids.device)
    for offset in range(n):
        hashed = (hashed * _HASH_PRIME + windows[..., offset]) % table_size
    valid = torch.arange(seq_len, device=byte_ids.device).ge(n - 1).unsqueeze(0).expand(bsz, -1)
    return hashed, valid


def _local_configs(config: MemoryMachineConfig) -> tuple[MemoryMachineConfig, MemoryMachineConfig, int]:
    r"""_local_configs(config) -> (local_cfg, cross_cfg, local_hidden)

    Derive the byte-width config (byte transformer) and the cross-attention
    config (no CVPE/FRVM, no positional bias in the cross-attention block).
    """
    local_hidden = config.blt_local_hidden_size or config.hidden_size
    if local_hidden % config.num_attention_heads != 0:
        raise ValueError(f"blt_local_hidden_size ({local_hidden}) must be divisible by num_attention_heads ({config.num_attention_heads})")
    max_positions = config.blt_local_max_position_embeddings or config.max_position_embeddings * config.blt_patch_size
    local_cfg = replace(
        config,
        hidden_size=local_hidden,
        max_position_embeddings=max(max_positions, 1),
        window_size=config.blt_local_window_size,
    )
    cross_heads = config.blt_cross_attn_heads or config.num_attention_heads
    cross_cfg = replace(
        local_cfg,
        num_attention_heads=cross_heads,
        num_key_value_heads=cross_heads,
        csa_num_output_groups=1,
        cvpe_enabled=False,
        frvm_enabled=False,
    )
    return local_cfg, cross_cfg, local_hidden


class _LocalEncoderLayer(nn.Module):
    """Byte transformer layer followed by byte-to-patch cross-attention."""

    def __init__(self, config: MemoryMachineConfig, local_cfg: MemoryMachineConfig, cross_cfg: MemoryMachineConfig):
        super().__init__()
        self.norm = build_norm(config.blt_local_norm, local_cfg.hidden_size, eps=config.rms_norm_eps)
        self.attn = GroupedQueryAttention(local_cfg, config.blt_local_window_size)
        self.ffn = DenseFFN(
            local_cfg,
            config.blt_local_ffn,
            norm_kind=config.blt_local_norm,
            intermediate_size=config.blt_local_ffn_intermediate_size or None,
        )
        self.cross_query_norm = build_norm(config.blt_local_norm, local_cfg.hidden_size, eps=config.rms_norm_eps)
        self.cross_kv_norm = build_norm(config.blt_local_norm, local_cfg.hidden_size, eps=config.rms_norm_eps)
        self.cross = GroupedQueryAttention(cross_cfg)

    def reset_parameters(self) -> None:
        """Re-apply every sub-module's neutral-start initialization."""
        for module in (self.norm, self.attn, self.ffn, self.cross_query_norm, self.cross_kv_norm, self.cross):
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()


class VocabularyEncoder(nn.Module):
    r"""VocabularyEncoder(config, vocab_size=None, local_hidden_size=None)

    BLT local encoder: bytes -> patch representations.

    Each byte is embedded and optionally augmented with hash n-gram
    embeddings (``config.blt_ngram_sizes``, paper Eq. 3). ``blt_local_layers``
    layers of causal sliding-window byte attention + dense FFN transform the
    byte states; after each layer a cross-attention block pools the byte
    states into patch queries. Patch queries are initialized by max-pooling
    the byte embeddings and projected to the local width. Patch ``j`` attends
    only to the bytes with ``patch_ids == j`` (Eq. 7), so patch
    representations are causal across patches.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides the ``blt_*`` local-model settings plus the shared
            attention/FFN/norm settings.
        vocab_size (int, optional): byte vocabulary size. Default: ``None``
            (``config.blt_vocab_size`` or ``config.vocab_size``).
        local_hidden_size (int, optional): byte model width ``h_E``.
            Default: ``None`` (``config.blt_local_hidden_size`` or
            ``config.hidden_size``).

    Examples::

        >>> import torch
        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> cfg = MemoryMachineConfig(hidden_size=16, num_attention_heads=2,
        ...     num_key_value_heads=1, csa_num_output_groups=2,
        ...     blt_local_hidden_size=16, blt_local_layers=1,
        ...     blt_local_window_size=8, blt_ngram_hash_size=64)
        >>> ids = torch.randint(0, 32, (1, 6))
        >>> pids = compute_patch_ids(ids, 2)
        >>> reps, hidden = VocabularyEncoder(cfg)(ids, pids)
        >>> reps.shape, hidden.shape
        (torch.Size([1, 3, 16]), torch.Size([1, 6, 16]))
    """

    def __init__(
        self,
        config: MemoryMachineConfig,
        vocab_size: int | None = None,
        local_hidden_size: int | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size if vocab_size is not None else (config.blt_vocab_size or config.vocab_size)
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size ({self.vocab_size}) must be positive")
        if local_hidden_size is not None:
            config = replace(config, blt_local_hidden_size=local_hidden_size)
        self.config = config
        local_cfg, cross_cfg, local_hidden = _local_configs(config)
        self.local_hidden_size = local_hidden
        self.byte_embeddings = nn.Embedding(self.vocab_size, local_hidden)
        self.ngram_embeddings = nn.ModuleList([nn.Embedding(config.blt_ngram_hash_size, local_hidden) for _ in config.blt_ngram_sizes] if config.blt_use_ngram_embeddings else [])
        self.layers = nn.ModuleList([_LocalEncoderLayer(config, local_cfg, cross_cfg) for _ in range(config.blt_local_layers)])
        self.patch_init_proj = nn.Linear(local_hidden, local_hidden, bias=False)
        self.patch_out_proj = nn.Linear(local_hidden, config.hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Re-apply byte/n-gram embedding and local-layer initialization."""
        std = self.config.initializer_range
        nn.init.trunc_normal_(self.byte_embeddings.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        for table in cast("list[nn.Embedding]", list(self.ngram_embeddings)):
            nn.init.trunc_normal_(table.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        for layer in cast("list[_LocalEncoderLayer]", list(self.layers)):
            layer.reset_parameters()
        for projection in (self.patch_init_proj, self.patch_out_proj):
            init_linear(projection, std)

    def _embeddings(self, byte_ids: torch.Tensor | None, byte_embeds: torch.Tensor | None) -> torch.Tensor:
        if byte_embeds is not None:
            if byte_embeds.dim() != 3 or byte_embeds.shape[-1] != self.local_hidden_size:
                raise ValueError(f"byte_embeds must be (B, N, {self.local_hidden_size}), got {tuple(byte_embeds.shape)}")
            embedded = byte_embeds
        elif byte_ids is None:
            raise ValueError("VocabularyEncoder requires byte_ids or byte_embeds")
        else:
            embedded = self.byte_embeddings(byte_ids)
        if byte_ids is None or not self.ngram_embeddings:
            return embedded
        total = embedded
        for size, table in zip(self.config.blt_ngram_sizes, cast("list[nn.Embedding]", list(self.ngram_embeddings))):
            hashed, valid = _ngram_hashes(byte_ids, size, table.num_embeddings)
            ngram = table(hashed)
            total = total + torch.where(valid.unsqueeze(-1), ngram, torch.zeros_like(ngram))
        # Eq. 3: normalize by the number of n-gram sizes plus the byte term.
        return total / (len(self.ngram_embeddings) + 1)

    def forward(
        self,
        byte_ids: torch.Tensor | None = None,
        patch_ids: torch.Tensor | None = None,
        byte_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""forward(byte_ids=None, patch_ids=None, byte_embeds=None) -> (Tensor, Tensor)

        Encode bytes into patch representations.

        Args:
            byte_ids (Tensor, optional): byte ids ``(B, N)``. Required unless
                ``byte_embeds`` is given (then used only for n-grams).
            patch_ids (Tensor, optional): contiguous byte-to-patch map
                ``(B, N)``. Default: ``None`` (strided patching with
                ``config.blt_patch_size``).
            byte_embeds (Tensor, optional): pre-embedded bytes
                ``(B, N, h_E)``; skips the byte embedding lookup (n-grams are
                still added when ``byte_ids`` is also given).

        Returns:
            tuple[Tensor, Tensor]: patch representations ``(B, P, hidden)``
            and final byte states ``(B, N, h_E)``.
        """
        if byte_ids is None and byte_embeds is None:
            raise ValueError("VocabularyEncoder requires byte_ids or byte_embeds")
        if byte_embeds is not None:
            if byte_embeds.dim() != 3:
                raise ValueError(f"byte_embeds must be (B, N, h_E), got {tuple(byte_embeds.shape)}")
            batch, num_bytes = byte_embeds.shape[:2]
            if byte_ids is not None and byte_ids.shape[:2] != (batch, num_bytes):
                raise ValueError(f"byte_ids {tuple(byte_ids.shape)} must match byte_embeds {tuple(byte_embeds.shape[:2])}")
        else:
            assert byte_ids is not None
            if byte_ids.dim() != 2:
                raise ValueError(f"byte_ids must be (B, N), got {tuple(byte_ids.shape)}")
            batch, num_bytes = byte_ids.shape
        if patch_ids is None:
            if byte_ids is None:
                raise ValueError("patch_ids is required when byte_embeds is given (the patch map cannot be derived from embeddings)")
            patch_ids = compute_patch_ids(byte_ids, self.config.blt_patch_size)
        num_patches = _validate_patch_ids(patch_ids, num_bytes)
        if patch_ids.shape[0] != batch:
            raise ValueError(f"patch_ids batch ({patch_ids.shape[0]}) does not match byte batch ({batch})")
        embedded = self._embeddings(byte_ids, byte_embeds)
        if embedded.shape != (batch, num_bytes, self.local_hidden_size):
            raise ValueError(f"embeddings {tuple(embedded.shape)} do not match byte batch {batch} x {num_bytes} x {self.local_hidden_size}")

        hidden = embedded
        patches = self.patch_init_proj(_max_pool_patches(embedded, patch_ids, num_patches))
        patch_positions = torch.arange(num_patches, device=hidden.device).unsqueeze(0).expand(hidden.shape[0], -1)
        membership = patch_ids.unsqueeze(1) != torch.arange(num_patches, device=hidden.device).view(1, -1, 1)
        for layer in cast("list[_LocalEncoderLayer]", list(self.layers)):
            hidden = hidden + layer.attn(layer.norm(hidden))[0]
            hidden = layer.ffn(hidden)
            cross, _, _ = layer.cross(
                layer.cross_query_norm(patches),
                encoder_hidden_states=layer.cross_kv_norm(hidden),
                attention_mask=membership,
                position_ids=patch_positions,
                encoder_position_ids=patch_ids,
                is_causal=False,
            )
            patches = patches + cross
        return self.patch_out_proj(patches), hidden


class _LocalDecoderLayer(nn.Module):
    """Patch-to-byte cross-attention followed by a byte transformer layer."""

    def __init__(self, config: MemoryMachineConfig, local_cfg: MemoryMachineConfig, cross_cfg: MemoryMachineConfig):
        super().__init__()
        self.cross_query_norm = build_norm(config.blt_local_norm, local_cfg.hidden_size, eps=config.rms_norm_eps)
        self.cross_kv_norm = build_norm(config.blt_local_norm, local_cfg.hidden_size, eps=config.rms_norm_eps)
        self.cross = GroupedQueryAttention(cross_cfg)
        self.norm = build_norm(config.blt_local_norm, local_cfg.hidden_size, eps=config.rms_norm_eps)
        self.attn = GroupedQueryAttention(local_cfg, config.blt_local_window_size)
        self.ffn = DenseFFN(
            local_cfg,
            config.blt_local_ffn,
            norm_kind=config.blt_local_norm,
            intermediate_size=config.blt_local_ffn_intermediate_size or None,
        )

    def reset_parameters(self) -> None:
        """Re-apply every sub-module's neutral-start initialization."""
        for module in (self.cross_query_norm, self.cross_kv_norm, self.cross, self.norm, self.attn, self.ffn):
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()


class VocabularyDecoder(nn.Module):
    r"""VocabularyDecoder(config, vocab_size=None, local_hidden_size=None)

    BLT local decoder: patch representations -> byte logits.

    Byte queries start from the local encoder's final byte states (Eq. 9) and
    the latent patch representations provide keys/values. Each of
    ``config.blt_local_layers`` layers applies patch cross-attention followed
    by a causal sliding-window byte transformer layer (Eqs. 10-12), then a
    final norm and a projection to the byte vocabulary. Keys are
    ``[begin_patch] + patch_reps``, so ``patch_ids[i] = p`` reads latent patch
    ``p - 1`` (``p = 0`` reads the learned begin-of-bytes patch). Callers pass
    the patch id of byte ``i`` itself: patch ``p(i)`` includes byte ``i`` but
    never byte ``i + 1``, so next-byte prediction stays causal.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides the ``blt_*`` local-model settings.
        vocab_size (int, optional): byte vocabulary size. Default: ``None``
            (``config.blt_vocab_size`` or ``config.vocab_size``).
        local_hidden_size (int, optional): byte model width ``h_D``.
            Default: ``None`` (``config.blt_local_hidden_size`` or
            ``config.hidden_size``).

    Examples::

        >>> import torch
        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> cfg = MemoryMachineConfig(hidden_size=16, num_attention_heads=2,
        ...     num_key_value_heads=1, csa_num_output_groups=2,
        ...     blt_local_hidden_size=16, blt_local_layers=1,
        ...     blt_local_window_size=8)
        >>> dec = VocabularyDecoder(cfg, vocab_size=32)
        >>> logits = dec(torch.randn(1, 6, 16), torch.randn(1, 3, 16),
        ...     torch.tensor([[0, 0, 1, 1, 2, 2]]))
        >>> logits.shape
        torch.Size([1, 6, 32])
    """

    def __init__(
        self,
        config: MemoryMachineConfig,
        vocab_size: int | None = None,
        local_hidden_size: int | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size if vocab_size is not None else (config.blt_vocab_size or config.vocab_size)
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size ({self.vocab_size}) must be positive")
        if local_hidden_size is not None:
            config = replace(config, blt_local_hidden_size=local_hidden_size)
        self.config = config
        local_cfg, cross_cfg, local_hidden = _local_configs(config)
        self.local_hidden_size = local_hidden
        self.max_patch_bytes = config.blt_max_patch_bytes
        self.patch_proj = nn.Linear(config.hidden_size, local_hidden, bias=False)
        self.begin_patch = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.byte_slot_embed = nn.Embedding(self.max_patch_bytes, local_hidden)
        self.layers = nn.ModuleList([_LocalDecoderLayer(config, local_cfg, cross_cfg) for _ in range(config.blt_local_layers)])
        self.norm = build_norm(config.blt_local_norm, local_hidden, eps=config.rms_norm_eps)
        self.output = nn.Linear(local_hidden, self.vocab_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Re-apply projection, begin-patch, slot, and local-layer initialization."""
        std = self.config.initializer_range
        init_linear(self.patch_proj, std)
        nn.init.trunc_normal_(self.begin_patch, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.trunc_normal_(self.byte_slot_embed.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        for layer in cast("list[_LocalDecoderLayer]", list(self.layers)):
            layer.reset_parameters()
        reset = getattr(self.norm, "reset_parameters", None)
        if callable(reset):
            reset()
        nn.init.trunc_normal_(self.output.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)

    def _run(self, byte_states: torch.Tensor, patch_reps: torch.Tensor, patch_ids: torch.Tensor) -> torch.Tensor:
        r"""_run(byte_states, patch_reps, patch_ids) -> Tensor

        Shared decode loop over byte queries and patch key/values.

        Keys are ``[begin_patch] + patch_reps`` for every caller: id ``0``
        selects the learned begin patch and ``p > 0`` selects
        ``patch_reps[p - 1]``.
        """
        bsz, num_patches, _ = patch_reps.shape
        if patch_ids.shape != byte_states.shape[:2]:
            raise ValueError(f"patch_ids {tuple(patch_ids.shape)} must match byte queries {tuple(byte_states.shape[:2])}")
        if patch_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"patch_ids must be an integer tensor, got {patch_ids.dtype}")
        if bool((patch_ids < 0).any()) or bool((patch_ids > num_patches).any()):
            raise ValueError(f"patch_ids must be in [0, {num_patches}] for {num_patches} patch representations")
        if bsz != byte_states.shape[0]:
            raise ValueError(f"patch_reps batch ({bsz}) must match byte queries ({byte_states.shape[0]})")
        keys = self.patch_proj(torch.cat([self.begin_patch.expand(bsz, -1, -1), patch_reps], dim=1))
        patch_index = torch.arange(num_patches + 1, device=byte_states.device)
        membership = patch_ids.unsqueeze(-1) != patch_index.view(1, 1, -1)
        hidden = byte_states
        for layer in cast("list[_LocalDecoderLayer]", list(self.layers)):
            cross, _, _ = layer.cross(
                layer.cross_query_norm(hidden),
                encoder_hidden_states=layer.cross_kv_norm(keys),
                attention_mask=membership,
                position_ids=patch_ids,
                is_causal=False,
            )
            hidden = hidden + cross
            hidden = hidden + layer.attn(layer.norm(hidden))[0]
            hidden = layer.ffn(hidden)
        return self.output(self.norm(hidden))

    def forward(self, byte_states: torch.Tensor, patch_reps: torch.Tensor, patch_ids: torch.Tensor) -> torch.Tensor:
        r"""forward(byte_states, patch_reps, patch_ids) -> Tensor

        Decode next-byte logits from encoder byte states and patch latents.

        Args:
            byte_states (Tensor): local-encoder byte states ``(B, N, h_D)``.
            patch_reps (Tensor): latent patch representations
                ``(B, P, hidden)``.
            patch_ids (Tensor): patch id of each byte query, ``(B, N)`` with
                values in ``[0, P]``. ``0`` selects the learned begin-of-bytes
                patch and ``p > 0`` selects ``patch_reps[p - 1]``. Pass the
                encoder patch id of byte ``i`` unchanged.

        Returns:
            Tensor: next-byte logits ``(B, N, vocab_size)``.
        """
        if byte_states.dim() != 3 or byte_states.shape[-1] != self.local_hidden_size:
            raise ValueError(f"byte_states must be (B, N, {self.local_hidden_size}), got {tuple(byte_states.shape)}")
        if patch_reps.dim() != 3 or patch_reps.shape[-1] != self.config.hidden_size:
            raise ValueError(f"patch_reps must be (B, P, {self.config.hidden_size}), got {tuple(patch_reps.shape)}")
        return self._run(byte_states, patch_reps, patch_ids)

    def decode_masked_patches(self, masked_states: torch.Tensor, patch_reps: torch.Tensor, first_patch_index: int) -> torch.Tensor:
        r"""decode_masked_patches(masked_states, patch_reps, first_patch_index) -> Tensor

        Decode the bytes of Perceiver-AR masked patches.

        There are no input byte states for future patches, so each masked
        patch's latent representation seeds ``config.blt_max_patch_bytes``
        byte-slot queries (a learned per-slot embedding breaks the symmetry).
        Every slot attends to its own patch's latent representation, i.e.
        ``patch_reps[first_patch_index + j]`` for the ``j``-th masked patch.

        Args:
            masked_states (Tensor): latent outputs for the masked patches
                ``(B, L, hidden)``.
            patch_reps (Tensor): latent patch representations
                ``(B, P + L, hidden)``; entries from ``first_patch_index``
                onward are the masked patches' own latents.
            first_patch_index (int): index of the first masked patch in
                ``patch_reps`` (``P``).

        Returns:
            Tensor: byte logits ``(B, L * blt_max_patch_bytes, vocab_size)``.
        """
        if masked_states.dim() != 3 or masked_states.shape[-1] != self.config.hidden_size:
            raise ValueError(f"masked_states must be (B, L, {self.config.hidden_size}), got {tuple(masked_states.shape)}")
        if patch_reps.dim() != 3 or patch_reps.shape[-1] != self.config.hidden_size:
            raise ValueError(f"patch_reps must be (B, P, {self.config.hidden_size}), got {tuple(patch_reps.shape)}")
        if patch_reps.shape[0] != masked_states.shape[0]:
            raise ValueError(f"patch_reps batch ({patch_reps.shape[0]}) must match masked_states batch ({masked_states.shape[0]})")
        num_masked = masked_states.shape[1]
        if num_masked < 1:
            raise ValueError("masked_states must hold at least one patch")
        if first_patch_index < 0 or first_patch_index + num_masked > patch_reps.shape[1]:
            raise ValueError(f"first_patch_index ({first_patch_index}) + masked patches ({num_masked}) exceeds {patch_reps.shape[1]} patch representations")
        slots = self.max_patch_bytes
        seed = self.patch_proj(masked_states).unsqueeze(2).expand(-1, -1, slots, -1).reshape(masked_states.shape[0], num_masked * slots, -1)
        slot_ids = torch.arange(num_masked * slots, device=seed.device) % slots
        hidden = seed + self.byte_slot_embed(slot_ids).unsqueeze(0)
        # Shifted key convention: id ``first_patch_index + 1 + j`` reads
        # ``patch_reps[first_patch_index + j]`` (own patch).
        patch_ids = first_patch_index + 1 + torch.arange(num_masked * slots, device=hidden.device) // slots
        patch_ids = patch_ids.unsqueeze(0).expand(hidden.shape[0], -1)
        return self._run(hidden, patch_reps, patch_ids)
