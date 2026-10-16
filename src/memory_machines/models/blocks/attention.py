r"""Dense attention: GQA with built-in sliding window.

:class:`GroupedQueryAttention` is the grouped-query baseline with
optional CVPE QK bias, FRVM OV gating, and a built-in sliding window:
``window_size > 0`` bounds causal attention to the last ``window_size``
keys and keeps only that window in the returned KV cache, while
``window_size <= 0`` runs full attention with an untruncated cache.
The dense feed-forward block lives in
:mod:`memory_machines.models.blocks.ffn`.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from memory_machines.utils.config_schema import MemoryMachineConfig

from .positioning import CVPEContextBias, FRVMValueGate
from .shared import _check_position_bounds, _resolve_positions, _window_mask, residual_init_scale


def _causal_key_mask(q_pos: torch.Tensor, k_pos: torch.Tensor) -> torch.Tensor:
    r"""_causal_key_mask(q_pos, k_pos) -> Tensor

    Build a broadcastable causal keep-mask from absolute positions.

    Args:
        q_pos (Tensor): query positions of shape ``(B, Lq)``.
        k_pos (Tensor): key positions of shape ``(B, Lk)``.

    Returns:
        Tensor: bool mask of shape ``(B, 1, Lq, Lk)`` where ``True`` means
        the key is not in the future (``k <= q``).

    Examples::

        >>> import torch
        >>> _causal_key_mask(torch.tensor([[1, 2]]), torch.tensor([[0, 1, 2]]))
        tensor([[[[ True,  True, False],
                  [ True,  True,  True]]]])
    """
    return (k_pos.unsqueeze(1) <= q_pos.unsqueeze(-1)).unsqueeze(1)


def _prepare_attention_mask(
    attention_mask: torch.Tensor | None,
    bsz: int,
    q_len: int,
    kv_len: int,
    num_heads: int,
    logits: torch.Tensor,
) -> torch.Tensor | None:
    r"""_prepare_attention_mask(attention_mask, bsz, q_len, kv_len, num_heads, logits) -> Tensor | None

    Normalize a user mask to an additive broadcastable form.

    Accepted layouts (``torch.nn.MultiheadAttention`` conventions):

    - float additive masks (added directly; ``-inf`` = masked out);
    - bool masks where ``True`` = masked out (do not attend);
    - shapes ``(Lq, Lk)``, ``(B, Lq, Lk)``, ``(B, 1, Lk)`` (key padding), or
      ``(B, 1/H, Lq, Lk)`` with leading singleton dims for broadcasting.

    A 2D mask is always ``(Lq, Lk)``; pass padding as 3D/4D.

    Args:
        attention_mask (Tensor, optional): user mask or ``None``.
        bsz (int): batch size.
        q_len (int): query length ``Lq``.
        kv_len (int): key length ``Lk``.
        num_heads (int): attention heads ``H`` (for 4D validation).
        logits (Tensor): attention logits fixing dtype/device of the output.

    Returns:
        Tensor or None: additive mask broadcastable to ``(B, H, Lq, Lk)``, or
        ``None`` when no mask was given.

    Examples::

        >>> import torch
        >>> logits = torch.zeros(1, 2, 3, 4)
        >>> _prepare_attention_mask(None, 1, 3, 4, 2, logits) is None
        True
        >>> _prepare_attention_mask(torch.ones(3, 4, dtype=torch.bool), 1, 3, 4, 2, logits).shape
        torch.Size([3, 4])
    """
    if attention_mask is None:
        return None
    mask = attention_mask
    if mask.dim() == 2:
        if tuple(mask.shape) != (q_len, kv_len):
            raise ValueError(f"2D attention_mask must be (Lq, Lk) = ({q_len}, {kv_len}), got {tuple(mask.shape)}; pass key-padding masks as 3D (B, 1, Lk) or (B, Lq, Lk)")
    elif mask.dim() == 3:
        if tuple(mask.shape) not in ((bsz, q_len, kv_len), (bsz, 1, kv_len)):
            raise ValueError(f"3D attention_mask must be (B, Lq, Lk) = ({bsz}, {q_len}, {kv_len}) or (B, 1, Lk), got {tuple(mask.shape)}")
        mask = mask.unsqueeze(1)
    elif mask.dim() == 4:
        if mask.shape[0] not in (1, bsz) or mask.shape[1] not in (1, num_heads) or tuple(mask.shape[2:]) != (q_len, kv_len):
            raise ValueError(f"4D attention_mask must be (B|1, 1|H, Lq, Lk) = ({bsz}|1, 1|{num_heads}, {q_len}, {kv_len}), got {tuple(mask.shape)}")
    else:
        raise ValueError(f"attention_mask must be 2D/3D/4D, got {mask.dim()}D {tuple(mask.shape)}")
    if mask.dtype == torch.bool:
        # True = masked out.
        return torch.where(mask, torch.tensor(float("-inf"), dtype=logits.dtype, device=logits.device), torch.zeros((), dtype=logits.dtype, device=logits.device))
    if not mask.is_floating_point():
        raise ValueError(f"attention_mask must be bool or a floating additive mask, got {mask.dtype}")
    return mask


def _sanitize_attention_weights(weights: torch.Tensor) -> torch.Tensor:
    r"""_sanitize_attention_weights(weights) -> Tensor

    Zero fully-masked rows whose softmax over all-``-inf`` yields NaN.

    Args:
        weights (Tensor): post-softmax weights of shape ``(..., Lq, Lk)``.

    Returns:
        Tensor: weights with the same shape, with non-finite rows replaced by
        zeros.

    Examples::

        >>> import torch
        >>> w = torch.tensor([[[0.5, 0.5], [float("nan"), float("nan")]]])
        >>> _sanitize_attention_weights(w)
        tensor([[[0.5000, 0.5000],
                 [0.0000, 0.0000]]])
    """
    row_ok = torch.isfinite(weights).any(dim=-1, keepdim=True)
    return torch.where(row_ok, weights, 0.0)


def _apply_masks_and_softmax(
    attn: torch.Tensor,
    q_pos: torch.Tensor,
    k_pos: torch.Tensor,
    window_size: int,
    is_causal: bool,
    windowed: bool,
) -> torch.Tensor:
    r"""_apply_masks_and_softmax(attn, q_pos, k_pos, window_size, is_causal, windowed) -> Tensor

    Apply the causal and sliding-window bounds to raw logits.

    The window bound subsumes the causal half (``_window_mask`` keeps
    ``0 <= q - k < window``) when causal; with ``is_causal=False`` the
    bound is symmetric (``|q - k| < window``), matching ``attend``.

    Args:
        attn (Tensor): raw logits of shape ``(B, H, Lq, Lk)``.
        q_pos (Tensor): query positions ``(B, Lq)``.
        k_pos (Tensor): key positions ``(B, Lk)``.
        window_size (int): window width; ``<= 0`` disables the window.
        is_causal (bool): whether to mask future keys.
        windowed (bool): whether the window bound applies.

    Returns:
        Tensor: masked logits of the same shape as ``attn``.
    """
    if windowed and window_size > 0:
        if is_causal:
            win = _window_mask(q_pos, k_pos, window_size).unsqueeze(1)
            return attn.masked_fill(~win, float("-inf"))
        rel = q_pos.unsqueeze(-1) - k_pos.unsqueeze(1)
        return attn.masked_fill((rel.abs() >= window_size).unsqueeze(1), float("-inf"))
    if is_causal:
        return attn.masked_fill(~_causal_key_mask(q_pos, k_pos), float("-inf"))
    return attn


class GroupedQueryAttention(nn.Module):
    r"""GroupedQueryAttention(config, window_size=None)

    Grouped-query attention (GQA) with CVPE QK bias, FRVM OV gating,
    and a built-in sliding window.

    Queries have ``H`` heads; keys/values have ``Hkv`` heads expanded by
    repetition to ``H``. Scores are scaled dot products plus the optional
    :class:`~memory_machines.models.blocks.positioning.CVPEContextBias`;
    the weighted reduction is either plain matmul or
    :class:`~memory_machines.models.blocks.positioning.FRVMValueGate`.

    Every attention is SWA-enabled by design: ``window_size > 0`` masks
    each query to the last ``window_size`` keys (sliding-window
    attention) and the returned KV cache keeps only that window;
    ``window_size <= 0`` disables the window, running full attention
    with an untruncated cache. The window is a hyperparameter, not an
    architectural constraint — compressed/sparse variants pass a
    mechanism-scaled default (see ``MemoryMachineConfig``).

    .. math::
        \mathrm{Attn}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}} + b\right) V

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``hidden_size``, ``num_attention_heads`` (``H``),
            ``num_key_value_heads`` (``Hkv``), ``max_position_embeddings``,
            ``window_size``, ``cvpe_*``/``frvm_*`` toggles, and
            ``attention_dropout``.
        window_size (int, optional): window width; ``<= 0`` means full
            attention. Default: ``None`` (falls back to
            ``config.window_size``).

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> attn = GroupedQueryAttention(MemoryMachineConfig(hidden_size=32,
        ...     num_attention_heads=4, num_key_value_heads=2, window_size=8,
        ...     csa_num_output_groups=4))
        >>> out, w, kv = attn(torch.randn(1, 6, 32), is_causal=True)
        >>> out.shape
        torch.Size([1, 6, 32])
    """

    cvpe_bias: CVPEContextBias | None
    frvm_gate: FRVMValueGate | None

    def __init__(self, config: MemoryMachineConfig, window_size: int | None = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.window_size = window_size if window_size is not None else config.window_size
        self.initializer_range: float = config.initializer_range
        self.num_layers: int = config.num_layers
        self.residual_init_scale: bool = config.residual_init_scale

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})")

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(f"num_attention_heads ({self.num_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})")

        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

        self.cvpe_enabled = config.cvpe_enabled
        self.frvm_enabled = config.frvm_enabled
        self.frvm_chunk_size = config.frvm_chunk_size
        self.cvpe_bias = (
            CVPEContextBias(
                self.head_dim,
                mlp_hidden=config.cvpe_mlp_hidden,
                use_log_transform=config.cvpe_use_log_transform,
            )
            if self.cvpe_enabled
            else None
        )
        self.frvm_gate = FRVMValueGate(self.head_dim, self.num_heads) if self.frvm_enabled else None

        self.attention_dropout = nn.Dropout(config.attention_dropout)
        GroupedQueryAttention.reset_parameters(self)

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Q/K/V input projections use truncated normal
        ``N(0, initializer_range^2)`` clipped to
        ``[-2*initializer_range, 2*initializer_range]``
        (DeepSeek's ``initializer_range = 0.006``; the
        standard Transformer default is 0.02). The output
        projection is either zero-initialized (identity
        residual start, Fixup/ReZero) or depth-scaled to
        ``initializer_range / sqrt(2N)`` (Megatron-LM/Qwen
        residual scaling) per ``config.residual_init_scale``.
        Submodules (CVPE/FRVM) keep their own neutral-start
        inits and are left untouched.
        """
        std = self.initializer_range
        scale = residual_init_scale(std, self.num_layers, self.residual_init_scale)
        nn.init.trunc_normal_(self.q_proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.trunc_normal_(self.k_proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.trunc_normal_(self.v_proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        if scale is None:
            nn.init.zeros_(self.o_proj.weight)
        else:
            nn.init.trunc_normal_(self.o_proj.weight, mean=0.0, std=scale, a=-2 * scale, b=2 * scale)

    def _project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""_project_qkv(hidden_states) -> (Tensor, Tensor, Tensor)

        Project inputs to per-head queries, keys, and values.

        Args:
            hidden_states (Tensor): input of shape ``(B, Lq, hidden)``.

        Returns:
            Tensor: queries ``(B, H, Lq, Dh)``, keys ``(B, Hkv, Lq, Dh)``,
            and current values ``(B, Hkv, Lq, Dh)``.

        Examples::

            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> attn = GroupedQueryAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, csa_num_output_groups=2))
            >>> [t.shape for t in attn._project_qkv(__import__("torch").randn(1, 4, 16))]
            [torch.Size([1, 2, 4, 8]), torch.Size([1, 1, 4, 8]), torch.Size([1, 1, 4, 8])]
        """
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_current = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        return query_states, key_states, value_current

    def _expand_kv(
        self,
        key_states: torch.Tensor,
        value_full: torch.Tensor,
        value_current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""_expand_kv(key_states, value_full, value_current) -> (Tensor, Tensor, Tensor)

        Repeat KV heads to the full query-head count.

        Args:
            key_states (Tensor): keys of shape ``(B, Hkv, Lk, Dh)``.
            value_full (Tensor): full value store of shape ``(B, Hkv, Lk, Dh)``.
            value_current (Tensor): current-step values of shape
                ``(B, Hkv, Lq, Dh)``.

        Returns:
            Tensor: expanded keys, full values, and current values, each with
            ``H = Hkv * groups`` heads.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> attn = GroupedQueryAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, csa_num_output_groups=2))
            >>> [t.shape for t in attn._expand_kv(torch.randn(1, 1, 5, 8),
            ...     torch.randn(1, 1, 5, 8), torch.randn(1, 1, 3, 8))]
            [torch.Size([1, 2, 5, 8]), torch.Size([1, 2, 5, 8]), torch.Size([1, 2, 3, 8])]
        """
        return (
            key_states.repeat_interleave(self.num_key_value_groups, dim=1),
            value_full.repeat_interleave(self.num_key_value_groups, dim=1),
            value_current.repeat_interleave(self.num_key_value_groups, dim=1),
        )

    def _score_qk(
        self,
        query_states: torch.Tensor,
        key_exp: torch.Tensor,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
    ) -> torch.Tensor:
        r"""_score_qk(query_states, key_exp, q_pos, k_pos) -> Tensor

        Compute scaled dot-product logits plus the optional CVPE bias.

        Args:
            query_states (Tensor): queries of shape ``(B, H, Lq, Dh)``.
            key_exp (Tensor): expanded keys of shape ``(B, H, Lk, Dh)``.
            q_pos (Tensor): query positions ``(B, Lq)``.
            k_pos (Tensor): key positions ``(B, Lk)``.

        Returns:
            Tensor: logits of shape ``(B, H, Lq, Lk)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> attn = GroupedQueryAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, csa_num_output_groups=2))
            >>> attn._score_qk(torch.randn(1, 2, 3, 8), torch.randn(1, 2, 5, 8),
            ...     torch.tensor([[5, 6, 7]]), torch.tensor([[0, 1, 2, 3, 4]])).shape
            torch.Size([1, 2, 3, 5])
        """
        logits = torch.matmul(query_states, key_exp.transpose(2, 3)) / math.sqrt(self.head_dim)
        if self.cvpe_bias is not None:
            logits = logits + self.cvpe_bias(query_states, q_pos, k_pos).to(logits.dtype)
        return logits

    def _weights_to_heads(
        self,
        weights: torch.Tensor,
        value_full_exp: torch.Tensor,
        value_query_exp: torch.Tensor,
    ) -> torch.Tensor:
        r"""_weights_to_heads(weights, value_full_exp, value_query_exp) -> Tensor

        Reduce attention weights against values (plain or FRVM-gated).

        Args:
            weights (Tensor): weights of shape ``(B, H, Lq, Lk)``.
            value_full_exp (Tensor): expanded full values ``(B, H, Lk, Dh)``.
            value_query_exp (Tensor): per-query reference values
                ``(B, H, Lq, Dh)`` used only by the FRVM gate.

        Returns:
            Tensor: per-head outputs of shape ``(B, H, Lq, Dh)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> attn = GroupedQueryAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, frvm_enabled=False, csa_num_output_groups=2))
            >>> attn._weights_to_heads(torch.softmax(torch.randn(1, 2, 3, 5), -1),
            ...     torch.randn(1, 2, 5, 8), torch.randn(1, 2, 3, 8)).shape
            torch.Size([1, 2, 3, 8])
        """
        gate = self.frvm_gate
        if gate is not None:
            return gate.gated_reduction(weights, value_full_exp, value_query_exp, chunk_size=self.frvm_chunk_size)
        return torch.matmul(weights, value_full_exp)

    def _project_heads(self, heads: torch.Tensor, bsz: int, q_len: int) -> torch.Tensor:
        r"""_project_heads(heads, bsz, q_len) -> Tensor

        Merge heads and apply the output projection.

        Args:
            heads (Tensor): per-head outputs of shape ``(B, H, Lq, Dh)``.
            bsz (int): batch size.
            q_len (int): query length.

        Returns:
            Tensor: model-dimension output of shape ``(B, Lq, hidden)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> attn = GroupedQueryAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, csa_num_output_groups=2))
            >>> attn._project_heads(torch.randn(1, 2, 3, 8), 1, 3).shape
            torch.Size([1, 3, 16])
        """
        out = heads.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(out)

    def project_heads(self, heads: torch.Tensor, bsz: int, q_len: int) -> torch.Tensor:
        r"""project_heads(heads, bsz, q_len) -> Tensor

        Public alias of :meth:`_project_heads`.

        Args:
            heads (Tensor): per-head outputs of shape ``(B, H, Lq, Dh)``.
            bsz (int): batch size.
            q_len (int): query length.

        Returns:
            Tensor: model-dimension output of shape ``(B, Lq, hidden)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> attn = GroupedQueryAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, csa_num_output_groups=2))
            >>> attn.project_heads(torch.randn(1, 2, 3, 8), 1, 3).shape
            torch.Size([1, 3, 16])
        """
        return self._project_heads(heads, bsz, q_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_len_override: int | None = None,
        past_positions: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        r"""forward(hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, past_len_override=None, past_positions=None, is_causal=True) -> (Tensor, Tensor | None, tuple | None)

        Run attention with the built-in window and optional caching.

        With ``is_causal=True``, keys are masked by ``k_pos <= q_pos``
        so cached prefixes stay correct under incremental decoding.
        ``window_size > 0`` further bounds each query to the last
        ``window_size`` keys, and the returned cache keeps only that
        window; ``window_size <= 0`` runs full attention with an
        untruncated cache. Replaying a truncated cache (``past_key_value``
        set while ``window_size > 0``) requires ``past_positions`` with
        the cached keys' absolute positions — without them the window
        would silently attend the wrong keys. ``past_len_override``
        (when given) must agree with ``past_positions``.

        Args:
            hidden_states (Tensor): input of shape ``(B, Lq, hidden)``.
            attention_mask (Tensor, optional): bool (``True`` = masked out) or
                float additive mask; 2D ``(Lq, Lk)``, 3D ``(B, Lq, Lk)`` /
                ``(B, 1, Lk)``, or 4D ``(B|1, 1|H, Lq, Lk)``. Default: ``None``.
            position_ids (Tensor, optional): absolute positions. Default: ``None``.
            past_key_value (tuple of Tensor, optional): cached
                ``(keys, values)`` of shape ``(B, Hkv, Lp, Dh)``.
                Default: ``None``.
            output_attentions (bool, optional): whether to return weights.
                Default: ``False``.
            use_cache (bool, optional): whether to return updated KV.
                Default: ``False``.
            past_len_override (int, optional): explicit past length
                overriding the cached width; must agree with
                ``past_positions`` when both are given, and requires
                ``past_key_value``. Default: ``None``.
            past_positions (Tensor, optional): explicit cached positions
                of shape ``(B, Wprev)``; required when replaying a
                truncated ``past_key_value``. Default: ``None``.
            is_causal (bool, optional): whether to apply the causal mask.
                Default: ``True``.

        Returns:
            Tensor: hidden outputs ``(B, Lq, hidden)``, attention weights
            ``(B, H, Lq, Lk)`` or ``None``, and the updated ``(keys, values)``
            cache (truncated to the window when ``window_size > 0``) or
            ``None``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> attn = GroupedQueryAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, window_size=4,
            ...     csa_num_output_groups=2))
            >>> out, w, _ = attn(torch.randn(1, 4, 16), is_causal=True,
            ...     output_attentions=True)
            >>> out.shape, w.shape
            (torch.Size([1, 4, 16]), torch.Size([1, 2, 4, 4]))
        """
        if not is_causal and (use_cache or past_key_value is not None):
            raise ValueError("GroupedQueryAttention with is_causal=False cannot use KV caching (truncated cache is only valid for causal attention)")
        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device

        query_states, key_states, value_current = self._project_qkv(hidden_states)

        cached_len = past_key_value[0].shape[-2] if past_key_value is not None else 0

        # NOTE: past_positions.max() incurs one GPU->CPU sync; required because
        # windowed caches are truncated and width alone does not give absolute positions.
        def _implied_len(pos: torch.Tensor) -> int:
            if pos.dim() not in (1, 2):
                raise ValueError(f"past_positions must be 1D (Lk,) or 2D (B, Lk), got {pos.dim()}D {tuple(pos.shape)}")
            p = pos.to(device) if pos.dim() == 2 else pos.to(device).unsqueeze(0).expand(bsz, -1)
            if p.shape[0] != bsz:
                raise ValueError(f"past_positions batch ({p.shape[0]}) does not match input batch ({bsz})")
            return int(p.max().item()) + 1

        if past_len_override is not None:
            if past_key_value is None and past_len_override != 0:
                raise ValueError("past_len_override requires past_key_value (a nonzero override with no cached history is meaningless; omit it for a fresh call)")
            if past_positions is not None:
                implied_len = _implied_len(past_positions)
                if past_len_override != implied_len:
                    raise ValueError(f"past_len_override ({past_len_override}) disagrees with past_positions (max {implied_len - 1}); pass one or make them consistent")
                if past_len_override != cached_len and self.window_size <= 0:
                    # Full-attention caches are untruncated: override must match cached width.
                    raise ValueError(f"past_len_override ({past_len_override}) does not match cached width ({cached_len})")
            true_past_len = past_len_override
        elif past_positions is not None:
            true_past_len = _implied_len(past_positions)
        else:
            true_past_len = cached_len
        total_len = true_past_len + q_len

        if total_len > self.max_position_embeddings:
            raise ValueError(f"Sequence length ({total_len}) exceeds configured max_position_embeddings ({self.max_position_embeddings})")

        if past_key_value is None:
            q_pos, k_pos = _resolve_positions(hidden_states, q_len, true_past_len, position_ids)
        else:
            # A windowed cache is truncated by construction, so
            # the caller must pass the cached keys' absolute
            # positions. Full-attention caches are untruncated
            # and keep dense positions, so arange is correct.
            if self.window_size > 0 and past_positions is None:
                raise ValueError(
                    "replaying a past_key_value with window_size > 0 requires past_positions (the cached keys' absolute positions); the windowed cache is truncated and its key positions are otherwise unknowable"
                )
            if position_ids is not None:
                if position_ids.dim() == 2:
                    if position_ids.shape[0] != bsz:
                        raise ValueError(f"position_ids batch size ({position_ids.shape[0]}) does not match input batch size ({bsz})")
                    q_pos = position_ids[:, -q_len:].to(device)
                else:
                    q_pos = position_ids[-q_len:].to(device).unsqueeze(0).expand(bsz, -1)
            else:
                q_pos = torch.arange(true_past_len, true_past_len + q_len, device=device).unsqueeze(0).expand(bsz, -1)
            if past_positions is not None:
                pp = past_positions.to(device)
                if pp.dim() == 1:
                    pp = pp.unsqueeze(0).expand(bsz, -1)
                elif pp.dim() != 2:
                    raise ValueError(f"past_positions must be 1D (Lk,) or 2D (B, Lk), got {pp.dim()}D {tuple(pp.shape)}")
                k_pos = torch.cat([pp, q_pos], dim=1)
            else:
                k_pos = torch.arange(true_past_len + q_len, device=device).unsqueeze(0).expand(bsz, -1)
        _check_position_bounds(q_pos, k_pos, self.max_position_embeddings, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_full = torch.cat([past_key_value[1], value_current], dim=2)
        else:
            value_full = value_current

        key_exp, value_full_exp, value_query_exp = self._expand_kv(key_states, value_full, value_current)

        attn_weights = self._score_qk(query_states, key_exp, q_pos, k_pos)

        attn_weights = _apply_masks_and_softmax(attn_weights, q_pos, k_pos, self.window_size, is_causal, windowed=True)

        additive_mask = _prepare_attention_mask(attention_mask, bsz, q_len, key_states.shape[2], self.num_heads, attn_weights)
        if additive_mask is not None:
            attn_weights = attn_weights + additive_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = _sanitize_attention_weights(attn_weights)
        # Dropout after sanitize is safe: sanitized rows are exact zeros and
        # 0 * dropout_scale == 0, so fully-masked rows stay zero (not NaN).
        attn_weights = self.attention_dropout(attn_weights)

        attn_output_heads = self._weights_to_heads(attn_weights, value_full_exp, value_query_exp)

        attn_output = self._project_heads(attn_output_heads, bsz, q_len)

        past_key_value = None
        if use_cache:
            if self.window_size > 0:
                past_key_value = (
                    key_states[:, :, -self.window_size :, :],
                    value_full[:, :, -self.window_size :, :],
                )
            else:
                past_key_value = (key_states, value_full)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def attend(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_full: torch.Tensor,
        value_query: torch.Tensor,
        q_pos: torch.Tensor,
        k_pos_full: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        windowed: bool = True,
        is_causal: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""attend(query_states, key_states, value_full, value_query, q_pos, k_pos_full, attention_mask=None, windowed=True, is_causal=True) -> (Tensor, Tensor)

        Batched head-parallel attention core over explicit QKV states.

        Args:
            query_states (Tensor): queries ``(B, H, Lq, Dh)``.
            key_states (Tensor): keys ``(B, Hkv, Lk, Dh)``.
            value_full (Tensor): full values ``(B, Hkv, Lk, Dh)``.
            value_query (Tensor): per-query reference values
                ``(B, Hkv, Lq, Dh)``.
            q_pos (Tensor): query positions ``(B, Lq)``.
            k_pos_full (Tensor): key positions ``(B, Lk)``.
            attention_mask (Tensor, optional): additive/bool mask.
                Default: ``None``.
            windowed (bool, optional): whether to apply the window bound.
                With ``is_causal=True`` the bound is causal
                (``0 <= q - k < window_size``); with
                ``is_causal=False`` it is bidirectional within the
                window (``|q - k| < window_size``). ``False`` with
                ``is_causal=True`` is pure causal. Default: ``True``.
            is_causal (bool, optional): whether to mask futures.
                Default: ``True``.

        Returns:
            Tensor: per-head outputs ``(B, H, Lq, Dh)`` and weights
            ``(B, H, Lq, Lk)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> attn = GroupedQueryAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, window_size=4,
            ...     csa_num_output_groups=2))
            >>> out, w = attn.attend(torch.randn(1, 2, 3, 8), torch.randn(1, 1, 5, 8),
            ...     torch.randn(1, 1, 5, 8), torch.randn(1, 1, 3, 8),
            ...     torch.tensor([[3, 4, 5]]), torch.tensor([[1, 2, 3, 4, 5]]))
            >>> out.shape, w.shape
            (torch.Size([1, 2, 3, 8]), torch.Size([1, 2, 3, 5]))
        """
        key_exp, val_exp, vq_exp = self._expand_kv(key_states, value_full, value_query)
        attn = self._score_qk(query_states, key_exp, q_pos, k_pos_full)
        # windowed, is_causal=False is bidirectional within the
        # window: |q - k| < window_size in BOTH directions (the
        # caller explicitly asked for non-causal attention, so the
        # bound is symmetric). All other cells are causal-bounded.
        if windowed and self.window_size > 0 and is_causal:
            win = _window_mask(q_pos, k_pos_full, self.window_size).unsqueeze(1)
            attn = attn.masked_fill(~win, float("-inf"))
        elif windowed and self.window_size > 0:
            rel = q_pos.unsqueeze(-1) - k_pos_full.unsqueeze(1)
            attn = attn.masked_fill((rel.abs() >= self.window_size).unsqueeze(1), float("-inf"))
        elif is_causal:
            attn = attn.masked_fill(~_causal_key_mask(q_pos, k_pos_full), float("-inf"))
        additive_mask = _prepare_attention_mask(attention_mask, query_states.shape[0], query_states.shape[2], key_exp.shape[2], self.num_heads, attn)
        if additive_mask is not None:
            attn = attn + additive_mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn = _sanitize_attention_weights(attn)
        attn = self.attention_dropout(attn)
        return self._weights_to_heads(attn, val_exp, vq_exp), attn

    def truncate_for_replay(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""truncate_for_replay(key_states, value_states, positions) -> (Tensor, Tensor, Tensor)

        Truncate a KV window to the last ``window_size`` entries.

        With ``window_size <= 0`` (full attention) the inputs are
        returned unchanged; otherwise the trailing ``window_size``
        entries are kept for bounded replay.

        Args:
            key_states (Tensor): keys of shape ``(B, Hkv, Lk, Dh)``.
            value_states (Tensor): values with the same shape as
                ``key_states``.
            positions (Tensor): positions of shape ``(B, Lk)`` or
                ``(Lk,)``.

        Returns:
            Tensor: truncated keys, values, and positions.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> attn = GroupedQueryAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, window_size=4,
            ...     csa_num_output_groups=2))
            >>> k, v, p = attn.truncate_for_replay(torch.randn(1, 1, 6, 8),
            ...     torch.randn(1, 1, 6, 8), torch.arange(6))
            >>> k.shape, p.shape
            (torch.Size([1, 1, 4, 8]), torch.Size([4]))
        """
        if self.window_size <= 0:
            return key_states, value_states, positions
        if positions.dim() == 2:
            positions = positions[:, -self.window_size :]
        else:
            positions = positions[-self.window_size :]
        return key_states[:, :, -self.window_size :, :], value_states[:, :, -self.window_size :, :], positions
