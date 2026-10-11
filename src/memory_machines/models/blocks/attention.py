r"""Dense attention: GQA, sliding-window attention, SwiGLU.

:class:`GroupedQueryAttention` is the dense grouped-query baseline with
optional CVPE QK bias and FRVM OV gating.
:class:`SlidingWindowAttention` adds a transient causal window cache, and
:class:`SwiGLU` is the gated feed-forward block.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from memory_machines.utils.config_schema import MemoryMachineConfig

from .positioning import CVPEContextBias, FRVMValueGate
from .shared import _check_position_bounds, _resolve_positions, _window_mask


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
        return torch.zeros((), dtype=logits.dtype, device=logits.device).masked_fill(mask, float("-inf"))
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
    return torch.where(row_ok, weights, torch.zeros_like(weights))


class GroupedQueryAttention(nn.Module):
    r"""GroupedQueryAttention(config)

    Grouped-query attention (GQA) with CVPE QK bias and FRVM OV gating.

    Queries have ``H`` heads; keys/values have ``Hkv`` heads expanded by
    repetition to ``H``. Scores are scaled dot products plus the optional
    :class:`~memory_machines.models.blocks.positioning.CVPEContextBias`;
    the weighted reduction is either plain matmul or
    :class:`~memory_machines.models.blocks.positioning.FRVMValueGate`.

    .. math::
        \mathrm{Attn}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}} + b\right) V

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``hidden_size``, ``num_attention_heads`` (``H``),
            ``num_key_value_heads`` (``Hkv``), ``max_position_embeddings``,
            ``cvpe_*``/``frvm_*`` toggles, and ``attention_dropout``.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> attn = GroupedQueryAttention(MemoryMachineConfig(hidden_size=32,
        ...     num_attention_heads=4, num_key_value_heads=2))
        >>> out, w, kv = attn(torch.randn(1, 6, 32), is_causal=True)
        >>> out.shape
        torch.Size([1, 6, 32])
    """

    def __init__(self, config: MemoryMachineConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

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
            ...     num_attention_heads=2, num_key_value_heads=1))
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
            ...     num_attention_heads=2, num_key_value_heads=1))
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
            ...     num_attention_heads=2, num_key_value_heads=1))
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
            ...     num_attention_heads=2, num_key_value_heads=1, frvm_enabled=False))
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
            ...     num_attention_heads=2, num_key_value_heads=1))
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
            ...     num_attention_heads=2, num_key_value_heads=1))
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
        is_causal: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        r"""forward(hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, is_causal=False) -> (Tensor, Tensor | None, tuple | None)

        Run dense grouped-query attention with optional caching.

        With ``is_causal=True``, keys are masked by ``k_pos <= q_pos`` so
        cached prefixes stay correct under incremental decoding.

        Args:
            hidden_states (Tensor): input of shape ``(B, Lq, hidden)``.
            attention_mask (Tensor, optional): bool (``True`` = masked out) or
                float additive mask; 2D ``(Lq, Lk)``, 3D ``(B, Lq, Lk)`` /
                ``(B, 1, Lk)``, or 4D ``(B|1, 1|H, Lq, Lk)``. Default: ``None``.
            position_ids (Tensor, optional): absolute positions. Default: ``None``.
            past_key_value (tuple of Tensor, optional): cached ``(keys, values)``
                of shape ``(B, Hkv, Lp, Dh)``. Default: ``None``.
            output_attentions (bool, optional): whether to return weights.
                Default: ``False``.
            use_cache (bool, optional): whether to return updated KV.
                Default: ``False``.
            is_causal (bool, optional): whether to apply the causal mask.
                Default: ``False``.

        Returns:
            Tensor: hidden outputs ``(B, Lq, hidden)``, attention weights
            ``(B, H, Lq, Lk)`` or ``None``, and the updated ``(keys, values)``
            cache or ``None``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> attn = GroupedQueryAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1))
            >>> out, w, _ = attn(torch.randn(1, 4, 16), is_causal=True,
            ...     output_attentions=True)
            >>> out.shape, w.shape
            (torch.Size([1, 4, 16]), torch.Size([1, 2, 4, 6]))
        """
        bsz, q_len, _ = hidden_states.size()

        query_states, key_states, value_current = self._project_qkv(hidden_states)

        past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
        total_len = past_len + q_len

        if total_len > self.max_position_embeddings:
            raise ValueError(f"Sequence length ({total_len}) exceeds configured max_position_embeddings ({self.max_position_embeddings})")

        q_pos, k_pos = _resolve_positions(hidden_states, q_len, past_len, position_ids)
        _check_position_bounds(q_pos, k_pos, self.max_position_embeddings, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_full = torch.cat([past_key_value[1], value_current], dim=2)
        else:
            value_full = value_current

        past_key_value = (key_states, value_full) if use_cache else None

        key_states, value_full_exp, value_query_exp = self._expand_kv(key_states, value_full, value_current)

        attn_weights = self._score_qk(query_states, key_states, q_pos, k_pos)

        if is_causal:
            attn_weights = attn_weights.masked_fill(~_causal_key_mask(q_pos, k_pos), float("-inf"))

        additive_mask = _prepare_attention_mask(attention_mask, bsz, q_len, key_states.shape[2], self.num_heads, attn_weights)
        if additive_mask is not None:
            attn_weights = attn_weights + additive_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = _sanitize_attention_weights(attn_weights)
        attn_weights = self.attention_dropout(attn_weights)

        attn_output_heads = self._weights_to_heads(attn_weights, value_full_exp, value_query_exp)

        attn_output = self._project_heads(attn_output_heads, bsz, q_len)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class SwiGLU(nn.Module):
    r"""SwiGLU(config)

    SwiGLU gated feed-forward block.

    .. math::
        \mathrm{SwiGLU}(x) = W_{down}(\mathrm{SiLU}(W_{gate}x) \odot W_{up}x)

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides ``hidden_size`` and ``intermediate_size``.

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> mlp = SwiGLU(MemoryMachineConfig(hidden_size=32, intermediate_size=64))
        >>> mlp(torch.randn(2, 5, 32)).shape
        torch.Size([2, 5, 32])
    """

    def __init__(self, config: MemoryMachineConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x) -> Tensor

        Apply the gated feed-forward transformation.

        Args:
            x (Tensor): input of shape ``(B, L, hidden)``.

        Returns:
            Tensor: output with the same shape ``(B, L, hidden)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> SwiGLU(MemoryMachineConfig(hidden_size=16, intermediate_size=32))(torch.randn(1, 4, 16)).shape
            torch.Size([1, 4, 16])
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        swish_gate = F.silu(gate)
        return self.down_proj(swish_gate * up)


class SlidingWindowAttention(GroupedQueryAttention):
    r"""SlidingWindowAttention(config, window_size=None)

    Sliding-window attention with a transient fixed-width KV cache.

    Inherits QKV projections, CVPE bias, and FRVM gating from
    :class:`GroupedQueryAttention`; the forward pass keeps only the last
    ``window_size`` keys/values in the returned cache. The :meth:`attend` core
    also supports ``windowed=False`` pure-causal attention over explicit
    query/key states (used by :class:`~memory_machines.models.blocks.liv.LivPerceiver`).

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            provides the GQA hyperparameters plus ``swa_window_size``.
        window_size (int, optional): window width. Default: ``None`` (falls
            back to ``config.swa_window_size``).

    Examples::

        >>> from memory_machines.utils.config_schema import MemoryMachineConfig
        >>> swa = SlidingWindowAttention(MemoryMachineConfig(hidden_size=32,
        ...     num_attention_heads=4, num_key_value_heads=2, swa_window_size=8))
        >>> out, _, present = swa(torch.randn(1, 6, 32), use_cache=True)
        >>> out.shape, present[0].shape
        (torch.Size([1, 6, 32]), torch.Size([1, 2, 6, 8]))
    """

    def __init__(self, config: MemoryMachineConfig, window_size: int | None = None):
        super().__init__(config)
        self.window_size = window_size if window_size is not None else config.swa_window_size

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
            value_query (Tensor): per-query reference values ``(B, Hkv, Lq, Dh)``.
            q_pos (Tensor): query positions ``(B, Lq)``.
            k_pos_full (Tensor): key positions ``(B, Lk)``.
            attention_mask (Tensor, optional): additive/bool mask. Default: ``None``.
            windowed (bool, optional): whether to apply the window bound.
                ``windowed=False, is_causal=True`` is pure causal. Default: ``True``.
            is_causal (bool, optional): whether to mask futures. Default: ``True``.

        Returns:
            Tensor: per-head outputs ``(B, H, Lq, Dh)`` and weights
            ``(B, H, Lq, Lk)``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> swa = SlidingWindowAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, swa_window_size=4))
            >>> out, w = swa.attend(torch.randn(1, 2, 3, 8), torch.randn(1, 1, 5, 8),
            ...     torch.randn(1, 1, 5, 8), torch.randn(1, 1, 3, 8),
            ...     torch.tensor([[3, 4, 5]]), torch.tensor([[1, 2, 3, 4, 5]]))
            >>> out.shape, w.shape
            (torch.Size([1, 2, 3, 8]), torch.Size([1, 2, 3, 5]))
        """
        key_exp, val_exp, vq_exp = self._expand_kv(key_states, value_full, value_query)
        attn = self._score_qk(query_states, key_exp, q_pos, k_pos_full)
        if windowed and is_causal:
            win = _window_mask(q_pos, k_pos_full, self.window_size).unsqueeze(1)
            attn = attn.masked_fill(~win, float("-inf"))
        elif windowed:
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

    def forward(  # type: ignore[override]
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
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]:
        r"""forward(hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, past_len_override=None, past_positions=None, is_causal=True) -> (Tensor, Tensor | None, tuple | None)

        Run sliding-window attention; ``is_causal=False`` forbids KV caching.

        The transient cache keeps only the last ``window_size`` entries so
        memory stays bounded; ``past_len_override``/``past_positions`` let
        callers correct positions when the cache was truncated.

        Args:
            hidden_states (Tensor): input of shape ``(B, Lq, hidden)``.
            attention_mask (Tensor, optional): bool/additive mask. Default: ``None``.
            position_ids (Tensor, optional): absolute positions. Default: ``None``.
            past_key_value (tuple of Tensor, optional): cached windowed
                ``(keys, values)``. Default: ``None``.
            output_attentions (bool, optional): whether to return weights.
                Default: ``False``.
            use_cache (bool, optional): whether to return the truncated window
                cache. Default: ``False``.
            past_len_override (int, optional): explicit past length overriding
                the cached width. Default: ``None``.
            past_positions (Tensor, optional): explicit cached positions of
                shape ``(B, Wprev)``. Default: ``None``.
            is_causal (bool, optional): whether attention is causal. Default: ``True``.

        Returns:
            Tensor: outputs ``(B, Lq, hidden)``, weights or ``None``, and the
            truncated ``(keys, values)`` window cache or ``None``.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> swa = SlidingWindowAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, swa_window_size=4))
            >>> out, _, present = swa(torch.randn(1, 6, 16), use_cache=True)
            >>> out.shape, present[0].shape
            (torch.Size([1, 6, 16]), torch.Size([1, 1, 4, 8]))
        """
        if not is_causal and (use_cache or past_key_value is not None):
            raise ValueError("SlidingWindowAttention with is_causal=False cannot use KV caching (truncated cache is only valid for causal attention)")
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_current = self._project_qkv(hidden_states)

        cached_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
        if past_len_override is not None:
            true_past_len = past_len_override
        elif past_positions is not None:
            true_past_len = int(past_positions.max().item()) + 1
        else:
            true_past_len = cached_len
        device = hidden_states.device
        q_pos, _ = _resolve_positions(hidden_states, q_len, true_past_len, position_ids)
        if past_key_value is None:
            k_pos_full = q_pos
        elif past_positions is not None:
            k_pos_full = torch.cat([past_positions.to(device), q_pos], dim=1)
        else:
            k_pos_full = torch.arange(true_past_len + q_len, device=device).unsqueeze(0).expand(bsz, -1)
        _check_position_bounds(q_pos, k_pos_full, self.max_position_embeddings, position_ids)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_full = torch.cat([past_key_value[1], value_current], dim=2)
        else:
            value_full = value_current
        out_h, attn_w = self.attend(query_states, key_states, value_full, value_current, q_pos, k_pos_full, attention_mask, is_causal=is_causal)
        full_k, full_v = key_states, value_full
        out = self._project_heads(out_h, bsz, q_len)
        present = None
        if use_cache:
            present = (
                full_k[:, :, -self.window_size :, :],
                full_v[:, :, -self.window_size :, :],
            )
        return out, (attn_w if output_attentions else None), present

    def truncate_for_replay(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""truncate_for_replay(key_states, value_states, positions) -> (Tensor, Tensor, Tensor)

        Truncate a KV window to the last ``window_size`` entries for replay.

        Args:
            key_states (Tensor): keys of shape ``(B, Hkv, Lk, Dh)``.
            value_states (Tensor): values with the same shape as ``key_states``.
            positions (Tensor): positions of shape ``(B, Lk)`` or ``(Lk,)``.

        Returns:
            Tensor: truncated keys, values, and positions.

        Examples::

            >>> import torch
            >>> from memory_machines.utils.config_schema import MemoryMachineConfig
            >>> swa = SlidingWindowAttention(MemoryMachineConfig(hidden_size=16,
            ...     num_attention_heads=2, num_key_value_heads=1, swa_window_size=4))
            >>> k, v, p = swa.truncate_for_replay(torch.randn(1, 1, 6, 8),
            ...     torch.randn(1, 1, 6, 8), torch.arange(6))
            >>> k.shape, p.shape
            (torch.Size([1, 1, 4, 8]), torch.Size([4]))
        """
        return key_states[:, :, -self.window_size :, :], value_states[:, :, -self.window_size :, :], positions[-self.window_size :]
