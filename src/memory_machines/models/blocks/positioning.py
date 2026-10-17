r"""CVPE QK-circuit bias and FRVM OV-circuit gating.

CVPE (:class:`CVPEContextBias`) adds a content-dependent, position-dependent
bias to attention logits. FRVM (:class:`FRVMValueGate`) replaces plain
weighted averaging of values with an antisymmetric gated reduction.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class CVPEContextBias(nn.Module):
    r"""CVPEContextBias(head_dim, mlp_hidden=32, use_log_transform=True, spatial_dims=1, axis_max=None)

    Content-versus-position-dependent (CVPE) bias added to attention logits.

    1D legacy path (``q_pos``/``k_pos`` are ``(B, L)``): for query head ``q``
    and relative offset :math:`r = q - k \ge 0`, the bias is

    .. math::
        b_{q,k} = \gamma(q)\, f(\tilde{r}) + \beta(q), \quad
        \tilde{r} = \frac{r}{q}

    N-dim path (``q_pos``/``k_pos`` are ``(B, L, N)``): the bounded distance
    grid ``[0, 1]^N`` from ``_multi_axis_distance`` feeds an N-wide MLP
    (``R^N -> R^1``) with fixed width ``spatial_dims`` (zero-padded unused
    axes), so one weight set serves every modality (text=(t,0,0), video=(t,h,w)).

    .. note::
        This module only adds the bias; the caller must still mask future
        positions.

    Args:
        head_dim (int): per-head dimension ``Dh``.
        mlp_hidden (int, optional): hidden width of the offset MLP. Default: ``32``.
        use_log_transform (bool, optional): whether to log-warp the normalized
            offset before the MLP. Default: ``True``.
        spatial_dims (int, optional): fixed N-dim input width ``N_max``.
            Default: ``1`` (pure 1D, backward compatible).
        axis_max (tuple, optional): per-axis normalization caps; empty means
            ``max_position_embeddings``-style default ``32768`` per axis.
            Default: ``None``.

    Examples::

        >>> import torch
        >>> mod = CVPEContextBias(head_dim=8)
        >>> q = torch.randn(2, 4, 5, 8)
        >>> q_pos = torch.tensor([[4, 5, 6, 7, 8], [4, 5, 6, 7, 8]])
        >>> k_pos = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8]])
        >>> mod(q, q_pos, k_pos).shape
        torch.Size([2, 4, 5, 9])
    """

    def __init__(
        self,
        head_dim: int,
        mlp_hidden: int = 32,
        use_log_transform: bool = True,
        spatial_dims: int = 1,
        axis_max: tuple[int, ...] | list[int] | torch.Tensor | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.use_log_transform = use_log_transform
        if spatial_dims <= 0:
            raise ValueError(f"spatial_dims ({spatial_dims}) must be positive")
        self.spatial_dims = spatial_dims
        if axis_max is None:
            self.axis_max: tuple[int, ...] | list[int] | torch.Tensor = (32768,) * spatial_dims
        else:
            if len(tuple(axis_max)) != spatial_dims:
                raise ValueError(f"axis_max must hold {spatial_dims} entries (one per axis), got {tuple(axis_max)}")
            self.axis_max = axis_max

        self.gamma_proj = nn.Linear(head_dim, 1)
        self.beta_proj = nn.Linear(head_dim, 1)

        self.mlp_in = nn.Linear(1, mlp_hidden)
        self.mlp_act = nn.ReLU()
        self.offset_out = nn.Linear(mlp_hidden, 1)
        # N-dim path: fixed-width MLP R^N -> R^1 (zero-padded unused axes).
        self.mlp_in_nd = nn.Linear(spatial_dims, mlp_hidden)
        self.offset_out_nd = nn.Linear(mlp_hidden, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        All-zero init so the bias starts at exactly zero and attention
        begins as the vanilla scaled dot-product; the content/position
        modulation is learned from a disabled start (cf. zero-initialized
        conditioning branches: ControlNet, Zhang et al., 2023; ReZero).
        The offset MLP input layer uses a truncated normal
        ``N(0, (1/sqrt(fan_in))^2)`` (ReLU-compatible scale) so re-init
        is idempotent; its output is multiplied by the zeroed
        ``offset_out``/``gamma`` at step 0 regardless.
        """
        fan_in = self.mlp_in.in_features
        mlp_std = 1 / math.sqrt(fan_in)
        nn.init.trunc_normal_(self.mlp_in.weight, mean=0.0, std=mlp_std, a=-2 * mlp_std, b=2 * mlp_std)
        nn.init.zeros_(self.mlp_in.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        nn.init.zeros_(self.offset_out.weight)
        nn.init.zeros_(self.offset_out.bias)
        # N-dim branch zero-starts identically (neutral attention at step 0).
        fan_nd = self.mlp_in_nd.in_features
        mlp_std_nd = 1 / math.sqrt(fan_nd)
        nn.init.trunc_normal_(self.mlp_in_nd.weight, mean=0.0, std=mlp_std_nd, a=-2 * mlp_std_nd, b=2 * mlp_std_nd)
        nn.init.zeros_(self.mlp_in_nd.bias)
        nn.init.zeros_(self.offset_out_nd.weight)
        nn.init.zeros_(self.offset_out_nd.bias)

    def forward(
        self,
        query_states: torch.Tensor,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
    ) -> torch.Tensor:
        r"""forward(query_states, q_pos, k_pos) -> Tensor

        Compute the CVPE bias for every query/key pair.

        Args:
            query_states (Tensor): queries of shape ``(B, H, Lq, Dh)``.
            q_pos (Tensor): query absolute positions — legacy ``(B, Lq)`` /
                ``(Lq,)`` 1D, or N-dim ``(B, Lq, N)`` coordinates.
            k_pos (Tensor): key absolute positions — legacy ``(B, Lk)`` /
                ``(Lk,)`` 1D, or N-dim ``(B, Lk, N)`` coordinates.

        Returns:
            Tensor: bias of shape ``(B, H, Lq, Lk)`` in the query dtype.

        Examples::

            >>> import torch
            >>> mod = CVPEContextBias(head_dim=4)
            >>> mod(torch.randn(1, 2, 3, 4), torch.tensor([[2, 3, 4]]), torch.tensor([[0, 1, 2, 3, 4]])).shape
            torch.Size([1, 2, 3, 5])
        """
        bsz, _, lq, _ = query_states.shape

        if q_pos.dim() == 3 or k_pos.dim() == 3:
            # N-dim path: separate query/key coordinate streams (self-attn
            # collapses to symmetric distances; cross-attn is asymmetric LxS).
            from .shared import _multi_axis_distance

            q3 = q_pos.to(dtype=torch.float32)
            k3 = k_pos.to(dtype=torch.float32)
            if q3.dim() == 2:
                q3 = q3.unsqueeze(-1)
            if k3.dim() == 2:
                k3 = k3.unsqueeze(-1)
            if q3.dim() != 3 or q3.shape[0] != bsz or q3.shape[1] != lq:
                raise ValueError(f"q_pos shape {tuple(q_pos.shape)} does not match (batch, Lq, N) = ({bsz}, {lq}, N)")
            if k3.dim() != 3 or k3.shape[0] != bsz:
                raise ValueError(f"k_pos shape {tuple(k_pos.shape)} does not match (batch, Lk, N) with batch ({bsz})")
            nq, nk = q3.shape[-1], k3.shape[-1]
            if nq != nk:
                raise ValueError(f"q_pos dims ({nq}) != k_pos dims ({nk}); both streams must share N coordinate axes")
            n = nq
            if n < self.spatial_dims:
                q3 = F.pad(q3, (0, self.spatial_dims - n))
                k3 = F.pad(k3, (0, self.spatial_dims - n))
            elif n > self.spatial_dims:
                q3 = q3[..., : self.spatial_dims]
                k3 = k3[..., : self.spatial_dims]
            dist = _multi_axis_distance(
                q3, k3,
                self.axis_max if len(tuple(self.axis_max)) == self.spatial_dims else (32768,) * self.spatial_dims,
                self.use_log_transform,
            )  # (B, Lq, Lk, N)
            mlp_dtype = self.mlp_in_nd.weight.dtype
            f = self.offset_out_nd(self.mlp_act(self.mlp_in_nd(dist.to(mlp_dtype)))).squeeze(-1)
            f = f.float().unsqueeze(1).to(query_states.dtype)
            gamma = self.gamma_proj(query_states).squeeze(-1).unsqueeze(-1)
            beta = self.beta_proj(query_states).squeeze(-1).unsqueeze(-1)
            return gamma * f + beta

        q_pos_2d = q_pos.unsqueeze(0).expand(bsz, -1) if q_pos.dim() == 1 else q_pos
        k_pos_2d = k_pos.unsqueeze(0).expand(bsz, -1) if k_pos.dim() == 1 else k_pos
        if q_pos_2d.shape != (bsz, lq):
            raise ValueError(f"q_pos shape {tuple(q_pos_2d.shape)} does not match (batch, Lq) = ({bsz}, {lq})")
        if k_pos_2d.dim() != 2 or k_pos_2d.shape[0] != bsz:
            raise ValueError(f"k_pos shape {tuple(k_pos_2d.shape)} does not match (batch, Lk) with batch ({bsz})")

        q_pos_f32 = q_pos_2d.float()
        k_pos_f32 = k_pos_2d.float()

        denom_f32 = q_pos_f32.clamp(min=1.0).unsqueeze(-1)
        rel_f32 = (q_pos_f32.unsqueeze(-1) - k_pos_f32.unsqueeze(1)).clamp(min=0.0)
        tilde_f32 = rel_f32 / denom_f32

        if self.use_log_transform:
            tilde_f32 = torch.log1p(tilde_f32 * 9.0) / math.log(10.0)

        # Run the offset MLP in its own weight dtype: after model.to(bf16) the
        # weights are bf16 while tilde is fp32 -> runtime dtype error otherwise.
        mlp_dtype = self.mlp_in.weight.dtype
        f_f32 = self.offset_out(self.mlp_act(self.mlp_in(tilde_f32.unsqueeze(-1).to(mlp_dtype)))).squeeze(-1)
        f = f_f32.float().unsqueeze(1).to(query_states.dtype)

        gamma = self.gamma_proj(query_states).squeeze(-1).unsqueeze(-1)
        beta = self.beta_proj(query_states).squeeze(-1).unsqueeze(-1)

        return gamma * f + beta

    def precompute_bias(
        self,
        query_states: torch.Tensor,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Precompute the ``(B, H, Lq, Lk)`` CVPE bias for the functional op.

        Thin alias of :meth:`forward` — the learnable QK-circuit part stays
        in the module; the functional only adds the returned tensor.
        """
        return self.forward(query_states, q_pos, k_pos)


class FRVMSpatialOV(nn.Module):
    """N-dim spatial framing for the OV circuit: ``MLP_OV(R^N -> R^Dh)``.

    Zero-initialized so the spatial term starts disabled (vanilla F-RVM).
    The raw ``weight`` is exposed for the functional op, which evaluates
    ``F.linear(spatial_grid, weight)`` inside memory-bounded query chunks.
    """

    def __init__(self, spatial_dims: int, head_dim: int):
        super().__init__()
        if spatial_dims <= 0:
            raise ValueError(f"spatial_dims ({spatial_dims}) must be positive")
        self.proj = nn.Linear(spatial_dims, head_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    @property
    def weight(self) -> torch.Tensor:
        return self.proj.weight

    def forward(self, spatial_grid: torch.Tensor) -> torch.Tensor:
        return self.proj(spatial_grid)


class FRVMValueGate(nn.Module):
    r"""FRVMValueGate(head_dim, num_heads)

    Functional relative-value modulation (FRVM): antisymmetric value gating.

    Instead of :math:`\sum_k w_{qk} v_k`, each query mixes values through a
    per-head map :math:`\phi_h`:

    .. math::
        o_q = \sum_k w_{qk}\, \left[v_k \odot \sigma(v_k + \phi_h(v_q - v_k))\right]

    with :math:`\sigma` the SiLU activation.

    .. warning::
        The per-head ``(Dh, Dh)`` maps are not load-compatible with a shared
        ``phi`` checkpoint.

    Args:
        head_dim (int): per-head dimension ``Dh``.
        num_heads (int): number of attention heads ``H``; must be positive.

    Attributes:
        phi_weight (torch.nn.Parameter): per-head maps of shape ``(H, Dh, Dh)``.

    Examples::

        >>> import torch
        >>> gate = FRVMValueGate(head_dim=8, num_heads=2)
        >>> w = torch.softmax(torch.randn(1, 2, 3, 5), dim=-1)
        >>> v_full = torch.randn(1, 2, 5, 8)
        >>> v_q = torch.randn(1, 2, 3, 8)
        >>> gate.gated_reduction(w, v_full, v_q).shape
        torch.Size([1, 2, 3, 8])
    """

    _MAX_CHUNK_ELEMENTS = 32 * 1024 * 1024

    def __init__(self, head_dim: int, num_heads: int, cross_heads: bool = False):
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads ({num_heads}) must be positive")
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.cross_heads = cross_heads
        self.phi_weight = nn.Parameter(torch.empty(num_heads, head_dim, head_dim))
        # Cross-attention path: separate modality-specific heads Phi_q/Phi_k.
        # Self-attention shares one phi (backward compatible).
        self.phi_q_weight: nn.Parameter | None = None
        self.phi_k_weight: nn.Parameter | None = None
        if cross_heads:
            self.phi_q_weight = nn.Parameter(torch.empty(num_heads, head_dim, head_dim))
            self.phi_k_weight = nn.Parameter(torch.empty(num_heads, head_dim, head_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Zero-initialized per-head maps, so gating starts as pure
        self-gating ``v * SiLU(v)`` with no pairwise term; the
        query-relative modulation ``phi(v_q - v_k)`` grows from a disabled
        start instead of injecting large random pairwise terms at step 0
        (cf. ReZero/gated-attention identity-start practice). This
        replaces the previous Kaiming init, which gave the pairwise term
        full random magnitude before any value geometry was learned.
        """
        nn.init.zeros_(self.phi_weight)
        if self.phi_q_weight is not None:
            nn.init.zeros_(self.phi_q_weight)
        if self.phi_k_weight is not None:
            nn.init.zeros_(self.phi_k_weight)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        r"""_phi(x) -> Tensor

        Apply the per-head linear map.

        Args:
            x (Tensor): input of shape ``(B, H, L, Dh)``.

        Returns:
            Tensor: mapped output of the same shape ``(B, H, L, Dh)``.

        Examples::

            >>> import torch
            >>> gate = FRVMValueGate(head_dim=4, num_heads=2)
            >>> gate._phi(torch.randn(1, 2, 3, 4)).shape
            torch.Size([1, 2, 3, 4])
        """
        if x.dim() != 4 or x.shape[1] != self.num_heads or x.shape[-1] != self.head_dim:
            raise ValueError(f"expected (B, H={self.num_heads}, L, Dh={self.head_dim}), got {tuple(x.shape)}")
        return torch.einsum("bhld,hed->bhle", x, self.phi_weight)

    def pairwise_delta(self, v_i: torch.Tensor, v_j: torch.Tensor) -> torch.Tensor:
        r"""pairwise_delta(v_i, v_j) -> Tensor

        Map a pairwise value difference through :math:`\phi`.

        Args:
            v_i (Tensor): first values of shape ``(B, H, L, Dh)``.
            v_j (Tensor): second values broadcast-compatible with ``v_i``.

        Returns:
            Tensor: :math:`\phi_h(v_i - v_j)` with the broadcast shape.

        Examples::

            >>> import torch
            >>> gate = FRVMValueGate(head_dim=4, num_heads=2)
            >>> gate.pairwise_delta(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)).shape
            torch.Size([1, 2, 3, 4])
        """
        return self._phi(v_i - v_j)

    def _phi_q(self, x: torch.Tensor) -> torch.Tensor:
        """Query-stream projection (cross-attn Phi_q, else shared phi)."""
        if self.phi_q_weight is not None:
            if x.dim() != 4 or x.shape[1] != self.num_heads or x.shape[-1] != self.head_dim:
                raise ValueError(f"expected (B, H={self.num_heads}, L, Dh={self.head_dim}), got {tuple(x.shape)}")
            return torch.einsum("bhld,hed->bhle", x, self.phi_q_weight)
        return self._phi(x)

    def _phi_k(self, x: torch.Tensor) -> torch.Tensor:
        """Key-stream projection (cross-attn Phi_k, else shared phi)."""
        if self.phi_k_weight is not None:
            if x.dim() != 4 or x.shape[1] != self.num_heads or x.shape[-1] != self.head_dim:
                raise ValueError(f"expected (B, H={self.num_heads}, L, Dh={self.head_dim}), got {tuple(x.shape)}")
            return torch.einsum("bhld,hed->bhle", x, self.phi_k_weight)
        return self._phi(x)

    def project_qk(
        self, v_query: torch.Tensor, v_full: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute ``(phi_vq, phi_vk)`` for the functional op."""
        return self._phi_q(v_query), self._phi_k(v_full)

    def gated_reduction(
        self,
        attn_weights: torch.Tensor,
        v_full: torch.Tensor,
        v_query: torch.Tensor,
        chunk_size: int = 128,
    ) -> torch.Tensor:
        r"""gated_reduction(attn_weights, v_full, v_query, chunk_size=128) -> Tensor

        Chunked FRVM-gated weighted reduction over values.

        Queries are chunked along ``Lq`` so the ``(Lq, Lk, Dh)`` intermediate
        stays bounded; the effective chunk size is reduced automatically when
        ``B * H * Lk * Dh`` is large.

        Args:
            attn_weights (Tensor): attention weights of shape ``(B, H, Lq, Lk)``.
            v_full (Tensor): full value store of shape ``(B, H, Lk, Dh)``.
            v_query (Tensor): per-query reference values of shape
                ``(B, H, Lq, Dh)``.
            chunk_size (int, optional): maximum queries per chunk. Default: ``128``.

        Returns:
            Tensor: gated outputs of shape ``(B, H, Lq, Dh)``.

        Examples::

            >>> import torch
            >>> gate = FRVMValueGate(head_dim=4, num_heads=2)
            >>> w = torch.softmax(torch.randn(2, 2, 4, 6), dim=-1)
            >>> gate.gated_reduction(w, torch.randn(2, 2, 6, 4), torch.randn(2, 2, 4, 4)).shape
            torch.Size([2, 2, 4, 4])
        """
        bsz, num_heads, lq, lk = attn_weights.shape
        if attn_weights.shape[-1] != v_full.shape[-2]:
            raise ValueError(f"attn_weights Lk ({attn_weights.shape[-1]}) does not match v_full Lk ({v_full.shape[-2]})")
        if attn_weights.shape[-2] != v_query.shape[-2]:
            raise ValueError(f"attn_weights Lq ({attn_weights.shape[-2]}) does not match v_query Lq ({v_query.shape[-2]})")
        if v_full.shape[-1] != v_query.shape[-1]:
            raise ValueError(f"v_full head_dim ({v_full.shape[-1]}) does not match v_query head_dim ({v_query.shape[-1]})")
        if attn_weights.shape[:2] != v_full.shape[:2] or attn_weights.shape[:2] != v_query.shape[:2]:
            raise ValueError(f"batch/heads mismatch: attn {tuple(attn_weights.shape[:2])} vs v_full {tuple(v_full.shape[:2])} vs v_query {tuple(v_query.shape[:2])}")

        phi_vq = self._phi(v_query)
        phi_vk = self._phi(v_full)

        per_query_elements = bsz * num_heads * lk * v_full.shape[-1]
        effective_chunk_size = min(chunk_size, max(1, self._MAX_CHUNK_ELEMENTS // max(1, per_query_elements)))

        from memory_machines.ops.references.attention_reference import reference_frvm_gated_reduction

        if self.phi_q_weight is not None or self.phi_k_weight is not None:
            phi_vq = self._phi_q(v_query)
            phi_vk = self._phi_k(v_full)
        return reference_frvm_gated_reduction(
            attn_weights, v_full, phi_vq, phi_vk, chunk_size=effective_chunk_size
        )
