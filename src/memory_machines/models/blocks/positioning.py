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
    r"""CVPEContextBias(head_dim, mlp_hidden=32, use_log_transform=True)

    Content-versus-position-dependent (CVPE) bias added to attention logits.

    For query head ``q`` and relative offset :math:`r = q - k \ge 0`, the bias is

    .. math::
        b_{q,k} = \gamma(q)\, f(\tilde{r}) + \beta(q), \quad
        \tilde{r} = \frac{r}{q}

    where :math:`\gamma, \beta` are per-query linear projections and
    :math:`f` is a small MLP over the normalized offset, optionally warped by
    :math:`\log_{10}(1 + 9\tilde{r})`.

    .. note::
        This module only adds the bias; the caller must still mask future
        positions.

    Args:
        head_dim (int): per-head dimension ``Dh``.
        mlp_hidden (int, optional): hidden width of the offset MLP. Default: ``32``.
        use_log_transform (bool, optional): whether to log-warp the normalized
            offset before the MLP. Default: ``True``.

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
    ):
        super().__init__()
        self.head_dim = head_dim
        self.use_log_transform = use_log_transform

        self.gamma_proj = nn.Linear(head_dim, 1)
        self.beta_proj = nn.Linear(head_dim, 1)

        self.mlp_in = nn.Linear(1, mlp_hidden)
        self.mlp_act = nn.ReLU()
        self.offset_out = nn.Linear(mlp_hidden, 1)

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
            q_pos (Tensor): query absolute positions of shape ``(B, Lq)`` or
                broadcastable ``(Lq,)``.
            k_pos (Tensor): key absolute positions of shape ``(B, Lk)`` or
                broadcastable ``(Lk,)``.

        Returns:
            Tensor: bias of shape ``(B, H, Lq, Lk)`` in the query dtype.

        Examples::

            >>> import torch
            >>> mod = CVPEContextBias(head_dim=4)
            >>> mod(torch.randn(1, 2, 3, 4), torch.tensor([[2, 3, 4]]), torch.tensor([[0, 1, 2, 3, 4]])).shape
            torch.Size([1, 2, 3, 5])
        """
        bsz, _, lq, _ = query_states.shape

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

        f_f32 = self.offset_out(self.mlp_act(self.mlp_in(tilde_f32.unsqueeze(-1)))).squeeze(-1)
        f = f_f32.unsqueeze(1).to(query_states.dtype)

        gamma = self.gamma_proj(query_states).squeeze(-1).unsqueeze(-1)
        beta = self.beta_proj(query_states).squeeze(-1).unsqueeze(-1)

        return gamma * f + beta


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

    def __init__(self, head_dim: int, num_heads: int):
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads ({num_heads}) must be positive")
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.phi_weight = nn.Parameter(torch.empty(num_heads, head_dim, head_dim))
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

        outs: list[torch.Tensor] = []

        for start in range(0, lq, effective_chunk_size):
            end = min(start + effective_chunk_size, lq)
            w = attn_weights[:, :, start:end, :]
            pvq = phi_vq[:, :, start:end, :].unsqueeze(3)
            pvk = phi_vk.unsqueeze(2)
            vj = v_full.unsqueeze(2)

            delta = pvq - pvk
            gated = vj * F.silu(vj + delta)
            outs.append((w.unsqueeze(-1) * gated).sum(dim=3))

        return torch.cat(outs, dim=2)
