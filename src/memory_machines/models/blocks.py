"""Flat composable blocks for MemoryMachine architectures.

Includes dual-circuit relative positioning:
- QK circuit: CVPE dynamic context bias (Shin & Zhang 2025) with FIRE-style progressive normalization (Li et al. 2024).
- OV circuit: F-RVM antisymmetric value gating (Höftmann et al. ICLR 2026).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.config_schema import MemoryMachineConfig


class CVPEContextBias(nn.Module):
    """QK-circuit dynamic context bias B_{i,j}.

    B_{i,j} = gamma(q_i) * MLP_QK(tilde{Delta}_{i,j}) + beta(q_i),
    Attention_Logit = q_i^T k_j / sqrt(d) + B_{i,j}.

    tilde{Delta}_{i,j} = clamp(i - j, min=0) / max(i, 1) in [0, 1]
    (progressive normalization, FIRE-style: bounded for any length).

    Note: Future positions (j > i) clamp to 0. Callers must still provide a causal
    attention mask if future tokens should be masked out.
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

        self.offset_mlp = nn.Sequential(
            self.mlp_in,
            self.mlp_act,
            self.offset_out,
        )

        # Zero-init so the model starts as vanilla dot-product attention.
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
        """Compute bias of shape (B, H, Lq, Lk).

        Args:
            query_states: Tensor of shape (B, H, Lq, D)
            q_pos: Tensor of shape (Lq,) or (B, Lq)
            k_pos: Tensor of shape (Lk,) or (B, Lk)

        Returns:
            Tensor of shape (B, H, Lq, Lk)
        """
        bsz, _, lq, _ = query_states.shape

        q_pos_2d = q_pos.unsqueeze(0).expand(bsz, -1) if q_pos.dim() == 1 else q_pos
        k_pos_2d = k_pos.unsqueeze(0).expand(bsz, -1) if k_pos.dim() == 1 else k_pos
        if q_pos_2d.shape != (bsz, lq):
            raise ValueError(f"q_pos shape {tuple(q_pos_2d.shape)} does not match (batch, Lq) = ({bsz}, {lq})")
        if k_pos_2d.dim() != 2 or k_pos_2d.shape[0] != bsz:
            raise ValueError(f"k_pos shape {tuple(k_pos_2d.shape)} does not match (batch, Lk) with batch ({bsz})")

        # Compute position grid in float32 to prevent precision collapse at scale (bf16/fp16)
        q_pos_f32 = q_pos_2d.float()
        k_pos_f32 = k_pos_2d.float()

        denom_f32 = q_pos_f32.clamp(min=1.0).unsqueeze(-1)  # [B, Lq, 1]
        rel_f32 = (q_pos_f32.unsqueeze(-1) - k_pos_f32.unsqueeze(1)).clamp(min=0.0)  # [B, Lq, Lk]
        tilde_f32 = rel_f32 / denom_f32  # [B, Lq, Lk]

        if self.use_log_transform:
            # log1p warp remapped to [0, 1]: amplifies near-field differences.
            tilde_f32 = torch.log1p(tilde_f32 * 9.0) / math.log(10.0)

        f_f32 = self.offset_mlp(tilde_f32.unsqueeze(-1)).squeeze(-1)  # [B, Lq, Lk]
        f = f_f32.unsqueeze(1).to(query_states.dtype)  # [B, 1, Lq, Lk]

        gamma = self.gamma_proj(query_states).squeeze(-1).unsqueeze(-1)  # [B, H, Lq, 1]
        beta = self.beta_proj(query_states).squeeze(-1).unsqueeze(-1)  # [B, H, Lq, 1]

        return gamma * f + beta


class FRVMValueGate(nn.Module):
    """OV-circuit antisymmetric relative value modulation.

    Uses Phi linearity (Phi(a - b) = Phi(a) - Phi(b)) to project in 4D,
    then broadcasts the difference per query chunk. Per-chunk 5D volume
    is bounded by _MAX_CHUNK_ELEMENTS; a fused kernel would remove it.
    """

    # Hard HBM guard for the per-chunk broadcast.
    _MAX_CHUNK_ELEMENTS = 32 * 1024 * 1024

    def __init__(self, head_dim: int):
        super().__init__()
        self.phi = nn.Linear(head_dim, head_dim, bias=False)

    def pairwise_delta(self, v_i: torch.Tensor, v_j: torch.Tensor) -> torch.Tensor:
        """Delta(i,j) for broadcastable V_i [..., D], V_j [..., D]."""
        return self.phi(v_i - v_j)

    def gated_reduction(
        self,
        attn_weights: torch.Tensor,  # [B, H, Lq, Lk]
        v_full: torch.Tensor,  # [B, H, Lk, D]
        v_query: torch.Tensor,  # [B, H, Lq, D]
        chunk_size: int = 128,
    ) -> torch.Tensor:
        """Memory-efficient gated reduction with 4D pre-projection."""
        bsz, num_heads, lq, lk = attn_weights.shape
        if attn_weights.shape[-1] != v_full.shape[-2]:
            raise ValueError(f"attn_weights Lk ({attn_weights.shape[-1]}) does not match v_full Lk ({v_full.shape[-2]})")
        if attn_weights.shape[-2] != v_query.shape[-2]:
            raise ValueError(f"attn_weights Lq ({attn_weights.shape[-2]}) does not match v_query Lq ({v_query.shape[-2]})")
        if v_full.shape[-1] != v_query.shape[-1]:
            raise ValueError(f"v_full head_dim ({v_full.shape[-1]}) does not match v_query head_dim ({v_query.shape[-1]})")
        if attn_weights.shape[:2] != v_full.shape[:2] or attn_weights.shape[:2] != v_query.shape[:2]:
            raise ValueError(f"batch/heads mismatch: attn {tuple(attn_weights.shape[:2])} vs v_full {tuple(v_full.shape[:2])} vs v_query {tuple(v_query.shape[:2])}")

        # Pre-project queries and keys once.
        phi_vq = self.phi(v_query)  # [B, H, Lq, D]
        phi_vk = self.phi(v_full)  # [B, H, Lk, D]

        # Bound the per-chunk broadcast to the element budget.
        per_query_elements = bsz * num_heads * lk * v_full.shape[-1]
        effective_chunk_size = min(chunk_size, max(1, self._MAX_CHUNK_ELEMENTS // max(1, per_query_elements)))

        outs: list[torch.Tensor] = []

        for start in range(0, lq, effective_chunk_size):
            end = min(start + effective_chunk_size, lq)
            w = attn_weights[:, :, start:end, :]  # [B, H, C, Lk]
            pvq = phi_vq[:, :, start:end, :].unsqueeze(3)  # [B, H, C, 1, D]
            pvk = phi_vk.unsqueeze(2)  # [B, H, 1, Lk, D]
            vj = v_full.unsqueeze(2)  # [B, H, 1, Lk, D]

            delta = pvq - pvk  # [B, H, C, Lk, D]
            gated = vj * F.silu(vj + delta)
            outs.append((w.unsqueeze(-1) * gated).sum(dim=3))

        return torch.cat(outs, dim=2)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with dual-circuit relative positioning.

    QK circuit: CVPE dynamic context bias (no rotation, no extra MatMul).
    OV circuit: F-RVM antisymmetric value gating (1x KV-cache, dynamic).
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

        # Linear projections
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

        # Dual-circuit relative positioning (replaces RoPE).
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
        self.frvm_gate = FRVMValueGate(self.head_dim) if self.frvm_enabled else None

        # Attention dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        bsz, q_len, _ = hidden_states.size()

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_flat = self.v_proj(hidden_states)

        # Reshape for attention computation
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_current = value_flat.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
        total_len = past_len + q_len

        if total_len > self.max_position_embeddings:
            raise ValueError(f"Sequence length ({total_len}) exceeds configured max_position_embeddings ({self.max_position_embeddings})")

        device = hidden_states.device
        if position_ids is not None:
            if position_ids.dim() == 2:
                if position_ids.shape[0] != bsz:
                    raise ValueError(f"position_ids batch size ({position_ids.shape[0]}) does not match input batch size ({bsz})")
                q_pos = position_ids[:, -q_len:].to(device)
                past_pos = torch.arange(past_len, device=device, dtype=q_pos.dtype).unsqueeze(0).expand(bsz, -1)
                k_pos = torch.cat([past_pos, q_pos], dim=1) if past_len > 0 else q_pos
            else:
                # 1D ids describe current tokens only; past is assumed
                # contiguous 0..past_len-1. Pass 2D position_ids for
                # offset/packed decoding.
                q_pos = position_ids[-q_len:].to(device)
                past_pos = torch.arange(past_len, device=device, dtype=q_pos.dtype)
                k_pos = torch.cat([past_pos, q_pos]) if past_len > 0 else q_pos
        else:
            q_pos = torch.arange(past_len, past_len + q_len, device=device)
            k_pos = torch.arange(total_len, device=device)

        if int(q_pos.max().item()) >= self.max_position_embeddings or int(k_pos.max().item()) >= self.max_position_embeddings:
            raise ValueError(f"Absolute position id exceeds max_position_embeddings ({self.max_position_embeddings})")

        # 1x KV-cache: store raw K/V (no rotation, no expansion).
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_full = torch.cat([past_key_value[1], value_current], dim=2)
        else:
            value_full = value_current

        past_key_value = (key_states, value_full) if use_cache else None

        # Repeat key and value states for grouped query attention
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_full_exp = value_full.repeat_interleave(self.num_key_value_groups, dim=1)
        value_query_exp = value_current.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # QK circuit: CVPE dynamic context bias (pre-softmax logit injection).
        if self.cvpe_bias is not None:
            attn_weights = attn_weights + self.cvpe_bias(query_states, q_pos, k_pos).to(attn_weights.dtype)

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)

        # OV circuit: F-RVM antisymmetric gated reduction (dynamic, cache stays 1x).
        gate = self.frvm_gate
        if gate is not None:
            attn_output_heads = gate.gated_reduction(
                attn_weights,
                value_full_exp,
                value_query_exp,
                chunk_size=self.frvm_chunk_size,
            )
        else:
            attn_output_heads = torch.matmul(attn_weights, value_full_exp)

        # Reshape and project output
        attn_output = attn_output_heads.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class SwiGLU(nn.Module):
    """SwiGLU activation function (bias-free).

    SwiGLU(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    where SiLU(x) = x * sigmoid(x)

    Args:
        config: memory_machines configuration
    """

    def __init__(self, config: MemoryMachineConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of SwiGLU.

        Args:
            x: Input tensor

        Returns:
            Output tensor after SwiGLU activation
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        swish_gate = F.silu(gate)  # SiLU is equivalent to Swish
        return self.down_proj(swish_gate * up)


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + attn_out)


class LivConvBlock(nn.Module):
    """double-gated short-range convolution block from https://arxiv.org/abs/2511.23404

    This implements the LIV (Linear Input-Varying) convolution as described:
    ```
    def liv_conv(x):
        pre_gate, post_gate, x = linear(x)    # input projection
        x = pre_gate*x                        # gating (gate depends on input)
        x = conv(x)                           # short conv
        x = post_gate*x                       # gating
        x = linear(x)
        return x
    ```
    """

    def __init__(self, config: MemoryMachineConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # Input projection to gates and values
        self.input_projection = nn.Linear(
            self.hidden_size,
            3 * self.hidden_size,  # pre_gate, post_gate, x
            bias=False,
        )

        # Short convolution
        self.conv = nn.Conv1d(
            self.hidden_size,
            self.hidden_size,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
            groups=self.hidden_size,  # Depthwise convolution for efficiency
            bias=False,
        )

        # Output projection
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the LiV convolution block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.shape

        # Input projection: pre_gate, post_gate, x = linear(x)
        projected = self.input_projection(x)  # (pre_gate, L, 3*H)
        pre_gate, post_gate, x_proj = projected.chunk(3, dim=-1)  # Each: (pre_gate, L, H)

        # First gating: x = pre_gate*x
        x_gated = pre_gate * x_proj

        # Apply short convolution
        # Convert to (pre_gate, H, L) for conv1d
        x_conv_input = x_gated.transpose(1, 2)  # (pre_gate, H, L)
        x_conv = self.conv(x_conv_input)  # (pre_gate, H, L)
        x_conv = x_conv.transpose(1, 2)  # (pre_gate, L, H)

        # Second gating: x = post_gate*x
        x_gated_2 = post_gate * x_conv

        # Apply dropout
        x_gated_2 = self.dropout(x_gated_2)

        # Output projection
        output = self.output_projection(x_gated_2)

        return output
