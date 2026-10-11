"""Flat composable blocks. No inheritance hierarchies — assemble with ModuleList.

These two are the minimal vocabulary; delete or extend them for your
domain (conv blocks, mixer blocks, attention variants). The rule is
composition: a top-level nn.Module owns a ModuleList/ModuleDict of
cohesive blocks, nothing more.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.config_schema import MemoryMachineConfig


# temporary
class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE).

    Args:
        dim: Dimension of the embedding
        max_position_embeddings: Maximum sequence length
        base: Base frequency for RoPE
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute frequency inverse
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Generate position indices
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention implementation.

    Args:
        config: MemoryMachine configuration
    """

    def __init__(self, config: MemoryMachineConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})")

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

        # Rotary positional embedding
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

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
        """Forward pass of grouped query attention.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_value: Optional cached key-value pairs
            output_attentions: Whether to output attention weights
            use_cache: Whether to use key-value caching

        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        bsz, q_len, _ = hidden_states.size()

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention computation
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Get sequence length for RoPE
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # Apply rotary positional embedding
        cos, sin = self.rotary_emb(value_states, kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle past key-value cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat key and value states for grouped query attention
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class SwiGLU(nn.Module):
    """SwiGLU activation function.

    SwiGLU(x) = Swish(xW + b) ⊙ (xV + c)
    where Swish(x) = x * sigmoid(x)

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


class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or 4 * embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


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
