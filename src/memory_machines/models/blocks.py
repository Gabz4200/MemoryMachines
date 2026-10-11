"""Flat composable blocks. No inheritance hierarchies — assemble with ModuleList.

These two are the minimal vocabulary; delete or extend them for your
domain (conv blocks, mixer blocks, attention variants). The rule is
composition: a top-level nn.Module owns a ModuleList/ModuleDict of
cohesive blocks, nothing more.
"""

import torch
from torch import nn


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
