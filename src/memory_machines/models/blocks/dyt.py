import torch
import torch.nn.functional as F
from torch import nn


class DyT(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.saturation = nn.Parameter(torch.ones(1, hidden_dim))
        self.layer_scale = nn.Parameter(torch.ones(1, hidden_dim))
        self.shift = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, x):
        # Apply saturation and shift
        x = x * self.saturation + self.shift
        # Apply Tanh activation
        x = F.tanh(x)
        # Apply layer scale
        x = x * self.layer_scale
        return x
