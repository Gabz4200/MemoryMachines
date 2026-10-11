"""Top-level research backbone: pure tensor math, nothing else.

This is a deliberately minimal EXAMPLE (a 2-block residual stack) that
exercises every scaffold invariant. Replace it with your model — keep
the invariants, swap the math:

  - No training loops, no loss, no optimizer, no .to(device), no file I/O.
  - Accepts typed kwargs (embed_dim: int), never a DictConfig.
  - State is passed in explicitly and returned explicitly (no self.state).
  - Primitive ops go through the dispatcher, never a raw kernel import.
"""

from typing import cast

import torch
from torch import nn

from project.models.blocks import AttentionBlock, FeedForwardBlock
from project.models.io import ModelInput, ModelOutput
from project.models.registry import register_model
from project.ops.dispatch import dispatch_dw_scale


@register_model("default")
@register_model("example")
class ExampleBackbone(nn.Module):
    """Minimal flat composition: attn + ffn blocks, a norm, a head."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        vocab_size: int = 1000,
        backend: str = "naive",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.backend = backend
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": AttentionBlock(embed_dim, num_heads),
                        "ff": FeedForwardBlock(embed_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.input_proj = nn.Linear(vocab_size, embed_dim, bias=False)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, inputs: ModelInput) -> ModelOutput:
        x = inputs.inputs_embeds
        if x is None:
            if inputs.input_ids is None:
                raise ValueError("ModelInput requires input_ids or inputs_embeds")
            x = torch.nn.functional.one_hot(inputs.input_ids, self.vocab_size).to(torch.float32)
            x = self.input_proj(x)
        for block in self.blocks:
            # ModuleList is typed as Module, so indexing a ModuleDict element
            # needs the cast to keep a type checker honest.
            pair = cast(nn.ModuleDict, block)
            x = pair["ff"](pair["attn"](x))
        x = self.norm(x)
        if x.dtype == torch.float32:  # primitive op via the dispatcher, never a raw kernel
            x = dispatch_dw_scale(x, x, backend=self.backend)
        return ModelOutput(logits=self.head(x), new_state=None)
