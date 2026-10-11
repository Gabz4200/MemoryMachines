"""Minimal example backbone exercising scaffold invariants (typed kwargs, explicit state, dispatcher ops)."""

from typing import cast

import torch
from torch import nn

from memory_machines.models.blocks import GroupedQueryAttention, SwiGLU
from memory_machines.models.io import ModelInput, ModelOutput
from memory_machines.models.registry import register_model
from memory_machines.ops.dispatch import dispatch_dw_scale
from memory_machines.utils.config_schema import MemoryMachineConfig


class _GQAWrapper(nn.Module):
    """GroupedQueryAttention returning hidden states only (drops weights/cache)."""

    def __init__(self, attn: GroupedQueryAttention):
        super().__init__()
        self.attn = attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _, _ = self.attn(x)
        return out


@register_model("default")
@register_model("example")
class ExampleBackbone(nn.Module):
    """Minimal flat composition: GQA + SwiGLU blocks, a norm, a head."""

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
        cfg = MemoryMachineConfig(
            hidden_size=embed_dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            # Unused by this backbone, but validated at boot: keep 1 head/group.
            csa_num_output_groups=num_heads,
        )
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": _GQAWrapper(GroupedQueryAttention(cfg)),
                        "ff": SwiGLU(cfg),
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
            # Cast keeps the type checker honest on ModuleDict indexing.
            pair = cast(nn.ModuleDict, block)
            x = pair["ff"](pair["attn"](x))
        x = self.norm(x)
        if x.dtype == torch.float32:
            x = dispatch_dw_scale(x, x, backend=self.backend)
        return ModelOutput(logits=self.head(x), new_state=None)
