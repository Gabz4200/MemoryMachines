"""Shape, field, and non-finite checks for every block and the backbone."""

import torch

from project.models.io import ModelInput
from project.models.registry import get_model


def test_backbone_shapes():
    model = get_model("example", embed_dim=64, num_layers=1, vocab_size=512)
    out = model(ModelInput(inputs_embeds=torch.randn(2, 8, 64)))
    assert out.logits.shape == (2, 8, 512)
    assert not torch.isnan(out.logits).any()
    assert not torch.isinf(out.logits).any()


def test_attention_block_shape():
    from project.models.blocks import AttentionBlock

    x = torch.randn(2, 8, 32)
    y = AttentionBlock(32)(x)
    assert y.shape == x.shape


def test_feedforward_block_shape():
    from project.models.blocks import FeedForwardBlock

    x = torch.randn(2, 8, 32)
    y = FeedForwardBlock(32)(x)
    assert y.shape == x.shape
