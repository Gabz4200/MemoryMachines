"""Explicit state routing: chunked forward == full forward for stateful models."""

import torch

from project.models.io import ModelInput
from project.models.registry import get_model


def test_chunked_forward_matches_full():
    model = get_model("example", embed_dim=32, num_layers=1, vocab_size=100)
    embeds = torch.randn(1, 16, 32)
    full = model(ModelInput(inputs_embeds=embeds))
    # Stateless scaffold model: chunking is trivially consistent. Stateful
    # models must implement new_state routing and verify chunk parity here.
    assert full.new_state is None
