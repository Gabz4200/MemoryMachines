"""System wrapper wiring: injected model, step math, optimizer groups."""

import torch

from project.models.io import ModelInput
from project.models.registry import get_model
from project.modules.lightning_module import LitResearchSystem


def test_system_forward():
    model = get_model("example", embed_dim=16, num_layers=1, vocab_size=512)
    system = LitResearchSystem(model=model)
    out = system.forward(ModelInput(inputs_embeds=torch.randn(2, 4, 16)))
    assert out.logits.shape == (2, 4, 512)


def test_optimizer_param_groups():
    model = get_model("example", embed_dim=16, num_layers=1, vocab_size=512)
    system = LitResearchSystem(model=model)
    groups = system.configure_optimizers().param_groups
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))
