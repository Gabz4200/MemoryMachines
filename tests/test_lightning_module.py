"""Tests for src/memory_machines/modules/lightning_module.py.

Covers forward delegation, the shared ``_step`` loss path (ModelInput building,
CE on flattened logits, optional aux loss, ignore_index), step logging names, and
the rule-based AdamW parameter grouping.
"""

import pytest
import torch
from torch import nn

from memory_machines.models.io import ModelInput, ModelOutput
from memory_machines.modules.lightning_module import LitResearchSystem

VOCAB = 7
EMBED = 4


class _FakeModel(nn.Module):
    """Real nn.Module (not a Mock) whose parameters span ndim 1 and ndim 2."""

    def __init__(self, aux_loss: torch.Tensor | None = None) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, EMBED)
        self.proj = nn.Linear(EMBED, VOCAB)
        self.scale = nn.Parameter(torch.ones(VOCAB))
        self.frozen = nn.Parameter(torch.zeros(2, 2))
        self.aux_loss = aux_loss
        self.seen: list[ModelInput] = []
        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(VOCAB, EMBED, generator=gen))
            self.proj.weight.copy_(torch.randn(VOCAB, EMBED, generator=gen))
            self.proj.bias.copy_(torch.randn(VOCAB, generator=gen))

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.seen.append(inputs)
        if inputs.input_ids is None:
            raise ValueError("fake model needs input_ids")
        logits = self.proj(self.embed(inputs.input_ids)) * self.scale
        return ModelOutput(logits=logits, aux_loss=self.aux_loss)


def _batch(labels: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 0]])
    out = {"input_ids": ids}
    if labels is None:
        labels = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 1]])
    out["labels"] = labels
    if mask is not None:
        out["attention_mask"] = mask
    return out


def _logits(model: _FakeModel, input_ids: torch.Tensor) -> torch.Tensor:
    return model(ModelInput(input_ids=input_ids)).logits


def _ce(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    return nn.CrossEntropyLoss(ignore_index=ignore_index)(logits.view(-1, logits.size(-1)), labels.view(-1))


def test_forward_delegates_to_injected_model() -> None:
    model = _FakeModel()
    layer = LitResearchSystem(model)
    inputs = ModelInput(input_ids=torch.tensor([[0, 1, 2]]))

    output = layer(inputs)

    assert model.seen == [inputs]
    assert torch.equal(output.logits, model(inputs).logits)


def test_step_builds_model_input_from_batch() -> None:
    model = _FakeModel()
    layer = LitResearchSystem(model)
    mask = torch.ones(2, 4, dtype=torch.bool)
    batch = _batch(mask=mask)

    layer._step(batch)

    received = model.seen[0]
    assert received.input_ids is batch["input_ids"]
    assert received.attention_mask is mask


def test_step_without_attention_mask_leaves_it_none() -> None:
    model = _FakeModel()
    layer = LitResearchSystem(model)

    layer._step(_batch())

    assert model.seen[0].attention_mask is None


def test_step_ce_loss_matches_manual_cross_entropy() -> None:
    model = _FakeModel()
    layer = LitResearchSystem(model)
    batch = _batch()

    loss = layer._step(batch)

    assert loss.shape == ()
    assert torch.allclose(loss, _ce(_logits(model, batch["input_ids"]), batch["labels"]))


def test_step_adds_aux_loss() -> None:
    aux = torch.tensor(0.5)
    model = _FakeModel(aux_loss=aux)
    layer = LitResearchSystem(model)
    batch = _batch()

    loss = layer._step(batch)

    assert torch.allclose(loss, _ce(_logits(model, batch["input_ids"]), batch["labels"]) + aux)


def test_step_skips_none_aux_loss() -> None:
    model = _FakeModel(aux_loss=None)
    layer = LitResearchSystem(model)
    batch = _batch()

    loss = layer._step(batch)

    assert torch.allclose(loss, _ce(_logits(model, batch["input_ids"]), batch["labels"]))


def test_step_honors_ignore_index() -> None:
    model = _FakeModel()
    layer = LitResearchSystem(model)
    labels = torch.tensor([[1, -100, 3, -100], [5, -100, 0, 1]])
    batch = _batch(labels=labels)

    loss = layer._step(batch)

    expected = _ce(_logits(model, batch["input_ids"]), labels)
    valid = labels.view(-1) != -100
    subset = nn.CrossEntropyLoss()(
        _logits(model, batch["input_ids"]).view(-1, VOCAB)[valid],
        labels.view(-1)[valid],
    )
    assert torch.allclose(loss, expected)
    assert torch.allclose(loss, subset)
    assert torch.isfinite(loss)


def test_constructor_stores_hyperparameters_and_criterion() -> None:
    layer = LitResearchSystem(_FakeModel(), learning_rate=3e-4, weight_decay=0.1, ignore_index=-7)

    assert layer.learning_rate == 3e-4
    assert layer.weight_decay == 0.1
    assert layer.criterion.ignore_index == -7


@pytest.mark.parametrize(
    ("method", "log_name"),
    [("training_step", "train/loss"), ("validation_step", "val/loss"), ("test_step", "test/loss")],
    ids=["train", "val", "test"],
)
def test_steps_return_loss_and_log_under_expected_name(monkeypatch: pytest.MonkeyPatch, method: str, log_name: str) -> None:
    model = _FakeModel()
    layer = LitResearchSystem(model)
    calls: list[tuple[str, torch.Tensor, dict]] = []

    def _record(name: str, value: torch.Tensor, **kwargs) -> None:
        calls.append((name, value, kwargs))

    monkeypatch.setattr(layer, "log", _record)

    loss = getattr(layer, method)(_batch(), 0)

    assert len(calls) == 1
    name, value, kwargs = calls[0]
    assert name == log_name
    assert value is loss
    assert kwargs["prog_bar"] is True


def test_configure_optimizers_returns_adamw_with_rule_based_groups() -> None:
    model = _FakeModel()
    model.frozen.requires_grad_(False)
    layer = LitResearchSystem(model, learning_rate=3e-4, weight_decay=0.1)

    optimizer = layer.configure_optimizers()

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == pytest.approx(3e-4)
    assert len(optimizer.param_groups) == 2

    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == pytest.approx(0.1)
    assert no_decay["weight_decay"] == 0.0
    assert decay["params"]
    assert no_decay["params"]
    assert all(p.ndim >= 2 for p in decay["params"])
    assert all(p.ndim < 2 for p in no_decay["params"])

    grouped = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert grouped == {id(p) for p in model.parameters() if p.requires_grad}
    assert id(model.frozen) not in grouped
