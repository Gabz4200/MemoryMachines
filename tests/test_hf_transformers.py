"""Tests for src/memory_machines/integrations/transformers/.

Covers the HF config adapter (defaults + typed accessors), the LM adapter
(backbone construction, single-registered head aliasing, embedding/head
accessors, forward/loss/return_dict translation), auto-class registration, and a
local save/load round-trip.
"""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

import memory_machines.integrations.transformers  # noqa: F401  (import registers the "project" auto classes)
from memory_machines.integrations.transformers.configuration_project import CustomResearchConfig
from memory_machines.integrations.transformers.modeling_project import CustomModelForCausalLM
from memory_machines.models.backbone import ExampleBackbone

VOCAB = 11
EMBED = 16


def _config(**kwargs) -> CustomResearchConfig:
    params = {"embed_dim": EMBED, "num_layers": 1, "vocab_size": VOCAB, "backend": "naive"}
    params.update(kwargs)
    return CustomResearchConfig(**params)


def _ids(B: int = 2, T: int = 5) -> torch.Tensor:
    return torch.arange(B * T).view(B, T) % VOCAB


def _manual_shifted_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].reshape(-1, logits.size(-1))
    shift_labels = labels[..., 1:].reshape(-1)
    return nn.CrossEntropyLoss()(shift_logits, shift_labels)


def test_config_defaults() -> None:
    config = CustomResearchConfig()

    assert config.model_type == "project"
    assert config.typed_embed_dim == 512
    assert config.typed_num_layers == 6
    assert config.typed_backend == "naive"
    assert config.typed_vocab_size == 50257


def test_config_typed_accessors_return_declared_types() -> None:
    config = _config(embed_dim=64, num_layers=3, backend="ced", vocab_size=1000)

    assert (config.typed_embed_dim, config.typed_num_layers, config.typed_vocab_size) == (64, 3, 1000)
    assert config.typed_backend == "ced"
    assert type(config.typed_embed_dim) is int
    assert type(config.typed_num_layers) is int
    assert type(config.typed_vocab_size) is int
    assert type(config.typed_backend) is str


def test_config_is_pretrained_config_subclass() -> None:
    assert issubclass(CustomResearchConfig, PretrainedConfig)


def test_model_builds_backbone_from_config_dims() -> None:
    config = _config(embed_dim=8, num_layers=2, vocab_size=13)

    model = CustomModelForCausalLM(config)

    backbone = model.backbone
    assert isinstance(backbone, ExampleBackbone)
    assert backbone.embed_dim == 8
    assert backbone.num_layers == 2
    assert backbone.vocab_size == 13
    assert model.vocab_size == 13
    assert backbone.embed.weight.shape == (13, 8)
    assert len(backbone.blocks) == 2


def test_lm_head_is_the_same_object_as_backbone_head() -> None:
    model = CustomModelForCausalLM(_config())

    assert model.lm_head is model.backbone.head

    head_keys = [key for key in model.state_dict() if key.endswith("head.weight")]
    assert head_keys == ["backbone.head.weight"]


def test_lm_head_setter_updates_backbone_head() -> None:
    model = CustomModelForCausalLM(_config())
    new_head = nn.Linear(EMBED, VOCAB, bias=False)

    model.lm_head = new_head

    assert model.lm_head is new_head
    assert model.backbone.head is new_head
    assert model.get_output_embeddings() is new_head


def test_input_embedding_accessors_round_trip() -> None:
    model = CustomModelForCausalLM(_config())
    assert model.get_input_embeddings() is model.backbone.embed

    replacement = nn.Embedding(VOCAB, EMBED)
    model.set_input_embeddings(replacement)

    assert model.get_input_embeddings() is replacement
    assert model.backbone.embed is replacement


def test_set_output_embeddings_replaces_head() -> None:
    model = CustomModelForCausalLM(_config())
    replacement = nn.Linear(EMBED, VOCAB, bias=False)

    model.set_output_embeddings(replacement)

    assert model.get_output_embeddings() is replacement
    assert model.lm_head is replacement
    assert model.backbone.head is replacement


def test_forward_returns_causal_lm_output_with_vocab_logits() -> None:
    model = CustomModelForCausalLM(_config())
    ids = _ids()

    output = model(input_ids=ids, return_dict=True)

    assert isinstance(output, CausalLMOutputWithPast)
    assert output.logits is not None
    assert output.logits.shape == (2, 5, VOCAB)
    assert output.loss is None
    assert output.past_key_values is None


def test_forward_loss_matches_manual_shifted_cross_entropy() -> None:
    model = CustomModelForCausalLM(_config())
    ids = _ids()
    labels = (_ids() + 1) % VOCAB

    output = model(input_ids=ids, labels=labels, return_dict=True)
    logits = model(input_ids=ids, return_dict=True).logits

    assert output.loss is not None
    assert output.loss.shape == ()
    assert torch.allclose(output.loss, _manual_shifted_ce(logits, labels))


def test_forward_without_return_dict_returns_logits_and_state() -> None:
    model = CustomModelForCausalLM(_config())
    ids = _ids()

    expected = model(input_ids=ids, return_dict=True).logits
    output = model(input_ids=ids, return_dict=False)

    assert isinstance(output, tuple)
    assert len(output) == 2
    assert torch.equal(output[0], expected)
    assert output[1] is None


def test_forward_without_return_dict_prepends_loss_when_labels_given() -> None:
    model = CustomModelForCausalLM(_config())
    ids = _ids()
    labels = (_ids() + 1) % VOCAB

    expected = model(input_ids=ids, labels=labels, return_dict=True).loss
    output = model(input_ids=ids, labels=labels, return_dict=False)

    assert isinstance(output, tuple)
    assert len(output) == 3
    assert expected is not None
    assert torch.allclose(output[0], expected)
    assert output[1].shape == (2, 5, VOCAB)


def test_forward_accepts_attention_mask() -> None:
    model = CustomModelForCausalLM(_config())
    ids = _ids()
    mask = torch.ones(2, 5, dtype=torch.long)

    unmasked = model(input_ids=ids, return_dict=True).logits
    masked = model(input_ids=ids, attention_mask=mask, return_dict=True).logits

    # ExampleBackbone documents padding as unsupported: the mask is accepted for
    # API parity and must not change the math.
    assert torch.equal(unmasked, masked)


def test_auto_config_for_model_resolves_adapter_config() -> None:
    config = AutoConfig.for_model("project", embed_dim=EMBED, num_layers=1, vocab_size=VOCAB)

    assert isinstance(config, CustomResearchConfig)
    assert config.typed_embed_dim == EMBED


def test_auto_model_from_config_resolves_adapter_model() -> None:
    config = AutoConfig.for_model("project", embed_dim=EMBED, num_layers=1, vocab_size=VOCAB)

    model = AutoModelForCausalLM.from_config(config)

    assert isinstance(model, CustomModelForCausalLM)
    assert model.lm_head is model.backbone.head


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    model = CustomModelForCausalLM(_config())
    ids = _ids()
    expected = model(input_ids=ids, return_dict=True).logits

    model.save_pretrained(tmp_path)
    loaded = CustomModelForCausalLM.from_pretrained(tmp_path)

    assert json.loads((tmp_path / "config.json").read_text())["model_type"] == "project"
    assert isinstance(loaded, CustomModelForCausalLM)
    assert loaded.config.typed_embed_dim == EMBED
    assert loaded.config.typed_vocab_size == VOCAB
    assert loaded.lm_head is loaded.backbone.head
    assert torch.allclose(loaded(input_ids=ids, return_dict=True).logits, expected)
