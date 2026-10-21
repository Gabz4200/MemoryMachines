"""Backbone behavior not covered by test_backbone.py / test_blt.py / test_ced_asymmetric.py.

Covers input-contract errors, byte-vocabulary selection, embedding accessors,
decoder view properties, the decode wrappers, symmetric/asymmetric
``decoder_in_proj`` construction, and ``reset_parameters`` on the untied head
and on the latent-sampling path.
"""

from typing import Any

import pytest
import torch
from torch import nn

from memory_machines.models.backbone import BLTBackbone, CheapLLM, ExampleBackbone
from memory_machines.models.io import ModelInput
from memory_machines.utils.config_schema import MemoryMachineConfig

VOCAB = 32
LOCAL = 16


def _cfg(**kw: Any) -> MemoryMachineConfig:
    base: dict[str, Any] = {
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 32,
        "ced_num_encoder_layers": 1,
        "ced_num_decoder_layers": 1,
        "csa_top_k": 2,
        "csa_num_output_groups": 2,
        "csa_output_group_dim": 8,
        "csa_indexer_heads": 2,
        "csa_indexer_dim": 8,
        "csa_hier_pool_blocks": 2,
        "csa_hier_pool_positions": 8,
        "csa_hier_block_size": 2,
        "vocab_size": VOCAB,
        "window_size": 8,
        "blt_local_hidden_size": LOCAL,
        "blt_local_layers": 1,
        "blt_local_window_size": 8,
        "blt_local_ffn_intermediate_size": 32,
        "blt_ngram_hash_size": 64,
        "blt_max_patch_bytes": 4,
        "blt_patch_size": 2,
    }
    base.update(kw)
    return MemoryMachineConfig(**base)


# ------------------------------------------------------------ input contract --
def test_example_backbone_requires_input_ids_or_embeds() -> None:
    model = ExampleBackbone(embed_dim=16, num_layers=1, num_heads=2, vocab_size=VOCAB)
    with pytest.raises(ValueError, match="input_ids or inputs_embeds"):
        model(ModelInput())


def test_blt_backbone_requires_input_ids_or_embeds() -> None:
    model = BLTBackbone(_cfg())
    with pytest.raises(ValueError, match="input_ids or inputs_embeds"):
        model(ModelInput())


def test_blt_backbone_inputs_embeds_without_patch_ids_raises() -> None:
    model = BLTBackbone(_cfg())
    with pytest.raises(ValueError, match="patch_ids is required"):
        model(ModelInput(inputs_embeds=torch.randn(1, 6, LOCAL)))


# ----------------------------------------------------------- embedding API ---
def test_example_backbone_embedding_accessors() -> None:
    model = ExampleBackbone(embed_dim=16, num_layers=1, num_heads=2, vocab_size=VOCAB)
    assert model.get_input_embeddings() is model.embed
    replacement = nn.Embedding(VOCAB, 16)
    model.set_input_embeddings(replacement)
    assert model.get_input_embeddings() is replacement
    out = model(ModelInput(input_ids=torch.randint(0, VOCAB, (1, 5))))
    assert tuple(out.logits.shape) == (1, 5, VOCAB)


# ---------------------------------------------------------- byte vocabulary --
def test_blt_vocab_size_falls_back_to_vocab_size() -> None:
    cfg = _cfg()
    assert cfg.blt_vocab_size == 0
    model = BLTBackbone(cfg).eval()
    assert model.vocab_size == VOCAB
    with torch.no_grad():
        logits = model(ModelInput(input_ids=torch.randint(0, VOCAB, (1, 6)))).logits
    assert tuple(logits.shape) == (1, 6, VOCAB)


def test_blt_vocab_size_override_wins() -> None:
    model = BLTBackbone(_cfg(blt_vocab_size=48)).eval()
    assert model.vocab_size == 48
    with torch.no_grad():
        logits = model(ModelInput(input_ids=torch.randint(0, VOCAB, (1, 6)))).logits
    assert tuple(logits.shape) == (1, 6, 48)


def test_blt_constructor_vocab_size_wins_over_config() -> None:
    model = BLTBackbone(_cfg(blt_vocab_size=48), vocab_size=64).eval()
    assert model.vocab_size == 64
    with torch.no_grad():
        logits = model(ModelInput(input_ids=torch.randint(0, VOCAB, (1, 6)))).logits
    assert tuple(logits.shape) == (1, 6, 64)


# -------------------------------------------------------- decoder views -----
def test_blt_decoder_view_properties_match_the_decoder() -> None:
    model = BLTBackbone(_cfg())
    assert model.decoder_layers is model.decoder.layers
    assert model.global_projs is model.decoder.global_projs
    assert model.decoder_ffns is model.decoder.ffns


# ---------------------------------------------------------- decode wrappers --
def test_blt_prefill_delegates_to_the_decoder() -> None:
    cfg = _cfg(window_size=4)
    model = BLTBackbone(cfg).eval()
    hidden, enc = torch.randn(1, 8, 16), torch.randn(1, 8, 16)
    with torch.no_grad():
        via_model = model.prefill(hidden, enc)
        via_decoder = model.decoder.prefill(hidden, enc)
    assert tuple(via_model.shape) == (1, cfg.window_size, 16)
    torch.testing.assert_close(via_model, via_decoder, atol=0, rtol=0)


def test_cheapllm_prefill_delegates_to_the_decoder() -> None:
    cfg = _cfg(window_size=4)
    model = CheapLLM(cfg).eval()
    hidden, enc = torch.randn(1, 8, 16), torch.randn(1, 8, 16)
    with torch.no_grad():
        via_model = model.prefill(hidden, enc)
        via_decoder = model.decoder.prefill(hidden, enc)
    assert tuple(via_model.shape) == (1, cfg.window_size, 16)
    torch.testing.assert_close(via_model, via_decoder, atol=0, rtol=0)


def test_blt_decode_full_context_delegates_to_the_decoder() -> None:
    model = BLTBackbone(_cfg()).eval()
    hidden, enc = torch.randn(1, 4, 16), torch.randn(1, 6, 16)
    with torch.no_grad():
        via_model = model.decode_full_context(hidden, enc, start_pos=2, exit_threshold=1.0)
        via_decoder = model.decoder.decode_full_context(hidden, enc, start_pos=2, exit_threshold=1.0)
    torch.testing.assert_close(via_model, via_decoder, atol=0, rtol=0)


# --------------------------------------------------------- decoder_in_proj ---
@pytest.mark.parametrize("model_cls", [CheapLLM, BLTBackbone], ids=["cheapllm", "blt"])
def test_symmetric_width_builds_identity_decoder_in_proj(model_cls: type[CheapLLM] | type[BLTBackbone]) -> None:
    model = model_cls(_cfg())
    assert isinstance(model.decoder_in_proj, nn.Identity)


@pytest.mark.parametrize("model_cls", [CheapLLM, BLTBackbone], ids=["cheapllm", "blt"])
def test_asymmetric_width_builds_linear_decoder_in_proj_and_forwards(model_cls: type[CheapLLM] | type[BLTBackbone]) -> None:
    cfg = _cfg(ced_decoder_hidden_size=8)
    model = model_cls(cfg).eval()
    proj = model.decoder_in_proj
    assert isinstance(proj, nn.Linear)
    assert (proj.in_features, proj.out_features) == (16, 8)
    model.reset_parameters()
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), name
    with torch.no_grad():
        logits = model(ModelInput(input_ids=torch.randint(0, VOCAB, (1, 6)))).logits
    assert tuple(logits.shape) == (1, 6, VOCAB)


# -------------------------------------------------------- reset_parameters ---
def test_cheapllm_reset_reinitializes_the_untied_head() -> None:
    model = CheapLLM(_cfg(tie_word_embeddings=False))
    assert model.head.weight is not model.embed.weight
    with torch.no_grad():
        model.head.weight.fill_(7.0)
    model.reset_parameters()
    assert not torch.all(model.head.weight == 7.0)
    assert torch.isfinite(model.head.weight).all()


def test_blt_reset_parameters_with_sampling_stays_finite() -> None:
    cfg = _cfg(blt_latent_sampling=True, blt_latent_target_std=0.3)
    model = BLTBackbone(cfg)
    with torch.no_grad():
        sampler = model.latent_sampler
        assert sampler is not None
        sampler.stats_proj.bias.fill_(-50.0)
    model.reset_parameters()
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), name
    hidden = torch.randn(1, 3, cfg.hidden_size)
    sample = sampler(hidden, noise=torch.zeros_like(hidden))
    assert torch.isfinite(sample.latent).all()
    assert torch.isfinite(sample.std).all()
    assert torch.isfinite(sample.regularization).item()
