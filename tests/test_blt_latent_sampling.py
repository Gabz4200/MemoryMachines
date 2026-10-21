"""Reparameterized Gaussian sampling at the BLT causal -> vocabulary boundary.

The sampler predicts a per-patch mean and standard deviation, draws one
Gaussian vector per patch, and reparameterizes the draw so byte cross-entropy
backpropagates into both predictions.
"""

import math
from typing import Any

import pytest
import torch

from memory_machines.models.backbone import BLTBackbone
from memory_machines.models.blocks.latent_sampling import GaussianLatentSampler
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


def _sampler(**kw: Any) -> GaussianLatentSampler:
    return GaussianLatentSampler(_cfg(blt_latent_sampling=True, **kw))


# ---------------------------------------------------------------- sampling --
def test_sampler_preserves_shape_and_returns_scalar_regularizer() -> None:
    cfg = _cfg(blt_latent_sampling=True)
    sampler = GaussianLatentSampler(cfg)
    hidden = torch.randn(2, 3, cfg.hidden_size)
    sample = sampler(hidden, noise=torch.randn_like(hidden))
    assert tuple(sample.latent.shape) == (2, 3, cfg.hidden_size)
    assert tuple(sample.mean.shape) == (2, 3, cfg.hidden_size)
    assert tuple(sample.std.shape) == (2, 3, cfg.hidden_size)
    assert sample.regularization.ndim == 0


def test_sampler_starts_at_identity_mean_and_target_std() -> None:
    cfg = _cfg(blt_latent_sampling=True, blt_latent_target_std=0.2)
    sampler = GaussianLatentSampler(cfg)
    hidden = torch.randn(1, 4, cfg.hidden_size)
    sample = sampler(hidden, noise=torch.zeros_like(hidden))
    torch.testing.assert_close(sample.mean, hidden)
    torch.testing.assert_close(sample.std, torch.full_like(hidden, 0.2))


def test_sampler_reparameterization_matches_closed_form() -> None:
    sampler = _sampler(blt_latent_target_std=0.3)
    hidden = torch.randn(2, 5, 16)
    noise = torch.randn_like(hidden)
    sample = sampler(hidden, temperature=0.7, noise=noise)
    torch.testing.assert_close(sample.latent, sample.mean + noise * sample.std * 0.7)


def test_zero_temperature_is_deterministic() -> None:
    sampler = _sampler()
    hidden = torch.randn(2, 3, 16)
    cold = sampler(hidden, temperature=0.0, noise=torch.randn_like(hidden))
    colder = sampler(hidden, temperature=0.0, noise=torch.randn_like(hidden))
    torch.testing.assert_close(cold.latent, cold.mean)
    torch.testing.assert_close(cold.latent, colder.latent, atol=0, rtol=0)


def test_temperature_scales_the_noise_range() -> None:
    sampler = _sampler(blt_latent_target_std=0.5)
    hidden = torch.randn(1, 4, 16)
    noise = torch.rand_like(hidden)
    base = sampler(hidden, temperature=1.0, noise=noise)
    warm = sampler(hidden, temperature=2.0, noise=noise)
    torch.testing.assert_close(warm.latent - warm.mean, 2 * (base.latent - base.mean))


def test_none_temperature_uses_the_configured_default() -> None:
    sampler = _sampler(blt_latent_temperature=0.4, blt_latent_target_std=0.5)
    hidden = torch.randn(2, 4, 16)
    noise = torch.randn_like(hidden)
    implicit = sampler(hidden, noise=noise)
    explicit = sampler(hidden, temperature=0.4, noise=noise)
    torch.testing.assert_close(implicit.latent, explicit.latent, atol=0, rtol=0)
    assert (implicit.latent - implicit.mean).abs().max().item() > 0


def test_none_temperature_uses_a_configured_zero() -> None:
    sampler = _sampler(blt_latent_temperature=0.0, blt_latent_target_std=0.5)
    hidden = torch.randn(1, 3, 16)
    sample = sampler(hidden, noise=torch.randn_like(hidden))
    torch.testing.assert_close(sample.latent, sample.mean, atol=0, rtol=0)


@pytest.mark.parametrize("temperature", [0.0, 1e6], ids=["zero", "huge"])
def test_temperature_boundaries_keep_the_latent_finite(temperature: float) -> None:
    sampler = _sampler(blt_latent_target_std=0.5)
    hidden = torch.randn(1, 3, 16)
    noise = torch.ones_like(hidden)
    sample = sampler(hidden, temperature=temperature, noise=noise)
    torch.testing.assert_close(sample.latent, sample.mean + noise * sample.std * temperature)
    assert torch.isfinite(sample.latent).all()


def test_std_clamps_to_the_configured_floor() -> None:
    sampler = _sampler(blt_latent_target_std=0.1, blt_latent_min_std=1e-4)
    d = sampler.hidden_size
    with torch.no_grad():
        sampler.stats_proj.weight[d:].zero_()
        sampler.stats_proj.bias[d:].fill_(-100.0)
    hidden = torch.randn(2, 3, d)
    sample = sampler(hidden, noise=torch.ones_like(hidden))
    # exp(clamp(-100, -8, 8)) * 0.1 = 3.4e-6, below the 1e-4 floor.
    torch.testing.assert_close(sample.std, torch.full_like(hidden, 1e-4), atol=0, rtol=0)
    assert torch.isfinite(sample.latent).all()
    assert torch.isfinite(sample.regularization).item()
    assert sample.regularization.item() > 0


def test_default_noise_is_standard_normal() -> None:
    sampler = _sampler(blt_latent_target_std=0.5)
    hidden = torch.zeros(8, 16, 16)
    torch.manual_seed(0)
    sample = sampler(hidden, temperature=1.0)
    eps = (sample.latent - sample.mean) / sample.std
    assert eps.mean().item() == pytest.approx(0.0, abs=0.1)
    assert eps.std().item() == pytest.approx(1.0, abs=0.1)
    assert eps.min().item() < 0.0
    assert eps.max().item() > 0.0


def test_sampler_rejects_negative_temperature() -> None:
    sampler = _sampler()
    with pytest.raises(ValueError, match="temperature"):
        sampler(torch.randn(1, 2, 16), temperature=-0.1)


def test_sampler_rejects_mismatched_noise_shape() -> None:
    sampler = _sampler()
    with pytest.raises(ValueError, match="noise"):
        sampler(torch.randn(1, 3, 16), noise=torch.zeros(1, 2, 16))


def test_reparameterized_sample_backpropagates_to_mean_and_std_heads() -> None:
    sampler = _sampler(blt_latent_target_std=0.5)
    hidden = torch.randn(2, 3, 16)
    sample = sampler(hidden, noise=torch.ones_like(hidden))
    sample.latent.sum().backward()
    grad = sampler.stats_proj.weight.grad
    assert grad is not None
    assert grad[:16].abs().sum().item() > 0
    assert grad[16:].abs().sum().item() > 0


# ------------------------------------------------------------ regularizer --
def _force_std(sampler: GaussianLatentSampler, std_value: float, target_std: float) -> None:
    d = sampler.hidden_size
    with torch.no_grad():
        sampler.stats_proj.weight[d:].zero_()
        sampler.stats_proj.bias[d:].fill_(math.log(std_value / target_std))


def test_regularizer_is_zero_at_the_target_std() -> None:
    sampler = _sampler(blt_latent_target_std=0.2)
    sample = sampler(torch.randn(2, 4, 16), noise=torch.zeros(2, 4, 16))
    assert sample.regularization.item() == pytest.approx(0.0, abs=1e-6)


def test_regularizer_penalizes_collapsed_and_exploded_std() -> None:
    target = 0.1
    hidden = torch.randn(1, 3, 16)
    collapsed = _sampler(blt_latent_target_std=target)
    _force_std(collapsed, target / 1000, target)
    exploded = _sampler(blt_latent_target_std=target)
    _force_std(exploded, target * 100, target)
    cold = collapsed(hidden, noise=torch.zeros_like(hidden))
    hot = exploded(hidden, noise=torch.zeros_like(hidden))
    assert cold.regularization.item() > 1.0
    assert hot.regularization.item() > 1.0


def test_regularizer_depends_only_on_std_not_mean() -> None:
    sampler = _sampler(blt_latent_target_std=0.1)
    d = sampler.hidden_size
    with torch.no_grad():
        sampler.stats_proj.weight.zero_()
        sampler.stats_proj.bias[d:].fill_(math.log(0.5))
    first = sampler(torch.randn(1, 2, d), noise=torch.zeros(1, 2, d))
    second = sampler(torch.randn(1, 2, d) + 10, noise=torch.zeros(1, 2, d))
    assert (first.mean - second.mean).abs().max().item() > 0
    torch.testing.assert_close(first.regularization, second.regularization, atol=0, rtol=0)


# ------------------------------------------------------------------ config --
@pytest.mark.parametrize(
    "kw",
    [
        {"blt_latent_target_std": 0.0},
        {"blt_latent_target_std": -0.1},
        {"blt_latent_temperature": -0.1},
        {"blt_latent_min_std": 0.0},
        {"blt_latent_std_reg_weight": -0.1},
        {"blt_latent_target_std": 0.1, "blt_latent_min_std": 0.2},
    ],
    ids=["zero-target", "negative-target", "negative-temperature", "zero-floor", "negative-reg-weight", "floor-above-target"],
)
def test_config_rejects_invalid_latent_sampling_values(kw: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="blt_latent"):
        _cfg(**kw)


# ---------------------------------------------------------------- backbone --
def test_backbone_samples_when_enabled() -> None:
    model = BLTBackbone(_cfg(blt_latent_sampling=True, blt_latent_target_std=0.5)).eval()
    ids = torch.randint(0, VOCAB, (1, 8))
    torch.manual_seed(0)
    with torch.no_grad():
        first = model(ModelInput(input_ids=ids)).logits
        second = model(ModelInput(input_ids=ids)).logits
    assert (first - second).abs().max().item() > 0


def test_backbone_temperature_override_is_deterministic() -> None:
    model = BLTBackbone(_cfg(blt_latent_sampling=True, blt_latent_target_std=0.5)).eval()
    ids = torch.randint(0, VOCAB, (1, 8))
    with torch.no_grad():
        first = model(ModelInput(input_ids=ids, temperature=0.0)).logits
        second = model(ModelInput(input_ids=ids, temperature=0.0)).logits
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_backbone_config_temperature_zero_is_deterministic() -> None:
    model = BLTBackbone(_cfg(blt_latent_sampling=True, blt_latent_temperature=0.0)).eval()
    ids = torch.randint(0, VOCAB, (1, 8))
    with torch.no_grad():
        first = model(ModelInput(input_ids=ids)).logits
        second = model(ModelInput(input_ids=ids)).logits
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_backbone_sampling_preserves_full_and_masked_shapes() -> None:
    cfg = _cfg(blt_latent_sampling=True, ced_masked_query=True, ced_masked_token_count=4, blt_max_patch_bytes=3)
    model = BLTBackbone(cfg).eval()
    ids = torch.randint(0, VOCAB, (1, 8))
    with torch.no_grad():
        full = model(ModelInput(input_ids=ids)).logits
        masked = model(ModelInput(input_ids=ids, num_masked_tokens=2)).logits
    assert tuple(full.shape) == (1, 8, VOCAB)
    assert tuple(masked.shape) == (1, 2 * 3, VOCAB)


def test_backbone_aux_loss_is_present_only_when_sampling_is_enabled() -> None:
    ids = torch.randint(0, VOCAB, (1, 8))
    plain = BLTBackbone(_cfg()).eval()
    sampled = BLTBackbone(_cfg(blt_latent_sampling=True)).eval()
    assert plain.latent_sampler is None
    with torch.no_grad():
        assert plain(ModelInput(input_ids=ids)).aux_loss is None
        aux = sampled(ModelInput(input_ids=ids)).aux_loss
    assert aux is not None
    assert aux.ndim == 0
    assert torch.isfinite(aux).all()
    assert aux.item() >= 0


def test_backbone_reset_parameters_restores_sampler_start() -> None:
    cfg = _cfg(blt_latent_sampling=True, blt_latent_target_std=0.3)
    model = BLTBackbone(cfg)
    sampler = model.latent_sampler
    assert sampler is not None
    with torch.no_grad():
        sampler.stats_proj.bias.fill_(3.0)
    model.reset_parameters()
    hidden = torch.randn(1, 2, cfg.hidden_size)
    sample = sampler(hidden, noise=torch.zeros_like(hidden))
    torch.testing.assert_close(sample.mean, hidden)
    torch.testing.assert_close(sample.std, torch.full_like(hidden, 0.3))


def test_backbone_training_backpropagates_into_both_sampling_heads() -> None:
    cfg = _cfg(blt_latent_sampling=True, blt_latent_target_std=0.5)
    model = BLTBackbone(cfg).train()
    out = model(ModelInput(input_ids=torch.randint(0, VOCAB, (1, 8))))
    assert model.latent_sampler is not None
    assert out.aux_loss is not None
    loss = out.logits.square().mean() + out.aux_loss
    loss.backward()
    grad = model.latent_sampler.stats_proj.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad[: cfg.hidden_size].abs().sum().item() > 0
    assert grad[cfg.hidden_size :].abs().sum().item() > 0
