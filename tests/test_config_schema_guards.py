"""Config guards for the schema module's non-block surface.

Covers what ``tests/test_config_schema_coverage.py`` deliberately does not:
``MemoryMachineConfig.from_pretrained_size`` sizes and its unknown-size error,
the nested ``ModelConfig``/``DataConfig``/``TrainerConfig``/``ExperimentConfig``
dataclasses, and the remaining ``MemoryMachineConfig.__post_init__`` bounds that
no other test names.
"""

import pytest

from memory_machines.utils.config_schema import (
    DataConfig,
    ExperimentConfig,
    MemoryMachineConfig,
    ModelConfig,
    TrainerConfig,
)

PRESET_SIZES = [
    # size, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, csa_num_output_groups
    ("350M", 768, 2048, 12, 3, 6),
    ("700M", 1024, 2816, 16, 4, 8),
    ("1.2B", 1536, 4096, 24, 6, 8),
]


@pytest.mark.parametrize(
    "size,hidden_size,intermediate_size,num_attention_heads,num_key_value_heads,csa_num_output_groups",
    PRESET_SIZES,
    ids=["350M", "700M", "1.2B"],
)
def test_from_pretrained_size_exposes_documented_shape(
    size: str,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    csa_num_output_groups: int,
) -> None:
    cfg = MemoryMachineConfig.from_pretrained_size(size)
    assert cfg.hidden_size == hidden_size
    assert cfg.intermediate_size == intermediate_size
    assert cfg.num_attention_heads == num_attention_heads
    assert cfg.num_key_value_heads == num_key_value_heads
    assert cfg.csa_num_output_groups == csa_num_output_groups


def test_from_pretrained_size_unknown_lists_available_keys() -> None:
    with pytest.raises(ValueError, match=r"Unknown model size: 42B\. Available: \['350M', '700M', '1\.2B'\]"):
        MemoryMachineConfig.from_pretrained_size("42B")


DEFAULTS = [
    (ModelConfig, "embed_dim", 256),
    (ModelConfig, "num_layers", 6),
    (ModelConfig, "backend", "naive"),
    (DataConfig, "data_dir", "data"),
    (DataConfig, "batch_size", 32),
    (DataConfig, "num_workers", 4),
    (TrainerConfig, "max_epochs", 10),
    (TrainerConfig, "accelerator", "auto"),
    (TrainerConfig, "devices", "auto"),
    (TrainerConfig, "precision", "bf16-mixed"),
    (TrainerConfig, "strategy", "auto"),
    (TrainerConfig, "fast_dev_run", False),
]


@pytest.mark.parametrize(
    "factory,field,expected",
    DEFAULTS,
    ids=[f"{factory.__name__.lower()}_{field}" for factory, field, _ in DEFAULTS],
)
def test_other_dataclasses_declare_defaults(factory: type, field: str, expected: object) -> None:
    assert getattr(factory(), field) == expected


OVERRIDES = [
    (ModelConfig, {"embed_dim": 64, "num_layers": 2, "backend": "recurrent"}, "backend", "recurrent"),
    (DataConfig, {"data_dir": "/tmp/d", "batch_size": 8, "num_workers": 0}, "batch_size", 8),
    (TrainerConfig, {"max_epochs": 3, "precision": "32", "fast_dev_run": True}, "fast_dev_run", True),
]


@pytest.mark.parametrize(
    "factory,kwargs,field,expected",
    OVERRIDES,
    ids=[factory.__name__.lower() for factory, _, _, _ in OVERRIDES],
)
def test_other_dataclasses_accept_overrides(factory: type, kwargs: dict, field: str, expected: object) -> None:
    assert getattr(factory(**kwargs), field) == expected


@pytest.mark.parametrize(
    "attr,expected",
    [("model", ModelConfig()), ("data", DataConfig()), ("trainer", TrainerConfig())],
    ids=["model", "data", "trainer"],
)
def test_experiment_config_builds_nested_defaults(attr: str, expected: object) -> None:
    assert getattr(ExperimentConfig(), attr) == expected


def test_experiment_config_instances_do_not_share_nested_config() -> None:
    first, second = ExperimentConfig(), ExperimentConfig()
    first.model.backend = "recurrent"
    assert second.model.backend == "naive"


# The ``ced_*_ffn_intermediate_size`` negatives are already asserted in
# tests/test_ced_ffn.py::test_config_rejects_bad_ffn_fields (same branch).
INVALID_FIELDS = [
    ({"num_layers": 0}, "num_layers"),
    ({"hidden_dropout": 1.5}, "hidden_dropout"),
    ({"cvpe_mlp_hidden": 0}, "cvpe_mlp_hidden"),
    ({"csa_q_latent_dim": -1}, "csa_q_latent_dim"),
    ({"mhc_sinkhorn_iters": 0}, "mhc_sinkhorn_iters"),
    ({"titan_conv_kernel_size": 0}, "titan_conv_kernel_size"),
]


@pytest.mark.parametrize(
    "overrides,match",
    INVALID_FIELDS,
    ids=[f"{next(iter(overrides))}" for overrides, _ in INVALID_FIELDS],
)
def test_rejects_out_of_range_override(overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MemoryMachineConfig(**overrides)
