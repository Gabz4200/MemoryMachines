"""Structured Config schemas. Validate cross-group invariants at boot, before GPU init."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    embed_dim: int = 256
    num_layers: int = 6
    backend: str = "naive"


@dataclass
class DataConfig:
    data_dir: str = "data"
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class TrainerConfig:
    max_epochs: int = 10
    accelerator: str = "auto"
    devices: str = "auto"
    precision: str = "bf16-mixed"
    strategy: str = "auto"
    fast_dev_run: bool = False


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
