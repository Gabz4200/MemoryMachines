"""LightningDataModule stage wiring and DataLoader configuration.

The module is synthetic, so no files are touched; ``data_dir`` only names where
real ingestion would look.
"""

from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from memory_machines.data.datamodule import ResearchDataModule
from memory_machines.data.dataset import SimpleTensorDataset


def _dm(tmp_path: Path, **kw: Any) -> ResearchDataModule:
    return ResearchDataModule(data_dir=str(tmp_path), **kw)


def test_prepare_data_is_a_no_op_returning_none(tmp_path: Path) -> None:
    dm = _dm(tmp_path)
    assert dm.prepare_data() is None
    assert not hasattr(dm, "train_dataset")


@pytest.mark.parametrize(
    "stage,expected",
    [
        ("fit", ("train_dataset", "val_dataset")),
        ("test", ("test_dataset",)),
        (None, ("train_dataset", "val_dataset", "test_dataset")),
        ("predict", ()),
    ],
    ids=["fit", "test", "none", "predict"],
)
def test_setup_builds_only_the_datasets_for_the_stage(tmp_path: Path, stage: str | None, expected: tuple[str, ...]) -> None:
    dm = _dm(tmp_path)
    dm.setup(stage)
    for attr in ("train_dataset", "val_dataset", "test_dataset"):
        assert hasattr(dm, attr) is (attr in expected), attr


def test_setup_fit_datasets_are_synthetic_training_splits(tmp_path: Path) -> None:
    dm = _dm(tmp_path)
    dm.setup("fit")
    assert isinstance(dm.train_dataset, SimpleTensorDataset)
    assert isinstance(dm.val_dataset, SimpleTensorDataset)
    assert dm.train_dataset.split == "train"
    assert dm.val_dataset.split == "val"


def test_setup_test_dataset_uses_test_split(tmp_path: Path) -> None:
    dm = _dm(tmp_path)
    dm.setup("test")
    assert isinstance(dm.test_dataset, SimpleTensorDataset)
    assert dm.test_dataset.split == "test"


@pytest.mark.parametrize("loader_name", ["train_dataloader", "val_dataloader"], ids=["train", "val"])
def test_dataloader_batch_size_and_workers_are_propagated(tmp_path: Path, loader_name: str) -> None:
    dm = _dm(tmp_path, batch_size=8, num_workers=2)
    dm.setup("fit")
    loader: DataLoader[Any] = getattr(dm, loader_name)()
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 8
    assert loader.num_workers == 2


def test_train_dataloader_shuffles_and_val_does_not(tmp_path: Path) -> None:
    dm = _dm(tmp_path, batch_size=4, num_workers=0)
    dm.setup("fit")
    # DataLoader does not retain ``shuffle``; the sampler type is the observable effect.
    assert isinstance(dm.train_dataloader().sampler, RandomSampler)
    assert isinstance(dm.val_dataloader().sampler, SequentialSampler)


@pytest.mark.parametrize("loader_name", ["train_dataloader", "val_dataloader"], ids=["train", "val"])
def test_first_batch_has_input_ids_and_labels(tmp_path: Path, loader_name: str) -> None:
    dm = _dm(tmp_path, batch_size=4, num_workers=0)
    dm.setup("fit")
    loader: DataLoader[Any] = getattr(dm, loader_name)()
    with torch.random.fork_rng():  # train shuffle draws from the global RNG
        batch = next(iter(loader))
    assert set(batch) == {"input_ids", "labels"}
    # SimpleTensorDataset defaults: 256 samples of 32 tokens.
    assert tuple(batch["input_ids"].shape) == (4, 32)
    assert tuple(batch["labels"].shape) == (4, 32)
