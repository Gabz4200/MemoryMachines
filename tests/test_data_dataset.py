"""Synthetic dataset wrapper and its process-independent per-split seeding.

``_stable_seed`` exists because ``hash()`` is salted per process, which made the
synthetic data change every run. The pinned literals below are the contract: a
seed change silently changes every training input, so it must be deliberate.
"""

import pytest
import torch

from memory_machines.data.dataset import SimpleTensorDataset, _stable_seed

SPLITS = ("train", "val", "test")


def _dataset(split: str = "train", **kw: int) -> SimpleTensorDataset:
    return SimpleTensorDataset("data", split=split, **kw)


# ------------------------------------------------------------------ seeding --
@pytest.mark.parametrize(
    "split,expected",
    [("train", 292508868), ("val", 2548024927), ("test", 2676412545)],
    ids=["train", "val", "test"],
)
def test_stable_seed_is_pinned_per_split(split: str, expected: int) -> None:
    assert _stable_seed(split) == expected


def test_stable_seed_is_repeatable_across_calls() -> None:
    first = _stable_seed("train")
    second = _stable_seed("train")
    assert first == second


def test_stable_seed_differs_per_split() -> None:
    seeds = {_stable_seed(split) for split in SPLITS}
    assert len(seeds) == len(SPLITS)


@pytest.mark.parametrize("split", SPLITS, ids=list(SPLITS))
def test_stable_seed_fits_in_32_bits(split: str) -> None:
    seed = _stable_seed(split)
    assert isinstance(seed, int)
    assert 0 <= seed <= 0xFFFFFFFF


# ------------------------------------------------------------------ dataset --
def test_len_matches_num_samples() -> None:
    assert len(_dataset(num_samples=17, seq_len=5, vocab_size=11)) == 17


def test_item_is_batch_dict_with_expected_shapes_and_dtype() -> None:
    ds = _dataset(num_samples=4, seq_len=6, vocab_size=13)
    item = ds[2]
    assert set(item) == {"input_ids", "labels"}
    assert tuple(item["input_ids"].shape) == (6,)
    assert tuple(item["labels"].shape) == (6,)
    assert item["input_ids"].dtype == torch.long
    assert item["labels"].dtype == torch.long


def test_item_values_stay_inside_vocab() -> None:
    ds = _dataset(num_samples=32, seq_len=16, vocab_size=7)
    assert ds.input_ids.min().item() >= 0
    assert ds.input_ids.max().item() < 7
    assert ds.labels.min().item() >= 0
    assert ds.labels.max().item() < 7


@pytest.mark.parametrize("index", [0, 1, 5], ids=["first", "second", "last"])
def test_getitem_returns_its_own_row(index: int) -> None:
    ds = _dataset(num_samples=6, seq_len=4)
    item = ds[index]
    assert torch.equal(item["input_ids"], ds.input_ids[index])
    assert torch.equal(item["labels"], ds.labels[index])


def test_two_instances_with_same_split_are_identical() -> None:
    first = _dataset(split="train", num_samples=8, seq_len=4)
    second = _dataset(split="train", num_samples=8, seq_len=4)
    assert torch.equal(first.input_ids, second.input_ids)
    assert torch.equal(first.labels, second.labels)


def test_train_and_val_splits_differ() -> None:
    train = _dataset(split="train", num_samples=8, seq_len=4)
    val = _dataset(split="val", num_samples=8, seq_len=4)
    assert not torch.equal(train.input_ids, val.input_ids)
    assert not torch.equal(train.labels, val.labels)


def test_labels_are_not_a_copy_of_input_ids() -> None:
    ds = _dataset(num_samples=16, seq_len=8)
    assert not torch.equal(ds.input_ids, ds.labels)


@pytest.mark.parametrize(
    "num_samples,seq_len,vocab_size",
    [(3, 2, 5), (256, 32, 1000), (10, 17, 2)],
    ids=["tiny", "defaults-size", "binary-vocab"],
)
def test_custom_sizes_are_honoured(num_samples: int, seq_len: int, vocab_size: int) -> None:
    ds = _dataset(num_samples=num_samples, seq_len=seq_len, vocab_size=vocab_size)
    assert len(ds) == num_samples
    assert tuple(ds.input_ids.shape) == (num_samples, seq_len)
    assert tuple(ds.labels.shape) == (num_samples, seq_len)
    assert ds.input_ids.max().item() < vocab_size
    assert ds.labels.max().item() < vocab_size
