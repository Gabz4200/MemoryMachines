"""Trainer factory and its tolerant config reader.

``create_trainer`` mirrors ``configs/trainer/default.yaml`` for callers that
build a Trainer directly. Only construction is exercised; ``fit`` is never
called, so nothing trains and no checkpoint is written.
"""

from pathlib import Path
from typing import Any

import pytest
import pytorch_lightning as pl

from memory_machines.callbacks.logging import MetricLoggerCallback
from memory_machines.training.engine import _get, create_trainer

FAST = {"accelerator": "cpu", "precision": "32-true"}


class _Mapping:
    """Mapping-like object exposing ``.get`` (DictConfig-style access)."""

    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values
        self.get_calls: list[tuple[str, Any]] = []

    def get(self, key: str, default: Any = None) -> Any:
        self.get_calls.append((key, default))
        return self._values.get(key, default)


class _Record:
    """Plain object without ``.get``; reads fall through to ``getattr``."""

    def __init__(self, **kw: Any) -> None:
        for name, value in kw.items():
            setattr(self, name, value)


class _DummyCallback(pl.Callback):
    pass


# --------------------------------------------------------------- config read --
@pytest.mark.parametrize(
    "cfg,key,default,expected",
    [
        ({"max_epochs": 3}, "max_epochs", 10, 3),
        ({"max_epochs": None}, "max_epochs", 10, 10),
        ({"other": 1}, "max_epochs", 10, 10),
        ({}, "max_epochs", None, None),
    ],
    ids=["present", "present-none", "missing", "missing-no-default"],
)
def test_get_reads_dicts(cfg: dict[str, Any], key: str, default: Any, expected: Any) -> None:
    assert _get(cfg, key, default) == expected


@pytest.mark.parametrize(
    "values,expected",
    [({"max_epochs": 3}, 3), ({"max_epochs": None}, 10), ({}, 10)],
    ids=["present", "present-none", "missing"],
)
def test_get_uses_mapping_get(values: dict[str, Any], expected: int) -> None:
    mapping = _Mapping(values)
    assert _get(mapping, "max_epochs", 10) == expected
    assert mapping.get_calls == [("max_epochs", 10)]


@pytest.mark.parametrize(
    "record,expected",
    [({"max_epochs": 3}, 3), ({"max_epochs": None}, 10), ({}, 10)],
    ids=["present", "present-none", "missing"],
)
def test_get_falls_back_to_attribute(record: dict[str, Any], expected: int) -> None:
    assert _get(_Record(**record), "max_epochs", 10) == expected


def test_get_treats_present_none_as_default() -> None:
    # A None value means "not configured", not "configure the value None".
    assert _get({"callbacks": None}, "callbacks", ["fallback"]) == ["fallback"]


def test_get_does_not_treat_falsey_values_as_missing() -> None:
    assert _get({"fast_dev_run": False}, "fast_dev_run", True) is False
    assert _get(_Record(fast_dev_run=False), "fast_dev_run", True) is False


# ------------------------------------------------------------- create_trainer --
@pytest.mark.parametrize(
    "cfg",
    [
        dict(FAST),
        _Mapping(dict(FAST)),
        _Record(**FAST),
    ],
    ids=["dict", "mapping", "object"],
)
def test_create_trainer_accepts_every_config_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cfg: Any) -> None:
    monkeypatch.chdir(tmp_path)
    trainer: Any = create_trainer(cfg)
    assert isinstance(trainer, pl.Trainer)


def test_create_trainer_always_includes_metric_logger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    trainer: Any = create_trainer(dict(FAST))
    metric_loggers = [cb for cb in trainer.callbacks if isinstance(cb, MetricLoggerCallback)]
    assert len(metric_loggers) == 1


def test_create_trainer_appends_extra_callbacks_after_metric_logger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    first, second = _DummyCallback(), _DummyCallback()
    trainer: Any = create_trainer({**FAST, "callbacks": [first, second]})
    ours = [cb for cb in trainer.callbacks if isinstance(cb, (MetricLoggerCallback, _DummyCallback))]
    assert ours == [trainer.callbacks[0], first, second]
    assert isinstance(ours[0], MetricLoggerCallback)


def test_create_trainer_ignores_empty_callback_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    trainer: Any = create_trainer({**FAST, "callbacks": []})
    assert sum(isinstance(cb, MetricLoggerCallback) for cb in trainer.callbacks) == 1


def test_create_trainer_applies_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    trainer: Any = create_trainer({"accelerator": "cpu"})
    assert trainer.max_epochs == 10
    assert trainer.log_every_n_steps == 10
    assert trainer.fast_dev_run is False
    assert trainer.precision == "bf16-mixed"


def test_create_trainer_applies_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    trainer: Any = create_trainer({**FAST, "max_epochs": 3, "log_every_n_steps": 5})
    assert trainer.max_epochs == 3
    assert trainer.log_every_n_steps == 5
    assert trainer.precision == "32-true"


def test_create_trainer_passes_fast_dev_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    trainer: Any = create_trainer({**FAST, "fast_dev_run": True})
    assert trainer.fast_dev_run is True
