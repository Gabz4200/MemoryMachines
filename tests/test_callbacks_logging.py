"""Side-effect callback seam.

``MetricLoggerCallback`` must stay a harmless no-op: it is attached to every
trainer built by ``create_trainer``, so a broken hook would break validation on
every run.
"""

from typing import Any

import pytorch_lightning as pl

from memory_machines.callbacks.logging import MetricLoggerCallback


def test_metric_logger_callback_is_a_lightning_callback() -> None:
    callback = MetricLoggerCallback()
    assert isinstance(callback, pl.Callback)
    assert isinstance(callback, MetricLoggerCallback)


def test_on_validation_epoch_end_returns_none_with_none_args() -> None:
    callback = MetricLoggerCallback()
    trainer: Any = None
    module: Any = None
    assert callback.on_validation_epoch_end(trainer, module) is None


def test_on_validation_epoch_end_returns_none_with_real_objects() -> None:
    callback = MetricLoggerCallback()

    class _Trainer:
        current_epoch = 0

    class _Module:
        def __getattr__(self, name: str) -> Any:  # any attribute access is a bug here
            raise AssertionError(f"callback touched module.{name}")

    trainer: Any = _Trainer()
    module: Any = _Module()
    assert callback.on_validation_epoch_end(trainer, module) is None


def test_on_validation_epoch_end_is_callable_repeatedly() -> None:
    callback = MetricLoggerCallback()
    trainer: Any = None
    module: Any = None
    assert callback.on_validation_epoch_end(trainer, module) is None
    assert callback.on_validation_epoch_end(trainer, module) is None
