"""Trainer factory for the imperative (non-Hydra) path.

This mirrors ``configs/trainer/default.yaml`` for callers that build a Trainer
directly instead of via Hydra ``instantiate``. LR schedules and gradient
clipping are not configured here yet; they live in the model/system only when
added.
"""

from typing import Any

import pytorch_lightning as pl

from memory_machines.callbacks.logging import MetricLoggerCallback


def _get(cfg, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dataclass, DictConfig, or mapping without KeyError."""
    if isinstance(cfg, dict):
        value = cfg.get(key, default)
    else:
        getter = getattr(cfg, "get", None)
        if callable(getter):
            value = getter(key, default)
        else:
            value = getattr(cfg, key, default)
    return default if value is None else value


def create_trainer(cfg) -> pl.Trainer:
    # pl.Callback is the supertype of every callback; MetricLoggerCallback and
    # anything instantiate() returns must fit it, so annotate at the boundary.
    callbacks: list[pl.Callback] = [MetricLoggerCallback()]
    extra = _get(cfg, "callbacks")
    if extra:
        callbacks.extend(extra)
    return pl.Trainer(
        max_epochs=_get(cfg, "max_epochs", 10),
        accelerator=_get(cfg, "accelerator", "auto"),
        devices=_get(cfg, "devices", "auto"),
        precision=_get(cfg, "precision", "bf16-mixed"),
        strategy=_get(cfg, "strategy", "auto"),
        log_every_n_steps=_get(cfg, "log_every_n_steps", 10),
        fast_dev_run=_get(cfg, "fast_dev_run", False),
        callbacks=callbacks,
    )
