"""Trainer factory. LR schedules and gradient clipping live here, never in models/."""

import pytorch_lightning as pl

from project.callbacks.logging import MetricLoggerCallback


def create_trainer(cfg) -> pl.Trainer:
    # pl.Callback is the supertype of every callback; MetricLoggerCallback and
    # anything instantiate() returns must fit it, so annotate at the boundary.
    callbacks: list[pl.Callback] = [MetricLoggerCallback()]
    if cfg.get("callbacks"):
        callbacks.extend(cfg.callbacks)
    return pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        strategy=cfg.strategy,
        log_every_n_steps=cfg.log_every_n_steps,
        fast_dev_run=cfg.get("fast_dev_run", False),
        callbacks=callbacks,
    )
