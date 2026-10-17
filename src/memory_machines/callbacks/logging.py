"""Side-effect callbacks only. No plotting, saving, or telemetry in training_step."""

import pytorch_lightning as pl


class MetricLoggerCallback(pl.Callback):
    """Lightweight validation hook.

    Kept as a no-op seam so side-effect logging can be added without touching
    the model or ``training_step``. It intentionally registers no per-batch
    hook, so it adds no per-step overhead until implemented.
    """

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Placeholder: hook up sample generation / custom logging here.
        return
