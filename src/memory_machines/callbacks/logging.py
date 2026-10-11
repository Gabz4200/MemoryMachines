"""Side-effect callbacks only. No plotting, saving, or telemetry in training_step."""

from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


class MetricLoggerCallback(pl.Callback):
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == 0 and trainer.logger:
            # Generate sample inference and log cleanly via trainer.logger.
            pass
