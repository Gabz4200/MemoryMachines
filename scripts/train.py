"""Declarative Hydra entry point: instantiate() wires model, system, data, callbacks, trainer.

Override the dev-run flag Hydra-style:
  uv run python scripts/train.py trainer.fast_dev_run=true
"""

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    # 1. Instantiate the pure PyTorch model (receives kwargs, NOT DictConfig).
    model = instantiate(cfg.model)
    # 2. Instantiate the Lightning system wrapper, injecting the pure model.
    system = instantiate(cfg.system, model=model)
    # 3. Instantiate the DataModule.
    datamodule = instantiate(cfg.data)
    # 4. Instantiate callbacks.
    callbacks = [instantiate(c) for c in cfg.callbacks.values()] if "callbacks" in cfg else []
    # 5. Instantiate the Trainer.
    trainer: pl.Trainer = instantiate(cfg.trainer, callbacks=callbacks)
    # 6. Fit.
    trainer.fit(system, datamodule=datamodule)


if __name__ == "__main__":
    main()
