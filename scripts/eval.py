"""Load a checkpoint and run the test set."""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from memory_machines.modules.lightning_module import LitResearchSystem


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    datamodule = instantiate(cfg.data)
    datamodule.setup(stage="test")
    # The model is injected via DI and is not serialized in the checkpoint, so
    # rebuild it from the same config and hand it to load_from_checkpoint.
    model = instantiate(cfg.model)
    system = LitResearchSystem.load_from_checkpoint(cfg.checkpoint, model=model)
    trainer = instantiate(cfg.trainer)
    trainer.test(system, datamodule=datamodule)


if __name__ == "__main__":
    main()
