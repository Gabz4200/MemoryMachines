"""Load a checkpoint and run the test set."""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from project.modules.lightning_module import LitResearchSystem


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    datamodule = instantiate(cfg.data)
    datamodule.setup(stage="test")
    system = LitResearchSystem.load_from_checkpoint(cfg.checkpoint)
    trainer = instantiate(cfg.trainer)
    trainer.test(system, datamodule=datamodule)


if __name__ == "__main__":
    main()
