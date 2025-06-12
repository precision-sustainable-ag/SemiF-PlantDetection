import logging
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from src.preprocess import main as preprocess
from src.train import main as train
from src.export import main as export

sys.path.append("src")

log = logging.getLogger(__name__)

MODE_REGISTRY = {
    "preprocess": preprocess,
    "train": train,
    "export" : export
}

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)
    log.info(f"************************************************")
    log.info(f"Pipeline mode: {cfg.mode}")
    log.info(f"************************************************")

    if cfg.mode not in MODE_REGISTRY:
        log.error(f"Mode {cfg.mode} not found in MODE_REGISTRY")
        raise ValueError(f"Mode {cfg.mode} not found in MODE_REGISTRY")
    try:
        MODE_REGISTRY[cfg.mode](cfg)
    except Exception as e:
        log.error(f"Failed to execute pipeline mode {cfg.mode}: {e}")
        raise ValueError(f"Failed to execute pipeline mode {cfg.mode}")


if __name__ == "__main__":
    run()
