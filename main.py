import logging
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from src.preprocess import main as preprocess


sys.path.append("src")

log = logging.getLogger(__name__)

MODE_REGISTRY = {
    "preprocess": preprocess
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
        log.exception(f"Failed to execute pipelinemode {cfg.mode}: {e}")
        return


if __name__ == "__main__":
    run()
