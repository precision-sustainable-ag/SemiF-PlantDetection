import logging
import os
import sys

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf
from src.preprocess import main as preprocess


sys.path.append("src")

log = logging.getLogger(__name__)

TASK_REGISTRY = {
    "preprocess": preprocess
}

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)
    log.info(f"Starting mode: {cfg.mode}")

    if cfg.mode not in TASK_REGISTRY:
        log.error(f"Mode {cfg.mode} not found in TASK_REGISTRY")
        raise ValueError(f"Mode {cfg.mode} not found in TASK_REGISTRY")
    
    TASK_REGISTRY[cfg.mode](cfg)
    # log.info(f"Starting task {','.join(cfg.tasks)}")

    # for tsk in cfg.tasks:
    #     try:
    #         # Map task names to module names if needed
    #         task_mapping = {
    #             "training_dataset": "training_dataset.main",
    #             "download_images": "download_images.main"
    #         }
            
    #         task_function = task_mapping.get(tsk, f"{tsk}.main")
    #         task = get_method(task_function)
    #         task(cfg)

    #     except Exception as e:
    #         log.exception(f"Failed to execute task {tsk}: {e}")
    #         return


if __name__ == "__main__":
    run()
