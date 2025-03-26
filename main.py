import logging
import os
import sys

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

sys.path.append("src")

# Import task modules for registration
import training_dataset
import download_images

log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)
    log.info(f"Starting task {','.join(cfg.tasks)}")

    for tsk in cfg.tasks:
        try:
            # Map task names to module names if needed
            task_mapping = {
                "training_dataset": "training_dataset.main",
                "download_images": "download_images.main"
            }
            
            task_function = task_mapping.get(tsk, f"{tsk}.main")
            task = get_method(task_function)
            task(cfg)

        except Exception as e:
            log.exception(f"Failed to execute task {tsk}: {e}")
            return


if __name__ == "__main__":
    run()
