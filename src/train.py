from omegaconf import DictConfig
import logging
from src.training.prepare_dataset import main as prepare_dataset_main

log = logging.getLogger(__name__)
TASK_REGISTRY = {
    "prepare_dataset": prepare_dataset_main
}

def main(cfg: DictConfig):
    """
    Main entrypoint for train mode
    - prepare_dataset
    """
    for task in cfg.train.tasks:
        if task in TASK_REGISTRY:
            log.info(f"Running task: {task}")
            TASK_REGISTRY[task](cfg)
        else:
            log.error(f"Task {task} not found in TASK_REGISTRY")
            raise ValueError(f"Task {task} not found in TASK_REGISTRY")
        try:
            TASK_REGISTRY[task](cfg)
        except Exception as e:
            log.error(f"Failed to execute task {task}: {e}")
            raise ValueError(f"Failed to execute task {task}: {e}")

if __name__ == "__main__":
    main()