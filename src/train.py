from omegaconf import DictConfig
import logging
from src.training.prepare_dataset import main as prepare_dataset_main
from src.training.train_model import main as train_model_main

log = logging.getLogger(__name__)
TASK_REGISTRY = {
    "prepare_dataset": prepare_dataset_main,
    "train_model": train_model_main
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

if __name__ == "__main__":
    main()