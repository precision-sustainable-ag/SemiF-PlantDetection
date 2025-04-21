from omegaconf import DictConfig
import logging
from src.preprocessing.training_dataset import main as training_dataset_main
from src.preprocessing.download_images import main as download_images_main
from src.preprocessing.cvat_formatter import main as cvat_formatter_main
from src.preprocessing.cvat_importer import main as cvat_importer_main

log = logging.getLogger(__name__)

# Map task names to their corresponding functions
TASK_REGISTRY = {
    "training_dataset": training_dataset_main,
    "download_images": download_images_main,
    "cvat_formatter": cvat_formatter_main,
    "cvat_importer": cvat_importer_main
}

def main(cfg: DictConfig):
    """
    Main entrypoint for preprocess mode
    - training_dataset
    - download_images
    - cvat_formatter
    - cvat_importer
    """
    log.info(f"Tasks: {cfg.preprocess.tasks}")
    
    for task in cfg.preprocess.tasks:
        if task in TASK_REGISTRY:
            log.info(f"Running task: {task}")
            TASK_REGISTRY[task](cfg)
        else:
            log.error(f"Task {task} not found in TASK_REGISTRY")
            raise ValueError(f"Task {task} not found in TASK_REGISTRY")