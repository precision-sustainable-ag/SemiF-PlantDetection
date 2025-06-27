from omegaconf import DictConfig
import logging
from src.exporting.cvat_exporter import main as cvat_exporter_main
from src.exporting.rebuild_dataset import main as rebuild_dataset_main

log = logging.getLogger(__name__)

TASK_REGISTRY = {
    "cvat_exporter": cvat_exporter_main,
    "rebuild_dataset": rebuild_dataset_main,
}

def main(cfg: DictConfig):
    """
    Main entrypoint for export mode
    - cvat_exporter
    """
    log.info(f"Tasks: {cfg.export.tasks}")

    for task in cfg.export.tasks:
        if task in TASK_REGISTRY:
            log.info(f"Running export task: {task}")
            TASK_REGISTRY[task](cfg)
        else:
            log.error(f"Task {task} not found in TASK_REGISTRY")
            raise ValueError(f"Task {task} not found in TASK_REGISTRY")
