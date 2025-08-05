from omegaconf import DictConfig
import logging

from src.evaluating.evaluate_model import main as evaluate_model_main
from src.evaluating.benchmark_nms import main as benchmark_nms_main
from src.evaluating.visualize_nms import main as visualize_nms_main 

log = logging.getLogger(__name__)

TASK_REGISTRY = {
    "evaluate_model": evaluate_model_main,
    "benchmark_nms": benchmark_nms_main,
    "visualize_nms": visualize_nms_main,
}

def main(cfg: DictConfig):
    """
    Main entrypoint for evaluate mode.
    - evaluate_model
    - benchmark_nms
    """
    log.info(f"Tasks: {cfg.evaluate.tasks}")

    for task in cfg.evaluate.tasks:
        if task in TASK_REGISTRY:
            log.info(f"Running evaluate task: {task}")
            TASK_REGISTRY[task](cfg)
        else:
            log.error(f"Task {task} not found in TASK_REGISTRY")
            raise ValueError(f"Task {task} not found in TASK_REGISTRY")
