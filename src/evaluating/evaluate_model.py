from pathlib import Path
import os
import logging
from omegaconf import DictConfig
from ultralytics import YOLO

from src.utils.utils import get_latest_checkpoint

log = logging.getLogger(__name__)


class EvaluateModel:
    def __init__(self, cfg: DictConfig):
        """
        Initialize evaluation for YOLO model

        Args:
            cfg (DictConfig): Hydra configuration
        """
        self.cfg = cfg

        self.model_dir = Path(self.cfg.paths.train.model_dir)
        self.data_yaml = self.cfg.paths.evaluate.data_yaml
        self.save_dir = Path(self.cfg.paths.evaluate.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        self.split = self.cfg.evaluate.split
        self.save_json = self.cfg.evaluate.save_json

        self.weights = self._resolve_checkpoint()
        if not self.weights:
            raise FileNotFoundError(
                f"No valid checkpoint found in {self.model_dir} or specified in config."
            )

    def _resolve_checkpoint(self) -> Path | None:
        """
        Decides which checkpoint to use:
        - if cfg.paths.train.checkpoint is set → use it if exists
        - else → fallback to latest checkpoint
        """
        ckpt = self.cfg.paths.train.checkpoint
        if ckpt:
            ckpt = Path(ckpt)
            if ckpt.exists():
                log.info(f"Using specified checkpoint: {ckpt}")
                return ckpt
            else:
                log.warning(f"Specified checkpoint does not exist: {ckpt}")
        latest = get_latest_checkpoint(self.model_dir)
        if latest:
            log.info(f"Using latest checkpoint: {latest}")
            return latest
        return None

    def evaluate(self):
        """
        Run YOLO evaluation
        """
        log.info(f"Evaluating {self.weights} on {self.data_yaml} [split={self.split}]")

        model = YOLO(self.weights)

        results = model.val(
            data=self.data_yaml,
            split=self.split,
            save_json=self.save_json,
            project=self.save_dir,
            name="evaluation_results"
        )

        log.info(f"Evaluation complete. Results saved to: {results.save_dir}")
        log.info(f"mAP50: {results.box.map50:.4f}, mAP50-95: {results.box.map:.4f}")
        return results


def main(cfg: DictConfig):
    """
    Entry point for evaluation task
    """
    evaluator = EvaluateModel(cfg)
    results = evaluator.evaluate()
    return results


if __name__ == "__main__":
    main()