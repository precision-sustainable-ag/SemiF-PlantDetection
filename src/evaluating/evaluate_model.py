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
        Resolves which checkpoint to use for evaluation:
        Priority:
        1. cfg.evaluate.custom_path (if set & exists)
        2. cfg.evaluate.run_version (if set & exists)
        3. Latest checkpoint in model_dir
        """
        # custom path
        custom_ckpt = getattr(self.cfg.evaluate, "custom_path", None)
        if custom_ckpt:
            custom_ckpt = Path(custom_ckpt)
            if custom_ckpt.exists():
                log.info(f"Using custom evaluation checkpoint: {custom_ckpt}")
                return custom_ckpt
            else:
                raise FileNotFoundError(f"Custom evaluation checkpoint does not exist: {custom_ckpt}")

        # run version
        run_version = getattr(self.cfg.evaluate, "run_version", None)
        if run_version is not None:
            if run_version in [0, 1]:
                run_dir = self.model_dir / "run"
            else:
                run_dir = self.model_dir / f"run{run_version}"

            ckpt_path = run_dir / "weights" / "best.pt"
            if ckpt_path.exists():
                log.info(f"Using checkpoint from run_version={run_version}: {ckpt_path}")
                return ckpt_path
            else:
                raise FileNotFoundError(f"Checkpoint for run_version={run_version} does not exist at: {ckpt_path}")

        # latest available
        latest = get_latest_checkpoint(self.model_dir)
        if latest:
            log.info(f"Using latest available checkpoint: {latest}")
            return latest

        # None found
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