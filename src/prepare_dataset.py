from omegaconf import DictConfig
from src.utils.utils import get_human_annotations
from pathlib import Path
import logging

log = logging.getLogger(__name__)

class PrepareDataset:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.lts_human_annotations = self.cfg.paths.lts_human_annotations
        self.human_annotations = get_human_annotations(self.lts_human_annotations)

    def run(self):
        log.info(f"Found {len(self.human_annotations)} human annotations")


def main(cfg: DictConfig):
    prepare_dataset = PrepareDataset(cfg)
    prepare_dataset.run()

if __name__ == "__main__":
    main()
