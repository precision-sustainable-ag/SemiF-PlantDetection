import logging
import os
from pathlib import Path

import torch
import yaml
from omegaconf import DictConfig
from ultralytics import YOLO


log = logging.getLogger(__name__)

class TrainModel:
    def __init__(self, cfg: DictConfig, data_path=None):
        """
        Initialize training for YOLOv11 model
        
        Args:
            cfg (DictConfig): Configuration
            data_path (Path, optional): Path to the dataset. If None, will use the path from the last run
        """
        self.cfg = cfg
        
        # Set up paths
        self.data_path = (Path(self.cfg.paths.train.model_data_dir))
        log.info(f"Using dataset at {self.data_path}")
        self.output_dir = Path(self.cfg.paths.train.model_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create data.yaml
        self.create_data_yaml()

    def get_latest_checkpoint(self):
        """
        Finds the most recent best.pt checkpoint under paths.train.model_dir
        """
        runs = sorted(
            self.output_dir.glob("run*"),
            key=os.path.getmtime,
            reverse=True
        )
        for run in runs:
            best_ckpt = run / "weights" / "best.pt"
            if best_ckpt.exists():
                log.info(f"Found latest checkpoint: {best_ckpt}")
                return best_ckpt
        log.warning("No checkpoint found in model_dir")
        return None
        
    def create_data_yaml(self):
        """Create data.yaml file required by YOLO"""
        data = {
            'path': str(self.data_path),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 2,  # Number of classes
            'names': {
                0: 'plant',
                1: 'colorchecker'
            }
        }
        
        yaml_path = self.data_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        log.info(f"Created data.yaml at {yaml_path}")
        self.yaml_path = yaml_path
        
    def train(self):
        """Train the YOLOv11 model"""
        log.info("Starting YOLOv11 training")
        
        if self.cfg.train.train_from_checkpoint:
            if self.cfg.paths.train.checkpoint:
                checkpoint = self.cfg.paths.train.checkpoint
                log.info(f"Resuming from specified checkpoint: {checkpoint}")
            else:
                checkpoint = self.get_latest_checkpoint()
                if checkpoint:
                    log.info(f"Resuming from latest checkpoint: {checkpoint}")
                else:
                    checkpoint = self.cfg.train.model_name
                    log.info(f"No checkpoint found — starting from model_name: {checkpoint}")
        else:
            checkpoint = self.cfg.train.model_name
            log.info(f"Starting fresh from model_name: {checkpoint}")

        model = YOLO(checkpoint)
        
        # Train the model
        results = model.train(
            data=str(self.yaml_path),
            epochs=self.cfg.train.epochs,
            imgsz=self.cfg.train.image_size,
            batch=self.cfg.train.batch_size,
            # workers=self.cfg.train.num_workers,
            # TODO: change device id (s) to be in config to use different gpu(s)
            device=0 if torch.cuda.is_available() else 'cpu',
            # device='cpu',
            project=str(self.output_dir),
            name='run',
            save=True,
            # patience=self.cfg.train.patience,
            # lr0=self.cfg.train.lr,
            # weight_decay=self.cfg.train.weight_decay
        )
        
        # Save the model
        model.export()
        log.info(f"Model trained and saved to {results.save_dir}")
        
        return results
        
def main(cfg: DictConfig, data_path=None):
    """
    Main entrypoint for model training
    
    Args:
        cfg (DictConfig): Configuration
        data_path (Path, optional): Path to the dataset
    """
    trainer = TrainModel(cfg, data_path)
    results = trainer.train()
    return results

if __name__ == "__main__":
    main() 