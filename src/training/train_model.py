from omegaconf import DictConfig
import logging
import os
from pathlib import Path
import yaml
from datetime import datetime
import torch
from ultralytics import YOLO
from ..utils.utils import find_most_recent_dataset_path

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
        self.data_path = (Path(self.cfg.train.model_data))
        log.info(f"Using dataset at {self.data_path}")
        self.output_dir = Path(self.cfg.train.models_path)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create data.yaml
        self.create_data_yaml()
        
    def create_data_yaml(self):
        """Create data.yaml file required by YOLO"""
        data = {
            'path': str(self.data_path),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 3,  # Number of classes
            'names': {
                0: 'plant',
                1: 'non_target',
                2: 'colorchecker'
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
        
        # Load a pretrained YOLO model
        model = YOLO(self.cfg.train.model_name)
        
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