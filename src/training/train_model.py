import logging
import os
from pathlib import Path

import torch
import yaml
from omegaconf import DictConfig
from ultralytics import YOLO
from src.utils.utils import get_latest_checkpoint
from src.utils.constants import CLASS_MAPPING


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
        
    def create_data_yaml(self):
        """Create data.yaml file required by YOLO"""
        if self.cfg.train.prepare_dataset.ignore_non_targets:
            # exclude non_target
            filtered_mapping = {
                k: v for k, v in CLASS_MAPPING.items() if k != "non_target"
            }
        else:
            filtered_mapping = CLASS_MAPPING

        # Invert to get index → name
        names_list = sorted(filtered_mapping.items(), key=lambda x: x[1])
        names = {i: name for i, (name, _) in enumerate(names_list)}

        data = {
            'path': str(self.data_path),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(names),
            'names': names
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
                log.error("train_from_checkpoint is True but no checkpoint was provided.")
                raise FileNotFoundError(
                    "No checkpoint specified in cfg.paths.train.checkpoint while train_from_checkpoint=True. "
                    "Please specify a checkpoint path or set train_from_checkpoint to False."
                )
        else:
            checkpoint = self.cfg.train.model_name
            log.info(f"Starting fresh from model_name: {checkpoint}")

        model = YOLO(checkpoint)

        # Ensure model is trainable
        model.model.train()
        for p in model.model.parameters():
            p.requires_grad = True

        trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        log.info(f"Trainable parameters: {trainable:,}")

        results = model.train(
            data=str(self.yaml_path),
            epochs=self.cfg.train.epochs,
            imgsz=self.cfg.train.image_size,
            batch=self.cfg.train.batch_size,
            device = self.cfg.train.device if torch.cuda.is_available() else 'cpu',
            project=str(self.output_dir),
            name='run',
            save=True,
            flipud=self.cfg.train.flipud,
            fliplr=self.cfg.train.fliplr,
            mosaic=self.cfg.train.mosaic,
            mixup=self.cfg.train.mixup,
            scale=self.cfg.train.scale,
            shear=self.cfg.train.shear,
            degrees=self.cfg.train.degrees,
            perspective=self.cfg.train.perspective,
            translate=self.cfg.train.translate,
            hsv_h=self.cfg.train.hsv_h,
            hsv_s=self.cfg.train.hsv_s,
            hsv_v=self.cfg.train.hsv_v,
            lr0=self.cfg.train.lr0,
            lrf=self.cfg.train.lrf,
            weight_decay=self.cfg.train.weight_decay,
            warmup_epochs=self.cfg.train.warmup_epochs,
            patience=self.cfg.train.patience,
            cache=self.cfg.train.cache,
            box=self.cfg.train.box,
            cls=self.cfg.train.cls,
        )
        
        # Save the model
        model.export()
        
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