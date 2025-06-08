import os
import json
import logging
import pandas as pd
import zipfile
from pathlib import Path
from typing import List, Dict
from omegaconf import DictConfig
import cv2
import shutil
from multiprocessing import Pool, cpu_count

from src.utils.utils import find_most_recent_dataset_path, convert_bbox_to_yolo_format

# Setup logging
log = logging.getLogger(__name__)

class CVATFormatter:
    """
    Class to format images and annotations in the Ultralytics YOLO Detection format for CVAT.
    
    The Ultralytics YOLO Detection format requires:
    - Bounding box format: [class_id, center_x, center_y, width, height] (all normalized)
    - Directory structure: 
        - images/train/
        - labels/train/
        - data.yaml 
        - train.txt
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the CVAT formatter.
        
        Args:
            cfg (DictConfig): Hydra configuration
        """
        self.dataset_path = Path(cfg.database.dataset.output_path)
        
        self.csv_file_path = Path(self.dataset_path, "training_images.csv")
        
        # Image source directory
        self.image_source_folder = Path(cfg.images.output_path)
        
        # CVAT output directory
        self.cvat_output_folder = Path(cfg.cvat.output_path)
        
        # Create CVAT output directory structure
        self.images_dir = self.cvat_output_folder / "images" / "train"
        self.labels_dir = self.cvat_output_folder / "labels" / "train"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        # Set default image dimensions if not available in image files
        self.default_image_width = cfg.cvat.default_image_width
        self.default_image_height = cfg.cvat.default_image_height
        
        # Get resize factor from config (default to 1.0 if not specified)
        self.resize_factor = cfg.cvat.get('resize_factor', 1.0)
        
        # Store class mapping from config
        self.class_mapping = cfg.cvat.class_mapping

        # Parallel processing configuration
        self.parallel = cfg.cvat.get('parallel', False)
        self.parallel_workers = min(cfg.cvat.get('parallel_workers', cpu_count()), cpu_count())
        
        log.info(f"Initialized CVAT formatter with output directory: {self.cvat_output_folder}")
        log.info(f"Using class mapping from config: {self.class_mapping}")
        log.info(f"Images will be resized by factor: {self.resize_factor}")
        if self.parallel:
            log.info(f"Parallel processing: {self.parallel} with {self.parallel_workers} workers")

    def cleanup(self):
        """
        Destructor that deletes the CVAT dataset folder when the object is deleted.
        """
        try:
            if self.cvat_output_folder.exists():
                shutil.rmtree(self.cvat_output_folder)
                log.info(f"Deleted CVAT dataset folder: {self.cvat_output_folder}")
        except Exception as e:
            log.error(f"Error deleting CVAT dataset folder: {e}")

    def load_dataset(self) -> pd.DataFrame:
        """
        Load the dataset from the CSV file.
        
        Returns:
            pd.DataFrame: DataFrame containing training image information
        """
        try:
            df = pd.read_csv(self.csv_file_path)
            log.info(f"Successfully loaded dataset from {self.csv_file_path}")
            return df
        except FileNotFoundError as e:
            log.error(f"CSV file not found: {self.csv_file_path} - {e}")
            raise
        except Exception as e:
            log.error(f"Error loading CSV file: {self.csv_file_path} - {e}")
            raise

    def process_image(self, row: pd.Series) -> None:
        """
        Process a single image and its annotations.
        
        Args:
            row (pd.Series): Row from DataFrame containing image information
        """
        image_id = row['image_id']
        
        # Source image path
        source_image_path = self.image_source_folder / f"{image_id}.jpg"
        
        # Check if source image exists
        if not source_image_path.exists():
            log.warning(f"Source image not found: {source_image_path}")
            return
        
        # Destination image path
        dest_image_path = self.images_dir / f"{image_id}.jpg"
        
        # Load and resize image
        image = cv2.imread(str(source_image_path))
        if image is None:
            log.warning(f"Failed to load image: {source_image_path}")
            return
        
        # Get original dimensions
        original_height, original_width = image.shape[:2]
        
        # Calculate resize factor (e.g., 0.75 for 75% of original size)
        new_width = int(original_width * self.resize_factor)
        new_height = int(original_height * self.resize_factor)
        
        # Resize image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Save resized image
        cv2.imwrite(str(dest_image_path), resized_image)
        
        # Parse annotations
        try:
            annotations = json.loads(row['annotations'])
        except (json.JSONDecodeError, TypeError):
            log.error(f"Error parsing annotations for image {image_id}")
            return
        
        # Create annotation file
        annotation_file_path = self.labels_dir / f"{image_id}.txt"
        
        with open(annotation_file_path, 'w') as f:
            for annotation in annotations:
                # Get bounding box
                bbox = annotation.get('bbox_xywh')
                if not bbox:
                    continue
                
                # Scale bounding box coordinates based on resize factor
                x, y, width, height = bbox
                scaled_bbox = [
                    int(x * self.resize_factor),
                    int(y * self.resize_factor),
                    int(width * self.resize_factor),
                    int(height * self.resize_factor)
                ]
                
                # Apply class mapping from config
                if annotation.get('non_target_weed') is True:
                    if annotation.get('non_target_weed_pred_conf') > 0.99:
                        mapped_class_id = int(self.class_mapping.non_target)
                    else:
                        mapped_class_id = int(self.class_mapping.plant)
                elif annotation.get('category_class_id') == 28:
                    mapped_class_id = int(self.class_mapping.color_checker)
                else:
                    mapped_class_id = int(self.class_mapping.plant)
                
                center_x, center_y, norm_width, norm_height = convert_bbox_to_yolo_format(
                    scaled_bbox, new_width, new_height
                )
                
                # Write to annotation file with mapped class ID
                f.write(f"{mapped_class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
        
        log.debug(f"Processed image {image_id} (resized to {new_width}x{new_height})")

    def process_img_wrapper(self, row):
        try:
            self.process_image(row)
        except Exception as e:
            log.error(f"Error processing image {row['image_id']}")
    def create_train_txt(self) -> None:
        """
        Create the train.txt file containing paths to all images.
        """
        train_txt_path = self.cvat_output_folder / "train.txt"
        
        with open(train_txt_path, 'w') as f:
            for image_file in self.images_dir.glob('*.jpg'):
                relative_path = f"images/train/{image_file.name}"
                f.write(f"{relative_path}\n")
        
        log.info(f"Created train.txt with {len(list(self.images_dir.glob('*.jpg')))} images")

    def create_data_yaml(self, class_names: Dict[int, str]) -> None:
        """
        Create the data.yaml configuration file.
        
        Args:
            class_names (Dict[int, str]): Dictionary mapping class IDs to class names
        """
        data_yaml_path = self.cvat_output_folder / "data.yaml"
        
        data = {
            "path": "./",
            "train": "train.txt",
            "names": class_names
        }
        
        with open(data_yaml_path, 'w') as f:
            # Simple YAML formatting
            f.write("path: ./\n")
            f.write("train: train.txt\n\n")
            f.write("# Classes\n")
            f.write("names:\n")
            for class_id, class_name in class_names.items():
                f.write(f"  {class_id}: {class_name}\n")
        
        log.info(f"Created data.yaml with {len(class_names)} classes")

    def get_unique_class_ids(self, df: pd.DataFrame) -> Dict[int, str]:
        """
        Return the class mapping defined in the configuration.
        
        Args:
            df (pd.DataFrame): DataFrame containing image information (not used)
            
        Returns:
            Dict[int, str]: Dictionary mapping class IDs to class names
        """
        # Create class mapping from config
        class_names = {
            int(self.class_mapping.plant): "plant",
            int(self.class_mapping.non_target): "non_target",
            int(self.class_mapping.color_checker): "colorchecker"
        }
        
        return class_names

    def format_for_cvat(self) -> None:
        """
        Format the dataset for CVAT import in Ultralytics YOLO Detection format.
        """
        log.info("Starting CVAT formatting process")
        
        # Load dataset
        df = self.load_dataset()
        
        # Get unique class IDs
        class_names = self.get_unique_class_ids(df)

        # Process images in parallel or sequentially
        if self.parallel:
            log.info(f"Processing images in parallel with {self.parallel_workers} workers")
            with Pool(processes=self.parallel_workers) as pool:
                pool.map(self.process_image, [row for _, row in df.iterrows()])
        else:
            log.info("Processing images sequentially")
            for _, row in df.iterrows():
                self.process_image(row)
        
        # Create train.txt
        self.create_train_txt()
        
        # Create data.yaml
        self.create_data_yaml(class_names)
        
        # Create zip archive
        self.create_zip_archive()
        
        log.info(f"CVAT formatting completed. Output saved to {self.cvat_output_folder}")

    def create_zip_archive(self) -> None:
        """
        Create a zip archive of the CVAT dataset.
        """
        # Output zip file path
        zip_path = self.cvat_output_folder.parent / f"{self.cvat_output_folder.name}.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add all files in the CVAT output folder
            for root, _, files in os.walk(self.cvat_output_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.cvat_output_folder.parent)
                    zipf.write(file_path, arcname)
        
        log.info(f"Created zip archive: {zip_path}")
    
    

def main(cfg: DictConfig) -> None:
    """
    Main entry point for generating a training dataset.
    Steps:
    - Load dataset (by identifying most recent dataset csv file)
    - Get unique class IDs
    - Resize images, change bbox coordinates accordingly
    - Convert bbox coordinates to YOLO format
    - Create cvat dataset (in specified folder structure)
    - Create zip archive of cvat dataset

    Args:
        cfg (DictConfig): Hydra configuration
    """
    
    log.info("Starting CVAT formatter task")
    
    formatter = CVATFormatter(cfg)
    formatter.format_for_cvat()
    formatter.cleanup()
    
    log.info("CVAT formatter task completed")


if __name__ == "__main__":
    main() 