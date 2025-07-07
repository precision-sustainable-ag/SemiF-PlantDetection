import os
import cv2
import json
import shutil
import logging
import pandas as pd

from pathlib import Path
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig

from src.utils.constants import CLASS_MAPPING
from src.utils.utils import (
    get_annotated_image_ids,
    find_most_recent_dataset_path,
    convert_bbox_to_yolo_format
)

log = logging.getLogger(__name__)

class PrepareDataset:
    def __init__(self, cfg: DictConfig):
        # TODO: save dataset to a different location - causing issues with getting latest csv
        self.cfg = cfg
        self.human_annotations = get_annotated_image_ids(self.cfg.paths.lts_human_annotations)
        self.dataset_path = find_most_recent_dataset_path(self.cfg.paths.preprocess.csv_dir)
        self.data_csv = self.dataset_path / 'training_images.csv'
        self.class_mapping = CLASS_MAPPING

        self.random_seed = self.cfg.train.random_seed
        self.validation_split = self.cfg.train.validation_split

        self.train_data_path = Path(self.cfg.paths.train.model_data_dir)
        os.makedirs(self.train_data_path, exist_ok=True)

        self.preprocess_images_path = Path(self.cfg.paths.preprocess.image_dir)
        self.preprocess_annotations_path = Path(self.cfg.paths.preprocess.label_dir)

        # Parallel processing configuration
        self.parallel = self.cfg.train.get('parallel', False)
        self.parallel_workers = min(self.cfg.train.get('parallel_workers', cpu_count()), cpu_count())
        if self.parallel:
            log.info(f"Parallel processing enabled with {self.parallel_workers} workers")

        if self.cfg.train.prepare_dataset.ignore_non_targets:
            log.info("Preparing training dataset: manual annotations will be filtered to ignore non_target labels.")
        else:
            log.info("Preparing training dataset: manual annotations will include all labels, including non_target.")

        self.remove_after_split = self.cfg.train.prepare_dataset.remove_after_split  # Flag to indicate that this is a prepare dataset run
        
    def _cleanup_preprocess_dirs(self):
        if self.remove_after_split:
            # Sanity check to avoid accidental deletion of important directories
            forbidden_keywords = [
                'screberg', 'longterm_images2', 'GROW_DATA',
                'longterm_images', 'semifield-upload',
                'semifield-developed-images', 'semifield-cutouts'
            ]

            try:
                image_path = self.preprocess_images_path
                annotations_path = self.preprocess_annotations_path

                # Check if both exist
                if image_path.exists() and annotations_path.exists():
                    # Check if both are safe to delete
                    if (
                        all(keyword not in image_path.stem for keyword in forbidden_keywords) and
                        all(keyword not in annotations_path.stem for keyword in forbidden_keywords)
                    ):
                        shutil.rmtree(image_path)
                        log.info(f"Deleted preprocess image directory: {image_path}")
                        shutil.rmtree(annotations_path)
                        log.info(f"Deleted preprocess annotations directory: {annotations_path}")
                    else:
                        log.warning("Skipping deletion due to forbidden keyword match in directory names.")
                else:
                    log.info("Skipping cleanup: One or both preprocess directories do not exist.")
            except Exception as e:
                log.warning(f"Failed to clean up preprocess directories: {e}")

    def copy_and_filter_manual_annotation(self, src_path, dest_path, image_id):
        non_target_id = int(self.cfg.cvat.class_mapping.non_target)
        colorchecker_id = int(self.cfg.cvat.class_mapping.color_checker)

        with open(src_path, 'r') as src, open(dest_path, 'w') as dest:
            for line in src:
                if not line.strip():
                    continue
                parts = line.strip().split()
                class_id = int(parts[0])
                
                if class_id == non_target_id:
                    log.debug(f"Skipping non_target in manual annotation for {image_id}")
                    continue
                
                # remap colorchecker from original to 1
                if class_id == colorchecker_id:
                    mapped_id = 1
                else:  # plant stays 0
                    mapped_id = 0
                
                parts[0] = str(mapped_id)
                dest.write(" ".join(parts) + "\n")
    
    def map_class_ids(self, annotation):
        """
        Maps annotation to YOLO class id or None if it should be ignored.
        """
        if annotation.get('non_target_weed') is True:
            if annotation.get('non_target_weed_pred_conf', 0) > 0.99:
                return None  # ignore non_target
            else:
                return int(self.cfg.cvat.class_mapping.plant)
        elif annotation.get('category_class_id') == 28:
            return int(self.cfg.cvat.class_mapping.color_checker)
        else:
            return int(self.cfg.cvat.class_mapping.plant)

    def process_image(self, row, type):
        """
        Process a single image and its annotations.
        
        Args:
            row (pd.Series): Row from DataFrame containing image information
            type (str): Type of dataset ('train' or 'val')
        """
        image_id = row['image_id']
        
        # Find source image - assuming .jpg extension
        train_image_path = self.train_data_path / type / 'images' / f"{image_id}.jpg"
        preprocess_image_path = Path(self.cfg.paths.preprocess.image_dir) / f"{image_id}.jpg"

        # Destination paths for the image and label in Ultralytics format
        dest_image_path = self.train_data_path / type / 'images' / f"{image_id}.jpg"
        dest_label_path = self.train_data_path / type / 'labels' / f"{image_id}.txt"
        
        # Copy image if it exists
        if train_image_path.exists():
            source_image_path = train_image_path
        elif preprocess_image_path.exists():
            source_image_path = preprocess_image_path
        else:
            log.warning(f"Source image not found in train or preprocess: {image_id}, will need to copy from LTS")
            # TODO: try to get it from LTS if not exists
            return
        
        # Only copy if source and destination are different paths
        if source_image_path.resolve() != dest_image_path.resolve():
            shutil.copy(source_image_path, dest_image_path)
        
        if image_id in self.human_annotations:
            src = self.human_annotations[image_id]
            if self.cfg.train.prepare_dataset.ignore_non_targets:
                self.copy_and_filter_manual_annotation(src, dest_label_path, image_id)
            else:
                shutil.copy(src, dest_label_path)
        else:
            log.warning(f'Manual annotation not found for {image_id}')
            # Only read image if annotations need to be normalized
            img = cv2.imread(str(source_image_path))
            if img is None:
                log.error(f"Unable to read image dimensions: {source_image_path}")
                raise ValueError(f"Unable to read image dimensions: {source_image_path}")
            image_height, image_width = img.shape[:2]
            if 'annotations' in row:
                try:
                    annotations = json.loads(row['annotations'])
                    with open(dest_label_path, 'w') as f:
                        for annotation in annotations:
                            # Get bounding box
                            bbox = annotation.get('bbox_xywh')
                            if not bbox:
                                continue
                            
                            mapped_class_id = self.map_class_ignore_non_target(annotation)

                            if mapped_class_id is None:
                                log.debug(f"Skipping non_target annotation for image {image_id}")
                                continue
                            
                            # Convert bounding box to YOLO format
                            center_x, center_y, norm_width, norm_height = convert_bbox_to_yolo_format(
                                bbox, image_width, image_height
                            )
                            
                            # Write to annotation file
                            f.write(f"{mapped_class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                except (json.JSONDecodeError, TypeError):
                    log.error(f"Error parsing annotations for image {image_id}")
            else:
                log.error(f'No annotations found for {image_id}')

    def process_image_wrapper(self, args):
        """
        Wrapper function for process_image to be used with multiprocessing.
        
        Args:
            args (tuple): Tuple containing (row, type)
        """
        try:
            row, type = args
            self.process_image(row, type)
        except Exception as e:
            log.error(f"Error processing image {row['image_id']}: {e}")

    def prepare_from_df(self, df, type='train'):
        """
        Read the dataframe, get image ids, copy/download images, copy annotations
        """
        os.makedirs(self.train_data_path / type / 'images', exist_ok=True)
        os.makedirs(self.train_data_path / type / 'labels', exist_ok=True)

        if self.parallel:
            log.info(f"Processing {type} images in parallel with {self.parallel_workers} workers")
            with Pool(processes=self.parallel_workers) as pool:
                # Create list of (row, type) tuples for each image
                args = [(row, type) for _, row in df.iterrows()]
                pool.map(self.process_image_wrapper, args)
        else:
            log.info(f"Processing {type} images sequentially")
            for _, row in df.iterrows():
                self.process_image(row, type)
                
        log.info(f"Prepared {type} dataset with {len(df)} images")

    def identify_training_data(self):
        """
        - Read csv
        - Identify images with human annotations
        - Identify images without human annotations (warn)
        - Images and annotations together - train/val split
        """

        df = pd.read_csv(self.data_csv)
        for _, row in df.iterrows():
            image_id = row['image_id']
            if image_id not in self.human_annotations.keys():
                log.warning(f"{image_id} - annotations not verified")
        
        # Split into train and validation sets
        train_ids, val_ids = train_test_split(
            df['image_id'].tolist(),
            test_size=self.validation_split,
            random_state=self.random_seed,
            shuffle=True
        )
        
        # Add split column to dataframe
        df['split'] = 'unused'
        df.loc[df['image_id'].isin(train_ids), 'split'] = 'train'
        df.loc[df['image_id'].isin(val_ids), 'split'] = 'val'
        
        # Save updated dataframe (also include timestamp to indicate data subset used)
        output_path = self.train_data_path / f'train_images_{str(self.data_csv.parent.name)}_{str(self.data_csv.name)}'
        df.to_csv(output_path, index=False)
        log.info(f"Split dataset into {len(train_ids)} training and {len(val_ids)} validation images")
        return df

    def structure_data(self, df):
        """
        Function to create the folder structure expected by Ultralytics for training/testing
        - Read df, identify train and val images
        - Get appropriate images from local folder, check LTS locations if not found
        - Get annotations for the relevant images from LTS location of human annotations, or from dataframe if not found
        """
        train_df = df.loc[df['split'] == 'train']
        val_df = df.loc[df['split'] == 'val']
        self.prepare_from_df(train_df, 'train')
        self.prepare_from_df(val_df, 'val')
   
    def run(self):
        log.info(f"Found {len(self.human_annotations)} human annotations")
        df = self.identify_training_data()
        self.structure_data(df)
        self._cleanup_preprocess_dirs()
        return self.train_data_path

def main(cfg: DictConfig):
    """
    Main entrypoint for preparing training dataset
     - Expects a path to a directory containing human annotations
     - Will take latest training_images csv and merge with human annotations
    """

    # checks for project and task before running the train pipeline
    project_dir = Path("projects") / cfg.project.name
    task_dir = project_dir / cfg.project.task_name

    if task_dir.exists():
        log.info(f"Project '{cfg.project.name}' and task '{cfg.project.task_name}' already exist. Proceeding with existing directory.")
    else:
        if project_dir.exists():
            log.info(f"Project '{cfg.project.name}' exists. Creating new task directory '{cfg.project.task_name}'.")
        else:
            log.info(f"Creating new project and task directories: {cfg.project.name}/{cfg.project.task_name}")
        task_dir.mkdir(parents=True, exist_ok=False)

    prepare_dataset = PrepareDataset(cfg)
    dataset_path = prepare_dataset.run()
    log.info(f"Dataset prepared at {dataset_path}")
    return dataset_path

if __name__ == "__main__":
    main()
