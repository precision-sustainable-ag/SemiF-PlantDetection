#!/usr/bin/env python
import sqlite3
import logging
import json
import random
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple, Set
import hydra
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path

# Setup logging
log = logging.getLogger(__name__)

class TrainingDatasetGenerator:
    """
    A class to generate training datasets from the SQLite database
    with specific selection criteria:
    1. Prioritize more recent batches (based on date in batch_id)
    2. Prioritize images with annotations having specified class_ids
    3. Ensure a good mix of other images
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the TrainingDatasetGenerator.
        
        Args:
            cfg (DictConfig): Hydra configuration containing database path and dataset settings.
        """
        self.db_path = cfg.db_path
        self.dataset_size = cfg.dataset.size
        self.priority_class_ids = cfg.dataset.priority_class_ids
        self.priority_ratio = cfg.dataset.priority_ratio
        self.min_per_class = cfg.dataset.min_per_class
        self.output_path = cfg.dataset.output_path
        self.random_seed = cfg.dataset.random_seed
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        
        # Connect to the database
        self.conn = sqlite3.connect(self.db_path)
        log.info(f"Connected to database: {self.db_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
    def fetch_validated_images(self) -> pd.DataFrame:
        """
        Fetch all validated images from the semif_developed_images table.
        Note: currently, not differentiating between seasons and bbot versions
        
        Returns:
            pd.DataFrame: DataFrame containing validated images.
        """
        query = """
        SELECT batch_id, image_id, validated, exif_meta, camera_info, annotations, categories
        FROM semif_developed_images
        WHERE validated = True
        """
        
        log.info("Fetching validated images from database")
        df = pd.read_sql_query(query, self.conn)
        log.info(f"Found {len(df)} validated images")
        return df
    
    def extract_date_from_batch_id(self, batch_id: str) -> datetime:
        """
        Extract date from batch_id
        
        Args:
            batch_id (str): Batch ID containing date
            
        Returns:
            datetime: Date extracted from batch_id
        """
        try:
            # Extract the date part (assuming format like MD_2023-05-07)
            date_str = batch_id.split('_')[1]
            return datetime.strptime(date_str, '%Y-%m-%d')
        except (IndexError, ValueError):
            # Return a default old date if format is not as expected
            # Hence, will be ignored.
            # TODO: add a warning
            return datetime(1900, 1, 1)
    
    def get_class_ids_from_annotations(self, annotations_str: str) -> Set[int]:
        """
        Extract class_ids from annotations JSON string.
        
        Args:
            annotations_str (str): JSON string containing annotations
            
        Returns:
            Set[int]: Set of unique class_ids in the annotations
        """
        class_ids = set()
        try:
            annotations = json.loads(annotations_str)
            for annotation in annotations:
                class_id = annotation.get('category_class_id')
                if class_id is not None:
                    class_ids.add(class_id)
        except (json.JSONDecodeError, TypeError):
            # TODO: add a warning
            pass
        
        return class_ids
    
    def select_training_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select images for the training dataset based on criteria:
        1. Prioritize recent batches
        2. Prioritize images with specified class_ids
        3. Ensure a good mix of other images
        
        Args:
            df (pd.DataFrame): DataFrame of validated images
            
        Returns:
            pd.DataFrame: DataFrame of selected images for training
        """
        # Extract class IDs and dates for each image
        df['class_ids'] = df['annotations'].apply(self.get_class_ids_from_annotations)
        df['batch_date'] = df['batch_id'].apply(self.extract_date_from_batch_id)
        
        # Flag images with priority classes
        df['has_priority_class'] = df['class_ids'].apply(
            lambda ids: any(c_id in self.priority_class_ids for c_id in ids)
        )
        
        # Sort by date (most recent first)
        # TODO: ensure a good mix of locations
        df_sorted = df.sort_values('batch_date', ascending=False)
        
        # Get priority images
        priority_images = df_sorted[df_sorted['has_priority_class']]
        other_images = df_sorted[~df_sorted['has_priority_class']]
        
        # Calculate how many images to take from each group
        priority_count = min(int(self.dataset_size * self.priority_ratio), len(priority_images))
        other_count = min(self.dataset_size - priority_count, len(other_images))
        
        log.info(f"Selecting {priority_count} priority images and {other_count} other images")
        
        # Select images
        # TODO: select random images from each group
        selected_priority = priority_images.head(priority_count)
        selected_other = other_images.head(other_count)
        
        # Combine and return
        selected = pd.concat([selected_priority, selected_other])
        
        # Ensure we have minimum images per class if possible
        # TODO: also add non target weeds, color checker data - either here or in a separate function
        # selected = self.ensure_class_balance(df_sorted, selected)
        
        return selected.sample(frac=1, random_state=self.random_seed)  # Shuffle the final selection
    
    def ensure_class_balance(self, all_images: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that we have at least min_per_class images for each class ID if possible.
        
        Args:
            all_images (pd.DataFrame): All validated images
            selected (pd.DataFrame): Currently selected images
            
        Returns:
            pd.DataFrame: DataFrame with balanced class representation
        """
        # TODO: improve this whole function

        # Get all unique class IDs across all images
        all_class_ids = set()
        for class_set in all_images['class_ids']:
            all_class_ids.update(class_set)
        
        # Check if we have the minimum number of images for each class
        for class_id in all_class_ids:
            # Count images with this class ID in the selected set
            class_count = sum(1 for ids in selected['class_ids'] if class_id in ids)
            
            # If we don't have enough, try to add more
            if class_count < self.min_per_class:
                # Find images with this class ID that aren't already selected
                additional_needed = self.min_per_class - class_count
                potential_images = all_images[~all_images['image_id'].isin(selected['image_id'])]
                potential_images = potential_images[
                    potential_images['class_ids'].apply(lambda ids: class_id in ids)
                ]
                
                # Sort by date and take what we need
                potential_images = potential_images.sort_values('batch_date', ascending=False)
                additional_images = potential_images.head(additional_needed)
                
                # Add to selected
                if not additional_images.empty:
                    log.info(f"Adding {len(additional_images)} images for class_id {class_id}")
                    selected = pd.concat([selected, additional_images])
        
        # If we've exceeded our desired size, trim down
        if len(selected) > self.dataset_size:
            log.info(f"Trimming dataset from {len(selected)} to {self.dataset_size} images")
            selected = selected.sample(self.dataset_size, random_state=self.random_seed)
        
        return selected
    
    
    def save_dataset(self, images: pd.DataFrame) -> None:
        """
        Save the selected images and cutouts to CSV files.
        
        Args:
            images (pd.DataFrame): Selected images for training
        """
        # Save to CSV
        output_images_path = os.path.join(self.output_path, "training_images.csv")
        
        images.to_csv(output_images_path, index=False)
        
        log.info(f"Saved {len(images)} images to {output_images_path}")
        
        # Save metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "dataset_size": len(images),
            "priority_class_ids": self.priority_class_ids,
            "priority_ratio": self.priority_ratio,
            "min_per_class": self.min_per_class,
            "random_seed": self.random_seed
        }
        
        metadata_path = os.path.join(self.output_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        log.info(f"Saved metadata to {metadata_path}")
    
    def generate(self) -> pd.DataFrame:
        """
        Generate the training dataset.
        
        Returns:
            pd.DataFrame: Selected images
        """
        # Fetch all validated images
        all_images = self.fetch_validated_images()
        
        # Select images for training
        selected_images = self.select_training_images(all_images)
        
        
        # Save the dataset
        self.save_dataset(selected_images)
        
        return selected_images
    
    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
        log.info("Closed database connection")


def main(cfg: DictConfig) -> None:
    """
    Main entry point for generating a training dataset.
    
    Args:
        cfg (DictConfig): Hydra configuration
    """
    # Print the configuration
    # log.info(f"Configuration:\n{cfg}")
    
    # Create and run the generator
    generator = TrainingDatasetGenerator(cfg.database)
    try:
        images, cutouts = generator.generate()
        log.info(f"Successfully generated training dataset with {len(images)} images and {len(cutouts)} cutouts")
    finally:
        generator.close()


if __name__ == "__main__":
    main() 