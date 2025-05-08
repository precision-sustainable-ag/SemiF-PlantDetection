#!/usr/bin/env python
import sqlite3
import logging
import json
import random
import os
from datetime import datetime
from typing import Tuple, Set
import math
from omegaconf import DictConfig
import pandas as pd

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
        self.priority_species = cfg.dataset.priority_species
        # self.priority_class_ids = None
        # self.priority_ratio = cfg.dataset.priority_ratio
        self.min_per_class = cfg.dataset.min_per_class
        self.random_seed = cfg.dataset.random_seed
        # self.priority_recent_ratio = cfg.dataset.selection.priority_recent_ratio
        self.other_species_recency_ratio = cfg.dataset.other_species_recency_ratio
        self.ratios = cfg.dataset.ratios
        
        # Define timestamped output directory
        base_output_path = cfg.dataset.output_path
        timestamp_date = datetime.now().strftime("%Y-%m-%d")
        timestamp_time = datetime.now().strftime("%H-%M-%S")
        self.output_path = os.path.join(base_output_path, timestamp_date, timestamp_time)

        # Set random seed for reproducibility
        random.seed(self.random_seed)
        
        # Connect to the database
        self.conn = sqlite3.connect(self.db_path)
        log.info(f"Connected to database: {self.db_path}")
        

    def fetch_validated_images_without_non_targets(self) -> pd.DataFrame:
        """
        Fetch all validated images from the semif_developed_images table.
        Note: currently, not differentiating between seasons and bbot versions
        Query:
        inner query gets all images with atleast one non-target weeds
        outer query filters out these images and only returns images without non-target weeds

        Returns:
            pd.DataFrame: DataFrame containing validated images.
        """

        query = """
        SELECT batch_id, image_id, validated, exif_meta, camera_info, annotations, categories, season
        FROM semif_developed_images
        WHERE NOT EXISTS (
            SELECT 1
            FROM json_each(annotations) AS annotation_each
            WHERE json_extract(annotation_each.value, '$.non_target_weed') = TRUE
            AND semif_developed_images.image_id = image_id
        )
        AND validated=True
        AND not json_array_length(annotations) = 0;
        """
        # query = """
        # SELECT batch_id, image_id, validated, exif_meta, camera_info, annotations, categories, season
        # FROM semif_developed_images
        # WHERE validated = True AND not json_array_length(annotations) = 0;
        # """
        
        log.info("Fetching validated images from database")
        df = pd.read_sql_query(query, self.conn)
        log.info(f"Found {len(df)} validated images")
        return df
    
    def fetch_validated_images_with_non_targets(self) -> pd.DataFrame:
        """
        Fetch all validated images from the semif_developed_images table which have atleast one non-target weed.
        Query:
        Inner query gets x images with most non-target weeds
        Outer query gets all the necessary data for those images
        """
        non_targets_count = int(self.dataset_size * self.ratios.non_targets)
        query = f"""
        select batch_id, image_id, validated, exif_meta, camera_info, annotations, categories, season
        from semif_developed_images where image_id in (
            select image_id
            from semif_developed_images, json_each(annotations) as annotation_each
            where json_extract(annotation_each.value, '$.non_target_weed') = TRUE
            and json_extract(annotation_each.value, '$.non_target_weed_pred_conf') > 0.8
            group by image_id
            order by count(*) desc
            limit {non_targets_count}
        );
        """
        df = pd.read_sql_query(query, self.conn)
        log.info(f"Found {len(df)} validated images with non-target weeds")
        return df
    
    def extract_date_from_batch_id(self, batch_id: str) -> datetime:
        """
        Extract date from batch_id
        Not using datetime field directly since it doesn't always match batch date
        
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
    
    @staticmethod
    def get_growing_season_bucket(season: str) -> str:
        """
        Get the growing season bucket for a given season
        """
        if "weed" in season.lower():
            return "weed crops"
        elif "cash" in season.lower():
            return "cash crops"
        elif "cover" in season.lower():
            return "cover crops"
        else:
            return "unknown"

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
            log.warning(f"Error decoding annotations for img")
            pass
        
        return class_ids
    
    def get_priority_species_and_batches(self, df: pd.DataFrame) -> Tuple[Set[int], Set[str]]:
        """
        Get class ids from latest batches across locations and seasons, and the batches themselves
        1. Group dataframe by growing season bucket and location
        2. Get the latest batch for each group
        3. Extract class ids from the annotations of these batches
        
        Args:
            df (pd.DataFrame): DataFrame of validated images
            
        Returns:
            Tuple[Set[int], Set[str]]: top class ids, batches
        """
        latest_batches = (
            df.sort_values('batch_date', ascending=False)  # Sort by batch_date first
            .groupby(['growing_season_bucket', 'location'])
            .first()  # Get the first entry for each group, which is an arbitrary image from the latest batch
            .reset_index()
        )
        # Extract class_ids from the annotations of these batches
        batches_to_check = df[df['batch_id'].isin(latest_batches['batch_id'])]

        class_ids_count = {}
        for annotations in batches_to_check['annotations']:
            class_ids = self.get_class_ids_from_annotations(annotations)
            for class_id in class_ids:
                if class_id in class_ids_count:
                    class_ids_count[class_id] += 1
                else:
                    class_ids_count[class_id] = 1

        # Get the top 10 most occurring classes
        top_classes = sorted(class_ids_count.items(), key=lambda x: x[1], reverse=True)[:10]
        top_class_ids, top_class_counts = zip(*top_classes) if top_classes else ([], [])

        # add specified priority class ids from config
        class_ids = set(top_class_ids).union(self.priority_species)
        return class_ids, set(latest_batches['batch_id'].unique())

    def create_balanced_dataset(self, df: pd.DataFrame, class_ids: Set[int], total_max_size: int) -> pd.DataFrame:
        """
        Create a balanced dataset by ensuring each specified class has at least min_count images,
        while respecting the total_max_size.

        Args:
            df (pd.DataFrame): DataFrame of images to select from.
            class_ids (Set[int]): Set of class IDs to balance.
            min_count (int): Minimum number of images required per class.
            total_max_size (int): Total maximum size of the dataset.

        Returns:
            pd.DataFrame: A balanced DataFrame of selected images.
        """
        if self.min_per_class == 0 or self.min_per_class > total_max_size:
            log.warning(f"No minimum count per species specified, or minimum count is greater than total max size, selecting random number of images")
            selected_images = df.sample(n=total_max_size, random_state=self.random_seed)
            return selected_images
        selected_images = pd.DataFrame()

        # TODO: when one image is selected, it can have multiple class_ids
        for class_id in class_ids:
            # Select images for the current class
            class_images = df[df['class_ids'].apply(lambda ids: class_id in ids)]
            if not selected_images.empty:
                class_images = class_images[~class_images['image_id'].isin(selected_images['image_id'])]

            # Ensure we have at least min_count images
            if len(class_images) < self.min_per_class:
                log.warning(f"Not enough images for class_id {class_id}: found {len(class_images)}, required {self.min_per_class}.")
                selected_images = pd.concat([selected_images, class_images])
            else:
                # Randomly sample min_count images from the class_images
                sampled_images = class_images.sample(n=self.min_per_class, random_state=self.random_seed)
                selected_images = pd.concat([selected_images, sampled_images])
                # log.info(f"dataframe classids totals: {selected_images['class_ids'].value_counts()}")

        # If the total size exceeds the maximum size, randomly sample to fit
        if len(selected_images) > total_max_size:
            log.info(f"Trimming dataset from {len(selected_images)} to {total_max_size} images.")
            selected_images = selected_images.sample(n=total_max_size, random_state=self.random_seed)
        elif len(selected_images) < total_max_size:
            log.warning(f"Not enough images to fill the dataset, selecting additional random number of images")
            unselected_images = df[~df['image_id'].isin(selected_images['image_id'])]
            additional_images = unselected_images.sample(n=total_max_size - len(selected_images), random_state=self.random_seed)
            selected_images = pd.concat([selected_images, additional_images])
        else:
            log.info(f"Selected {len(selected_images)} images")

        return selected_images

    def select_target_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select images for the training dataset based on criteria:
        1. Prioritize recent batches
        2. Prioritize images with specified class_ids
        3. Ensure a good mix of other images, including some from older batches
        
        Args:
            df (pd.DataFrame): DataFrame of validated images
            
        Returns:
            pd.DataFrame: DataFrame of selected images for training
        """
        # Extract class IDs and dates for each image
        df['class_ids'] = df['annotations'].apply(self.get_class_ids_from_annotations)
        df['batch_date'] = df['batch_id'].apply(self.extract_date_from_batch_id)
        df['growing_season_bucket'] = df['season'].apply(self.get_growing_season_bucket)
        # ignore unknown growing season buckets, ideally there'll be none since db is clean
        df = df[df['growing_season_bucket'] != 'unknown']
        # Extract location from batch_id (assuming format like 'location_YYYY-MM-DD')
        df['location'] = df['batch_id'].apply(lambda x: x.split('_')[0])

        # Flag images with priority classes
        self.priority_species, latest_batches = self.get_priority_species_and_batches(df)
        df['has_priority_species'] = df['class_ids'].apply(
            lambda ids: any(c_id in self.priority_species for c_id in ids)
        )
        # Get priority images from latest batches
        priority_images = df[df['has_priority_species'] & df['batch_id'].isin(latest_batches)]
        priority_count = min(int(self.dataset_size * self.ratios.priority_species), len(priority_images))
        selected_priority = self.create_balanced_dataset(priority_images, self.priority_species, priority_count)
        
        # Get other images
        # TODO: get class ids for non-priority species
        # TODO: thought - forget about priority species when selecting other images
        # balance - because that'll give us an equal mix of classes
        # then generate a balanced dataset of other_images (first give latest batches, then older batches - in recency ratio)
        df_sorted = df.sort_values('batch_date', ascending=False)
        # other_images = df_sorted[~df_sorted['has_priority_species'] & ~df_sorted['image_id'].isin(priority_images['image_id'])]
        other_images = df_sorted[~df_sorted['image_id'].isin(priority_images['image_id'])]
        other_count = min(int(self.dataset_size * self.ratios.other_species), len(other_images))
        log.info(f"Selecting {priority_count} priority images and {other_count} other images")
        
        # TODO: Better selection strategy to balance recent and older batches
        if len(other_images) > 0:
            recent_count = int(other_count * self.other_species_recency_ratio)
            diverse_count = other_count - recent_count
            
            # Get the most recent other images (will probably be from one batch only)
            selected_recent_other = other_images.head(recent_count)
            
            # Get a diverse sample from the remaining other images
            remaining_other = other_images.iloc[recent_count:] if recent_count < len(other_images) else pd.DataFrame()
            
            if not remaining_other.empty and diverse_count > 0:
                selected_diverse_other = remaining_other.sample(
                    min(diverse_count, len(remaining_other)), 
                    random_state=self.random_seed
                )
                selected_other = pd.concat([selected_recent_other, selected_diverse_other])
            else:
                selected_other = selected_recent_other
        else:
            selected_other = pd.DataFrame()
        # Combine and return
        selected = pd.concat([selected_priority, selected_other])
        
        return selected.sample(frac=1, random_state=self.random_seed)  # Shuffle the final selection
    
    def select_non_target_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select non-target images for the training dataset
        1. Get images with most non-target weeds from the database
        2. Add class_ids, batch_date, growing_season_bucket, location, has_priority_species (to match target images)
        """
        df = self.fetch_validated_images_with_non_targets()
        df['class_ids'] = df['annotations'].apply(self.get_class_ids_from_annotations)
        df['batch_date'] = df['batch_id'].apply(self.extract_date_from_batch_id)
        df['growing_season_bucket'] = df['season'].apply(self.get_growing_season_bucket)
        df = df[df['growing_season_bucket'] != 'unknown']

        # Extract location from batch_id (assuming format like 'location_YYYY-MM-DD')
        df['location'] = df['batch_id'].apply(lambda x: x.split('_')[0])

        df['has_priority_species'] = df['class_ids'].apply(
            lambda ids: any(c_id in self.priority_species for c_id in ids)
        )
        
        return df
    
    
    def save_dataset(self, images: pd.DataFrame) -> None:
        """
        Save the selected images and cutouts to CSV files.
        
        Args:
            images (pd.DataFrame): Selected images for training
        """
        os.makedirs(self.output_path, exist_ok=True)
        # Save to CSV
        output_images_path = os.path.join(self.output_path, "training_images.csv")
        images.to_csv(output_images_path, index=False)
        
        log.info(f"Saved {len(images)} images to {output_images_path}")
        
        # Save metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "dataset_size": len(images),
            "priority_class_ids": list(self.priority_species),
            "species_ratios": dict(self.ratios),
            "other_species_recency_ratio": self.other_species_recency_ratio,
            "min_per_class": self.min_per_class,
            "random_seed": self.random_seed,
            "output_path": self.output_path
        }
        
        metadata_path = os.path.join(self.output_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        log.info(f"Saved metadata to {metadata_path}")
    
    def generate(self) -> pd.DataFrame:
        """
        Generate the training dataset with the following strategy:
        1. Fetch all validated images from the database
        2. Select images based on:
           - Priority class IDs (specified in configuration)
           - Date-based selection (balancing recent and older batches):
             * For priority classes: priority_recent_ratio from recent batches
             * For other classes: other_recent_ratio from recent batches
           - The rest randomly sampled from older batches for diversity
        3. Ensure class balance with minimum images per class
        4. Save the dataset to the output path
        
        Returns:
            pd.DataFrame: Selected images for the training dataset
        """
        # Fetch all validated images
        target_images = self.fetch_validated_images_without_non_targets()
        
        # Select images for training
        selected_target_images = self.select_target_images(target_images)
        selected_non_target_images = self.select_non_target_images(target_images)
        
        selected_images = pd.concat([selected_target_images, selected_non_target_images])
        if len(selected_images) != selected_images['image_id'].nunique():
            log.error("Duplicate image_ids in the selected images")
            raise ValueError("Duplicate image_ids in the selected images")

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
    
    # check if ratios sum to 1.0
    # adding tolerance due to floating point precision errors
    if not math.isclose(sum(cfg.database.dataset.ratios.values()), 1.0, rel_tol=1e-5):
        log.error("Ratios must sum to 1.0")
        raise ValueError("Ratios must sum to 1.0")
    generator = TrainingDatasetGenerator(cfg.database)
    try:
        images = generator.generate()
        log.info(f"Successfully generated training data subset with {len(images)} images")
    except Exception as e:
        log.error(f"Error generating training data subset - {e}")
        raise ValueError(f"Error generating training data subset")
    finally:
        generator.close()


if __name__ == "__main__":
    main() 