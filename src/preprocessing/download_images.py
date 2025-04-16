#!/usr/bin/env python
import os
import json
import logging
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from omegaconf import DictConfig
from src.utils.utils import find_most_recent_dataset_path

# Setup logging
log = logging.getLogger(__name__)

class ImageDownloader:
    """
    Class to handle downloading images from long-term storage to a local directory.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the ImageDownloader with paths for CSV data, long-term storage, and local folder.

        Args:
            cfg (DictConfig): Hydra configuration
        """
        # Find the most recent dataset directory
        base_dataset_path = Path(cfg.database.dataset.output_path)
        self.dataset_path = find_most_recent_dataset_path(base_dataset_path)
        self.csv_file_path = Path(self.dataset_path, "training_images.csv")
        
        self.storage_bases = []
        for lts_location in cfg.paths.lts_locations:
            self.storage_bases.append(Path(lts_location, "semifield-developed-images"))
        
        log.info(f"Using LTS storage locations: {self.storage_bases}")
        
        # Local download folder
        self.image_download_folder = Path(cfg.images.output_path)
        
        # Configuration for parallel processing
        self.max_workers = cfg.images.parallel_workers
        self.parallel = cfg.images.parallel

        # Create the local download folder if it doesn't exist
        if not self.image_download_folder.exists():
            self.image_download_folder.mkdir(parents=True, exist_ok=True)
            log.info(f"Created local download folder: {self.image_download_folder}")

    def load_dataset(self) -> pd.DataFrame:
        """
        Loads the dataset from the CSV file.

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

    def download_image(self, batch_id: str, image_id: str) -> None:
        """
        Downloads an image corresponding to a given batch_id and image_id from the long-term storage.

        Args:
            batch_id (str): The batch ID to locate the image in the long-term storage.
            image_id (str): The ID of the image to download.
        """
        image_filename = f"{image_id}.jpg"

        # Construct the local path where the image will be saved
        local_image_path = Path(self.image_download_folder, image_filename)
        
        # Skip if the image already exists locally
        if local_image_path.exists():
            log.debug(f"Image already exists locally: {image_id}")
            return

        # Build a list of all possible storage paths
        storage_paths = []
        for i, storage_base in enumerate(self.storage_bases):
            storage_path = Path(storage_base, batch_id, "images", image_filename)
            storage_paths.append((f"Storage{i+1}", storage_path))

        # Try each storage location until the image is found and copied
        for storage_name, storage_path in storage_paths:
            if storage_path.exists():
                try:
                    shutil.copy(storage_path, local_image_path)
                    log.debug(f"Downloaded from {storage_name}: {image_id} to {local_image_path}")
                    return  # Exit after successful download
                except IOError as e:
                    log.error(f"Error copying file from {storage_name} ({storage_path}) to {local_image_path} - {e}")
                    # Continue to the next storage if copy fails
        
        # If we reach this point, the file was not found or could not be copied from any storage
        log.error(
            f"Image not found in any storage for image_id: {image_id}, batch_id: {batch_id}. Tried paths: " +
            ", ".join(f"{name}: {path}" for name, path in storage_paths)
        )

    def get_unique_images(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Extracts unique batch_id and image_id pairs from the DataFrame.
        Returns only images that don't exist locally.

        Args:
            df (pd.DataFrame): DataFrame containing image information.
            
        Returns:
            List[Dict[str, str]]: List of dictionaries with batch_id and image_id.
        """
        unique_images = []
        for _, row in df.iterrows():
            batch_id = row['batch_id']
            image_id = row['image_id']
            
            # Check if the image already exists locally
            local_image_path = Path(self.image_download_folder, f"{image_id}.jpg")
            if local_image_path.exists():
                log.debug(f"Image already exists locally: {image_id}")
                continue
                
            unique_images.append({"batch_id": batch_id, "image_id": image_id})
        
        return unique_images

    def process_images_sequentially(self) -> None:
        """
        Processes each image in the dataset and downloads it from long-term storage in serial mode.
        """
        df = self.load_dataset()
        unique_images = self.get_unique_images(df)
        log.info(f"Found {len(unique_images)} unique images to download.")

        for image_info in unique_images:
            self.download_image(image_info["batch_id"], image_info["image_id"])

        log.info("Download process completed in serial mode.")

    def process_images_concurrently(self) -> None:
        """
        Processes each image in the dataset and downloads it from long-term storage.
        This method uses multithreading to parallelize the download process.
        """
        df = self.load_dataset()
        unique_images = self.get_unique_images(df)
        log.info(f"Found {len(unique_images)} unique images to download.")

        # Thread pool executor for multithreading
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.download_image, image_info["batch_id"], image_info["image_id"])
                for image_info in unique_images
            ]

            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    future.result()  # This will re-raise exceptions if any occurred
                except Exception as e:
                    log.error(f"Error occurred during download task: {e}")

        log.info("Download process completed in parallel mode.")

    def download_all_images(self) -> None:
        """
        Main method to download all images based on the configuration.
        """
        if self.parallel:
            self.process_images_concurrently()
        else:
            self.process_images_sequentially()


def main(cfg: DictConfig) -> None:
    """
    Main entry point for downloading images.
    
    Args:
        cfg (DictConfig): Hydra configuration
    """
    log.info("Starting copying images locally")
    downloader = ImageDownloader(cfg)
    downloader.download_all_images()
    log.info("Copying images process completed")


if __name__ == "__main__":
    main() 