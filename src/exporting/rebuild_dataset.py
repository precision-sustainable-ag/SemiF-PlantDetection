import logging
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from omegaconf import DictConfig
from src.preprocessing.download_images import ImageDownloader
from src.preprocessing.training_dataset import TrainingDatasetGenerator

log = logging.getLogger(__name__)

class RebuildDataset:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.csv_path = Path(cfg.paths.preprocess.csv_dir) / "training_images.csv"
        self.image_dir = Path(cfg.paths.preprocess.image_dir)
        self.train_txt_path = Path(cfg.paths.preprocess.label_dir) / "train.txt"
        self.db_path = Path(cfg.paths.db_path)

    def parse_train_txt(self) -> list:
        """
        Parses train.txt to extract image IDs (stems of image filenames).
        """
        if not self.train_txt_path.exists():
            raise FileNotFoundError(f"train.txt not found at {self.train_txt_path}")
        
        with open(self.train_txt_path, 'r') as f:
            lines = f.read().splitlines()

        image_ids = [Path(line).stem for line in lines]
        log.info(f"Parsed {len(image_ids)} image IDs from train.txt")
        return image_ids

    def query_image_metadata(self, image_ids: list) -> pd.DataFrame:
        """
        Queries the DB for metadata associated with given image IDs.
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        placeholders = ','.join(['?'] * len(image_ids))
        query = f"""
        SELECT batch_id, image_id, validated, exif_meta, camera_info, annotations, categories, season
        FROM semif_developed_images
        WHERE image_id IN ({placeholders})
        """
        df = pd.read_sql_query(query, conn, params=image_ids)
        conn.close()

        log.info(f"Retrieved metadata for {len(df)} images from database")
        return df

    def process_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds derived fields to the image metadata.
        """
        generator = TrainingDatasetGenerator(self.cfg)
        df['class_ids'] = df['annotations'].apply(generator.get_class_ids_from_annotations)
        df['batch_date'] = df['batch_id'].apply(generator.extract_date_from_batch_id)
        df['growing_season_bucket'] = df['season'].apply(generator.get_growing_season_bucket)
        df = df[df['growing_season_bucket'] != 'unknown']
        df['location'] = df['batch_id'].apply(lambda x: x.split('_')[0])
        df['has_priority_species'] = df['class_ids'].apply(
            lambda ids: any(c_id in generator.priority_species for c_id in ids)
        )
        return df

    def save_training_csv(self, df: pd.DataFrame):
        """
        Saves the training metadata CSV and a metadata.json file.
        """
        output_dir = self.csv_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(self.csv_path, index=False)
        log.info(f"Saved rebuilt training CSV to {self.csv_path}")

        metadata = {
            "created_at": datetime.now().isoformat(),
            "num_images": len(df),
            "output_path": str(self.csv_path)
        }
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def all_images_exist(self, df: pd.DataFrame) -> bool:
        """
        Checks if all referenced image files exist locally.
        """
        return all((self.image_dir / f"{img_id}.jpg").exists() for img_id in df['image_id'])

    def run(self) -> None:
        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            if self.all_images_exist(df):
                log.info("All images and CSV already exist. Skipping dataset rebuild.")
                return
            log.warning("CSV exists but some images are missing. Rebuilding dataset...")

        log.info("Rebuilding training_images.csv from train.txt and DB metadata...")

        try:
            image_ids = self.parse_train_txt()
            if not image_ids:
                log.warning("No image IDs found in train.txt. Skipping rebuild.")
                return

            df = self.query_image_metadata(image_ids)

            # Check whether the size of the DataFrame is the same as the number of image IDs
            if df.shape[0] != len(image_ids):
                log.warning(f"Mismatch: {len(image_ids)} image IDs but {df.shape[0]} records in DB. Rebuilding dataset...")
            
            df = self.process_metadata(df)
            self.save_training_csv(df)

            downloader = ImageDownloader(self.cfg)
            downloader.download_all_images()

            log.info("Dataset reconstruction and image download complete.")
        except Exception as e:
            log.error(f"Failed to rebuild dataset from train.txt: {e}")
            raise

def main(cfg: DictConfig) -> None:
    log.info("Starting dataset rebuild step")
    rebuilder = RebuildDataset(cfg)
    rebuilder.run()
    log.info("Dataset rebuild step completed")

if __name__ == "__main__":
    main()