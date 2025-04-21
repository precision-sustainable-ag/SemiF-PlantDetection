from omegaconf import DictConfig
from src.utils.utils import get_annotated_image_ids, find_most_recent_dataset_path
from pathlib import Path
import pandas as pd
import logging
import os
import shutil
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

class PrepareDataset:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.human_annotations = get_annotated_image_ids(self.cfg.paths.lts_human_annotations)
        self.train_data_path = Path(self.cfg.paths.data_dir) / 'training_dataset'
        self.data_csv = find_most_recent_dataset_path(self.train_data_path) / 'training_images.csv'
        self.random_seed = self.cfg.train.random_seed
        self.validation_split = self.cfg.train.validation_split
    
    def indentify_training_data(self):
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
        
        # Save updated dataframe
        output_path = self.train_data_path / 'training_images_split.csv'
        df.to_csv(output_path, index=False)
        log.info(f"Split dataset into {len(train_ids)} training and {len(val_ids)} validation images")
        return df

    def prepare_from_df(self, df, type='train'):
        """
        Read the dataframe, get image ids, copy/download images, copy annotations
        """
        os.makedirs(self.train_data_path / 'data' / type / 'images', exist_ok=True)
        os.makedirs(self.train_data_path / 'data' / type / 'labels', exist_ok=True)

        for index, row in df.iterrows():
            image_id = row['image_id']
            
            # Find source image - assuming .jpg extension
            source_image_path = Path(self.cfg.paths.data_dir) / 'training_images' / f"{image_id}.jpg"
            
            # Destination paths for the image and label in Ultralytics format
            dest_image_path = self.train_data_path / 'data' / type / 'images' / f"{image_id}.jpg"
            dest_label_path = self.train_data_path / 'data' / type / 'labels' / f"{image_id}.txt"
            
            # Copy image if it exists
            if source_image_path.exists():
                shutil.copy(source_image_path, dest_image_path)
            else:
                log.warning(f"Source image not found: {source_image_path}")
                continue
            
            # if image_id not in self.human_annotations.keys():
            #     
            if image_id in self.human_annotations.keys():
                shutil.copy(self.human_annotations[image_id], dest_label_path)
            else:
                log.warn(f'Manual annotation not found for {image_id}')
                if 'annotations' in row:
                    annotations = row['annotations']
                    with open(dest_label_path, 'w') as f:
                        # implement
                        pass
                else:
                    log.error(f'No annotations fond for {image_id}')
                
        log.info(f"Prepared {type} dataset with {len(df)} images")

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


def main(cfg: DictConfig):
    """
    Main entrypoint for preparing training dataset
     - Expects a path to a directory containing human annotations
     - Will take latest training_images csv and merge with human annotations
    """
    prepare_dataset = PrepareDataset(cfg)
    prepare_dataset.indentify_training_data()

if __name__ == "__main__":
    main()
