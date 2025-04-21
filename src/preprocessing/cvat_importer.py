import logging
from omegaconf import DictConfig
from pathlib import Path
from cvat_sdk import Client, models
from src.utils.utils import read_secrets, find_most_recent_dataset_path

log = logging.getLogger(__name__)

class CVATImporter:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.cvat_secrets = read_secrets(cfg.secrets_path)['cvat']
        self.cvat_dataset_format = cfg.cvat.dataset_format
    
    def get_cvat_dataset(self) -> Path:
        """
        Returns path of the CVAT dataset to be uploaded
        """
        # TODO: Might include going through different dates
        dataset_path = Path(self.cfg.paths.data_dir, "cvat_dataset.zip")
        return str(dataset_path)
    
    def login(self):
        """
        Login to CVAT
        """
        self.cvat_client = Client(self.cvat_secrets['url'])
        self.cvat_client.login(credentials=[self.cvat_secrets['username'], self.cvat_secrets['password']])
    
    def create_project(self) -> str:
        """
        Create a new CVAT project using the most recent dataset
        """
        experiment_date_path = find_most_recent_dataset_path(self.cfg.database.dataset.output_path)
        
        # Extract date and time components from dataset path
        # path_parts = str(experiment_date_path).split('/')
        path_parts = experiment_date_path.parts
        date_str = path_parts[-2] if len(path_parts) >= 2 else "unknown-date"
        time_parts = path_parts[-1].split('-') if len(path_parts) >= 1 else ["00", "00", "00"]
        hour_min = f"{time_parts[0]}-{time_parts[1]}" if len(time_parts) >= 2 else "00-00"

        project_name = f"Plant detection {date_str} ({hour_min})"
        project_spec = models.ProjectWriteRequest(name=project_name)
        self.cvat_client.projects.create_from_dataset(spec=project_spec,
                                                      dataset_path=self.get_cvat_dataset(),
                                                      dataset_format=self.cvat_dataset_format)
        
        return project_name

def main(cfg: DictConfig) -> None:
    """
    Main entry point for uploading a CVAT dataset to CVAT server.
    Steps:
    - Get the appropriate training dataset (by identifying most recent dataset csv file)
    - Login to CVAT
    - Create a new project (name: Plant detection {date} ({hour_min}) - date derived from dataset path)
    - Upload the dataset
    
    Args:
        cfg (DictConfig): Hydra configuration
    """
    # TODO: delete the cvat_dataset folder and zip file after upload
    log.info("Starting CVAT dataset uploader")
    cvat_importer = CVATImporter(cfg)
    log.info(f"logging into cvat")
    cvat_importer.login()
    log.info(f"uploading dataset")
    project_name = cvat_importer.create_project()
    log.info(f"CVAT project created for manual annotation: {project_name}")


if __name__ == "__main__":
    main() 