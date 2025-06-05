import logging
import requests
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
        self.zip_path = Path(self.cfg.paths.project_dir, "cvat_dataset.zip")
    
    def login(self):
        """
        Login to CVAT
        """
        self.cvat_client = Client(self.cvat_secrets['url'])
        self.cvat_client.login(credentials=[self.cvat_secrets['username'], self.cvat_secrets['password']])

    def rename_task_via_rest(self, task_id: int, new_name: str):
        """Rename a CVAT task using a direct REST API PATCH call"""
        url = f"{self.cvat_secrets['url']}/api/tasks/{task_id}"
        auth = (self.cvat_secrets['username'], self.cvat_secrets['password'])

        response = requests.patch(
            url,
            auth=auth,
            headers={"Content-Type": "application/json"},
            json={"name": new_name}
        )

        if response.ok:
            log.info(f"Renamed task ID {task_id} to '{new_name}'")
        else:
            log.error(f"Failed to rename task ID {task_id}: {response.status_code}, {response.text}")
    
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

        project_name = self.cfg.project.name 
        task_name = self.cfg.project.task_name

        project = self.cvat_client.projects.create_from_dataset(
            spec=models.ProjectWriteRequest(name=project_name),
            dataset_path=str(self.zip_path),
            dataset_format=self.cvat_dataset_format
        )

        tasks = self.cvat_client.tasks.list()
        task = next((t for t in tasks if t.project_id == project.id), None)

        if not task:
            log.warning(f"No task found in project '{project_name}'")
        else:
            self.rename_task_via_rest(task.id, task_name)
        
        return project_name

    def __del__(self):
        """
        Destructor that deletes the zip file after it has been uploaded.
        """
        try:
            if self.zip_path.exists():
                self.zip_path.unlink()
                log.info(f"Deleted CVAT dataset zip file: {self.zip_path}")
        except Exception as e:
            log.error(f"Error deleting CVAT dataset zip file: {e}")

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