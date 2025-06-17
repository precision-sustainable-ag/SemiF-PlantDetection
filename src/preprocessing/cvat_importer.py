import logging
import requests
import time
from urllib.parse import quote_plus
from omegaconf import DictConfig
from pathlib import Path
from cvat_sdk import Client, models
from src.utils.utils import read_secrets

log = logging.getLogger(__name__)

class CVATImporter:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.cvat_secrets = read_secrets(cfg.paths.secrets)['cvat']
        self.cvat_dataset_format = cfg.cvat.dataset_format
        self.project_name = self.cfg.project.name
        self.task_name = self.cfg.project.task_name
        self.zip_path = Path(self.cfg.paths.project_dir, "cvat_dataset.zip")

        self.auth = (self.cvat_secrets['username'], self.cvat_secrets['password'])
        self.api_url = self.cvat_secrets['url']
    
    def login(self):
        """
        Login to CVAT
        """
        self.cvat_client = Client(self.cvat_secrets['url'])
        self.cvat_client.login(credentials=[self.cvat_secrets['username'], self.cvat_secrets['password']])

    def rename_task_via_rest(self, task_id: int, new_name: str):
        """
        Rename a CVAT task using a direct REST API PATCH call
        """
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

    def project_exists(self, name: str):
        """
        Check if a project with the given name exists on the CVAT server.
        """
        for project in self.cvat_client.projects.list():
            if project.name == name:
                return project
        return None

    def task_exists(self, project_id: int, task_name: str):
        """
        Check if a task with the given name exists within the specified project.
        """
        for task in self.cvat_client.tasks.list():
            if task.project_id == project_id and task.name == task_name:
                return task
        return None

    def import_dataset_to_project(self, project_id: int) -> str:
        """
        Import a ZIP dataset into the given CVAT project via REST API.

        Args:
            project_id (int): ID of the project to import the dataset into.

        Returns:
            str: Request ID (rq_id) for tracking import status.

        Raises:
            RuntimeError: If rq_id is missing in the response.
            HTTPError: If the POST request fails.
        """
        # properly encode format into query string
        format_encoded = quote_plus(self.cvat_dataset_format)
        url = f"{self.api_url}/api/projects/{project_id}/dataset/?format={format_encoded}&location=local"

        with open(self.zip_path, "rb") as f:
            files = {
                "dataset_file": (self.zip_path.name, f, "application/zip"),
            }

            try:
                response = requests.post(url, auth=self.auth, files=files)
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                print(f"Error: {e}")
                print("Response content:", response.content.decode())
                raise

            response_json = response.json()
            rq_id = response_json.get("rq_id")

            if not rq_id:
                raise RuntimeError("Request ID (rq_id) not found in the response.")

            log.info(f"Import request submitted with ID: {rq_id}")
            self.poll_request_status(rq_id)
            return rq_id
    
    def poll_request_status(self, rq_id: str, timeout=120):
        """
        Poll the status of a dataset import request until completion or timeout.
        """
        url = f"{self.cvat_secrets['url']}/api/requests/{rq_id}"
        auth = (self.cvat_secrets['username'], self.cvat_secrets['password'])

        for _ in range(timeout // 3):
            time.sleep(3)
            resp = requests.get(url, auth=auth)
            resp.raise_for_status()
            status = resp.json()["status"]
            log.info(f"Request {rq_id} status: {status}")
            if status == "finished":
                return
            elif status == "failed":
                raise RuntimeError(f"Dataset import failed: {resp.json()}")
        raise TimeoutError("Timed out waiting for dataset import.")

    def create_or_update_project(self) -> str:
        """
        Create a new CVAT project or update an existing one by importing a dataset and renaming the task.
        """
        project = self.project_exists(self.project_name)

        if project:
            log.info(f"Project '{self.project_name}' already exists.")

            task = self.task_exists(project.id, self.task_name)
            if task:
                log.info(f"Task '{self.task_name}' already exists in project '{self.project_name}'. Skipping dataset import.")
                return self.project_name

            # import dataset → creates task "train"
            rq_id = self.import_dataset_to_project(project.id)
            log.info(f"Import request submitted with ID: {rq_id}")

            # rename 'train' to the desired task name
            train_task = self.task_exists(project.id, "train")
            if train_task:
                self.rename_task_via_rest(train_task.id, self.task_name)
            else:
                log.warning(f"Could not find 'train' task to rename.")
            return self.project_name

        # create project if not found
        log.info(f"Creating new project '{self.project_name}'")
        new_project = self.cvat_client.projects.create(models.ProjectWriteRequest(name=self.project_name))

        # import dataset into new project
        rq_id = self.import_dataset_to_project(new_project.id)
        log.info(f"Import request submitted with ID: {rq_id}")

        # rename 'train' task
        train_task = self.task_exists(new_project.id, "train")
        if train_task:
            self.rename_task_via_rest(train_task.id, self.task_name)
        else:
            log.warning(f"Could not find 'train' task to rename.")

        return self.project_name

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

    log.info("Starting CVAT dataset uploader")
    cvat_importer = CVATImporter(cfg)
    log.info(f"Logging into cvat")
    cvat_importer.login()
    log.info(f"Uploading dataset")
    cvat_importer.create_or_update_project()

if __name__ == "__main__":
    main() 