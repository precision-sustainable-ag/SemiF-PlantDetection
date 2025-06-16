import logging
import requests
import time
import zipfile
import shutil
from pathlib import Path
from omegaconf import DictConfig
from cvat_sdk import Client
from src.utils.utils import read_secrets

log = logging.getLogger(__name__)

class CVATExporter:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.cvat_secrets = read_secrets(cfg.paths.secrets)['cvat']
        self.dataset_format = cfg.cvat.dataset_format
        self.project_name = cfg.project.name
        self.task_name = cfg.project.task_name

        # output directory for annotations
        self.project_annotations_dir = Path(cfg.paths.preprocess.label_dir)

        # LTS path for annotations
        self.lts_annotations_dir = Path(cfg.paths.lts_human_annotations)

        self.auth = (self.cvat_secrets['username'], self.cvat_secrets['password'])
        self.api_url = self.cvat_secrets['url']

    def login(self):
        self.cvat_client = Client(self.api_url)
        self.cvat_client.login([self.auth[0], self.auth[1]])

    def get_task_and_job(self):
        # find the project
        projects = self.cvat_client.projects.list()
        project = next((p for p in projects if p.name == self.project_name), None)
        if not project:
            log.warning(f"[Export Skipped] Project '{self.project_name}' not found.")
            return None

        # find the task within the project
        tasks = self.cvat_client.tasks.list()
        task = next((t for t in tasks if t.project_id == project.id and t.name == self.task_name), None)
        if not task:
            log.warning(f"[Export Skipped] Task '{self.task_name}' not found in project '{self.project_name}'.")
            return None

        # find the job for the task
        jobs = self.cvat_client.jobs.list()
        job = next((j for j in jobs if j.task_id == task.id), None)
        if not job:
            log.warning(f"[Export Skipped] No job found for task '{self.task_name}'.")
            return None

        return self.cvat_client.jobs.retrieve(job.id)

    def export_annotations(self, job_id: int):
        url = f"{self.api_url}/api/jobs/{job_id}/dataset/export"
        params = {
            "format": self.dataset_format,
            "save_images": "false"
        }

        log.info(f"Initiating export for job ID: {job_id}")
        response = requests.post(url, auth=self.auth, params=params)
        response.raise_for_status()

        rq_id = response.json().get("rq_id")
        if not rq_id:
            raise RuntimeError("Failed to get request ID from CVAT export API.")
        return rq_id
    
    def export_annotations_lts(self, target_dir: Path):
        source_dir = self.project_annotations_dir / "labels" / "train"
        if not source_dir.exists():
            log.warning(f"No annotations found in {source_dir}. Skipping export to LTS.")
            return

        target_dir.mkdir(parents=True, exist_ok=True)
        label_files = list(source_dir.glob("*.txt"))

        if not label_files:
            log.warning(f"No .txt label files found in {source_dir}.")
            return

        for label_file in label_files:
            shutil.copy2(label_file, target_dir / label_file.name)

        log.info(f"Exported {len(label_files)} image annotation(s) to '{target_dir}'.")

    def poll_export_status(self, rq_id: str, timeout=120) -> str:
        url = f"{self.api_url}/api/requests/{rq_id}"

        for _ in range(timeout // 3):
            time.sleep(3)
            response = requests.get(url, auth=self.auth)
            response.raise_for_status()
            status_data = response.json()
            status = status_data["status"]
            log.info(f"Export request {rq_id} status: {status}")

            if status == "finished":
                if "result_url" in status_data:
                    return status_data["result_url"]
                else:
                    raise RuntimeError(f"Export finished but no result_url found: {status_data}")

            elif status == "failed":
                raise RuntimeError(f"Export failed: {status_data}")

        raise TimeoutError("Timed out waiting for annotation export.")


    def download_and_extract(self, download_url: str):
        log.info(f"Downloading annotation zip from: {download_url}")
        zip_response = requests.get(download_url, auth=self.auth)
        zip_response.raise_for_status()

        zip_path = self.project_annotations_dir / "annotations.zip"
        with open(zip_path, "wb") as f:
            f.write(zip_response.content)

        # extract
        log.info(f"Unzipping to {self.project_annotations_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.project_annotations_dir)

        # delete zip
        zip_path.unlink()
        log.info(f"Deleted temporary zip: {zip_path}")

    def run(self):
        job = self.get_task_and_job()
        if job is None:
            return

        # make the annotations dir after making sure that the project/task exists in cvat
        self.project_annotations_dir.mkdir(parents=True, exist_ok=True)

        rq_id = self.export_annotations(job.id)
        download_url = self.poll_export_status(rq_id)
        self.download_and_extract(download_url)
        self.export_annotations_lts(self.lts_annotations_dir)

def main(cfg: DictConfig) -> None:
    log.info("Starting CVAT annotation export process")
    exporter = CVATExporter(cfg)
    exporter.login()
    exporter.run()
    log.info("Annotation export step completed (may have been skipped if project/task/job was missing)")

if __name__ == "__main__":
    main()