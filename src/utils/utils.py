import yaml
import logging
from pathlib import Path

log = logging.getLogger(__name__)

def read_secrets(keypath):
    with open(keypath, "r") as file:
        secrets = yaml.safe_load(file)
        
    return secrets

def find_most_recent_dataset_path(base_dataset_path):
    """
    Find the most recent dataset directory from a hierarchical date/time structure.
    
    The function looks for:
    1. The most recent date directory (YYYY-MM-DD)
    2. The most recent timestamp directory (hr-mm-ss) within that date directory
    
    Args:
        base_dataset_path (str or Path): Base directory path where dataset directories are located
        
    Returns:
        Path: The path to the most recent dataset directory
    """
    base_dataset_path = Path(base_dataset_path)
    
    if base_dataset_path.exists():
        date_dirs = [d for d in base_dataset_path.iterdir() if d.is_dir()]
        date_dirs.sort(reverse=True)
        if date_dirs:
            # Now find the timestamp (hr-mm-ss) folders inside the most recent date directory
            hr_min_sec_dirs = [d for d in date_dirs[0].iterdir() if d.is_dir()]
            hr_min_sec_dirs.sort(reverse=True)
            if hr_min_sec_dirs:
                dataset_path = hr_min_sec_dirs[0]
                log.info(f"Using most recent timestamp directory: {dataset_path}")
            else:
                dataset_path = date_dirs[0]
                log.warning(f"No timestamp (hr-mm-ss) directories found, using recent date path: {dataset_path}")
        else:
            dataset_path = base_dataset_path
            log.warning(f"No timestamp directories found, using base path: {dataset_path}")
    else:
        dataset_path = base_dataset_path
        log.warning(f"Dataset path does not exist: {dataset_path}")
    
    return dataset_path