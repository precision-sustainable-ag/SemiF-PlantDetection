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

def get_annotated_image_ids(lts_locations):
    """
    Get all annotated image ids from a base path.
    Read all directories in base_path for exported cvat annotations
    - failsafe - also include a local directory (and copy it to lts) - in case user doesn't upload there
    Validate format
    return list of image ids and/or the annotations themselves (since validating)
    Args:
        lts_locations List[str or Path]: List of base directory paths where human annotations are located
        
    Returns:
        List[str]: image ids for images already annotated
    """
    image_ids = {}
    # TODO: check data.yaml in each subdirectory to verify 3 classes
    # Check if base path exists
    for lts_location in lts_locations:
        base_path = Path(lts_location)
        if not base_path.exists():
            log.warning(f"Base path does not exist: {base_path}")
            continue
        # Go through all directories in the base path
        for directory in [d for d in base_path.iterdir() if d.is_dir()]:
            # Look for 'labels' subfolder
            labels_dir = directory / "labels"
            if labels_dir.exists() and labels_dir.is_dir():
                # Use recursive glob to get all txt files in labels directory and its subfolders
                txt_files = labels_dir.glob("**/*.txt")
                # Extract filenames without extension and add to image_ids
                for txt_file in txt_files:
                    image_id = txt_file.stem  # Get filename without extension
                    if image_id in image_ids.keys():
                        log.error(f"Found multiple annotations for {image_id}")
                        raise ValueError(f"Found multiple annotations for {image_id}")
                    
                    image_ids[image_id] = txt_file
                    # image_ids.append(image_id)
                    # full_paths.append(txt_file)
    
    log.info(f"Found {len(image_ids)} annotated image IDs")
    return image_ids

def convert_bbox_to_yolo_format(bbox, image_width, image_height):
    """
    Convert bounding box from [x, y, width, height] (top-left) to YOLO format 
    [center_x, center_y, width, height] (normalized).
    
    Args:
        bbox (List[int]): Bounding box in [x, y, width, height] format
        image_width (int): Width of the image
        image_height (int): Height of the image
        
    Returns:
        List[float]: Bounding box in YOLO format [center_x, center_y, width, height] (normalized)
    """
    x, y, width, height = bbox
    
    # Convert to center coordinates
    center_x = (x + width / 2) / image_width
    center_y = (y + height / 2) / image_height
    
    # Normalize width and height
    normalized_width = width / image_width
    normalized_height = height / image_height
    
    return [center_x, center_y, normalized_width, normalized_height]
