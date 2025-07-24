import logging
from pathlib import Path
import cv2
import torch
import yaml
from omegaconf import DictConfig
from ultralytics import YOLO
from torchvision.ops import box_iou
import torch.multiprocessing as mp
from GPUtil import getAvailable

from src.utils.utils import get_latest_checkpoint

log = logging.getLogger(__name__)


def run_nms(preds: torch.Tensor, iou_thres=0.5):
    """
    Runs Non-Maximum Suppression (NMS) on a set of predictions.
    Keeps the highest confidence boxes and removes overlapping boxes beyond `iou_thres`.

    Args:
        preds (torch.Tensor): (N, 6) Tensor of predictions [x1, y1, x2, y2, conf, cls]
        iou_thres (float): IoU threshold for suppression

    Returns:
        torch.Tensor: filtered predictions after NMS
    """
    keep = []
    idxs = preds[:, 4].argsort(descending=True)

    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i.item())
        if idxs.numel() == 1:
            break
        ious = box_iou(preds[i, :4].unsqueeze(0), preds[idxs[1:], :4]).squeeze()
        idxs = idxs[1:][ious <= iou_thres]

    return preds[keep]


class MultiScaleInferencer:
    def __init__(self, cfg: DictConfig):
        """
        Initialize the MultiScaleInferencer
        """
        self.cfg = cfg
        self.model_dir = Path(cfg.paths.train.model_dir)
        self.base_save_dir = Path(cfg.paths.evaluate.save_dir)
        self.conf_thres = cfg.evaluate.conf
        self.scales = cfg.evaluate.scales
        self.num_gpus = cfg.evaluate.gpus.n
        self.exclude_id = cfg.evaluate.gpus.exclude_id

        self.model_path = self._resolve_checkpoint()
        self.source_folder = self._resolve_test_images()

    def _resolve_checkpoint(self) -> Path:
        """
        Find the appropriate model checkpoint to use.

        Priority:
        1. `cfg.evaluate.custom_path`
        2. `cfg.evaluate.run_version`
        3. latest checkpoint

        Returns:
            Path to checkpoint file
        """
        custom_ckpt = getattr(self.cfg.evaluate, "custom_path", None)
        if custom_ckpt:
            custom_ckpt = Path(custom_ckpt)
            if custom_ckpt.exists():
                log.info(f"Using custom evaluation checkpoint: {custom_ckpt}")
                return custom_ckpt
            else:
                raise FileNotFoundError(f"Custom evaluation checkpoint does not exist: {custom_ckpt}")

        run_version = getattr(self.cfg.evaluate, "run_version", None)
        if run_version is not None:
            if run_version in [0, 1]:
                run_dir = self.model_dir / "run"
            else:
                run_dir = self.model_dir / f"run{run_version}"

            ckpt_path = run_dir / "weights" / "best.pt"
            if ckpt_path.exists():
                log.info(f"Using checkpoint from run_version={run_version}: {ckpt_path}")
                return ckpt_path
            else:
                raise FileNotFoundError(f"Checkpoint for run_version={run_version} does not exist at: {ckpt_path}")

        latest = get_latest_checkpoint(self.model_dir)
        if latest:
            log.info(f"Using latest available checkpoint: {latest}")
            return latest

        raise FileNotFoundError("No valid checkpoint found.")

    def _resolve_test_images(self) -> Path:
        """
        Resolve the path to the test images from data.yaml

        Returns:
            Path to test images folder
        """
        data_yaml_path = Path(self.cfg.paths.evaluate.data_yaml)
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")
        with open(data_yaml_path) as f:
            data_cfg = yaml.safe_load(f)

        base_path = Path(data_cfg.get("path", ".")).resolve()
        test_rel = data_cfg.get("test")
        if not test_rel:
            raise ValueError(f"No 'test:' field found in data.yaml at {data_yaml_path}")

        test_path = Path(test_rel)
        if not test_path.is_absolute():
            test_path = base_path / test_path

        if not test_path.is_dir():
            raise FileNotFoundError(f"'test:' path does not exist or is not a directory: {test_path}")

        return test_path

    def _get_unique_save_dir(self, base_dir: Path) -> Path:
        base_name = "multi_scale"
        candidate = base_dir / base_name
        idx = 1

        while candidate.exists():
            idx += 1
            candidate = base_dir / f"{base_name}{idx}"

        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def run(self):
        images = sorted(list(self.source_folder.glob("*.jpg")) + list(self.source_folder.glob("*.png")))
        if not images:
            raise FileNotFoundError(f"No images found in {self.source_folder}")

        log.info(f"Found {len(images)} images in {self.source_folder}")

        save_dir = self._get_unique_save_dir(self.base_save_dir)

        self._run_multi_gpu(images, save_dir)

    def _run_multi_gpu(self, images, save_dir):
        """
        Distribute images across multiple GPUs and run inference

        Args:
            images (list[Path]): List of image paths
            save_dir (Path): Directory to save annotated images
            
        Raises:
            RuntimeError: If the number of available GPUs (after excluding `exclude_id`) is less than `num_gpus`.
        """
        available_ids = getAvailable(order='memory', limit=100, excludeID=[self.exclude_id])
        if len(available_ids) < self.num_gpus:
            raise RuntimeError(f"Requested {self.num_gpus} GPUs, but only {len(available_ids)} available after excluding GPU {self.exclude_id}")
        selected_gpus = available_ids[:self.num_gpus]

        log.info(f"Using GPUs: {selected_gpus}")
        chunks = [images[i::self.num_gpus] for i in range(self.num_gpus)]
        mp.set_start_method("spawn", force=True)

        processes = []
        for i, gpu_id in enumerate(selected_gpus):
            p = mp.Process(target=self._worker, args=(gpu_id, chunks[i], save_dir))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        log.info(f"Multiscale inference completed on {self.num_gpus} GPUs.")

    def _worker(self, gpu_id, images, save_dir):
        log.info(f"GPU {gpu_id}: Starting inference on {len(images)} images.")
        device = f"cuda:{gpu_id}"
        model = YOLO(self.model_path)
        model.to(device)

        for img_path in images:
            self._process_image(img_path, model, device, save_dir)

        log.info(f"GPU {gpu_id}: Done.")

    def _process_image(self, img_path, model, device, save_dir):
        """
        Process a single image at multiple scales, perform inference, NMS and save result.

        Args:
            img_path (Path): Image path
            model (YOLO): Loaded YOLO model
            device (str): GPU device id
            save_dir (Path): Directory to save annotated images
        """
        log.info(f"Processing: {img_path.name}")
        orig_img = cv2.imread(str(img_path))
        if orig_img is None:
            log.warning(f"Could not read image: {img_path}")
            return

        h0, w0 = orig_img.shape[:2]
        all_preds = []
        max_allowed_size = 2048

        for scale in self.scales:
            # compute target size at this scale
            target_w, target_h = max(1, int(w0 * scale)), max(1, int(h0 * scale))
            # clamp to max_allowed_size while preserving aspect ratio
            ratio = min(max_allowed_size / target_w, max_allowed_size / target_h, 1.0)
            new_w, new_h = int(target_w * ratio), int(target_h * ratio)

            resized = cv2.resize(orig_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            imgsz = max(new_w, new_h)
            imgsz = (imgsz + 31) // 32 * 32

            results = model.predict(
                resized,
                imgsz=imgsz,
                conf=self.conf_thres,
                device=device,
                verbose=False
            )

            for r in results:
                if r.boxes is not None and r.boxes.xyxy is not None:
                    boxes = r.boxes.xyxy.clone()
                    confs = r.boxes.conf.clone()
                    clss = r.boxes.cls.clone()

                    # undo both scale *and* ratio
                    resize_factor_x = new_w / w0
                    resize_factor_y = new_h / h0

                    boxes[:, [0, 2]] /= resize_factor_x
                    boxes[:, [1, 3]] /= resize_factor_y

                    pred = torch.cat([boxes, confs.unsqueeze(1), clss.unsqueeze(1)], dim=1)
                    all_preds.append(pred)

        if not all_preds:
            log.info(f"No predictions for {img_path.name}")
            return

        all_preds = torch.cat(all_preds, dim=0)
        final_preds = run_nms(all_preds, iou_thres=0.5)

        log.info(f"Final predictions: {len(final_preds)} boxes after NMS for {img_path.name}")
        self.annotate_and_save(orig_img, final_preds, img_path, save_dir)

    def annotate_and_save(self, orig_img, preds, img_path, save_dir):
        annotated = orig_img.copy()
        model = YOLO(self.model_path)
        for box in preds:
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        out_path = save_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)
        log.info(f"Saved: {out_path}")


def main(cfg: DictConfig):
    inferencer = MultiScaleInferencer(cfg)
    inferencer.run()