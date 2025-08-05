import logging
from pathlib import Path
import cv2
import torch
import yaml
import pandas as pd
import numpy as np
from omegaconf import DictConfig, ListConfig
from ultralytics import YOLO
import torch.multiprocessing as mp
from multiprocessing import Value, Manager
from GPUtil import getAvailable
from torchvision.ops import box_iou

from src.utils.utils import get_latest_checkpoint
from src.utils.nms_methods import benchmark_nms

log = logging.getLogger(__name__)


class MultiScaleInferencer:
    def __init__(self, cfg: DictConfig):
        """
        Initialize the MultiScaleInferencer
        """
        self.cfg = cfg
        self.model_dir = Path(cfg.paths.train.model_dir)
        self.base_save_dir = Path(cfg.paths.evaluate.save_dir)
        self.scales = cfg.evaluate.scales
        self.num_gpus = cfg.evaluate.gpus.n
        self.exclude_id = cfg.evaluate.gpus.exclude_id

        conf_values = cfg.evaluate.conf
        iou_values = cfg.evaluate.iou
        self.conf_list = list(conf_values) if isinstance(conf_values, (ListConfig, list)) else [float(conf_values)]
        self.iou_list = list(iou_values) if isinstance(iou_values, (ListConfig, list)) else [float(iou_values)]

        self.model_path = self._resolve_checkpoint()
        self.source_folder = self._resolve_test_images()
        self.gt_label_folder = self.source_folder.parent / "labels"

        if isinstance(cfg.evaluate.nms_method, ListConfig) or isinstance(cfg.evaluate.nms_method, list):
            self.nms_methods = list(cfg.evaluate.nms_method)
        else:
            self.nms_methods = [cfg.evaluate.nms_method]

        self.metrics = []
        self.total_nms_time = 0.0

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
        parent_dir = base_dir / "nms_benchmark"
        parent_dir.mkdir(parents=True, exist_ok=True)

        base_name = "multi_scale"
        candidate = parent_dir / base_name
        idx = 1
        while candidate.exists():
            idx += 1
            candidate = parent_dir / f"{base_name}{idx}"

        candidate.mkdir(parents=True, exist_ok=False)
        (candidate / "images").mkdir()
        (candidate / "metrics").mkdir()
        (candidate / "plots").mkdir()
        return candidate
    
    def _load_gt_boxes(self, label_path: Path, img_w: int, img_h: int):
        """Reads YOLO format labels and converts to [x1,y1,x2,y2,class]."""
        labels = torch.tensor([list(map(float, line.split())) for line in open(label_path)], dtype=torch.float32)
        if labels.numel() == 0:
            return torch.zeros((0, 5))
        x_c, y_c, w, h = labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4]
        x1, y1 = (x_c - w / 2) * img_w, (y_c - h / 2) * img_h
        x2, y2 = (x_c + w / 2) * img_w, (y_c + h / 2) * img_h
        cls = labels[:, 0:1]
        return torch.cat([x1.unsqueeze(1), y1.unsqueeze(1), x2.unsqueeze(1), y2.unsqueeze(1), cls], dim=1)
    
    def _evaluate_nms(self, preds: torch.Tensor, gts: torch.Tensor, iou_threshold=0.5):
        """Compare predictions with ground truth boxes and return TP, FP, FN, precision, recall, F1."""
        device = preds.device
        gts = gts.to(device)

        if preds.numel() == 0:
            return 0, 0, gts.shape[0], 0.0, 0.0, 0.0
        if gts.numel() == 0:
            return 0, preds.shape[0], 0, 0.0, 0.0, 0.0

        pred_boxes, gt_boxes = preds[:, :4], gts[:, :4]
        ious = box_iou(pred_boxes, gt_boxes)

        max_iou, _ = ious.max(1)
        tp = (max_iou > iou_threshold).sum().item()
        fp = preds.shape[0] - tp
        fn = gts.shape[0] - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return tp, fp, fn, precision, recall, f1

    def run(self):
        images = sorted(list(self.source_folder.glob("*.jpg")) + list(self.source_folder.glob("*.png")))
        if not images:
            raise FileNotFoundError(f"No images found in {self.source_folder}")

        log.info(f"Found {len(images)} images in {self.source_folder}")
        save_dir = self._get_unique_save_dir(self.base_save_dir)
        metrics_csv = save_dir / "metrics" / "nms_benchmark_all.csv"

        for method in self.nms_methods:
            for conf in self.conf_list:
                for iou in self.iou_list:
                    log.info(f"Running benchmark → method={method}, conf={conf}, iou={iou}")
                    self.current_method = method
                    self.conf_thres = float(conf)
                    self.iou_thres = float(iou)

                    self._run_multi_gpu(images, save_dir)

                    # Add metadata to metrics
                    for record in self.metrics:
                        record["method"] = method
                        record["conf"] = conf
                        record["iou"] = iou

                    # Append results to CSV
                    df_current = pd.DataFrame(self.metrics)
                    if metrics_csv.exists():
                        df_existing = pd.read_csv(metrics_csv)
                        df_all = pd.concat([df_existing, df_current], ignore_index=True)
                    else:
                        df_all = df_current
                    df_all.to_csv(metrics_csv, index=False)

                    log.info(f"Metrics for {method}, conf={conf}, iou={iou} saved to {metrics_csv}")
                    self.metrics.clear()
                    self.total_nms_time = 0.0


    def _run_multi_gpu(self, images, save_dir):
        manager = Manager()
        shared_metrics = manager.list()
        shared_time = manager.Value('d', 0.0)

        available_ids = getAvailable(order='memory', limit=100, excludeID=[self.exclude_id])
        if len(available_ids) < self.num_gpus:
            raise RuntimeError(f"Requested {self.num_gpus} GPUs, but only {len(available_ids)} available")

        selected_gpus = available_ids[:self.num_gpus]
        chunks = [images[i::self.num_gpus] for i in range(self.num_gpus)]
        mp.set_start_method("spawn", force=True)

        processes = []
        for i, gpu_id in enumerate(selected_gpus):
            p = mp.Process(target=self._worker, args=(gpu_id, chunks[i], save_dir, shared_metrics, shared_time))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Merge shared data back
        self.metrics = list(shared_metrics)
        self.total_nms_time = shared_time.value
        log.info(f"Multiscale inference completed on {self.num_gpus} GPUs.")

    def _worker(self, gpu_id, images, save_dir, shared_metrics, shared_time):
        logging.basicConfig(
            level=logging.INFO,
            format=f"[GPU {gpu_id}][%(processName)s][%(levelname)s] - %(message)s"
        )
        log.info(f"Worker started on GPU {gpu_id} with {len(images)} images.")
        device = f"cuda:{gpu_id}"
        model = YOLO(self.model_path)
        model.to(device)

        for img_path in images:
            self._process_image(img_path, model, device, save_dir, shared_metrics, shared_time)     

        log.info(f"GPU {gpu_id}: Done.")

    def _process_image(self, img_path, model, device, save_dir, shared_metrics, shared_time):
        """
        Process a single image at multiple scales, perform inference, apply NMS, evaluate against GT, and save results.

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

        # Multi-scale inference
        for scale in self.scales:
            target_w, target_h = max(1, int(w0 * scale)), max(1, int(h0 * scale))
            ratio = min(max_allowed_size / target_w, max_allowed_size / target_h, 1.0)
            new_w, new_h = int(target_w * ratio), int(target_h * ratio)

            resized = cv2.resize(orig_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            imgsz = (max(new_w, new_h) + 31) // 32 * 32

            results = model.predict(
                resized,
                imgsz=imgsz,
                conf=0.0,
                iou=1.0,
                device=device,
                verbose=False
            )

            for r in results:
                if r.boxes is not None and r.boxes.xyxy is not None:
                    boxes = r.boxes.xyxy.clone()
                    confs = r.boxes.conf.clone()
                    clss = r.boxes.cls.clone()

                    # undo both scale and ratio
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

        # Load ground truth labels
        gt_path = self.gt_label_folder / f"{img_path.stem}.txt"
        gt_boxes = self._load_gt_boxes(gt_path, w0, h0) if gt_path.exists() else torch.zeros((0, 5))

        # Apply NMS
        final_preds, nms_time = benchmark_nms(
            all_preds,
            iou_thres=self.iou_thres,
            conf_thres=self.conf_thres,
            method=self.current_method
        )
        log.info(f"NMS [{self.current_method}] → {len(final_preds)} boxes kept in {nms_time:.2f} ms for {img_path.name}")

        # Evaluate against GT (TP, FP, FN)
        tp, fp, fn, precision, recall, f1 = self._evaluate_nms(final_preds, gt_boxes)
        shared_metrics.append({
            "image": img_path.name,
            "pre_nms_count": all_preds.shape[0],
            "post_nms_count": final_preds.shape[0],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "nms_time_ms": nms_time
        })
        shared_time.value += nms_time

        out_path = save_dir / "images" / f"{img_path.stem}_{self.current_method}.jpg"
        self.save_side_by_side(orig_img, all_preds, final_preds, out_path.parent, out_path.name)

    def save_side_by_side(self, orig_img, pre_preds, post_preds, save_dir, out_filename):
        """Draws Pre-NMS (blue) and Post-NMS (red) boxes and saves them side by side with thicker lines and bigger text."""
        
        def draw_boxes(img, preds, color):
            annotated = img.copy()
            for x1, y1, x2, y2, conf, cls in preds:
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 15) 
                cv2.putText(annotated, f"{conf:.2f}", (int(x1), max(0, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5) 
            return annotated

        # Pre-NMS: Blue, Post-NMS: Red
        left = draw_boxes(orig_img, pre_preds, (255, 0, 0))
        right = draw_boxes(orig_img, post_preds, (0, 0, 255)) 
        combined = np.hstack((left, right))

        # Labels with larger font
        cv2.putText(combined, "Pre-NMS", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)
        cv2.putText(combined, "Post-NMS", (orig_img.shape[1] + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)

        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / out_filename
        cv2.imwrite(str(out_path), combined)
        log.info(f"Saved side-by-side visualization: {out_path}")

    def annotate_and_save(self, orig_img, preds, save_dir, out_filename):
        annotated = orig_img.copy()
        model = YOLO(self.model_path)
        valid_preds = [b for b in preds if int(b[5]) in model.names]
        for box in valid_preds:
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / out_filename
        cv2.imwrite(str(out_path), annotated)
        log.info(f"Saved: {out_path}")


def main(cfg: DictConfig):
    inferencer = MultiScaleInferencer(cfg)
    inferencer.run()