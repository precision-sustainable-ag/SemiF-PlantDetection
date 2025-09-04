#!/usr/bin/env python3
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import os
import cv2
import GPUtil
import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO
from ultralytics.data.build import check_source
from ultralytics.data.loaders import LoadImagesAndVideos, SourceTypes
from ultralytics.engine.results import Results
from ultralytics.utils import ops

from torchvision.ops import batched_nms
from hydra.utils import get_original_cwd
from src.utils.utils import get_latest_checkpoint

log = logging.getLogger(__name__)

# ------------------------------- Config ---------------------------------------

def _to_abs(p: Path) -> Path:
    """Make `p` absolute using Hydra's original CWD when needed."""
    p = Path(p)
    return p if p.is_absolute() else Path(get_original_cwd()) / p

def pick_best_device(min_free_gb: float = 8.0, exclude: list[int] | None = None) -> str:
    """
    Choose among *logical* CUDA devices exposed by CUDA_VISIBLE_DEVICES.
    Avoid GPUtil physical IDs to prevent invalid device ordinal errors.
    """
    if not torch.cuda.is_available():
        log.info("CUDA not available; using CPU")
        return "cpu"

    exclude = set(exclude or [])
    n = torch.cuda.device_count()
    if n == 0:
        log.info("No logical CUDA devices; using CPU")
        return "cpu"

    # Gather free memory for each logical device id
    candidates = []
    for i in range(n):
        if i in exclude:
            continue
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(i)
            free_gb = free_bytes / (1024 ** 3)
            candidates.append((i, free_gb))
        except Exception as e:
            log.warning(f"Skipping device {i} due to mem_get_info error: {e}")

    if not candidates:
        # if everything excluded or errored, retry without exclude
        for i in range(n):
            try:
                free_bytes, _ = torch.cuda.mem_get_info(i)
                free_gb = free_bytes / (1024 ** 3)
                candidates.append((i, free_gb))
            except Exception:
                pass

    if not candidates:
        log.info("Could not query device memory; defaulting to cuda:0")
        return "cuda:0"

    # Prefer those meeting threshold; else take max free anyway
    meeting = [c for c in candidates if c[1] >= min_free_gb]
    chosen = max(meeting or candidates, key=lambda x: x[1])[0]

    # Extra sanity: ensure chosen < device_count
    if chosen >= torch.cuda.device_count():
        log.warning(f"Chosen device {chosen} >= logical count; falling back to 0")
        chosen = 0

    # Helpful debug
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    log.info(f"CUDA_VISIBLE_DEVICES={vis} | logical_count={n} | picked cuda:{chosen}")
    return f"cuda:{chosen}"

# ------------------------------- Geometry -------------------------------------

def iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tl = torch.max(a[:, None, :2], b[None, :, :2])
    br = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter + 1e-9
    return inter / union


@torch.no_grad()
def weighted_boxes_fusion(
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_thr: float = 0.55,
    score_power: float = 1.0,
    conf_type: str = "avg",
    skip_box_thr: float = 0.0,
) -> torch.Tensor:
    """
    Returns [M,6] (xyxy, conf, cls) after fusing duplicates per class.
    """
    device = boxes_xyxy.device
    boxes_xyxy = boxes_xyxy.detach().float()
    scores = scores.detach().float()
    labels = labels.detach().float()

    keep = scores >= skip_box_thr
    boxes_xyxy, scores, labels = boxes_xyxy[keep], scores[keep], labels[keep]

    out_boxes, out_scores, out_labels = [], [], []

    for cls in labels.unique():
        m = labels == cls
        if m.sum() == 0:
            continue
        b = boxes_xyxy[m]
        s = scores[m]

        order = torch.argsort(s, descending=True)
        b = b[order]
        s = s[order]

        clusters: list[list[int]] = []
        for i in range(b.size(0)):
            if not clusters:
                clusters.append([i])
                continue
            reps = b[torch.tensor([c[0] for c in clusters], device=b.device)]
            ious = iou_xyxy(b[i : i + 1], reps).squeeze(0)
            j = torch.argmax(ious)
            if ious[j] >= iou_thr:
                clusters[j].append(i)
            else:
                clusters.append([i])

        for idxs in clusters:
            idxs_t = torch.tensor(idxs, device=b.device)
            bb = b[idxs_t]
            ss = s[idxs_t]
            w = ss ** score_power
            w = w / (w.sum() + 1e-9)
            fused = (bb * w[:, None]).sum(dim=0)

            conf = ss.max() if conf_type == "max" else ss.mean()
            out_boxes.append(fused)
            out_scores.append(conf)
            out_labels.append(cls)

    if not out_boxes:
        return torch.zeros((0, 6), device=device, dtype=torch.float32)

    out_boxes = torch.stack(out_boxes).to(device)
    out_scores = torch.stack(out_scores).to(device)
    out_labels = torch.stack(out_labels).to(device)
    return torch.cat([out_boxes, out_scores[:, None], out_labels[:, None]], dim=1)

# ------------------------- Hydra helpers (resolvers) --------------------------

def _resolve_checkpoint(cfg: DictConfig, model_dir: Path) -> Path:
    """
    Priority:
      1) cfg.evaluate.custom_path
      2) cfg.evaluate.run_version
      3) latest checkpoint in model_dir
    """
    custom_ckpt = getattr(cfg.evaluate, "custom_path", None)
    if custom_ckpt:
        custom_ckpt = _to_abs(Path(custom_ckpt))
        if custom_ckpt.exists():
            logging.getLogger(__name__).info(f"Using custom evaluation checkpoint: {custom_ckpt}")
            return custom_ckpt
        raise FileNotFoundError(f"Custom evaluation checkpoint does not exist: {custom_ckpt}")

    run_version = getattr(cfg.evaluate, "run_version", None)
    if run_version is not None:
        run_dir = model_dir / ("run" if run_version in [0, 1] else f"run{run_version}")
        ckpt_path = run_dir / "weights" / "best.pt"
        if ckpt_path.exists():
            logging.getLogger(__name__).info(f"Using checkpoint from run_version={run_version}: {ckpt_path}")
            return ckpt_path
        raise FileNotFoundError(f"Checkpoint for run_version={run_version} does not exist at: {ckpt_path}")

    latest = get_latest_checkpoint(model_dir)
    if latest:
        logging.getLogger(__name__).info(f"Using latest available checkpoint: {latest}")
        return latest

    raise FileNotFoundError("No valid checkpoint found.")


def _resolve_test_images(cfg: DictConfig) -> Path:
    """
    Resolve the path to test images from data.yaml (cfg.paths.evaluate.data_yaml).
    """
    data_yaml_path = _to_abs(Path(cfg.paths.evaluate.data_yaml))
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")

    with open(data_yaml_path) as f:
        data_cfg = yaml.safe_load(f) or {}

    base_path = Path(data_cfg.get("path", "."))
    base_path = base_path if base_path.is_absolute() else data_yaml_path.parent / base_path
    test_rel = data_cfg.get("test")
    if not test_rel:
        raise ValueError(f"No 'test:' field found in data.yaml at {data_yaml_path}")

    test_path = Path(test_rel)
    if not test_path.is_absolute():
        test_path = base_path / test_path
    test_path = test_path.resolve()

    if not test_path.is_dir():
        raise FileNotFoundError(f"'test:' path does not exist or is not a directory: {test_path}")

    return test_path

def _next_available_subdir(base: Path, stem: str = "multi_scale") -> Path:
    """
    Return base/stem if it doesn't exist; otherwise base/stem{N} with the
    smallest N>=2 that doesn't exist.
    """
    first = base / stem
    if not first.exists():
        return first

    # Find max existing numeric suffix
    pat = re.compile(rf"^{re.escape(stem)}(\d+)$")
    max_n = 1
    for d in base.iterdir():
        if d.is_dir():
            m = pat.fullmatch(d.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return base / f"{stem}{max_n + 1}"


def _resolve_save_dir(cfg: DictConfig, source: Path) -> Path:
    # Base of the evaluate save dir
    if "paths" in cfg and "evaluate" in cfg.paths and "save_dir" in cfg.paths.evaluate:
        base = _to_abs(Path(cfg.paths.evaluate.save_dir))
    else:
        base = source / "results" / "ms_infer"

    run_root = _next_available_subdir(base, stem="multi_scale")
    return run_root / "images"

# ------------------------------- Predictor ------------------------------------

class YOLOMultiscalePredictor:
    """
    Multiscale predictor for Ultralytics YOLO (v8/v11).
    Strategy:
      - Run predict() at multiple imgsz values with conf=0, iou=1 (keep all)
      - Concatenate all detections (already mapped to original image coords)
      - (Optional) WBF fuse duplicates
      - Run exactly ONE class-aware NMS in postprocess
      - Visualize & save
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Resolve model + source + save dir
        model_dir = _to_abs(Path(cfg.paths.train.model_dir))
        model_path = _resolve_checkpoint(cfg, model_dir=model_dir)
        source = _resolve_test_images(cfg)
        save_dir = _resolve_save_dir(cfg, source)

        # Device
        user_device = getattr(cfg.evaluate, "device", "auto")
        if isinstance(user_device, str) and user_device.lower() == "auto":
            ex = []
            if "evaluate" in cfg and "gpus" in cfg.evaluate:
                ex_id = getattr(cfg.evaluate.gpus, "exclude_id", None)
                if ex_id is not None:
                    ex = [ex_id]
            device = pick_best_device(min_free_gb=10.0, exclude=ex)
        else:
            device = user_device

        # Store runtime fields
        self.model_path = model_path
        self.source = source
        self.save_dir = save_dir
        self.device = device

        # Inference knobs (all required in cfg.evaluate)
        ev = cfg.evaluate

        self.base_imgsz         = int(ev.base_imgsz)
        self.scales             = tuple(float(s) for s in ev.scales)

        self.per_scale_conf     = float(ev.per_scale_conf)
        self.per_scale_iou      = float(ev.per_scale_iou)
        self.per_scale_max_det  = int(ev.per_scale_max_det)

        self.final_conf         = float(ev.conf)
        self.final_iou          = float(ev.iou)
        self.final_max_det      = int(ev.final_max_det)

        pf = ev.post_fusion_nms
        self.post_fusion_nms_enabled = bool(pf.enabled)
        self.post_fusion_nms_iou     = float(pf.iou)

        # Visualization knobs (all required)
        self.draw_labels        = bool(ev.draw_labels)
        self.draw_boxes         = bool(ev.draw_boxes)
        self.draw_conf          = bool(ev.draw_conf)
        self.line_width         = int(ev.line_width)
        self.font_size          = float(ev.font_size)
        self.names              = ev.names

        # Model
        self.model = YOLO(str(self.model_path))
        try:
            self.model.to(self.device)
        except Exception as e:
            log.warning(f"Failed to move model to {self.device}: {e}; retrying on cuda:0 then cpu")
            try:
                self.model.to("cuda:0")
                self.device = "cuda:0"
            except Exception:
                self.model.to("cpu")
                self.device = "cpu"

        self.model.fuse()
        if self.names is None:
            try:
                self.names = self.model.names
            except Exception:
                self.names = {}

        # FS
        self.save_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving predictions to: {self.save_dir}")

    def run(self) -> None:
        source, stream, screenshot, from_img, in_memory, tensor = check_source(self.source)
        source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)

        dataset = LoadImagesAndVideos(path=source, batch=1, channels=3)
        setattr(dataset, "source_type", source_type)

        for batch in dataset:
            paths, im0s_list, _info = batch
            path = Path(paths[0])
            im0 = im0s_list[0]

            log.debug(f"Inferencing: {path}")
            merged = self._multiscale_raw_preds(im0)

            log.info(f"  {len(merged)} boxes before WBF")
            if merged.numel():
                boxes, scores, clses = merged[:, :4], merged[:, 4], merged[:, 5]
                merged = weighted_boxes_fusion(
                    boxes, scores, clses, iou_thr=0.65, score_power=1.0, conf_type="max"
                )
            log.info(f"  {len(merged)} boxes after WBF")

            # Optional third-tier NMS
            if self.post_fusion_nms_enabled and merged.numel():
                # torchvision expects integer group indices for class labels
                keep = batched_nms(
                    merged[:, :4],                  # boxes [N,4]
                    merged[:, 4],                   # scores [N]
                    merged[:, 5].to(torch.int64),   # class indices [N] as int64
                    iou_threshold=self.post_fusion_nms_iou,
                )
                merged = merged[keep]
                log.info(f"  {len(merged)} boxes after extra NMS")

            results, _raw = self._postprocess_det(merged, im0, path)
            log.info(f"Detections: {results[0].boxes.shape[0]} boxes")
            self._save_visualizations(results, path)

    @torch.no_grad()
    def _multiscale_raw_preds(self, im0: np.ndarray) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        for s in self.scales:
            imgsz = max(32, int(round(self.base_imgsz * float(s))))
            r = self.model.predict(
                source=im0,
                imgsz=imgsz,
                conf=self.per_scale_conf,
                iou=self.per_scale_iou,
                max_det=self.per_scale_max_det,
                verbose=False,
                device=self.device,
            )[0]

            if r.boxes is None or len(r.boxes) == 0:
                continue

            outs.append(
                torch.cat(
                    [
                        r.boxes.xyxy.detach().cpu(),
                        r.boxes.conf[:, None].detach().cpu(),
                        r.boxes.cls[:, None].detach().cpu(),
                    ],
                    dim=1,
                )
            )

        if not outs:
            return torch.zeros((0, 6), dtype=torch.float32)
        return torch.cat(outs, dim=0)

    def _postprocess_det(
        self,
        merged_xyxy_conf_cls: torch.Tensor,
        orig_img: np.ndarray,
        path: Path,
        classes: Optional[Sequence[int]] = None,
    ) -> Tuple[List[Results], List[torch.Tensor]]:
        if merged_xyxy_conf_cls is None or merged_xyxy_conf_cls.numel() == 0:
            preds_in = torch.zeros((1, 0, 6), dtype=torch.float32)
        else:
            preds_in = merged_xyxy_conf_cls[None, ...].float()

        filtered = ops.non_max_suppression(
            preds_in,
            conf_thres=self.final_conf,
            iou_thres=self.final_iou,
            classes=list(classes) if classes is not None else None,
            agnostic=False,
            max_det=self.final_max_det,
        )

        out = filtered[0]
        results = [Results(boxes=out, orig_img=orig_img, path=str(path), names=self.names)]
        return results, [out]

    def _save_visualizations(self, results: Iterable[Results], path: Path) -> None:
        out_path = self.save_dir / path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        for res in results:
            plot_img = res.plot(
                labels=self.draw_labels,
                boxes=self.draw_boxes,
                conf=self.draw_conf,
                line_width=self.line_width,
                font_size=self.font_size,
            )
            cv2.imwrite(str(out_path), plot_img)
            log.info(f"Saved: {out_path}")

# --------------------------------- CLI ----------------------------------------


def _ensure_task_logger():
    if not log.hasHandlers():
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
        ))
        log.addHandler(h)
        log.propagate = False

    if log.level == logging.NOTSET:
        log.setLevel(logging.INFO)

def main(cfg: DictConfig):
    _ensure_task_logger()
    log.info("Config:\n" + OmegaConf.to_yaml(cfg, resolve=True))
    log.info("Starting multiscale inference task (new_ms_inferencing)")
    predictor = YOLOMultiscalePredictor(cfg)
    predictor.run()
    log.info("Finished multiscale inference task")

if __name__ == "__main__":
    main()
