#!/usr/bin/env python3
from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import os
import json
import cv2
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
from src.evaluating.detection_evaluator import DetectionEvaluator
from src.utils.utils import get_latest_checkpoint
from src.evaluating.detection_evaluator import (
    draw_boxes_on_image_with_colors,
    match_tp_fp_fn,
    ui_scale_for_image,
    put_counts_legend,
    put_panel_title
)
log = logging.getLogger(__name__)

# ------------------------------- Config ---------------------------------------

def _to_abs(p: Path) -> Path:
    """Make `p` absolute using Hydra's original CWD when needed."""
    p = Path(p)
    return p if p.is_absolute() else Path(get_original_cwd()) / p

def _json_default(o):
    # numpy -> python
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    # torch -> python
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    # dataclasses or other odd types
    try:
        return o.__dict__
    except Exception:
        return str(o)

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
    test_rel = data_cfg.get(cfg.evaluate.split)

    if not test_rel:
        raise ValueError(f"No '{test_rel}:' field found in data.yaml at {data_yaml_path}")

    test_path = Path(test_rel)
    if not test_path.is_absolute():
        test_path = base_path / test_path
    test_path = test_path.resolve()

    if not test_path.is_dir():
        raise FileNotFoundError(f"'{test_rel}:' path does not exist or is not a directory: {test_path}")

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

    run_root = _next_available_subdir(base, stem=cfg.evaluate.save_subdir_stem)
    return run_root / "images"

# ------------------------------- Eval helpers ------------------------------------

def _label_path_from_image_path(img_path: Path, test_root: Path, labels_root: Optional[Path]) -> Path:
    """
    Resolve YOLO label .txt path corresponding to an image.
    Priority:
      1) If labels_root provided: labels_root / relpath(img, test_root) with .txt
      2) Else: swap first 'images' segment -> 'labels' and change suffix to .txt
    """
    img_path = img_path.resolve()
    test_root = test_root.resolve()
    if labels_root:
        labels_root = labels_root.resolve()
        rel = img_path.relative_to(test_root)
        return (labels_root / rel).with_suffix(".txt")

    parts = list(img_path.parts)
    for i, p in enumerate(parts):
        if p == "images":
            parts[i] = "labels"
            break
    lbl = Path(*parts).with_suffix(".txt")
    return lbl

def _read_yolo_labels(txt_path: Path, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Read YOLO txt -> (boxes_xyxy [G,4], classes [G]).
    Boxes are in absolute pixel coords.
    """
    if not txt_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    boxes = []
    clses = []
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip().split()
            if len(s) < 5:
                continue
            c = int(float(s[0]))
            cx, cy, w, h = map(float, s[1:5])
            # YOLO (normalized center) -> xyxy pixels
            px = cx * img_w
            py = cy * img_h
            pw = w * img_w
            ph = h * img_h
            x1 = max(0.0, px - pw / 2)
            y1 = max(0.0, py - ph / 2)
            x2 = min(float(img_w), px + pw / 2)
            y2 = min(float(img_h), py + ph / 2)
            boxes.append([x1, y1, x2, y2])
            clses.append(c)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.array(boxes, dtype=np.float32), np.array(clses, dtype=np.int64)

def edge_aware_filter(
    boxes_xyxy: np.ndarray,   # [N,4] absolute pixels
    scores: np.ndarray,       # [N]
    img_wh: tuple[int, int],  # (W, H)
    *,
    base_conf: float = 0.70,      # normal final conf
    edge_band_rel: float = 0.08,  # within 8% of the nearest edge = edge zone
    min_factor: float = 0.60,     # allow down to 60% of base_conf at the edge
    taper_rel: float = 0.20       # linearly ramp back to base_conf by 20% distance
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-box dynamic threshold:
        thr_i = base_conf * f(d_edge_rel)
      where d_edge_rel in [0, inf) is the center's normalized distance to the closest edge.
      If d_edge_rel <= edge_band_rel:
          thr_i = base_conf * min_factor
      If d_edge_rel >= taper_rel:
          thr_i = base_conf
      Else linearly interpolate between those.

    Returns:
      keep_mask: [N] bool
      dyn_thr:   [N] per-box thresholds used (float32)
    """
    if len(boxes_xyxy) == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float32)

    W, H = map(float, img_wh)
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) * 0.5

    # distance (in pixels) from box center to the nearest frame edge
    d_left   = cx
    d_right  = W - cx
    d_top    = cy
    d_bottom = H - cy
    d_edge_px = np.minimum.reduce([d_left, d_right, d_top, d_bottom])

    # normalize by the smaller image dimension so it’s scale-invariant
    min_side = min(W, H)
    d_edge_rel = d_edge_px / (min_side + 1e-9)  # in [0, ~0.5]

    # piecewise-linear threshold factor
    #   close to edge → min_factor
    #   far from edge → 1.0
    #   between edge_band_rel and taper_rel → linear ramp
    f = np.ones_like(d_edge_rel, dtype=np.float32)
    near = d_edge_rel <= edge_band_rel
    far  = d_edge_rel >= taper_rel
    mid  = ~(near | far)

    f[near] = float(min_factor)
    if np.any(mid):
        # linear interpolation from (edge_band_rel -> min_factor) to (taper_rel -> 1.0)
        t = (d_edge_rel[mid] - edge_band_rel) / max(taper_rel - edge_band_rel, 1e-6)
        f[mid] = min_factor + t * (1.0 - min_factor)

    dyn_thr = (base_conf * f).astype(np.float32)
    keep_mask = scores >= dyn_thr
    return keep_mask, dyn_thr

# ------------------------------- Predictor ------------------------------------

class YOLOMultiscalePredictor:
    """
    Multiscale predictor for Ultralytics YOLO.
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
        self.viz_side_by_side   = bool(ev.eval_labels.viz_side_by_side.enabled)
        self.sbs_save_full_res  = bool(ev.eval_labels.viz_side_by_side.save_full_res)
        self.sbs_save_preview    = bool(ev.eval_labels.viz_side_by_side.save_preview)
        self.viz_single_image_preds = bool(ev.eval_labels.viz_single_image_preds)

        # Eval labels
        evl = cfg.evaluate.eval_labels
        self.eval_enabled = bool(evl.enabled)
        self.eval_iou_thresholds = [float(t) for t in evl.iou_thresholds]
        self.eval_per_image = bool(evl.per_image)
        self.eval_save_json = bool(evl.save_json)
        self.labels_dir = None if evl.labels_dir in (None, "null") else _to_abs(Path(evl.labels_dir))

        # prepare evaluator
        if self.eval_enabled:
            self.evaluator = DetectionEvaluator(self.eval_iou_thresholds, per_image=self.eval_per_image)
            # Save root (one level above images/), e.g., .../multi_scale or multi_scale2
            self.run_root = self.save_dir.parent

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

        def _safe(x, default=None):
            try:
                return x() if callable(x) else x
            except Exception:
                return default

        ckpt = _safe(lambda: getattr(getattr(self.model, "ckpt", None), "filename", None)) \
            or str(self.model_path)

        # capture class names mapping
        try:
            names_map = dict(self.names) if isinstance(self.names, dict) else self.names
        except Exception:
            names_map = None

        # curated summary from cfg.evaluate
        ev_summary = {
            "base_imgsz": int(self.base_imgsz),
            "scales": list(self.scales),
            "per_scale": {
                "conf": float(self.per_scale_conf),
                "iou": float(self.per_scale_iou),
                "max_det": int(self.per_scale_max_det),
            },
            "final": {
                "conf": float(self.final_conf),
                "iou": float(self.final_iou),
                "max_det": int(self.final_max_det),
            },
            "post_fusion_nms": {
                "enabled": bool(self.post_fusion_nms_enabled),
                "iou": float(self.post_fusion_nms_iou),
            },
            "draw": {
                "labels": bool(self.draw_labels),
                "boxes": bool(self.draw_boxes),
                "conf": bool(self.draw_conf),
                "line_width": int(self.line_width),
                "font_size": float(self.font_size),
            },
            "eval_labels": {
                "enabled": bool(self.eval_enabled),
                "iou_thresholds": list(self.eval_iou_thresholds),
                "per_image": bool(self.eval_per_image),
                "save_json": bool(self.eval_save_json),
                "labels_dir": str(self.labels_dir) if self.labels_dir else None,
            },
            "nms_method": str(getattr(self.cfg, "nms_method", "ultralytics")),
            "gpus": OmegaConf.to_container(getattr(ev, "gpus", {}), resolve=True),
        }

        # assemble metadata
        self.run_meta = {
            "model": {
                "checkpoint": ckpt,
                "names": names_map,
            },
            "io": {
                "source": str(self.source),
                "save_dir": str(self.save_dir),
                "run_root": str(getattr(self, "run_root", self.save_dir.parent)),
            },
            "evaluate_cfg_summary": ev_summary,
        }

        # write alongside metrics/images
        meta_path = Path(self.save_dir.parent) / "inference_run_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.run_meta, f, indent=2)
        log.info(f"Wrote run metadata: {meta_path}")

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

            # first tier - multi-scale raw predictions
            merged = self._multiscale_raw_preds(im0)

            # second tier - edge aware conf filter
            H, W = im0.shape[:2]
            if merged.numel():
                m_np = merged.detach().cpu().numpy()
                boxes_xyxy = m_np[:, :4].astype(np.float32)
                scores     = m_np[:, 4].astype(np.float32)

                # Using edge-aware thresholds; base on final_conf
                keep_mask, dyn_thr = edge_aware_filter(
                    boxes_xyxy, scores, img_wh=(W, H),
                    base_conf=self.final_conf,  
                    edge_band_rel=0.08,          
                    min_factor=0.50,
                    taper_rel=0.20
                )
                if keep_mask.any():
                    merged = merged[torch.from_numpy(keep_mask).to(merged.device)]
                else:
                    merged = torch.zeros((0, 6), device=merged.device, dtype=merged.dtype)

            _raw = merged
            log.info(f"{_raw.shape[0]} boxes before WBF")
            if _raw.numel():
                _raw = weighted_boxes_fusion(
                    _raw[:, :4], _raw[:, 4], _raw[:, 5],
                    iou_thr=0.65, score_power=1.0, conf_type="max"
                )
            log.info(f"{_raw.shape[0]} boxes after WBF")

            # Optional third-tier NMS
            if self.post_fusion_nms_enabled and _raw.numel():
                # torchvision expects integer group indices for class labels
                keep = batched_nms(
                    _raw[:, :4],                  # boxes [N,4]
                    _raw[:, 4],                   # scores [N]
                    _raw[:, 5].to(torch.int64),   # class indices [N] as int64
                    iou_threshold=self.post_fusion_nms_iou,
                )
                _raw = _raw[keep]

                if _raw.shape[0] > self.final_max_det:
                    order = torch.argsort(_raw[:, 4], descending=True)
                    _raw = _raw[order[: self.final_max_det]]
                log.info(f"{_raw.shape[0]} boxes after final NMS")

            final_results = [Results(
                boxes=_raw.detach().cpu() if _raw.numel() else _raw,  # [N,6]: xyxy, conf, cls
                orig_img=im0,
                path=str(path),
                names=self.names
            )]

            if self.viz_single_image_preds:
                self._save_visualizations(final_results, path)

            # Ground-truth evaluation (if enabled)
            if self.eval_enabled:
                H, W = im0.shape[:2]

                # Predictions (numpy)
                if _raw.numel():
                    p_boxes = _raw[:, :4].cpu().numpy().astype(np.float32)
                    p_scores = _raw[:, 4].cpu().numpy().astype(np.float32)
                    p_classes = _raw[:, 5].cpu().numpy().astype(np.int64)
                else:
                    p_boxes = np.zeros((0, 4), dtype=np.float32)
                    p_scores = np.zeros((0,), dtype=np.float32)
                    p_classes = np.zeros((0,), dtype=np.int64)

                # Ground-truth
                lbl_path = _label_path_from_image_path(path, self.source, self.labels_dir)
                g_boxes, g_classes = _read_yolo_labels(lbl_path, img_w=W, img_h=H)

                self.evaluator.add_image(
                    image_id=str(path),
                    pred_boxes=p_boxes, pred_scores=p_scores, pred_classes=p_classes,
                    gt_boxes=g_boxes,   gt_classes=g_classes,
                )

                if self.viz_side_by_side:
                    self._save_side_by_side(
                        im0=im0,
                        path=path,
                        gt_boxes=g_boxes,
                        gt_classes=g_classes,
                        pred_xyxy_conf_cls=_raw.detach().cpu() if _raw.numel() else _raw,  # [N,6]
                    )
        
        # Summarize detection metrics
        if self.eval_enabled:
            metrics = self.evaluator.summarize()
            if self.eval_save_json:
                out_json = self.run_root / "metrics.json"
                with open(out_json, "w") as f:
                    json.dump(metrics, f, indent=2, default=_json_default)
                log.info(f"Saved metrics: {out_json}")

            # Pretty log a quick summary
            for t in self.eval_iou_thresholds:
                key = f"{t:.2f}"
                log.info(f"mAP@{key}: {metrics['mAP'].get(key)}")
            if len(self.eval_iou_thresholds) > 1:
                log.info(f"mAP@[{'/'.join(f'{t:.2f}' for t in self.eval_iou_thresholds)}]: {metrics['mAP'].get('avg')}")

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
    ) -> Tuple[List[Results], torch.Tensor]:
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

        out = filtered[0]  # shape [N,6] (xyxy, conf, cls) or empty [0,6]
        results = [Results(boxes=out, orig_img=orig_img, path=str(path), names=self.names)]
        return results, out

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

    def _save_side_by_side(
        self,
        im0: np.ndarray,
        path: Path,
        gt_boxes: np.ndarray, gt_classes: np.ndarray,
        pred_xyxy_conf_cls: torch.Tensor,  # [N,6] xyxy, conf, cls (orig coords)
    ) -> None:
        """
        Save [ GT overlay | Prediction overlay ] with TP/FP/FN coloring + small legend.
        Colors (BGR):
        TP = green (0,255,0), FP = red (0,0,255), FN = yellow (0,255,255)
        """
        out_dir = self.save_dir / "side_by_side"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}_gt_pred.jpg"

        H, W = im0.shape[:2]
        # ensure 8-bit BGR for drawing
        if im0.dtype != np.uint8:
            im0 = np.clip(im0, 0, 255).astype(np.uint8)
        # Convert preds to numpy arrays
        if pred_xyxy_conf_cls is not None and pred_xyxy_conf_cls.numel():
            p = pred_xyxy_conf_cls.detach().cpu().numpy().astype(np.float32)  # [N,6]
            p_boxes  = p[:, :4]
            p_scores = p[:, 4]
            p_cls    = p[:, 5].astype(np.int64)
        else:
            p_boxes  = np.zeros((0, 4), dtype=np.float32)
            p_scores = np.zeros((0,), dtype=np.float32)
            p_cls    = np.zeros((0,), dtype=np.int64)

        # Default IoU threshold: first configured eval IoU if available, else 0.50
        iou_thr = float(self.eval_iou_thresholds[0]) if getattr(self, "eval_iou_thresholds", None) else 0.50

        # Match to label TP/FP and GT matched (for FN)
        pred_status, gt_matched = match_tp_fp_fn(
            pred_boxes=p_boxes, pred_scores=p_scores, pred_classes=p_cls,
            gt_boxes=gt_boxes,   gt_classes=gt_classes,
            iou_thr=iou_thr,
        )

            # --- counts for legends ---
        tp_pred = int(np.sum(pred_status == "TP"))
        fp_pred = int(np.sum(pred_status == "FP"))
        fn_gt  = int(np.sum(~gt_matched))           # GT that weren't matched
        tp_gt  = int(np.sum(gt_matched))            # matched GT (for display symmetry)

        # --- Left: GT overlay --- matched GT -> green (TP from GT view), unmatched -> yellow (FN)
        ui = ui_scale_for_image(*im0.shape[:2])  # e.g., 6k px -> ~6.0
        gt_colors = [(0,255,0) if m else (0,255,255) for m in (gt_matched.tolist() if gt_boxes.size else [])]
        gt_img = draw_boxes_on_image_with_colors(
            im0,
            gt_boxes.astype(np.float32) if gt_boxes.size else np.zeros((0, 4), dtype=np.float32),
            gt_classes.astype(np.int64) if gt_boxes.size else np.zeros((0,), dtype=np.int64),
            gt_colors,
            names=self.names,
            label_texts=None,  # <- use class names (from `names`)
            line_thickness=max(2, int(self.line_width * ui)),
            font_scale=float(self.font_size * ui),
        )
        # --- Left: GT image and legend ---
        put_panel_title(gt_img, "Ground Truth", origin=(int(0.012*W), int(0.045*H)), scale=ui)
        # Legends (counts)
        put_counts_legend(
            gt_img,
            items=[
                (f"TP (matched GT): {tp_gt}", (0,255,0)),
                (f"FN (missed GT): {fn_gt}",  (0,255,255)),
            ],
            origin=(int(0.012*W), int(0.075*H)),  # a bit below the title
            scale=ui,
            bg_alpha=0.38,
        )
        
        # --- RIGHT (Pred): confidence only ---
        pred_colors = [(0,255,0) if s == "TP" else (0,0,255) for s in (pred_status.tolist() if len(pred_status) else [])]
        pred_conf_texts = [f"{sc:.4f}" for sc in (p_scores.tolist() if len(p_scores) else [])]

        pred_img = draw_boxes_on_image_with_colors(
            im0,
            p_boxes,
            p_cls,
            pred_colors,
            names=None,                 # <- ignore class names
            label_texts=pred_conf_texts,  # <- show confidence numbers only
            line_thickness=max(2, int(self.line_width * ui)),
            font_scale=float(self.font_size * ui),
        )
        # --- Right: Pred image and legend ---
        put_panel_title(pred_img, "Predictions", origin=(int(0.012*W), int(0.045*H)), scale=ui)
        put_counts_legend(
            pred_img,
            items=[
                (f"TP (correct): {tp_pred}",  (0,255,0)),
                (f"FP (spurious): {fp_pred}", (0,0,255)),
            ],
            origin=(int(0.012*W), int(0.075*H)),
            scale=ui,
            bg_alpha=0.38,
        )

        # --- Concat side by side
        if gt_img.shape[0] != pred_img.shape[0]:
            new_w = int(round(pred_img.shape[1] * (gt_img.shape[0] / pred_img.shape[0])))
            pred_img = cv2.resize(pred_img, (new_w, gt_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        side = np.concatenate([gt_img, pred_img], axis=1)

        # Save a preview capped to 2400 px width (keeps aspect)
        if self.sbs_save_preview and side.shape[1] > 2400:
            preview_w = 2400
            ratio = preview_w / side.shape[1]
            preview_h = max(1, int(round(side.shape[0] * ratio)))
            preview = cv2.resize(side, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
            preview_path = out_path.with_name(out_path.stem + "_preview.jpg")
            cv2.imwrite(str(preview_path), preview)
            log.info(f"Saved preview: {preview_path} ({preview_w}x{preview_h})")
        
        if self.sbs_save_full_res or side.shape[1] <= 2400:
            cv2.imwrite(str(out_path), side)
            log.info(f"Saved side-by-side GT|Pred (TP/FP/FN) @ IoU>={iou_thr:.2f}: {out_path}")

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
