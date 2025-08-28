#!/usr/bin/env python3
from __future__ import annotations
from torchvision.ops import batched_nms
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import torch
import GPUtil
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.build import check_source
from ultralytics.data.loaders import LoadImagesAndVideos, SourceTypes
from ultralytics.engine.results import Results
from ultralytics.utils import ops

from torchvision.ops import batched_nms
# ------------------------------- Config ---------------------------------------

def pick_best_device(min_free_gb: float = 8.0) -> str:
    """
    Return 'cuda:{idx}' for the GPU with the most free memory that meets min_free_gb.
    Fall back to the GPU with the most free memory, or 'cpu' if no CUDA / no GPUs.
    """
    if not torch.cuda.is_available():
        return "cpu"

    gpus = GPUtil.getGPUs()
    if not gpus:
        return "cpu"

    # candidates with free mem >= min_free_gb
    candidates = [(i, g.memoryFree/1024) for i, g in enumerate(gpus) if (g.memoryFree/1024) >= min_free_gb]
    if candidates:
        best = max(candidates, key=lambda x: x[1])[0]
        return f"cuda:{best}"

    # otherwise, take the most free overall
    best_any = max(enumerate(gpus), key=lambda x: x[1].memoryFree)[0]
    return f"cuda:{best_any}"

@dataclass
class InferenceConfig:
    model_path: Path
    source: Path
    save_dir: Path
    device: str = "cuda:0"
    base_imgsz: int = 4000                       # be careful with VRAM; 5k can blow up
    scales: Tuple[float, ...] = (0.15, 0.25, 0.5, 1.0, 1.5)

    # “No-op” per-scale settings to keep everything, we do the single NMS at the end:
    per_scale_conf: float = 0.0
    per_scale_iou: float = 1.0
    per_scale_max_det: int = 300

    # Final postprocess (single) NMS settings:
    final_conf: float = 0.75
    final_iou: float = 0.55
    final_max_det: int = 500

    # Visualization
    draw_labels: bool = True
    draw_boxes: bool = True
    draw_conf: bool = True
    line_width: int = 5
    font_size: float = 4.0

    # Class name map (optional). If None, uses model.names
    names: Optional[dict[int, str]] = field(default=None)



def iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [Na,4], b: [Nb,4]
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
    boxes_xyxy: torch.Tensor,      # [N,4]
    scores: torch.Tensor,          # [N]
    labels: torch.Tensor,          # [N]
    iou_thr: float = 0.55,
    score_power: float = 1.0,      # weight = score ** score_power
    conf_type: str = "avg",        # "avg" or "max"
    skip_box_thr: float = 0.0
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

    out_boxes = []
    out_scores = []
    out_labels = []

    for cls in labels.unique():
        m = labels == cls
        if m.sum() == 0:
            continue
        b = boxes_xyxy[m]
        s = scores[m]

        # sort by score desc for deterministic clustering
        order = torch.argsort(s, descending=True)
        b = b[order]
        s = s[order]

        clusters = []   # list of (indices_in_b) for each cluster
        for i in range(b.size(0)):
            if len(clusters) == 0:
                clusters.append([i])
                continue
            # compare with representative of each cluster (the first/highest score)
            reps = b[torch.tensor([c[0] for c in clusters], device=b.device)]
            ious = iou_xyxy(b[i:i+1], reps).squeeze(0)  # [num_clusters]
            j = torch.argmax(ious)
            if ious[j] >= iou_thr:
                clusters[j].append(i)
            else:
                clusters.append([i])

        # fuse each cluster by weighted average of coordinates
        for idxs in clusters:
            idxs = torch.tensor(idxs, device=b.device)
            bb = b[idxs]        # [K,4]
            ss = s[idxs]        # [K]
            w = ss ** score_power
            w = w / (w.sum() + 1e-9)
            fused = (bb * w[:, None]).sum(dim=0)

            if conf_type == "max":
                conf = ss.max()
            else:
                conf = ss.mean()

            out_boxes.append(fused)
            out_scores.append(conf)
            out_labels.append(cls)

    if not out_boxes:
        return torch.zeros((0, 6), device=device, dtype=torch.float32)

    out_boxes = torch.stack(out_boxes).to(device)
    out_scores = torch.stack(out_scores).to(device)
    out_labels = torch.stack(out_labels).to(device)

    return torch.cat([out_boxes, out_scores[:, None], out_labels[:, None]], dim=1)

# ------------------------------- Predictor ------------------------------------


class YOLOMultiscalePredictor:
    """
    Multiscale predictor for Ultralytics YOLO (v8/v11).
    Strategy:
      - Run predict() at multiple imgsz values with conf=0, iou=1 (keep all)
      - Concatenate all detections (already mapped to original image coords)
      - Run exactly ONE class-aware NMS in postprocess
      - Visualize & save
    """

    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.model = YOLO(str(cfg.model_path))
        self.model.to(cfg.device)
        self.model.fuse()  # safe noop for most models; speeds inference if supported

        # Resolve names: prefer user-provided mapping, else model.names
        try:
            self.names = cfg.names or self.model.names
        except Exception:
            self.names = cfg.names or {}

        self._setup_logging()
        self.cfg.save_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------- Public API -----------------------------------

    def run(self) -> None:
        """Run inference on the configured source and save visualizations."""
        source, stream, screenshot, from_img, in_memory, tensor = check_source(self.cfg.source)
        source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)

        dataset = LoadImagesAndVideos(path=source, batch=1, channels=3)
        setattr(dataset, "source_type", source_type)

        for batch in dataset:
            # dataset yields tuples; for images: (paths, im0s, info)
            paths, im0s_list, _info = batch
            # batch size is 1 (by construction), unwrap
            path = Path(paths[0])
            im0 = im0s_list[0]  # HWC, BGR uint8

            self.log.debug(f"Inferencing: {path}")

            # if self.cfg.per_scale_max_det > 0:
            merged = self._multiscale_raw_preds(im0)
            # else:
            #     merged = self._multiscale_nms_preds(im0)
            # After multiscale_raw_preds() -> merged [N,6] (xyxy, conf, cls)
            self.log.info(f"  {len(merged)} boxes before final NMS")
            if merged.numel():
                boxes, scores, clses = merged[:, :4], merged[:, 4], merged[:, 5]
                merged = weighted_boxes_fusion(
                    boxes, scores, clses,
                    iou_thr=0.65,         # try 0.55–0.65
                    score_power=1.0,      # try 1.0–2.0 if you want high-conf to dominate
                    conf_type="max"       # or "max" if you prefer the top score
                )
            # else:
                # fused = merged  # empty

            self.log.info(f"  {len(merged)} boxes after fusion")
            
            # Optionally run a light NMS after fusion to remove any stragglers:
            # keep = batched_nms(merged[:, :4], merged[:, 4], merged[:, 5], iou_threshold=0.6)
            # merged = merged[keep]
            # self.log.info(f"  {len(merged)} boxes after final NMS")

            results, _raw = self._postprocess_det(merged, im0, path)

            self.log.info(f"Detections: {results[0].boxes.shape[0]} boxes")

            self._save_visualizations(results, path)
            del results, merged, _raw

    # ------------------------- Core Operations --------------------------------

    @torch.no_grad()
    def _multiscale_raw_preds(self, im0: np.ndarray) -> torch.Tensor:
        """
        Run model.predict across scales, capturing all outputs (no per-scale NMS).
        Returns concatenated [N,6] tensor with columns [x1,y1,x2,y2,conf,cls] in original coords.
        """
        outs: List[torch.Tensor] = []
        for s in self.cfg.scales:
            imgsz = max(32, int(round(self.cfg.base_imgsz * s)))
            r = self.model.predict(
                source=im0,
                imgsz=imgsz,
                conf=self.cfg.per_scale_conf,  # keep all
                iou=self.cfg.per_scale_iou,    # suppress none
                max_det=self.cfg.per_scale_max_det,
                verbose=False,
                device=self.cfg.device,
            )[0]

            if r.boxes is None or len(r.boxes) == 0:
                continue

            # boxes already mapped to original image geometry by Ultralytics
            outs.append(torch.cat(
                [r.boxes.xyxy.detach().cpu(),
                 r.boxes.conf[:, None].detach().cpu(),
                 r.boxes.cls[:, None].detach().cpu()],
                dim=1
            ))

        if not outs:
            return torch.zeros((0, 6), dtype=torch.float32)

        return torch.cat(outs, dim=0)  # [N,6]
    
    @torch.no_grad()
    def _multiscale_nms_preds(self, im0):
        boxes_all, scores_all, cls_all = [], [], []

        for s in self.cfg.scales:
            imgsz = max(32, int(round(self.cfg.base_imgsz * float(s))))
            r = self.model.predict(
                source=im0,
                imgsz=imgsz,
                conf=self.cfg.conf,
                iou=self.cfg.iou,
                verbose=False,
                max_det=self.cfg.per_scale_max_det,
            )[0]  # Results for single image

            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes_all.append(r.boxes.xyxy.detach().cpu())   # [N,4] in original image coords
            scores_all.append(r.boxes.conf.detach().cpu())  # [N]
            cls_all.append(r.boxes.cls.detach().cpu())      # [N]

        if not boxes_all:
            return torch.zeros((0, 6), dtype=torch.float32)

        boxes = torch.cat(boxes_all)
        scores = torch.cat(scores_all)
        clses = torch.cat(cls_all)

        keep = batched_nms(boxes, scores, clses, self.cfg.iou)
        boxes, scores, clses = boxes[keep], scores[keep], clses[keep]

        if boxes.shape[0] > self.cfg.per_scale_max_det:
            topk = torch.topk(scores, k=self.cfg.per_scale_max_det).indices
            boxes, scores, clses = boxes[topk], scores[topk], clses[topk]

        return torch.cat([boxes, scores[:, None], clses[:, None]], dim=1).float()  # [N,6]


    def _postprocess_det(
        self,
        merged_xyxy_conf_cls: torch.Tensor,
        orig_img: np.ndarray,
        path: Path,
        classes: Optional[Sequence[int]] = None
    ) -> Tuple[List[Results], List[torch.Tensor]]:
        """
        Single NMS over concatenated predictions.
        Returns Ultralytics Results for easy plotting.
        """
        if merged_xyxy_conf_cls is None or merged_xyxy_conf_cls.numel() == 0:
            preds_in = torch.zeros((1, 0, 6), dtype=torch.float32)
        else:
            preds_in = merged_xyxy_conf_cls[None, ...].float()  # [1,N,6]

        filtered = ops.non_max_suppression(
            preds_in,
            conf_thres=self.cfg.final_conf,
            iou_thres=self.cfg.final_iou,
            classes=list(classes) if classes is not None else None,
            agnostic=False,
            max_det=self.cfg.final_max_det,
        )

        out = filtered[0]  # [M,6] in original coords
        results = [Results(boxes=out, orig_img=orig_img, path=str(path), names=self.names)]
        return results, [out]

    def _save_visualizations(self, results: Iterable[Results], path: Path) -> None:
        """
        Save annotated image(s) side-by-side with the input filename under save_dir.
        """
        out_path = self.cfg.save_dir / path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        for res in results:
            plot_img = res.plot(
                labels=self.cfg.draw_labels,
                boxes=self.cfg.draw_boxes,
                conf=self.cfg.draw_conf,
                line_width=self.cfg.line_width,
                font_size=self.cfg.font_size,
            )
            # plot() returns BGR np.ndarray
            cv2.imwrite(str(out_path), plot_img)
            self.log.info(f"Saved: {out_path}")

    # ----------------------------- Utils --------------------------------------

    def _setup_logging(self) -> None:
        self.log = logging.getLogger("YOLOMultiscalePredictor")
        if not self.log.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
            handler.setFormatter(fmt)
            self.log.addHandler(handler)
            self.log.setLevel(logging.INFO)


# ------------------------------- CLI -----------------------------------------


def make_config() -> InferenceConfig:

    model_path = Path("/home/psa_images/temp_data/temp_detection_model/best.pt")
    # model_path = Path("data/semifield-tools/detection_model/last.pt")
    source = Path("infer_compare/images") 
    save_dir = source / "results" / "new_msinf_anuraags_model"
    device = pick_best_device(min_free_gb=10.0)

    return InferenceConfig(
        model_path=model_path,
        source=source,
        save_dir=save_dir,
        device=device
        )


def main():
    cfg = make_config()
    predictor = YOLOMultiscalePredictor(cfg)
    predictor.run()


if __name__ == "__main__":
    main()
