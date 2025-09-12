from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Sequence

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False


def _iou_xyxy_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NumPy IoU for xyxy boxes. a:[Na,4], b:[Nb,4] -> [Na,Nb]"""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, a_min=0.0, a_max=None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union

def _voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """
    VOC-style AP (area under P-R curve with 11/continuous interpolation).
    Here we use the continuous interpolation implementation.
    """
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

# ------------------------------- Evaluator ------------------------------------

class DetectionEvaluator:
    """
    Accumulates predictions/ground-truth and computes AP/mAP across IoU thresholds.
    """
    def __init__(
        self,
        iou_thresholds: Sequence[float],
        per_image: bool = False,
        *,
        match_algo: str = "hungarian",            # "greedy" | "hungarian"
        hungarian_cost: str = "1-iou",         # "1-iou" | "1-(alpha*iou+(1-alpha)*score)"
        hungarian_alpha: float = 0.8,
        per_image_mode: str = "full",          # "full" | "lite"
        per_image_k: int = 50,                  # keep top-K FP/FN examples in lite mode
        keep_iou_hist: bool = False,            # store small IoU histogram bins
    ):
        self.iou_thresholds = [float(t) for t in iou_thresholds]
        self.per_image = per_image

        self.match_algo = match_algo.lower()
        self.hungarian_cost = hungarian_cost.lower()
        self.hungarian_alpha = float(hungarian_alpha)
        self.per_image_mode = per_image_mode.lower()
        self.per_image_k = int(per_image_k)
        self.keep_iou_hist = bool(keep_iou_hist)

        self.preds_by_class = defaultdict(list)
        self.gts_by_image_class = defaultdict(lambda: defaultdict(list))
        self.num_gt_by_class = defaultdict(int)
        self.per_image_stats = {}

    def add_image(
        self,
        image_id: str,
        pred_boxes: np.ndarray, pred_scores: np.ndarray, pred_classes: np.ndarray,
        gt_boxes: np.ndarray, gt_classes: np.ndarray,
    ):
        # store gts
        for c in np.unique(gt_classes):
            m = (gt_classes == c)
            b = gt_boxes[m]
            self.gts_by_image_class[image_id][int(c)].extend(b.tolist())
            self.num_gt_by_class[int(c)] += int(m.sum())

        # store preds
        for c in np.unique(pred_classes):
            m = (pred_classes == c)
            boxes = pred_boxes[m]
            scores = pred_scores[m]
            for box, s in zip(boxes, scores):
                self.preds_by_class[int(c)].append({"image_id": image_id, "score": float(s), "box": box.tolist()})

        if self.per_image:
            self.per_image_stats.setdefault(image_id, {"meta": {
                "num_pred": int(len(pred_boxes)),
                "num_gt": int(len(gt_boxes))
            }})
            # compute per-image TP/FP/FN for each IoU threshold (and per class)
            self._accumulate_per_image_stats(
                image_id,
                pred_boxes.astype(np.float32),
                pred_scores.astype(np.float32),
                pred_classes.astype(np.int64),
                gt_boxes.astype(np.float32),
                gt_classes.astype(np.int64),
            )

    def _compute_ap_for_class(self, c: int, iou_thr: float) -> dict:
        preds = self.preds_by_class.get(c, [])
        npos = self.num_gt_by_class.get(c, 0)
        if npos == 0:
            return {"AP": np.nan, "precision": [], "recall": []}

        # sort predictions by score desc
        preds = sorted(preds, key=lambda x: x["score"], reverse=True)
        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        # matched gt flags per image
        matched = {img_id: np.zeros(len(self.gts_by_image_class[img_id].get(c, [])), dtype=bool)
                   for img_id in self.gts_by_image_class}

        for i, p in enumerate(preds):
            img_id = p["image_id"]
            gt_list = self.gts_by_image_class[img_id].get(c, [])
            gt = np.array(gt_list, dtype=np.float32)
            box = np.array(p["box"], dtype=np.float32)[None, :]

            if gt.size == 0:
                fp[i] = 1.0
                continue

            ious = _iou_xyxy_np(box, gt).squeeze(0)
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not matched[img_id][j]:
                tp[i] = 1.0
                matched[img_id][j] = True
            else:
                fp[i] = 1.0

        # precision-recall
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / max(npos, 1)
        prec = np.divide(tp_cum, (tp_cum + fp_cum + 1e-9))
        ap = _voc_ap(rec, prec)
        return {"AP": ap, "precision": prec.tolist(), "recall": rec.tolist(), "npos": int(npos)}
    
    def _greedy_match_single_class(self, preds_c, gts_c, iou_thr: float):
        """
        preds_c: (P, 5) -> [x1,y1,x2,y2,score]
        gts_c:   (G, 4)
        Returns: tp, fp, fn, matches (list of (pi, gi)), ious_row (P,), best_iou_per_gt (G,)
        """
        P = len(preds_c); G = len(gts_c)
        if P == 0 and G == 0: return 0, 0, 0, [], np.array([]), np.array([])
        if P == 0:           return 0, 0, G, [], np.array([]), np.zeros(G, np.float32)
        if G == 0:           return 0, P, 0, [], np.zeros(P, np.float32), np.array([])

        # sort predictions by score desc (old behavior)
        order = np.argsort(-preds_c[:, 4])
        preds_c = preds_c[order]
        ious = _iou_xyxy_np(preds_c[:, :4], gts_c)  # (P,G)

        matched_g = np.zeros(G, dtype=bool)
        matches = []
        tp = 0; fp = 0
        for i in range(P):
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= iou_thr and not matched_g[j]:
                matched_g[j] = True
                matches.append((i, j))
                tp += 1
            else:
                fp += 1
        fn = int(G - matched_g.sum())

        # diagnostics: best IoU each pred and each gt
        best_iou_pred = ious.max(axis=1) if P else np.array([], np.float32)
        best_iou_gt   = ious.max(axis=0) if G else np.array([], np.float32)
        return tp, fp, fn, matches, best_iou_pred, best_iou_gt


    def _hungarian_match_single_class(self, preds_c, gts_c, iou_thr: float):
        """
        Optimal one-to-one assignment w.r.t. cost built from IoU (and optionally score).
        Returns same tuple as greedy version.
        """
        P = len(preds_c); G = len(gts_c)
        if P == 0 and G == 0: return 0, 0, 0, [], np.array([]), np.array([])
        if P == 0:           return 0, 0, G, [], np.array([]), np.zeros(G, np.float32)
        if G == 0:           return 0, P, 0, [], np.zeros(P, np.float32), np.array([])

        if not _SCIPY_OK:
            # fallback to greedy if SciPy is not available
            return self._greedy_match_single_class(preds_c, gts_c, iou_thr)

        # sort by score desc for consistent diagnostics only
        order = np.argsort(-preds_c[:, 4])
        preds_c = preds_c[order]

        ious = _iou_xyxy_np(preds_c[:, :4], gts_c)  # (P,G)

        # Build cost
        if self.hungarian_cost == "1-iou":
            cost = 1.0 - ious
        else:
            # 1 - (alpha*iou + (1-alpha)*sigmoid(score))
            alpha = self.hungarian_alpha
            scores = preds_c[:, 4:5]  # (P,1)
            sig = 1.0 / (1.0 + np.exp(-scores))
            combo = alpha * ious + (1 - alpha) * sig
            cost = 1.0 - combo

        # Disallow matches below IoU threshold by giving them huge cost
        cost = cost.copy()
        cost[ious < iou_thr] = 1e6

        # Hungarian on rectangular matrices is fine
        row_ind, col_ind = linear_sum_assignment(cost)

        matched = []
        tp = 0
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 1e6:  # valid match
                matched.append((r, c))
                tp += 1
        fp = P - tp
        fn = G - tp

        best_iou_pred = ious.max(axis=1) if P else np.array([], np.float32)
        best_iou_gt   = ious.max(axis=0) if G else np.array([], np.float32)

        return tp, fp, fn, matched, best_iou_pred, best_iou_gt
    
    def _match_one_class(self, preds_c, gts_c, iou_thr: float):
        if self.match_algo == "hungarian":
            return self._hungarian_match_single_class(preds_c, gts_c, iou_thr)
        return self._greedy_match_single_class(preds_c, gts_c, iou_thr)

    def _accumulate_per_image_stats(
        self, image_id: str,
        pred_boxes: np.ndarray, pred_scores: np.ndarray, pred_classes: np.ndarray,
        gt_boxes:   np.ndarray, gt_classes:   np.ndarray
    ):
        """
        Compute per-image TP/FP/FN at each IoU threshold in self.iou_thresholds.
        Stores:
        self.per_image_stats[image_id][f"{t:.2f}"]["all"] = {tp, fp, fn}
        and a class breakdown:
        self.per_image_stats[image_id][f"{t:.2f}"]["by_class"][c] = {tp, fp, fn, n_pred, n_gt}
        """
        # init container if first time
        if image_id not in self.per_image_stats:
            self.per_image_stats[image_id] = {}

        classes = sorted(set(list(pred_classes.astype(int)) + list(gt_classes.astype(int))))

        for t in self.iou_thresholds:
            tp_total = fp_total = fn_total = 0
            by_class = {}

            # Optional per-image IoU histogram across all preds/GTs (diagnostic)
            # We'll build from best IoU per pred
            all_best_iou_pred = []

            for c in classes:
                m_p = (pred_classes == c)
                preds_c = (np.concatenate([pred_boxes[m_p], pred_scores[m_p, None]], axis=1)
                        if m_p.any() else np.zeros((0, 5), dtype=np.float32))

                m_g = (gt_classes == c)
                gts_c = gt_boxes[m_g] if m_g.any() else np.zeros((0, 4), dtype=np.float32)

                tp_c, fp_c, fn_c, matches, best_iou_pred, best_iou_gt = self._match_one_class(preds_c, gts_c, float(t))

                tp_total += tp_c; fp_total += fp_c; fn_total += fn_c
                all_best_iou_pred.append(best_iou_pred)

                if self.per_image_mode == "full":
                    # old detailed mode
                    by_class[int(c)] = {
                        "tp": int(tp_c), "fp": int(fp_c), "fn": int(fn_c),
                        "n_pred": int(len(preds_c)), "n_gt": int(len(gts_c)),
                        "matches": matches,  # (pi, gi) in score-sorted pred space
                    }
                else:
                    # compact: keep only top-K examples of FP/FN and summary stats
                    # Identify FPs: indices not in matched pred rows
                    matched_pred_rows = set(pi for pi, _ in matches)
                    fp_rows = [i for i in range(len(preds_c)) if i not in matched_pred_rows]
                    # Rank FPs by score desc (they're already score-sorted)
                    fp_keep = fp_rows[: self.per_image_k]

                    # Identify FNs: GTs not matched
                    matched_g_cols = set(gi for _, gi in matches)
                    fn_cols = [j for j in range(len(gts_c)) if j not in matched_g_cols]
                    # Rank FNs by their best IoU (lowest first for interesting misses)
                    if len(best_iou_gt):
                        fn_cols = sorted(fn_cols, key=lambda j: best_iou_gt[j])
                    fn_keep = fn_cols[: self.per_image_k]

                    by_class[int(c)] = {
                        "tp": int(tp_c), "fp": int(fp_c), "fn": int(fn_c),
                        "n_pred": int(len(preds_c)), "n_gt": int(len(gts_c)),
                        "fp_examples": [
                            {
                                "score": float(preds_c[i, 4]),
                                "box": preds_c[i, :4].round(2).tolist(),
                                "best_iou": float(best_iou_pred[i]) if len(best_iou_pred) else 0.0,
                            }
                            for i in fp_keep
                        ],
                        "fn_examples": [
                            {
                                "gt_box": gts_c[j, :].round(2).tolist(),
                                "best_iou": float(best_iou_gt[j]) if len(best_iou_gt) else 0.0,
                            }
                            for j in fn_keep
                        ],
                    }

            # IoU histogram (tiny: 10 bins) across all preds for this image at this IoU threshold
            img_entry = {"all": {"tp": int(tp_total), "fp": int(fp_total), "fn": int(fn_total)},
                        "by_class": by_class}

            if self.keep_iou_hist:
                if len(all_best_iou_pred):
                    b = np.concatenate(all_best_iou_pred) if all_best_iou_pred else np.array([], np.float32)
                else:
                    b = np.array([], np.float32)
                hist, edges = np.histogram(b, bins=10, range=(0.0, 1.0))
                img_entry["best_iou_hist"] = {"bins": hist.astype(int).tolist(), "edges": np.round(edges, 2).tolist()}

            self.per_image_stats[image_id][f"{t:.2f}"] = img_entry

    def summarize(self) -> dict:
        # Compute AP per class per IoU, then mAP
        classes = sorted(set(list(self.preds_by_class.keys()) + list(self.num_gt_by_class.keys())))
        results = {"per_class": {}, "mAP": {}}

        for t in self.iou_thresholds:
            aps = []
            per_class = {}
            for c in classes:
                out = self._compute_ap_for_class(c, iou_thr=t)
                per_class[str(c)] = {"AP": out["AP"], "npos": out.get("npos", 0)}
                if not np.isnan(out["AP"]):
                    aps.append(out["AP"])
            results["mAP"][f"{t:.2f}"] = float(np.mean(aps)) if aps else float("nan")
            results["per_class"][f"{t:.2f}"] = per_class

        # macro across thresholds (COCO-style average if multiple thresholds provided)
        if len(self.iou_thresholds) > 1:
            results["mAP"]["avg"] = float(np.nanmean([results["mAP"][f"{t:.2f}"] for t in self.iou_thresholds]))

        # Optional per-image stats
        if self.per_image:
            results["per_image"] = self.per_image_stats

        return results