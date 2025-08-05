import torch
import time
import networkx as nx
import torchvision
from torchvision.ops import box_iou

import logging
log = logging.getLogger(__name__)


def custom_nms(preds: torch.Tensor, iou_thres=0.45, conf_thres=0.25,
               max_det=300, top_k=500, soft=True, merge=False):
    """
    Ultra-fast Custom NMS optimized for plant detection.

    - Drops very low-confidence boxes early (conf < conf_thres * 0.5)
    - Keeps only top_k boxes by confidence before expensive operations
    - Uses standard torchvision NMS for fast pruning
    - Applies lightweight Soft-NMS decay only on top 50 boxes
    - Merge-NMS disabled by default (optional)

    Args:
        preds (Tensor): (N,6) [x1,y1,x2,y2,conf,cls]
        iou_thres (float): IoU threshold for suppression
        conf_thres (float): Confidence threshold to keep boxes
        max_det (int): Maximum detections to return
        top_k (int): Keep only top_k boxes before Soft/Merge steps
        soft (bool): Whether to apply Soft-NMS decay
        merge (bool): Whether to merge overlapping boxes (optional)

    Returns:
        Tensor: Filtered boxes [x1,y1,x2,y2,conf,cls]
    """
    if preds.numel() == 0:
        return preds

    device = preds.device

    # Early confidence filtering
    preds = preds[preds[:, 4] > conf_thres * 0.5]
    if preds.numel() == 0:
        return preds

    # Keep only top-k highest confidence boxes
    preds = preds[preds[:, 4].argsort(descending=True)[:top_k]]

    # Hard NMS pruning using torchvision.nms
    boxes, scores, classes = preds[:, :4], preds[:, 4], preds[:, 5]
    offsets = classes * 4096  # separate classes
    keep = torch.ops.torchvision.nms(boxes + offsets.unsqueeze(1), scores, iou_thres)[:max_det]
    kept = preds[keep]

    # Lightweight Soft-NMS (only on top 50 boxes)
    if soft and kept.size(0) > 1:
        top_n = min(50, kept.size(0))
        ious = box_iou(kept[:top_n, :4], kept[:top_n, :4])
        decay = torch.exp(- (ious ** 2) / 0.5)
        decay.fill_diagonal_(1.0)
        kept[:top_n, 4] *= decay.mean(1)
        kept = kept[kept[:, 4].argsort(descending=True)]

    # Optional Merge-NMS (disabled by default)
    if merge and kept.size(0) > 1:
        ious = box_iou(kept[:, :4], kept[:, :4])
        adjacency = ious > iou_thres
        visited = torch.zeros(kept.size(0), dtype=torch.bool, device=device)
        merged = []
        for i in range(kept.size(0)):
            if visited[i]:
                continue
            cluster_idx = adjacency[i] & ~visited
            cluster = kept[cluster_idx]
            weights = cluster[:, 4:5]
            weighted_box = (cluster[:, :4] * weights).sum(0) / weights.sum()
            merged_conf = cluster[:, 4].max()
            merged_cls = kept[i, 5]
            merged.append(torch.cat([weighted_box, merged_conf.unsqueeze(0), merged_cls.unsqueeze(0)]))
            visited[cluster_idx] = True
        kept = torch.stack(merged)

    return kept

def ultralytics_nms(preds: torch.Tensor, iou_thres=0.45, conf_thres=0.25,
                    max_det=300, classes=None, agnostic=False):
    """
    applies ultralytics' default nms which is a simplified greedy suppression. 
    filters by confidence, sorts, and suppresses overlapping boxes.
    """
    if preds.ndim == 2:
        preds = preds.unsqueeze(0)
    device = preds.device

    output = [torch.zeros((0, 6), device=device)] * preds.shape[0]
    max_wh = 7680
    max_nms = 30000

    for xi, x in enumerate(preds):
        x = x[x[:, 4] > conf_thres]
        if not x.shape[0]:
            continue

        if classes is not None:
            cls_tensor = torch.tensor(classes, device=device)
            x = x[(x[:, 5:6] == cls_tensor.unsqueeze(0)).any(1)]

        x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        if not x.shape[0]:
            continue

        boxes_for_nms = x[:, :4].clone()
        if not agnostic:
            boxes_for_nms += x[:, 5:6] * max_wh
        scores = x[:, 4]

        keep = torchvision.ops.nms(boxes_for_nms, scores, iou_thres)[:max_det]
        x = x[keep]
        output[xi] = x

    return output[0] if len(output) == 1 else output


def graph_based_nms(preds: torch.Tensor, iou_thres=0.5):
    """
    uses a graph where boxes are nodes and edges connect overlapping boxes. 
    merges connected components into a single combined box.
    """
    if preds.numel() == 0:
        return preds

    bboxes = [{
        'xmin': b[0].item(), 'ymin': b[1].item(), 'xmax': b[2].item(),
        'ymax': b[3].item(), 'conf': b[4].item(), 'class': int(b[5].item())
    } for b in preds]

    n = len(bboxes)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            bi = torch.tensor([[bboxes[i]['xmin'], bboxes[i]['ymin'], bboxes[i]['xmax'], bboxes[i]['ymax']]])
            bj = torch.tensor([[bboxes[j]['xmin'], bboxes[j]['ymin'], bboxes[j]['xmax'], bboxes[j]['ymax']]])
            iou = box_iou(bi, bj).item()
            if iou >= iou_thres:
                G.add_edge(i, j)

    merged = []
    for comp in nx.connected_components(G):
        comp_boxes = [bboxes[k] for k in comp]
        xmin = min(b['xmin'] for b in comp_boxes)
        ymin = min(b['ymin'] for b in comp_boxes)
        xmax = max(b['xmax'] for b in comp_boxes)
        ymax = max(b['ymax'] for b in comp_boxes)
        conf = max(b['conf'] for b in comp_boxes)
        cls = comp_boxes[0]['class']
        merged.append([xmin, ymin, xmax, ymax, conf, cls])

    return torch.tensor(merged, device=preds.device)


def simple_nms(preds: torch.Tensor, iou_thres=0.5):
    """
    implements a basic greedy nms. 
    iteratively picks the highest score box and removes boxes with iou above threshold.
    """
    if preds.numel() == 0:
        return preds

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


def soft_nms(preds: torch.Tensor, iou_thres=0.5, sigma=0.5, conf_thres=0.25):
    """
    applies soft nms where scores of overlapping boxes are decayed rather than suppressed. 
    keeps boxes with scores above confidence threshold.
    """
    if preds.numel() == 0:
        return preds

    boxes = preds[:, :4]
    scores = preds[:, 4]
    keep_boxes = []

    while boxes.size(0) > 0:
        max_idx = torch.argmax(scores)
        max_box = boxes[max_idx].unsqueeze(0)
        max_score = scores[max_idx]
        cls = preds[max_idx, 5]
        keep_boxes.append(torch.cat([max_box.squeeze(), max_score.unsqueeze(0), cls.unsqueeze(0)]))

        boxes = torch.cat((boxes[:max_idx], boxes[max_idx + 1:]))
        scores = torch.cat((scores[:max_idx], scores[max_idx + 1:]))
        if boxes.size(0) == 0:
            break

        ious = box_iou(max_box, boxes).squeeze()
        decay = torch.ones_like(ious)

        # apply Gaussian decay only where IoU exceeds threshold
        high_iou_mask = ious > iou_thres
        decay[high_iou_mask] = torch.exp(-(ious[high_iou_mask] ** 2) / sigma)

        scores *= decay
        mask = scores > conf_thres
        boxes, scores = boxes[mask], scores[mask]

    return torch.stack(keep_boxes)


def merge_nms(preds: torch.Tensor, iou_thres=0.5, conf_thres=0.25, max_det=300):
    """
    performs merge nms where overlapping boxes are averaged using confidence weights. 
    produces a single refined box for each cluster.
    """
    if preds.numel() == 0:
        return preds
    preds = preds[preds[:, 4] > conf_thres]
    if preds.numel() == 0:
        return preds

    boxes, scores = preds[:, :4], preds[:, 4]
    keep = torchvision.ops.nms(boxes, scores, iou_thres)[:max_det]
    merged = []

    for i in keep:
        iou = box_iou(boxes[i].unsqueeze(0), boxes).squeeze()
        cluster = preds[iou > iou_thres]
        weights = cluster[:, 4:5]
        weighted_box = (cluster[:, :4] * weights).sum(0) / weights.sum()
        merged.append(torch.cat([weighted_box, cluster[:, 4].max().unsqueeze(0), cluster[0, 5].unsqueeze(0)]))

    return torch.stack(merged)


def matrix_nms(preds: torch.Tensor, iou_thres=0.5, conf_thres=0.25, sigma=0.5):
    """
    uses matrix nms which applies global gaussian decay to scores based on the iou matrix. 
    considers relationships between all boxes before suppression.
    """
    if preds.numel() == 0:
        return preds

    preds = preds[preds[:, 4] > conf_thres]
    if preds.numel() == 0:
        return preds

    boxes, scores = preds[:, :4], preds[:, 4]
    ious = box_iou(boxes, boxes)

    decay = torch.exp(-(ious ** 2) / sigma)
    decay_factor = decay.min(0).values
    new_scores = scores * decay_factor

    preds[:, 4] = new_scores
    order = new_scores.argsort(descending=True)
    preds = preds[order]

    keep = []
    while preds.size(0):
        current = preds[0]
        keep.append(current)
        if preds.size(0) == 1:
            break
        ious_cur = box_iou(current[:4].unsqueeze(0), preds[1:, :4]).squeeze(0)
        preds = preds[1:][ious_cur < iou_thres]

    kept = torch.stack(keep) if keep else torch.zeros((0, 6), device=preds.device)
    return kept


def benchmark_nms(preds: torch.Tensor, iou_thres=0.5, conf_thres=0.25, method="yolo"):
    start = time.time()

    if method == "custom":
        filtered = custom_nms(preds, iou_thres=iou_thres, conf_thres=conf_thres)
    elif method == "ultralytics":
        filtered = ultralytics_nms(preds, iou_thres=iou_thres, conf_thres=conf_thres)
    elif method == "graph":
        filtered = graph_based_nms(preds, iou_thres=iou_thres)
    elif method == "simple":
        filtered = simple_nms(preds, iou_thres=iou_thres)
    elif method == "soft":
        filtered = soft_nms(preds, iou_thres=iou_thres, conf_thres=conf_thres)
    elif method == "merge":
        filtered = merge_nms(preds, iou_thres=iou_thres, conf_thres=conf_thres)
    elif method == "matrix":
        filtered = matrix_nms(preds, iou_thres=iou_thres, conf_thres=conf_thres)
    else:
        raise ValueError(f"Unknown NMS method: {method}")

    duration = (time.time() - start) * 1000
    return filtered, duration