"""Evaluation metrics for the axis-aligned two-stage detector.

``box_iou`` (``torchvision.ops.box_iou``), ``calculate_ap`` (11-point VOC),
``calculate_map`` and ``evaluate_model``.
"""

import numpy as np
import torch
from torchvision.ops import box_iou


def calculate_ap(predictions, targets, iou_threshold=0.5):
    """AP for a single class via 11-point interpolation."""
    all_scores, all_tp, total_gt = [], [], 0
    for pred, gt_boxes in zip(predictions, targets):
        scores, boxes = pred["scores"], pred["boxes"]
        total_gt += len(gt_boxes)
        if len(scores) == 0:
            continue
        tp = torch.zeros(len(scores))
        if len(gt_boxes) > 0:
            ious = box_iou(boxes, gt_boxes)
            max_ious, _ = ious.max(dim=1)
            tp[max_ious >= iou_threshold] = 1
        all_scores.extend(scores.tolist())
        all_tp.extend(tp.tolist())

    if len(all_scores) == 0 or total_gt == 0:
        return 0.0

    order = np.argsort(all_scores)[::-1]
    sorted_tp = np.array(all_tp)[order]
    cumsum_tp = np.cumsum(sorted_tp)
    cumsum_fp = np.cumsum(1 - sorted_tp)
    precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-6)
    recall = cumsum_tp / (total_gt + 1e-6)

    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        p = 0 if np.sum(recall >= t) == 0 else np.max(precision[recall >= t])
        ap += p / 11.0
    return ap


def calculate_map(predictions, targets, num_classes=15, iou_threshold=0.5):
    """mAP across classes; returns ``(mAP, per_class_aps)``."""
    aps = []
    for class_id in range(num_classes):
        class_predictions, class_targets = [], []
        for pred, target in zip(predictions, targets):
            if len(pred["labels"]) > 0 and (pred["labels"] == class_id).any():
                mask = pred["labels"] == class_id
                class_predictions.append({"scores": pred["scores"][mask], "boxes": pred["boxes"][mask]})
            else:
                class_predictions.append({"scores": torch.tensor([]), "boxes": torch.zeros((0, 4))})
            if len(target["labels"]) > 0:
                class_targets.append(target["boxes"][target["labels"] == class_id])
            else:
                class_targets.append(torch.zeros((0, 4)))
        aps.append(calculate_ap(class_predictions, class_targets, iou_threshold))
    return float(np.mean(aps)), aps


@torch.no_grad()
def evaluate_model(model, dataloader, device, num_classes=15):
    """Run the model over a loader and return ``(mAP, per_class_aps)``."""
    model.eval()
    all_predictions, all_targets = [], []
    for images, targets in dataloader:
        images = images.to(device)
        targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
        predictions = model(images)

        if isinstance(predictions, list):
            for pred in predictions:
                if isinstance(pred, dict):
                    all_predictions.append(pred)
                else:
                    all_predictions.append({
                        "scores": torch.tensor([], device=device),
                        "boxes": torch.zeros((0, 4), device=device),
                        "labels": torch.tensor([], dtype=torch.long, device=device),
                    })
        else:
            for _ in range(len(targets_device)):
                all_predictions.append({
                    "scores": torch.tensor([], device=device),
                    "boxes": torch.zeros((0, 4), device=device),
                    "labels": torch.tensor([], dtype=torch.long, device=device),
                })
        all_targets.extend(targets_device)

    return calculate_map(all_predictions, all_targets, num_classes)
