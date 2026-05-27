"""mAP evaluation for the OBB detector (VOC-style, rotated IoU).

``compute_ap`` is the standard all-point VOC average-precision routine.

``evaluate_map`` runs the model, decodes detections via
``inference.decode_predictions`` (which returns one ``(boxes, scores, labels)``
triple per image), then matches predictions to ground truth per class using
rotated IoU (``common.rotated_ops.box_iou_rotated``).
"""

from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from common.rotated_ops import box_iou_rotated
from .inference import decode_predictions


def compute_ap(recall, precision):
    """All-point VOC average precision from recall/precision curves."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])


@torch.no_grad()
def evaluate_map(model, dataloader, device, anchors_per_level, class_names,
                 iou_thresh=0.5, conf_thresh=0.05):
    model.eval()
    all_detections = defaultdict(list)   # class -> [(image_id, score, box)]
    all_annotations = defaultdict(list)  # class -> [(image_id, box)]

    for idx, (images, targets) in enumerate(tqdm(dataloader, desc="Evaluating mAP")):
        images = images.to(device)
        preds = model(images)
        decoded = decode_predictions(preds, anchors_per_level,
                                     conf_thresh=conf_thresh, device=device)
        for b in range(images.shape[0]):
            image_id = idx * dataloader.batch_size + b
            gt_boxes = targets[b]["boxes"].cpu().numpy()
            gt_labels = targets[b]["labels"].cpu().numpy()
            for box, label in zip(gt_boxes, gt_labels):
                all_annotations[class_names[label]].append((image_id, box))

            boxes, scores, labels_pred = decoded[b]
            for box, score, label in zip(boxes, scores, labels_pred):
                all_detections[class_names[int(label)]].append((image_id, score, box))

    aps = {}
    for cls in class_names:
        detections = sorted(all_detections[cls], key=lambda x: -x[1])
        annotations = all_annotations[cls]
        if len(annotations) == 0:
            aps[cls] = np.nan
            continue

        npos = len(annotations)
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        detected = {}
        for i, (image_id, _, box_pred) in enumerate(detections):
            candidates = [ann for ann in annotations if ann[0] == image_id]
            ious = []
            for _, box_gt in candidates:
                iou = box_iou_rotated(
                    torch.tensor(box_pred[None, :], dtype=torch.float32),
                    torch.tensor(box_gt[None, :], dtype=torch.float32),
                )[0, 0].item()
                ious.append(iou)
            if ious and max(ious) > iou_thresh:
                max_idx = int(np.argmax(ious))
                gt_key = (image_id, max_idx)
                if gt_key not in detected:
                    tp[i] = 1
                    detected[gt_key] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        recall = tp / npos
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        aps[cls] = compute_ap(recall, precision)

    valid_aps = [ap for ap in aps.values() if not np.isnan(ap)]
    mAP = float(np.mean(valid_aps)) if valid_aps else 0.0

    print("Class-wise AP:")
    for cls, ap in aps.items():
        print(f"{cls}: {ap:.4f}")
    print(f"Overall mAP: {mAP:.4f}")
    return aps, mAP
