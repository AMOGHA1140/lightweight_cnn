"""Prediction decoding for the OBB detector.

Turns raw head outputs into oriented boxes in image coordinates:
  1. flatten cls/reg/obj across all FPN levels,
  2. sigmoid cls and obj, confidence = obj * max-cls,
  3. threshold on confidence,
  4. decode regression deltas against anchors (``decode_obb``),
  5. per-class rotated NMS (mmcv ``nms_rotated``; mmcv is required).

``decode_predictions`` returns one ``(boxes, scores, labels)`` triple per image.
"""

import numpy as np
import torch

try:
    from mmcv.ops import nms_rotated as _mmcv_nms_rotated
    _HAS_MMCV_NMS = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_MMCV_NMS = False

_MMCV_REQUIRED = (
    "mmcv is required for rotated NMS. Install it on the training machine, e.g.\n"
    "  pip install -U openmim && mim install mmcv"
)


def decode_obb(deltas, anchors):
    """Decode predicted deltas back to absolute boxes.

    Both inputs: ``[N, 5]`` as ``(cx, cy, w, h, angle)`` / deltas.
    """
    cx = deltas[:, 0] * anchors[:, 2] + anchors[:, 0]
    cy = deltas[:, 1] * anchors[:, 3] + anchors[:, 1]
    w = torch.exp(deltas[:, 2]) * anchors[:, 2]
    h = torch.exp(deltas[:, 3]) * anchors[:, 3]
    angle = deltas[:, 4] + anchors[:, 4]
    return torch.stack([cx, cy, w, h, angle], dim=1)


def _nms_rotated(boxes, scores, iou_thresh):
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    if not _HAS_MMCV_NMS:
        raise ImportError(_MMCV_REQUIRED)
    _, keep = _mmcv_nms_rotated(boxes.float(), scores.float(), iou_thresh)
    return keep


def decode_predictions(preds, anchors_per_level, conf_thresh=0.05, nms_thresh=0.1,
                       pre_nms_topk=2000, device="cpu"):
    """Decode a batch of predictions into per-image detections.

    Returns a list (length = batch size) of ``(boxes, scores, labels)`` numpy
    arrays, where ``boxes`` is ``[K, 5]`` in image coordinates.
    """
    cls_outs, reg_outs, obj_outs = preds
    batch_size = cls_outs[0].shape[0]

    # Flatten every level to [B, sum(H*W*A), *].
    cls_all, reg_all, obj_all, anchors_all = [], [], [], []
    for cls_out, reg_out, obj_out, anchors in zip(cls_outs, reg_outs, obj_outs, anchors_per_level):
        B, _, H, W = cls_out.shape
        A = obj_out.shape[1]
        num_classes = cls_out.shape[1] // A
        cls_all.append(cls_out.permute(0, 2, 3, 1).reshape(B, H * W * A, num_classes))
        reg_all.append(reg_out.permute(0, 2, 3, 1).reshape(B, H * W * A, 5))
        obj_all.append(obj_out.permute(0, 2, 3, 1).reshape(B, H * W * A))
        anchors_all.append(anchors.to(device))
    cls_all = torch.cat(cls_all, dim=1).detach()
    reg_all = torch.cat(reg_all, dim=1).detach()
    obj_all = torch.cat(obj_all, dim=1).detach()
    anchors_all = torch.cat(anchors_all, dim=0)

    results = []
    for b in range(batch_size):
        cls_prob = torch.sigmoid(cls_all[b])
        obj_prob = torch.sigmoid(obj_all[b])
        cls_scores, cls_labels = cls_prob.max(dim=1)
        conf = obj_prob * cls_scores

        mask = conf > conf_thresh
        if mask.sum() == 0:
            results.append((np.zeros((0, 5), np.float32), np.zeros((0,), np.float32),
                            np.zeros((0,), np.int64)))
            continue

        conf = conf[mask]
        labels = cls_labels[mask]
        boxes = decode_obb(reg_all[b][mask], anchors_all[mask])

        # Bound NMS cost: keep the top-scoring candidates first.
        if conf.numel() > pre_nms_topk:
            topk = conf.topk(pre_nms_topk).indices
            conf, labels, boxes = conf[topk], labels[topk], boxes[topk]

        keep_boxes, keep_scores, keep_labels = [], [], []
        for cls_id in labels.unique():
            cls_mask = labels == cls_id
            kept = _nms_rotated(boxes[cls_mask], conf[cls_mask], nms_thresh)
            keep_boxes.append(boxes[cls_mask][kept])
            keep_scores.append(conf[cls_mask][kept])
            keep_labels.append(labels[cls_mask][kept])

        boxes = torch.cat(keep_boxes).detach().cpu().numpy()
        scores = torch.cat(keep_scores).detach().cpu().numpy()
        labels = torch.cat(keep_labels).detach().cpu().numpy()
        results.append((boxes, scores, labels.astype(np.int64)))

    return results
