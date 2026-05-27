"""Detection loss for the one-stage OBB detector.

Composition:
  * classification: focal loss (alpha=0.25, gamma=2.0) via
    ``torchvision.ops.sigmoid_focal_loss``, normalised by the positive count.
  * box regression: Smooth L1 on positive anchors only, against anchor-relative
    deltas (``encode_obb``); the head predicts deltas, not absolute boxes.
  * objectness: binary cross-entropy over all anchors.
  * anchor assignment: each anchor takes its best-IoU GT, positive if IoU > 0.5.
  * the three terms are summed with equal weight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from common.rotated_ops import box_iou_rotated


def encode_obb(gt_boxes, anchors):
    """Encode GT boxes as deltas relative to anchors.

    Both inputs: ``[N, 5]`` as ``(cx, cy, w, h, angle)``.
    """
    dx = (gt_boxes[:, 0] - anchors[:, 0]) / anchors[:, 2]
    dy = (gt_boxes[:, 1] - anchors[:, 1]) / anchors[:, 3]
    dw = torch.log(gt_boxes[:, 2] / anchors[:, 2])
    dh = torch.log(gt_boxes[:, 3] / anchors[:, 3])
    da = gt_boxes[:, 4] - anchors[:, 4]
    return torch.stack([dx, dy, dw, dh, da], dim=1)


class DetectionLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")

    def focal_loss(self, logits, targets):
        """Sigmoid focal loss, summed and normalised by the #positive targets."""
        loss = sigmoid_focal_loss(
            logits, targets, alpha=self.alpha, gamma=self.gamma, reduction="sum"
        )
        return loss / max(1, int(targets.sum().item()))

    def forward(self, preds, targets, anchors_per_level, device):
        """
        Args:
            preds: ``(cls_outs, reg_outs, obj_outs)`` -- each a list per FPN level.
            targets: list of dicts with 'boxes' [num_gt, 5] and 'labels' [num_gt].
            anchors_per_level: list of ``[num_anchors, 5]`` per FPN level.
        """
        cls_outs, reg_outs, obj_outs = preds
        batch_size = cls_outs[0].shape[0]
        total_cls_loss = total_reg_loss = total_obj_loss = 0.0

        for b in range(batch_size):
            gt_boxes = targets[b]["boxes"].to(device)
            gt_labels = targets[b]["labels"].to(device)

            all_anchors, all_cls_preds, all_reg_preds, all_obj_preds = [], [], [], []
            for i, anchors in enumerate(anchors_per_level):
                cls_pred = cls_outs[i][b].permute(1, 2, 0).reshape(-1, self.num_classes)
                reg_pred = reg_outs[i][b].permute(1, 2, 0).reshape(-1, 5)
                obj_pred = obj_outs[i][b].permute(1, 2, 0).reshape(-1)
                all_anchors.append(anchors.to(device))
                all_cls_preds.append(cls_pred)
                all_reg_preds.append(reg_pred)
                all_obj_preds.append(obj_pred)

            all_anchors = torch.cat(all_anchors, dim=0)
            all_cls_preds = torch.cat(all_cls_preds, dim=0)
            all_reg_preds = torch.cat(all_reg_preds, dim=0)
            all_obj_preds = torch.cat(all_obj_preds, dim=0)

            n = all_anchors.shape[0]
            cls_targets = torch.zeros((n, self.num_classes), device=device)
            reg_targets = torch.zeros((n, 5), device=device)
            obj_targets = torch.zeros((n,), device=device)
            pos_mask = torch.zeros((n,), dtype=torch.bool, device=device)

            if gt_boxes.numel() > 0:
                ious = box_iou_rotated(gt_boxes, all_anchors)  # [num_gt, N]
                max_ious, max_ids = ious.max(dim=0)            # for each anchor, best GT
                pos_mask = max_ious > 0.5
                if pos_mask.sum() > 0:
                    pos_idx = torch.where(pos_mask)[0]
                    assigned_gt = max_ids[pos_mask]
                    cls_targets[pos_idx, gt_labels[assigned_gt]] = 1
                    reg_targets[pos_idx] = encode_obb(gt_boxes[assigned_gt], all_anchors[pos_idx])
                    obj_targets[pos_idx] = 1

            cls_loss = self.focal_loss(all_cls_preds, cls_targets)
            obj_loss = F.binary_cross_entropy_with_logits(all_obj_preds, obj_targets)
            if pos_mask.sum() > 0:
                reg_loss = self.smooth_l1(
                    all_reg_preds[pos_mask], reg_targets[pos_mask]
                ).sum() / max(1, int(pos_mask.sum().item()))
            else:
                reg_loss = torch.tensor(0.0, device=device)

            total_cls_loss = total_cls_loss + cls_loss
            total_reg_loss = total_reg_loss + reg_loss
            total_obj_loss = total_obj_loss + obj_loss

        return {
            "total_loss": total_cls_loss + total_reg_loss + total_obj_loss,
            "cls_loss": total_cls_loss,
            "bbox_loss": total_reg_loss,
            "obj_loss": total_obj_loss,
        }
