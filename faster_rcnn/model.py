"""Two-stage axis-aligned (HBB) detector -- EXPERIMENTAL / ABANDONED.

This pipeline is not the project's research direction and is incomplete: it is
axis-aligned (not oriented), its ROI head trains on random features rather than
pooled RoIs, ``forward_test`` returns raw top-k RPN outputs without decode/NMS, and
the multi-scale FPN input is faked by pooling a single backbone feature map. Kept
for reference only (see docs/secondary-pipelines.md).

Note: it expects ``backbone.forward_features`` to return a single tensor; the
current backbone returns three feature maps, so this pipeline needs adapting
before it can run.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou


class FPN(nn.Module):
    def __init__(self, in_channels, out_channel=256, num_levels=5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_levels = num_levels
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.lateral_convs.append(nn.Conv2d(in_channels[i], out_channel, 1))
            self.fpn_convs.append(nn.Conv2d(out_channel, out_channel, 3, padding=1))
        for _ in range(num_levels - len(in_channels)):
            self.fpn_convs.append(nn.Conv2d(out_channel, out_channel, 3, stride=2, padding=1))

    def forward(self, inputs):
        laterals = [lc(inputs[i]) for i, lc in enumerate(self.lateral_convs)]
        for i in range(len(laterals) - 1, 0, -1):
            target_size = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=target_size, mode="nearest"
            )
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        for i in range(len(laterals), self.num_levels):
            src = outs[-1] if i == len(laterals) else F.relu(outs[-1])
            outs.append(self.fpn_convs[i](src))
        return outs


class RPNHead(nn.Module):
    def __init__(self, in_channel=256, feat_channel=256, num_anchors=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, feat_channel, 3, padding=1)
        self.cls_conv = nn.Conv2d(feat_channel, num_anchors, 1)
        self.reg_conv = nn.Conv2d(feat_channel, num_anchors * 4, 1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        return self.cls_conv(x), self.reg_conv(x)


class ROIHead(nn.Module):
    def __init__(self, in_channel=256, num_classes=15):
        super().__init__()
        self.roi_pool = nn.AdaptiveAvgPool2d(7)
        self.fc1 = nn.Linear(in_channel * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_fc = nn.Linear(1024, num_classes + 1)
        self.reg_fc = nn.Linear(1024, num_classes * 4)

    def forward(self, features, rois):
        pooled = self.roi_pool(features).view(features.size(0), -1)
        x = F.relu(self.fc1(pooled))
        x = F.relu(self.fc2(x))
        return self.cls_fc(x), self.reg_fc(x)


class AnchorGenerator:
    def __init__(self, sizes=(32, 64, 128), aspect_ratios=(0.5, 1.0, 2.0), scales=(1.0,)):
        self.sizes = list(sizes)
        self.aspect_ratios = list(aspect_ratios)
        self.scales = list(scales)

    def generate_level_anchors(self, H, W, device, level=0):
        base_size = self.sizes[level] if level < len(self.sizes) else self.sizes[-1]
        base_anchors = []
        for ar in self.aspect_ratios:
            for scale in self.scales:
                w = base_size * scale * (ar ** 0.5)
                h = base_size * scale / (ar ** 0.5)
                base_anchors.append([-w / 2, -h / 2, w / 2, h / 2])
        base_anchors = torch.tensor(base_anchors, device=device, dtype=torch.float32)

        shift_x = torch.arange(0, W, device=device, dtype=torch.float32) * (800 // W)
        shift_y = torch.arange(0, H, device=device, dtype=torch.float32) * (800 // H)
        shift_xx, shift_yy = torch.meshgrid(shift_x, shift_y, indexing="ij")
        shifts = torch.stack(
            [shift_xx.ravel(), shift_yy.ravel(), shift_xx.ravel(), shift_yy.ravel()], dim=1
        )
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        return all_anchors.view(-1, 4)


class FasterRCNN(nn.Module):
    def __init__(self, backbone, num_classes=15):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            backbone_channels = self.backbone.forward_features(dummy).shape[1]

        self.fpn = FPN([backbone_channels] * 3, out_channel=256)
        self.rpn_head = RPNHead(in_channel=256, num_anchors=3)
        self.roi_head = ROIHead(in_channel=256, num_classes=num_classes)
        self.anchor_generator = AnchorGenerator()

        self.rpn_cls_weight = self.rpn_reg_weight = 1.0
        self.roi_cls_weight = self.roi_reg_weight = 1.0

    def forward(self, images, targets=None):
        features = self.backbone.forward_features(images)
        B, C, H, W = features.shape
        feature_maps = [
            F.adaptive_avg_pool2d(features, (H // 4, W // 4)),
            F.adaptive_avg_pool2d(features, (H // 2, W // 2)),
            features,
        ]
        fpn_features = self.fpn(feature_maps)

        rpn_outputs, featmap_sizes = [], []
        for feat in fpn_features:
            rpn_outputs.append(self.rpn_head(feat))
            featmap_sizes.append((feat.size(2), feat.size(3)))

        all_anchors = [
            self.anchor_generator.generate_level_anchors(h, w, images.device, level=i)
            for i, (h, w) in enumerate(featmap_sizes)
        ]

        if self.training and targets is not None:
            return self.forward_train(fpn_features, rpn_outputs, all_anchors, targets)
        return self.forward_test(fpn_features, rpn_outputs, all_anchors)

    def forward_train(self, features, rpn_outputs, all_anchors, targets):
        losses = {}
        device = features[0].device
        gt_boxes_list = [t["boxes"] for t in targets]
        gt_labels_list = [t["labels"] for t in targets]

        # --- RPN loss ---
        rpn_cls_losses, rpn_reg_losses = [], []
        for (cls_score, bbox_pred), level_anchors in zip(rpn_outputs, all_anchors):
            B = cls_score.shape[0]
            cls_score_flat = cls_score.permute(0, 2, 3, 1).reshape(B, -1, 1)
            bbox_pred_flat = bbox_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)

            batch_cls_losses, batch_reg_losses = [], []
            for bi in range(B):
                batch_cls_score = cls_score_flat[bi].squeeze(-1)
                batch_bbox_pred = bbox_pred_flat[bi]
                gt_boxes = gt_boxes_list[bi] if bi < len(gt_boxes_list) else torch.zeros((0, 4), device=device)

                rpn_labels, rpn_bbox_targets = self._assign_rpn_targets_single(level_anchors, gt_boxes, device)

                valid = rpn_labels >= 0
                if valid.sum() > 0:
                    cls_loss = F.binary_cross_entropy_with_logits(
                        batch_cls_score[valid], rpn_labels[valid].float())
                else:
                    cls_loss = torch.tensor(0.0, device=device, requires_grad=True)

                pos = rpn_labels == 1
                if pos.sum() > 0:
                    reg_loss = F.smooth_l1_loss(batch_bbox_pred[pos], rpn_bbox_targets[pos], reduction="mean")
                else:
                    reg_loss = torch.tensor(0.0, device=device, requires_grad=True)

                batch_cls_losses.append(cls_loss)
                batch_reg_losses.append(reg_loss)

            rpn_cls_losses.append(sum(batch_cls_losses) / len(batch_cls_losses))
            rpn_reg_losses.append(sum(batch_reg_losses) / len(batch_reg_losses))

        losses["rpn_cls_loss"] = sum(rpn_cls_losses) / len(rpn_cls_losses) * self.rpn_cls_weight
        losses["rpn_reg_loss"] = sum(rpn_reg_losses) / len(rpn_reg_losses) * self.rpn_reg_weight

        # --- ROI loss (placeholder: trained on random features, see module docstring) ---
        roi_cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
        roi_reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
        batch_size = len(targets)
        num_rois_per_batch = 256
        for bi in range(batch_size):
            roi_features = torch.randn(num_rois_per_batch, 256 * 7 * 7, device=device)
            gt_boxes = gt_boxes_list[bi] if bi < len(gt_boxes_list) else torch.zeros((0, 4), device=device)
            gt_labels = gt_labels_list[bi] if bi < len(gt_labels_list) else torch.zeros((0,), dtype=torch.long, device=device)
            roi_labels, roi_bbox_targets = self._prepare_roi_targets_single(gt_boxes, gt_labels, num_rois_per_batch, device)
            roi_cls_scores, roi_bbox_preds = self._roi_forward(roi_features)
            if len(roi_labels) > 0:
                roi_cls_loss = roi_cls_loss + F.cross_entropy(roi_cls_scores, roi_labels)
            pos_roi = roi_labels > 0
            if pos_roi.sum() > 0:
                roi_reg_loss = roi_reg_loss + F.smooth_l1_loss(roi_bbox_preds[pos_roi], roi_bbox_targets[pos_roi])

        losses["roi_cls_loss"] = roi_cls_loss / batch_size * self.roi_cls_weight
        losses["roi_reg_loss"] = roi_reg_loss / batch_size * self.roi_reg_weight
        losses["total_loss"] = (losses["rpn_cls_loss"] + losses["rpn_reg_loss"]
                                + losses["roi_cls_loss"] + losses["roi_reg_loss"])
        return losses

    def _assign_rpn_targets_single(self, anchors, gt_boxes, device):
        num_anchors = len(anchors)
        labels = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
        bbox_targets = torch.zeros((num_anchors, 4), device=device)
        if len(gt_boxes) == 0:
            labels[:min(256, num_anchors)] = 0
            return labels, bbox_targets

        ious = box_iou(anchors, gt_boxes)  # standard torchvision IoU
        max_ious, max_indices = ious.max(dim=1)
        labels[max_ious >= 0.7] = 1
        labels[max_ious < 0.3] = 0
        pos_mask = max_ious >= 0.7
        if pos_mask.sum() > 0:
            bbox_targets[pos_mask] = self._encode_boxes(anchors[pos_mask], gt_boxes[max_indices[pos_mask]])
        return labels, bbox_targets

    @staticmethod
    def _encode_boxes(anchors, gt_boxes):
        aw = anchors[:, 2] - anchors[:, 0]
        ah = anchors[:, 3] - anchors[:, 1]
        acx = anchors[:, 0] + 0.5 * aw
        acy = anchors[:, 1] + 0.5 * ah
        gw = gt_boxes[:, 2] - gt_boxes[:, 0]
        gh = gt_boxes[:, 3] - gt_boxes[:, 1]
        gcx = gt_boxes[:, 0] + 0.5 * gw
        gcy = gt_boxes[:, 1] + 0.5 * gh
        return torch.stack([
            (gcx - acx) / (aw + 1e-6),
            (gcy - acy) / (ah + 1e-6),
            torch.log(gw / (aw + 1e-6) + 1e-6),
            torch.log(gh / (ah + 1e-6) + 1e-6),
        ], dim=1)

    def _prepare_roi_targets_single(self, gt_boxes, gt_labels, num_rois, device):
        if len(gt_boxes) > 0:
            num_pos = min(num_rois // 4, len(gt_boxes))
            num_neg = num_rois - num_pos
            pos_indices = torch.randperm(len(gt_boxes), device=device)[:num_pos]
            pos_labels = gt_labels[pos_indices]
            pos_bbox_targets = torch.zeros((num_pos, self.num_classes * 4), device=device)
            for i, label in enumerate(pos_labels):
                if label > 0:
                    s = label * 4
                    pos_bbox_targets[i, s:s + 4] = torch.randn(4, device=device) * 0.1
            neg_labels = torch.zeros(num_neg, dtype=torch.long, device=device)
            neg_bbox_targets = torch.zeros((num_neg, self.num_classes * 4), device=device)
            labels = torch.cat([pos_labels, neg_labels])
            bbox_targets = torch.cat([pos_bbox_targets, neg_bbox_targets])
        else:
            labels = torch.zeros(num_rois, dtype=torch.long, device=device)
            bbox_targets = torch.zeros((num_rois, self.num_classes * 4), device=device)
        return labels, bbox_targets

    def _roi_forward(self, roi_features):
        x = F.relu(self.roi_head.fc1(roi_features))
        x = F.relu(self.roi_head.fc2(x))
        return self.roi_head.cls_fc(x), self.roi_head.reg_fc(x)

    def forward_test(self, features, rpn_outputs, all_anchors):
        detections = []
        for (cls_score, bbox_pred), _ in zip(rpn_outputs, all_anchors):
            B = cls_score.shape[0]
            for bi in range(B):
                scores = torch.sigmoid(cls_score[bi]).view(-1)
                boxes = bbox_pred[bi].view(-1, 4)
                top_scores, top_indices = scores.topk(min(1000, len(scores)), dim=0)
                detections.append({
                    "scores": top_scores,
                    "boxes": boxes[top_indices],
                    "labels": torch.zeros_like(top_scores, dtype=torch.long),
                })
        return detections
