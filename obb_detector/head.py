"""Rotated dense detection head (RetinaNet + YOLO style).

Three branches per FPN level:
  * classification subnet: 2x (conv3x3 + ReLU) -> conv3x3 -> A*num_classes
  * box regression subnet: 1x (conv3x3 + ReLU) -> conv3x3 -> A*5  (cx,cy,w,h,theta)
  * objectness branch:      conv3x3 -> A*1

The cls/obj prediction biases use the RetinaNet prior-probability init
(``-log((1-p)/p)``) so the sigmoid focal loss starts at a sane magnitude instead
of treating every anchor as a confident positive.
"""

import math

import torch.nn as nn


class RotatedDetectionHead(nn.Module):
    def __init__(self, num_classes, num_anchors=9, in_channels=128, prior_prob=0.01):
        super().__init__()
        self.cls_subnet = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(),
        )
        self.cls_score = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)

        self.bbox_subnet = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(),
        )
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 5, 3, padding=1)
        self.obj_pred = nn.Conv2d(in_channels, num_anchors, 3, padding=1)

        bias_init = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_score.bias, bias_init)
        nn.init.constant_(self.obj_pred.bias, bias_init)

    def forward(self, feats):
        cls_outs, reg_outs, obj_outs = [], [], []
        for feat in feats:
            cls_feat = self.cls_subnet(feat)
            reg_feat = self.bbox_subnet(feat)
            cls_outs.append(self.cls_score(cls_feat))
            reg_outs.append(self.bbox_pred(reg_feat))
            obj_outs.append(self.obj_pred(reg_feat))
        return cls_outs, reg_outs, obj_outs
