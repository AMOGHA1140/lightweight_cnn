"""Detector wrapper: backbone -> neck -> head."""

import torch.nn as nn


class RemoteDetector(nn.Module):
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        feats = self.neck(feats)
        return self.head(feats)
