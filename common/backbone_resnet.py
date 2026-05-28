"""ResNet-50 detection backbone.

A thin wrapper around torchvision's ResNet-50 that exposes the same
``forward_features(x) -> [C3, C4, C5]`` contract as the custom backbone, so it
drops into the existing FPN / head / anchor pipeline unchanged. C3/C4/C5 are the
outputs of ``layer2/layer3/layer4`` with channels ``[512, 1024, 2048]`` at
strides ``[8, 16, 32]``.

Fine-tuning follows the mmdetection/mmrotate convention used for DOTA (and by
Strip R-CNN's ResNet baseline): ImageNet-pretrained, ``frozen_stages=1`` (freeze
the stem and layer1) and ``norm_eval=True`` (all BatchNorm kept in eval mode so
running stats stay frozen). No per-layer LR scaling -- that is a transformer
backbone technique, not used for ResNet detection.
"""

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50Backbone(nn.Module):
    """ResNet-50 backbone returning the C3/C4/C5 feature pyramid.

    Args:
        pretrained: load ImageNet weights.
        frozen_stages: -1 frozen nothing, 0 freezes the stem, 1 also freezes
            layer1, etc. (mmdet convention). Default 1.
        norm_eval: keep all BatchNorm layers in eval mode during training.
    """

    out_channels = [512, 1024, 2048]
    strides = [8, 16, 32]

    def __init__(self, pretrained=True, frozen_stages=1, norm_eval=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = resnet50(weights=weights)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for p in self.stem.parameters():
                p.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f"layer{i}")
            layer.eval()
            for p in layer.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c3, c4, c5]

    def forward(self, x):
        return self.forward_features(x)
