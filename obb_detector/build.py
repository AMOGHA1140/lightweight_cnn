"""Build the detector and anchors from a config.

Shared by ``train.py`` and ``evaluate.py`` so neither imports the other.
"""

import torch

from common.registry import build_backbone, build_head, build_neck
from .anchors import generate_rotated_anchors
from .detector import RemoteDetector


def build_model(cfg, device, img_size):
    """Build the detector from config; return (model, feature_sizes)."""
    backbone = build_backbone(cfg)

    # Probe backbone output shapes to size the FPN and anchors.
    dummy = torch.randn(1, 3, img_size, img_size)
    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        feats = backbone.forward_features(dummy)
    backbone.train(was_training)
    fpn_in_channels = [f.shape[1] for f in feats]
    feature_sizes = [(f.shape[2], f.shape[3]) for f in feats]

    neck = build_neck(cfg, fpn_in_channels)
    head = build_head(cfg)
    bcfg, ncfg = cfg.model.backbone, cfg.model.neck
    print(f"Backbone: {bcfg.name} | Neck: FPN(out={ncfg.out_channels}, "
          f"smooth={ncfg.smooth_conv})")
    model = RemoteDetector(backbone, neck, head).to(device)
    return model, feature_sizes


def build_anchors(cfg, feature_sizes, img_size, device):
    """Generate per-level rotated anchors from config."""
    strides = [img_size // h for (h, _w) in feature_sizes]
    return generate_rotated_anchors(
        feature_sizes, strides,
        level_scales=cfg.anchors.level_scales,
        anchor_ratios=cfg.anchors.ratios,
        anchor_angles=cfg.anchors.angles,
        device=device,
    )
