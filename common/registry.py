"""Lightweight component registry + builders driven by config.

A thin name->class map lets configs select a component by string
(``model.backbone.name: resnet50``). Builder functions do the explicit wiring of
config fields to constructor arguments -- no generic kwarg spraying.
"""

import os

import torch

from common.backbone import GhostTriRemoteXProPP
from common.backbone_resnet import ResNet50Backbone
from common.classes import NUM_CLASSES
from obb_detector.fpn import FPN
from obb_detector.head import RotatedDetectionHead


class Registry:
    def __init__(self, name):
        self.name = name
        self._classes = {}

    def register(self, name):
        def deco(cls):
            self._classes[name] = cls
            return cls
        return deco

    def get(self, key):
        if key not in self._classes:
            raise KeyError(f"{key!r} not in {self.name} registry "
                           f"(have: {sorted(self._classes)})")
        return self._classes[key]


BACKBONES = Registry("backbone")
BACKBONES.register("resnet50")(ResNet50Backbone)
BACKBONES.register("custom")(GhostTriRemoteXProPP)

NECKS = Registry("neck")
NECKS.register("fpn")(FPN)

HEADS = Registry("head")
HEADS.register("rotated")(RotatedDetectionHead)


def build_backbone(cfg):
    bcfg = cfg.model.backbone
    cls = BACKBONES.get(bcfg.name)
    if bcfg.name == "resnet50":
        return cls(pretrained=bcfg.pretrained,
                   frozen_stages=bcfg.frozen_stages,
                   norm_eval=bcfg.norm_eval)

    backbone = cls(num_classes=200, width_mult=1.0)
    ckpt = cfg.paths.pretrained_backbone
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        backbone.load_state_dict(state, strict=False)
        print(f"Loaded pretrained backbone from {ckpt}")
    else:
        print(f"[warn] pretrained backbone not found at {ckpt}; using random init.")
    return backbone


def build_neck(cfg, in_channels):
    return NECKS.get("fpn")(in_channels,
                            out_channels=cfg.model.neck.out_channels,
                            smooth_conv=cfg.model.neck.smooth_conv)


def build_head(cfg):
    head_cfg = cfg.model.head
    return HEADS.get("rotated")(num_classes=NUM_CLASSES,
                                num_anchors=head_cfg.num_anchors,
                                in_channels=cfg.model.neck.out_channels,
                                prior_prob=head_cfg.get("prior_prob", 0.01))
