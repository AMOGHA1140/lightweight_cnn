"""Pipeline 1: custom one-stage oriented (OBB) detector.

GhostTriRemoteXProPP backbone -> vanilla FPN neck -> rotated dense head, trained
with focal + Smooth-L1 + objectness losses on DOTA oriented boxes.

This is the project's primary research pipeline. Train with:
    python -m obb_detector.train
"""

from .dataset import DOTADataset, collate_fn
from .detector import RemoteDetector
from .fpn import FPN
from .head import RotatedDetectionHead
from .loss import DetectionLoss
from .anchors import generate_rotated_anchors

__all__ = [
    "DOTADataset",
    "collate_fn",
    "RemoteDetector",
    "FPN",
    "RotatedDetectionHead",
    "DetectionLoss",
    "generate_rotated_anchors",
]
