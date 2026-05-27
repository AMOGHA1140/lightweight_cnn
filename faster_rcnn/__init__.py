"""Experimental two-stage axis-aligned (HBB) Faster R-CNN.

ABANDONED / reference-only: axis-aligned (not OBB) and incomplete. See ``model.py``
for the caveats.

Train with:  python -m faster_rcnn.train
"""

from .dataset import DOTADataset, custom_collate_fn, parse_dota_annotation
from .model import FasterRCNN, FPN, RPNHead, ROIHead, AnchorGenerator
from .metrics import box_iou, calculate_ap, calculate_map, evaluate_model

__all__ = [
    "DOTADataset",
    "custom_collate_fn",
    "parse_dota_annotation",
    "FasterRCNN",
    "FPN",
    "RPNHead",
    "ROIHead",
    "AnchorGenerator",
    "box_iou",
    "calculate_ap",
    "calculate_map",
    "evaluate_model",
]
