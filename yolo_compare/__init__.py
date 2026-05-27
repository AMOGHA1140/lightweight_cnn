"""Pipeline 2: Ultralytics YOLO comparison benchmark (axis-aligned/HBB).

Independent of the custom backbone -- fine-tunes off-the-shelf YOLO models on DOTA
converted to YOLO format, for comparison numbers only.

Run with:  python -m yolo_compare.benchmark
"""

from .dataset_processor import DOTADatasetProcessor
from .benchmark import YOLOModelTrainer

__all__ = ["DOTADatasetProcessor", "YOLOModelTrainer"]
