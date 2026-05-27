"""Shared components used by more than one detection pipeline."""

from .classes import CLASS2ID, DOTA_CLASSES, NUM_CLASSES
from .backbone import GhostTriRemoteXProPP
from .model_utils import clean_gpu, count_parameters, print_model_stats

__all__ = [
    "DOTA_CLASSES",
    "CLASS2ID",
    "NUM_CLASSES",
    "GhostTriRemoteXProPP",
    "count_parameters",
    "print_model_stats",
    "clean_gpu",
]
