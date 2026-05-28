"""Shared components used by more than one detection pipeline."""

from .classes import CLASS2ID, DOTA_CLASSES, NUM_CLASSES
from .backbone import GhostTriRemoteXProPP
from .backbone_resnet import ResNet50Backbone
from .gaconv import GAConv
from .config import Config, load_config
from .constants import IMAGENET_MEAN, IMAGENET_STD
from .run import append_results_row, create_run_dir
from .model_utils import clean_gpu, count_parameters, print_model_stats

# Note: registry/build_* live in common.registry and are imported directly
# (importing them here would create a cycle: registry -> obb_detector -> common).

__all__ = [
    "DOTA_CLASSES",
    "CLASS2ID",
    "NUM_CLASSES",
    "GhostTriRemoteXProPP",
    "ResNet50Backbone",
    "GAConv",
    "Config",
    "load_config",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "create_run_dir",
    "append_results_row",
    "count_parameters",
    "print_model_stats",
    "clean_gpu",
]
