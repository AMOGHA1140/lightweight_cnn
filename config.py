"""Central configuration: filesystem paths and training hyperparameters.

Edit the defaults for your machine, or override any value with an environment
variable of the same name.
"""

import os

# --- Filesystem paths -------------------------------------------------------
# Root of the DOTA dataset, expected to contain train/ val/ test/ subdirs with
# `images/` and `labelTxt-v1.0/labelTxt/` annotation folders.
DATA_ROOT = os.environ.get("DOTA_DATA_ROOT", "./data/dota_dataset")

# Where model checkpoints are written / read.
MODELS_DIR = os.environ.get("MODELS_DIR", "./models")

# Pretrained classification weights for the backbone (produced by
# pretrain_backbone.py). Loaded into the detector when present.
PRETRAINED_BACKBONE = os.environ.get(
    "PRETRAINED_BACKBONE", os.path.join(MODELS_DIR, "backbone_pretrained.pth")
)

# --- Common training defaults ----------------------------------------------
IMG_SIZE = 1024          # OBB detector input resolution
BATCH_SIZE = 4
NUM_WORKERS = 0
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# ImageNet normalisation (used by the dataset transforms).
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)
    return MODELS_DIR
