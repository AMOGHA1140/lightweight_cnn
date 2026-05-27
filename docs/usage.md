# Usage

All commands run from the project root so that `config`, `common`, and the pipeline
packages import as top-level modules.

## Dependencies

```bash
pip install -r requirements.txt
```

Rotated IoU needs **one** of:
- `mmcv` (preferred — CUDA/CPU op, also enables `nms_rotated` for inference), or
- `shapely` (CPU fallback).

## Configuration (`config.py`)

Every value can be overridden by an environment variable of the same name.

| Name | Default | Meaning |
|---|---|---|
| `DATA_ROOT` | `./data/dota_dataset` | DOTA dataset root |
| `MODELS_DIR` | `./models` | checkpoint output directory |
| `PRETRAINED_BACKBONE` | `${MODELS_DIR}/backbone_pretrained.pth` | backbone weights loaded by the detector |
| `IMG_SIZE` | `1024` | OBB detector input resolution |
| `BATCH_SIZE` | `4` | |
| `NUM_WORKERS` | `0` | DataLoader workers |
| `NUM_EPOCHS` | `100` | |
| `LEARNING_RATE` | `1e-3` | |
| `WEIGHT_DECAY` | `1e-4` | |
| `NORM_MEAN` / `NORM_STD` | ImageNet | normalisation |

## DOTA dataset layout

```
DATA_ROOT/
├── train/
│   ├── images/                    # *.png / *.jpg
│   └── labelTxt-v1.0/labelTxt/    # *.txt, one per image
├── val/
│   ├── images/
│   └── labelTxt-v1.0/labelTxt/
└── test/
    └── images/                    # no labels
```

A label line is `x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty`; header lines
(`imagesource:`, `gsd:`) are skipped. The dataset rescales box coordinates to the
resized image and converts each quadrilateral to `(cx, cy, w, h, θ)`.

## 1. Pretrain the backbone

The detector initialises from a backbone trained on image classification. The
dataset must be an `ImageFolder` with `train/` and `val/` splits:

```
<data-dir>/train/<class>/*.jpg
<data-dir>/val/<class>/*.jpg
```

```bash
python pretrain_backbone.py \
    --data-dir /path/to/classification_dataset \
    --num-classes 200 \
    --epochs 100 --batch-size 64 --lr 1e-3 \
    --out ./models/backbone_pretrained.pth
```

It uses AdamW with a cosine schedule and AMP, and saves the best checkpoint by
validation accuracy. `--num-classes` defaults to 200; if it disagrees with the
dataset, the dataset's class count is used. The saved file is the full backbone
state dict.

## 2. Train the OBB detector

```bash
export PRETRAINED_BACKBONE=./models/backbone_pretrained.pth
python -m obb_detector.train
```

`build_model` loads the pretrained backbone when the file exists (the classification
`fc` layer is dropped, since the detector does not use it) and falls back to random
initialisation otherwise. Training uses AMP, gradient clipping, AdamW, and a cosine
schedule, with `DataParallel` across multiple GPUs. It writes `best_detector.pth`
(best validation loss) and periodic `checkpoint_epoch_{n}.pth` to `MODELS_DIR`.

## 3. Evaluate

`obb_detector/evaluate.py` provides `evaluate_map(model, dataloader, device,
anchors_per_level, class_names)`, which reports per-class AP and the mAP using the
decode + rotated-NMS path.
