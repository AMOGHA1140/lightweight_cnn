# Lightweight Oriented Object Detection for Remote Sensing

A lightweight one-stage **oriented** (OBB) object detector for aerial imagery
(DOTA, 15 classes). Built around a custom efficient backbone
(`GhostTriRemoteXProPP`) feeding a multi-scale FPN and a rotated dense head that
predicts 5-parameter oriented boxes `(cx, cy, w, h, θ)`.

The repository also contains two secondary pipelines for comparison: an
experimental two-stage axis-aligned detector and an Ultralytics YOLO benchmark.

## Layout

```
config.py              # dataset/model paths + hyperparameters (edit this)
pretrain_backbone.py   # classification pretraining for the backbone
common/                # shared: backbone, DOTA classes, rotated IoU, model utils
obb_detector/          # primary one-stage oriented (OBB) detector
faster_rcnn/           # experimental two-stage axis-aligned detector (reference only)
yolo_compare/          # Ultralytics YOLO benchmark (comparison only)
docs/                  # detailed documentation
```

## Pipelines

| Pipeline | Boxes | Status |
|---|---|---|
| `obb_detector` | oriented (OBB) | **Primary.** Multi-scale FPN, per-level anchors, delta regression, rotated NMS. |
| `faster_rcnn`  | horizontal (HBB) | Experimental / reference only — see [docs/secondary-pipelines.md](docs/secondary-pipelines.md). |
| `yolo_compare` | horizontal (HBB) | Comparison only; does not use the custom backbone. |

## Quick start

Run everything from the project root (so `config` and `common` are importable):

```bash
pip install -r requirements.txt          # install mmcv (preferred) or shapely for rotated IoU
export DOTA_DATA_ROOT=/path/to/dota_dataset   # or edit config.py

# 1) pretrain the backbone on a classification dataset (ImageFolder layout)
python pretrain_backbone.py --data-dir /path/to/classification_dataset --num-classes 200

# 2) train the OBB detector (loads the pretrained backbone if present)
python -m obb_detector.train

# 3) the secondary pipelines
python -m faster_rcnn.train
python -m yolo_compare.benchmark
```

## Documentation

- [docs/architecture.md](docs/architecture.md) — backbone and detection-pipeline design.
- [docs/usage.md](docs/usage.md) — dataset layout, configuration, pretraining, training, evaluation.
- [docs/secondary-pipelines.md](docs/secondary-pipelines.md) — the Faster R-CNN and YOLO pipelines.
- [docs/roadmap.md](docs/roadmap.md) — open work and planned improvements.
