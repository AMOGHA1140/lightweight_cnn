# Lightweight Oriented Object Detection for Remote Sensing

A lightweight one-stage **oriented** (OBB) object detector for aerial imagery
(DOTA, 15 classes). Built around a custom efficient backbone
(`GhostTriRemoteXProPP`) feeding a multi-scale FPN and a rotated dense head that
predicts 5-parameter oriented boxes `(cx, cy, w, h, θ)`.

The repository also contains two secondary pipelines for comparison: an
experimental two-stage axis-aligned detector and an Ultralytics YOLO benchmark.

## Layout

```
configs/               # YAML configs: base.yaml + exp/*.yaml (no env vars)
pretrain_backbone.py   # classification pretraining for the custom backbone
common/                # shared: config loader, registry, run dirs, backbones, GAConv, rotated IoU
obb_detector/          # primary one-stage oriented (OBB) detector
faster_rcnn/           # experimental two-stage axis-aligned detector (reference only)
yolo_compare/          # Ultralytics YOLO benchmark (comparison only)
runs/                  # per-run outputs (gitignored except runs/README.md)
docs/                  # detailed documentation
CLAUDE.md              # entry point for agents; claude_notes/SETUP.md for full setup
```

## Pipelines

| Pipeline | Boxes | Status |
|---|---|---|
| `obb_detector` | oriented (OBB) | **Primary.** Multi-scale FPN, per-level anchors, delta regression, rotated NMS. |
| `faster_rcnn`  | horizontal (HBB) | Experimental / reference only — see [docs/secondary-pipelines.md](docs/secondary-pipelines.md). |
| `yolo_compare` | horizontal (HBB) | Comparison only; does not use the custom backbone. |

## Quick start

Run everything from the project root. Full setup (clone, env, data) is in
[claude_notes/SETUP.md](claude_notes/SETUP.md).

```bash
pip install -r requirements.txt          # then: pip install -U openmim && mim install mmcv
# place DOTA at configs/base.yaml -> data.root (default ./data/dota_dataset)

# train the OBB detector (config selects backbone / neck / etc.)
python -m obb_detector.train --config configs/exp/baseline.yaml      # ResNet-50 + standard FPN
python -m obb_detector.train --config configs/exp/gaconv_neck.yaml   # ResNet-50 + GAConv neck

# the secondary pipelines
python -m faster_rcnn.train
python -m yolo_compare.benchmark
```

Configuration is YAML only (`configs/base.yaml` + `configs/exp/*.yaml` overrides via a
`_base_` key); there are no environment-variable settings. Each run writes a
self-contained `runs/<name>_<timestamp>/` (config snapshot, checkpoints, TensorBoard,
NOTES.md) and a row in `runs/README.md`. The custom backbone needs
`pretrain_backbone.py` first; ResNet-50 works out of the box.

## Documentation

- [docs/architecture.md](docs/architecture.md) — backbone and detection-pipeline design.
- [docs/usage.md](docs/usage.md) — dataset layout, configuration, pretraining, training, evaluation.
- [docs/secondary-pipelines.md](docs/secondary-pipelines.md) — the Faster R-CNN and YOLO pipelines.
- [docs/roadmap.md](docs/roadmap.md) — open work and planned improvements.
