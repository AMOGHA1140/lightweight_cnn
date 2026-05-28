# Setup & run (training machine)

How to bring this repo up on a fresh machine and train the baseline. Machine-agnostic;
use that machine's Python (a venv/conda env with a CUDA-enabled PyTorch).

## 1. Clone + environment

```bash
git clone <repo-url> lightweight_cnn
cd lightweight_cnn
python -m venv .venv && source .venv/bin/activate    # or conda
pip install -r requirements.txt
```

`requirements.txt` pulls torch/torchvision, numpy, Pillow, tqdm, tensorboard, thop,
matplotlib, plus the YOLO-comparison extras. **mmcv is required** for rotated IoU/NMS
(loss assignment, NMS, mAP) — install it matched to your torch/CUDA:

```bash
pip install -U openmim && mim install mmcv
```

Sanity-check the install without any dataset:

```bash
python -m tests.test_config_and_run
python -m tests.test_gaconv
python -m tests.test_train_utils
python -m tests.test_pipeline_synthetic   # full loss/step path runs once mmcv is present
```

All should pass on CPU or GPU.

## 2. DOTA dataset

Download DOTA-v1.0 and place it at the path in `configs/base.yaml` -> `data.root`
(default `./data/dota_dataset`), in this layout:

```
data/dota_dataset/
├── train/
│   ├── images/                 # *.png / *.jpg
│   └── labelTxt-v1.0/labelTxt/ # *.txt, one per image
├── val/
│   ├── images/
│   └── labelTxt-v1.0/labelTxt/
└── test/
    └── images/                 # no labels
```

Label lines are `x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty`; the dataset converts
each quad to `(cx, cy, w, h, θ)` and rescales to `img_size`. To use a different path,
edit `data.root` in `configs/base.yaml` (do not use environment variables).

## 3. Train

```bash
# Baseline: ResNet-50 + standard 3x3 FPN smooth convs
python -m obb_detector.train --config configs/exp/baseline.yaml

# GAConv: ResNet-50 + GAConv FPN smooth convs (the research variant)
python -m obb_detector.train --config configs/exp/gaconv_neck.yaml
```

ResNet-50 loads ImageNet weights on first run (small download), with
`frozen_stages=1` + `norm_eval=True` (mmrotate DOTA convention). Validation computes
**mAP** every `train.eval_interval` epochs (and on the last) and the best checkpoint is
chosen by mAP. Each run writes a self-contained `runs/<name>_<YYYY_MM_DD-HHmm>/`
(config snapshot, meta.json, metrics.csv, NOTES.md, `checkpoints/{best,last}.pth` —
full state, `tb/`) and appends a row to `runs/README.md`.

Watch training: `tensorboard --logdir runs/`. Resume an interrupted run:

```bash
python -m obb_detector.train --resume runs/<run_dir>
```

**Tip:** before a full run (default `epochs:100`, `img_size:1024`, `batch_size:4` —
heavy on DOTA), do a quick smoke run on a few images / 1–2 epochs to confirm a full
epoch completes on the hardware. Override via a small experiment YAML, e.g.:

```yaml
_base_: ../base.yaml
experiment: {name: smoke, why: "pipeline smoke test", method: "ResNet-50 / FPN-GAConv"}
data: {img_size: 512, batch_size: 2}
train: {epochs: 2}
model: {neck: {smooth_conv: gaconv}}
```

## 4. Evaluate

mAP is computed during training (best checkpoint by mAP). To evaluate any checkpoint
standalone:

```bash
python -m obb_detector.evaluate --config configs/exp/gaconv_neck.yaml \
    --checkpoint runs/<run>/checkpoints/best.pth --split val
```

It prints the per-class AP table + mAP and writes `eval.json` beside the checkpoint.
Thresholds default to the config's `eval:` block and can be overridden with
`--conf-thresh / --iou-thresh / --nms-thresh`.

## 5. Custom backbone (optional, later)

`configs` default to `model.backbone.name: resnet50`. The custom backbone
(`GhostTriRemoteXProPP`) needs classification pretraining first:

```bash
python pretrain_backbone.py --data-dir /path/to/imagenet --out-dir backbone_weights --epochs 100
```

Then set `model.backbone.name: custom` and `paths.pretrained_backbone` to the produced
`backbone_weights/best/backbone.pth`. (`pretrain_backbone.py` keeps its own argparse;
`--config` can seed defaults from a YAML `pretrain:` block.)

## Notes

- Everything is config-driven YAML (`configs/`); there is no `config.py` and no env-var
  configuration.
- `runs/` is gitignored except `runs/README.md` (the shared results index).
- Reproduce any past run: `python -m obb_detector.train --config runs/<run>/config.yaml`.
