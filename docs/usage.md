# Usage

All commands run from the project root so that `common` and the pipeline packages
import as top-level modules.

## Dependencies

```bash
# Pinned stack (Python 3.10, torch 2.1.0, mmcv 2.1.0); order matters. Full details
# and the why are in claude_notes/SETUP.md.
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
pip install -r requirements.txt
```

`mmcv` is **required** for rotated IoU + NMS (loss assignment, NMS, mAP); the code raises
a clear error if it is missing (no slow CPU fallback). Prebuilt mmcv wheels exist only
for torch 2.1.0 and Python 3.8-3.11 — newer torch or Python 3.12+ force a failing source
build, so do not bump those without checking the wheel index.

## Configuration (YAML, no env vars)

All configuration is YAML under `configs/`, loaded by `common/config.py`. There is no
`config.py` module and no environment-variable configuration.

- `configs/base.yaml` holds every default, grouped under `data`, `model`, `anchors`,
  `train`, `paths`, `experiment`.
- An experiment file sets `_base_: ../base.yaml` and overrides only the keys it
  changes; bases are deep-merged and the experiment wins.

```yaml
# configs/exp/gaconv_neck.yaml
_base_: ../base.yaml
experiment: {name: gaconv_neck, why: "...", method: "ResNet-50 / FPN-GAConv"}
model: {neck: {smooth_conv: gaconv}}
```

Key fields: `data.{root,img_size,batch_size,num_workers}`,
`model.backbone.{name,pretrained,frozen_stages,norm_eval}` (name: `resnet50` | `custom`),
`model.neck.{out_channels,smooth_conv}` (smooth_conv: `standard` | `gaconv`),
`train.{epochs,lr,weight_decay,grad_clip}`, `paths.{pretrained_backbone,runs_dir}`.
Components are built by name through `common/registry.py`.

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

The detector initialises from a backbone trained on ImageNet-1k classification.
The dataset must be an `ImageFolder` with `train/` and `val/` splits:

```
<data-dir>/train/<class>/*.JPEG
<data-dir>/val/<class>/*.JPEG
```

```bash
python pretrain_backbone.py \
    --data-dir /path/to/imagenet \
    --out-dir backbone_weights \
    --epochs 100 --batch-size 256 --lr 1e-3
```

Recipe: RandomResizedCrop + horizontal flip + RandAugment + ColorJitter +
RandomErasing, label-smoothed cross-entropy, AdamW (no weight decay on norm/bias),
linear warmup → cosine decay, AMP, and a weight EMA. Each epoch the live and EMA
models are both evaluated and the better one becomes the saved "best". `--num-classes`
defaults to 200; the dataset's actual class count is always used. Multiple GPUs are
used automatically via `DataParallel`.

### Output directory (`--out-dir`)

```
<out-dir>/
  config.json        # resolved run arguments (reloaded on resume)
  metrics.csv        # per-epoch train/val/EMA loss & accuracy (crash-safe)
  tb/                # TensorBoard events: per-epoch curves + per-step train loss
                     #   (run: tensorboard --logdir <out-dir>/tb)
  best/
    checkpoint.pth   # record of the best epoch
    backbone.pth     # bare backbone weights for the detector
  epoch_<n>/
    checkpoint.pth   # full resumable state: model, optimizer, scheduler, scaler
```

A full checkpoint is written every `--save-every` epochs (default 1 = every epoch).

### Resuming

```bash
python pretrain_backbone.py --data-dir /path/to/imagenet --out-dir backbone_weights --resume auto
```

`--resume auto` picks the latest `epoch_<n>/`; you can also pass an explicit
checkpoint path or epoch directory. Run arguments are reloaded from `config.json`
(pass `--epochs N` to extend training). Model/optimizer/scheduler/AMP-scaler are
restored exactly; the EMA is re-seeded from the resumed weights and re-accumulates
(pass `--save-ema-in-ckpt` during training for exact EMA resume).

Key flags: `--save-every`, `--warmup-epochs`, `--ema-decay`, `--no-ema`,
`--rand-aug-magnitude` (≤0 disables), `--label-smoothing`, `--seed`,
`--amp-dtype {bfloat16,float16}` (default `bfloat16`; bf16 needs no loss scaling
and falls back to fp16 if the GPU lacks bf16 support). `--config <yaml>` optionally
seeds defaults from a `pretrain:` block (CLI flags still win); the argparse interface
is otherwise unchanged.

## 2. Train the OBB detector

```bash
python -m obb_detector.train --config configs/exp/gaconv_neck.yaml   # or baseline.yaml
```

The config selects the backbone (`resnet50` works out of the box; `custom` needs the
pretrained weights from step 1 at `paths.pretrained_backbone`), neck, and
hyperparameters. Training uses AMP, gradient clipping, AdamW, and a cosine schedule,
with `DataParallel` across multiple GPUs. Each run writes a self-contained
`runs/<experiment.name>_<YYYY_MM_DD-HHmm>/` holding the resolved `config.yaml`,
`meta.json` (git commit + command), `NOTES.md`, `checkpoints/{best,last}.pth`, and
`tb/` TensorBoard logs, and appends a row to `runs/README.md`. Reproduce a past run
with `--config runs/<run>/config.yaml`.

## 3. Evaluate

mAP (per-class AP + mean) is computed during training every `train.eval_interval`
epochs, logged to TensorBoard (`metrics/val_mAP`, `AP/<class>`) and `metrics.csv`, and
used to pick `checkpoints/best.pth`. To evaluate a checkpoint standalone:

```bash
python -m obb_detector.evaluate --config configs/exp/gaconv_neck.yaml \
    --checkpoint runs/<run>/checkpoints/best.pth --split val
```

This prints the per-class AP table + mAP and writes `eval.json` next to the checkpoint.
Thresholds come from the config `eval:` block, overridable via
`--conf-thresh / --iou-thresh / --nms-thresh`. `evaluate_map(...)` remains importable
for programmatic use.
