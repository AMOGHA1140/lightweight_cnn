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
and falls back to fp16 if the GPU lacks bf16 support).

## 2. Train the OBB detector

```bash
export PRETRAINED_BACKBONE=backbone_weights/best/backbone.pth
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
