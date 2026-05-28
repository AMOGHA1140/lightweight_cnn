# IMPLEMENTATION.md

Extensive implementation reference for the `lightweight_cnn` package — the
refactored form of `Pipeline-Object Detection.ipynb`. This is the document to read
first in a future session. It complements two shorter docs:

- `README.md` — quick start + pipeline status table.
- `REFACTORING_NOTES.md` — the change log (notebook → package) and justifications.
- `SUMMARY.md` — the research context / 6-week plan (predates this refactor).

> The notebook is the untouched source of record. Nothing here modifies it.

---

## 0. TL;DR

Three independent detection pipelines were extracted from one notebook into a
package with a shared `common/` core:

| Package | What it is | Boxes | Status |
|---|---|---|---|
| `obb_detector/` | Custom **one-stage oriented** detector (the research target) | OBB (5-param) | **Primary**, trains; eval/decode path buggy (flagged) |
| `faster_rcnn/` | Two-stage **axis-aligned** experiment (notebook cell 24) | HBB | **Abandoned**, kept for reference |
| `yolo_compare/` | **Ultralytics YOLO** fine-tuning benchmark | HBB | **Comparison only**, no custom backbone |

Run each from the project root: `python -m <package>.train` (or
`yolo_compare.benchmark`).

---

## 1. Repository layout

```
lightweight_cnn/
├── Pipeline-Object Detection.ipynb   # original notebook (DO NOT MODIFY)
├── SUMMARY.md                        # research context / plan
├── README.md                         # quick start
├── REFACTORING_NOTES.md              # notebook → package change log
├── IMPLEMENTATION.md                 # this file
├── requirements.txt
├── config.py                         # paths + hyperparameters (EDIT THIS)
│
├── common/                           # shared across pipelines
│   ├── __init__.py                   #   re-exports the common public API
│   ├── classes.py                    #   DOTA_CLASSES, CLASS2ID, NUM_CLASSES
│   ├── backbone.py                   #   GhostTriRemoteXProPP + all attention blocks
│   ├── rotated_ops.py                #   box_iou_rotated (mmcv→shapely), get_rotated_corners
│   └── model_utils.py                #   count_parameters, print_model_stats, clean_gpu
│
├── obb_detector/                     # PIPELINE 1 (primary)
│   ├── __init__.py
│   ├── dataset.py                    #   DOTADataset (OBB 5-param) + collate_fn
│   ├── fpn.py                        #   FPN neck (out=128)
│   ├── head.py                       #   RotatedDetectionHead (cls/reg/obj)
│   ├── anchors.py                    #   generate_rotated_anchors (radians)
│   ├── detector.py                   #   RemoteDetector (backbone→neck→head)
│   ├── loss.py                       #   DetectionLoss (focal + smoothL1 + BCE)
│   ├── inference.py                  #   decode_obb, decode_predictions (+ rotated NMS)
│   ├── evaluate.py                   #   compute_ap, evaluate_map
│   └── train.py                      #   train_epoch, validate, build_model, main
│
├── faster_rcnn/                      # PIPELINE 3 (abandoned)
│   ├── __init__.py
│   ├── dataset.py                    #   DOTADataset (HBB, cv2) + parse_dota_annotation
│   ├── model.py                      #   FPN/RPNHead/ROIHead/AnchorGenerator/FasterRCNN
│   ├── metrics.py                    #   box_iou(torchvision), calculate_ap/map, evaluate_model
│   └── train.py                      #   train_epoch, main
│
└── yolo_compare/                     # PIPELINE 2 (comparison)
    ├── __init__.py
    ├── dataset_processor.py          #   DOTADatasetProcessor (DOTA → YOLO format on disk)
    └── benchmark.py                  #   YOLOModelTrainer + main
```

Import rules: everything is run from the project root, so `config`, `common`,
`obb_detector`, `faster_rcnn`, `yolo_compare` are all importable as top-level
modules/packages. Pipelines import from `common` and `config`; they never import
each other.

---

## 2. `config.py`

Single editable place for what used to be hard-coded Windows paths (`D:/Abhi/...`).
Every path can also be overridden by an environment variable of the same name.

| Name | Default | Meaning |
|---|---|---|
| `DATA_ROOT` | `./data/dota_dataset` | DOTA root (see §3 for expected layout) |
| `MODELS_DIR` | `./models` | checkpoint output dir |
| `PRETRAINED_BACKBONE` | `${MODELS_DIR}/best_GBR_model.pth` | backbone classification weights |
| `IMG_SIZE` | `1024` | OBB detector input resolution |
| `BATCH_SIZE` | `4` | |
| `NUM_WORKERS` | `0` | DataLoader workers |
| `NUM_EPOCHS` | `100` | |
| `LEARNING_RATE` | `1e-3` | |
| `WEIGHT_DECAY` | `1e-4` | |
| `NORM_MEAN`/`NORM_STD` | ImageNet | normalisation |

`ensure_models_dir()` creates and returns `MODELS_DIR`.

The Faster R-CNN pipeline uses its own local `img_size=800`, `num_epochs=50`,
`lr=1e-3` inside `faster_rcnn/train.py:main()` (matching cell 24), not all of the
shared defaults.

---

## 3. Expected DOTA dataset layout

```
DATA_ROOT/
├── train/
│   ├── images/                       # *.png / *.jpg
│   └── labelTxt-v1.0/labelTxt/       # *.txt, one per image
├── val/
│   ├── images/
│   └── labelTxt-v1.0/labelTxt/       # NOTE: faster_rcnn expects val labels at
│   │                                 #       labelTxt-v1.0/ (one level up) — see §6.3
└── test/
    └── images/                       # no labels
```

A DOTA label line is `x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty`. Header lines
(`imagesource:...`, `gsd:...`) are skipped by all parsers.

---

## 4. `common/` — shared core

### 4.1 `classes.py`
`DOTA_CLASSES` (15 names, canonical order), `CLASS2ID` (name→int), `NUM_CLASSES=15`.
Order matters — class ids are positional indices into this list.

### 4.2 `backbone.py` — `GhostTriRemoteXProPP` and blocks

A lightweight backbone for OBB detection. The attention sub-modules follow the
**official paper implementations** (GhostNet / Coordinate Attention / CBAM); this
intentionally breaks compatibility with the old `best_GBR_model.pth` checkpoint,
which is why the backbone is retrained from scratch (`pretrain_backbone.py`). See
REFACTORING_NOTES.md §7 for the full before/after.

Building blocks:

| Block | Origin | Role |
|---|---|---|
| `GhostModule` | GhostNet, CVPR 2020 | intrinsic maps + **strictly depthwise** cheap op (`groups=init`) |
| `CoordAtt` | CVPR 2021 (houqb) | coordinate attention, `[B,C,H+W,1]` + BN + h_swish + `sigmoid(a)·sigmoid(b)` — used inside every `GhostBottle` |
| `MultiStripAttn` | Strip Pooling-inspired | 4 asymmetric DW convs (1×7,7×1,1×15,15×1) for elongated objects |
| `SEBlock` | CVPR 2018 | squeeze-excite channel attention (standalone; no longer used in the backbone) |
| `ChannelGate` / `SpatialGate` / `CBAM` | ECCV 2018 (Jongchan) | avg+max shared-MLP channel attn + 7×7 spatial attn (with BN) |
| `ChannelShuffleFusion` | ShuffleNet, CVPR 2018 | parameter-free channel shuffle (groups=4) |
| `RotationInvariantFusion` (RIF) | custom | fuses 0/90/180/270° rotations with learnable `alpha[4,C,1,1]` |
| `GhostBottle` | custom composite | Ghost→DWconv(stride)→Ghost→CoordAtt→shuffle→+residual |

Backbone forward (for the default `IMG_SIZE=1024`):

| Stage | Op | Out channels | Out spatial | Stride (cumulative) |
|---|---|---|---|---|
| stem | GhostModule k3 s2 | 48 | 512×512 | 2 |
| stage1 | 3× GhostBottle | 64 | 256×256 | 4 |
| stage2 | 4× GhostBottle | 128 | 128×128 | 8 |
| stage3 | 4× GhostBottle | 192 | 64×64 | 16 |
| rif | RotationInvariantFusion | 192 | 64×64 | 16 |
| stage4 | 2× GhostBottle | 256 | 32×32 | 32 |
| attn | MultiStripAttn→CBAM | 256 | 32×32 | 32 |

Channels come from `make_divisible(B(x), 8)` with `width_mult=1.0`:
48→64→128→192→256.

`forward_features(x)` returns **three feature maps** `[C3, C4, C5]` — C3 `[B,128,128,128]`
(stride 8), C4 `[B,192,64,64]` (stride 16, post-RIF), C5 `[B,256,32,32]` (stride 32,
post-attention) — feeding a real multi-scale FPN. `forward(x)` adds the classification
head (pool over C5 → dropout → FC) used for pretraining only.

### 4.3 `rotated_ops.py` — rotated geometry

- `get_rotated_corners(boxes[N,5]) -> [N,4,2]`: converts `(cx,cy,w,h,θ)` (θ radians)
  to corner points using a standard CCW rotation matrix.
- `box_iou_rotated(boxes1[N,5], boxes2[M,5]) -> [N,M]`: **replaces** the notebook's
  `O(N·M)` Python Sutherland–Hodgman polygon-clip double loop (the bottleneck that
  stalled training). Resolution order:
  1. `mmcv.ops.box_iou_rotated` (CUDA, field-standard) — preferred.
  2. `shapely` polygon intersection (CPU) — fallback.
  3. raises `ImportError` if neither installed.

  Angle-convention note: IoU is invariant to a *global* rotation of the plane, so
  any sign difference between our corner construction and mmcv's is harmless as
  long as every box uses the same convention (gt and anchors do).

### 4.4 `model_utils.py`
- `count_parameters(model) -> (total, trainable)`.
- `print_model_stats(model, input_size, device)` — prints trainable params and,
  if `thop` is installed, GFLOPs.
- `clean_gpu()` — frees cached memory on all visible CUDA devices.

### 4.5 `common/__init__.py`
Re-exports `DOTA_CLASSES, CLASS2ID, NUM_CLASSES, GhostTriRemoteXProPP,
count_parameters, print_model_stats, clean_gpu`. (`rotated_ops` is **not** eagerly
imported here, so importing `common` never triggers the mmcv/shapely import; modules
that need it import `common.rotated_ops` directly.)

---

## 5. Pipeline 1 — `obb_detector/` (PRIMARY)

End-to-end one-stage dense oriented detector. Data flow:

```
image [B,3,1024,1024]
  └─ backbone.forward_features ─► feats: [B,256,32,32]  (single level)
       └─ RemoteDetector wraps to list ─► FPN ─► [ [B,128,32,32] ]
            └─ RotatedDetectionHead ─► (cls_outs, reg_outs, obj_outs)
                 cls: [B, A*15, 32,32]   reg: [B, A*5, 32,32]   obj: [B, A, 32,32]
DetectionLoss(preds, targets, anchors_per_level, device)  with A=6 anchors/loc
```

### 5.1 `dataset.py` — `DOTADataset` (OBB)
Parses 8-point labels → 5-param OBB `(cx,cy,w,h,θ)`:
`cx,cy = mean(xs),mean(ys)`, `w=hypot(p1-p0)`, `h=hypot(p2-p1)`,
`θ=atan2(dy01,dx01)` (radians). Image resized to `img_size` + ImageNet-normalised.
`collate_fn` stacks images, keeps targets as a list of dicts
`{'boxes':[n,5], 'labels':[n]}`.

Box coordinates are rescaled by the per-axis resize factors (`img_size/orig_w`,
`img_size/orig_h`) before computing `(cx,cy,w,h,θ)`, so GT matches the resized image.

### 5.2 `fpn.py` — `FPN(in_channels, out_channels=128)`
Standard top-down FPN: 1×1 lateral → nearest-upsample + add → 3×3 smooth. Vanilla
on purpose (same family as Strip R-CNN's neck) so the backbone/head are what the
ablation isolates. Operates on the 3 backbone levels (C3/C4/C5) as a real pyramid.

### 5.3 `head.py` — `RotatedDetectionHead(num_classes, num_anchors=9, in_channels=128)`
Three branches per level:
- cls subnet: 2×(conv3×3+ReLU) → conv3×3 → `A*num_classes`
- reg subnet: 1×(conv3×3+ReLU) → conv3×3 → `A*5`
- obj branch: conv3×3 → `A`

### 5.4 `anchors.py` — `generate_rotated_anchors(...)`
Per-level anchors, each `(cx,cy,w,h,θ)`, angles in **radians**. With an FPN each
level handles a single scale, so `level_scales` gives the scales per level. Defaults
wired in `train.py`: `level_scales=[[32],[64],[128]]`, `ratios=[0.5,1,2]`,
`angles=[-60,0,60]°` → **9 anchors/location**. Strides are derived from the real
feature-map sizes (`img_size // H`) → `[8,16,32]` at 1024.

### 5.5 `detector.py` — `RemoteDetector(backbone, neck, head)`
`forward`: `feats = backbone.forward_features(x)` (always a 3-level list);
`feats = neck(feats)`; `return head(feats)`.

### 5.6 `loss.py` — `DetectionLoss(num_classes, alpha=0.25, gamma=2.0)`
Per image: flatten all levels/anchors, then
- **assignment**: `ious = box_iou_rotated(gt, anchors)`; each anchor takes its
  best-IoU gt; `pos = max_iou > 0.5` (naive threshold; ATSS planned).
- **cls**: focal loss via `torchvision.ops.sigmoid_focal_loss(reduction="sum")`
  normalised by `max(1, num_pos)` (equivalent to the notebook's hand-written focal).
- **reg**: `SmoothL1` on positive anchors only, against anchor-relative deltas
  (`encode_obb`); the head predicts deltas, not absolute boxes.
- **obj**: BCE-with-logits over all anchors.
- total = cls + reg + obj (equal weight). Returns dict
  `{total_loss, cls_loss, bbox_loss, obj_loss}`.

### 5.7 `inference.py` — `decode_obb`, `decode_predictions(...)`
- `decode_obb(deltas, anchors)`: inverse of `encode_obb`.
- `decode_predictions(...)`: flattens all levels, `sigmoid` on cls/obj,
  confidence `= obj * max-cls`, threshold, decode via `decode_obb`, per-class
  rotated NMS (mmcv `nms_rotated` if available, else greedy `box_iou_rotated`).
  Returns one `(boxes, scores, labels)` triple per image.

### 5.8 `evaluate.py` — `compute_ap`, `evaluate_map`
- `compute_ap(recall, precision)`: standard all-point VOC AP (kept; no clean lib
  replacement without pycocotools).
- `evaluate_map(...)`: VOC-style mAP using `common.rotated_ops.box_iou_rotated`,
  decoding via the fixed `decode_predictions`.

### 5.9 `train.py`
- `build_model(device, img_size)`: builds backbone (loads `PRETRAINED_BACKBONE`
  `strict=False` if present), probes feature shapes, builds `FPN(128)` + head,
  returns `(model, feature_sizes)`.
- `train_epoch(...)`: AMP via `torch.amp.autocast(device_type=...)` +
  `GradScaler`, grad-clip norm 10. (Modernised from the deprecated
  `torch.cuda.amp`.)
- `validate(...)`: loss-only eval.
- `main()`: builds loaders, model, `DataParallel` if >1 GPU, anchors,
  `AdamW(lr*num_gpus)`, `CosineAnnealingLR`; saves `best_detector.pth` on best val
  loss + `checkpoint_epoch_{n}.pth` every 10 epochs.

Run: `python -m obb_detector.train`.

---

## 6. Pipeline 3 — `faster_rcnn/` (ABANDONED, reference only)

Ported faithfully from cell 24 but it is **not** the research direction. Read the
caveats before touching it.

### 6.1 Why it is abandoned
- **Axis-aligned (HBB)**, contradicting the firm one-stage-OBB decision.
- The **ROI head trains on `torch.randn` features** with synthetic targets — it
  never pools real RoIs, so the second stage learns nothing.
- `forward_test` returns top-k raw RPN outputs with no decode/NMS.
- The "multi-scale FPN input" is faked by `adaptive_avg_pool2d` of the single
  backbone map at three resolutions.

### 6.2 `model.py`
`FPN(out=256, num_levels=5)`, `RPNHead`, `ROIHead`, `AnchorGenerator` (HBB anchors),
`FasterRCNN` (forward / forward_train / forward_test, box encode, RPN/ROI target
assignment). Internal IoU now uses `torchvision.ops.box_iou`.

### 6.3 `dataset.py`
OpenCV load → letterbox resize to `img_size=800` square + pad; boxes scaled by the
same factor. **`parse_dota_annotation` was reconstructed** — cell 24 called it but
never defined it, so the cell could not run. Returns enclosing axis-aligned box +
class id. Note: val annotations are read from `labelTxt-v1.0/` (one level up vs the
OBB pipeline's `labelTxt-v1.0/labelTxt/`) — preserved from cell 24; verify against
your disk layout. Only `custom_collate_fn` is kept (the two unused collate variants
in cell 24 were dropped).

### 6.4 `metrics.py`
`box_iou` (= `torchvision.ops.box_iou`), `calculate_ap` (11-point VOC),
`calculate_map`, `evaluate_model`.

Run: `python -m faster_rcnn.train`.

---

## 7. Pipeline 2 — `yolo_compare/` (COMPARISON only)

Independent of the custom backbone — fine-tunes off-the-shelf Ultralytics YOLO on
DOTA converted to YOLO format. For comparison numbers only.

### 7.1 `dataset_processor.py` — `DOTADatasetProcessor`
Converts DOTA 8-point labels → enclosing HBB → normalised YOLO
`<cls> <xc> <yc> <w> <h>`, copies images into `yolo_format/<split>/images` +
`/labels`, writes `dataset.yaml`. Idempotent (`prepare_dataset` reuses an existing
`yolo_format/`).

### 7.2 `benchmark.py` — `YOLOModelTrainer` + `main`
Trains the models in `self.models` (default `{"yolov8s": "yolov8s.pt"}` — add more
as needed), evaluates, writes `model_comparison.csv` + comparison/heatmap PNGs.
The notebook's `install_packages()` pip-subprocess hack was removed in favour of
`requirements.txt`. `main()` points both train and val at `DATA_ROOT` by default
(the notebook used a separate "balanced" dataset for training).

Run: `python -m yolo_compare.benchmark`.

---

## 8. Standard-implementation replacements (summary)

| Component | Was | Now | Behaviour change? |
|---|---|---|---|
| Rotated IoU | O(N·M) polygon-clip Python loop (×3 copies) | `common.rotated_ops` (mmcv→shapely) | None (same quantity, far faster) |
| Focal loss | hand-written | `torchvision.ops.sigmoid_focal_loss` + same norm | None |
| HBB IoU | hand-written `box_iou` | `torchvision.ops.box_iou` | None |
| AMP | `torch.cuda.amp` (deprecated) | `torch.amp` | None |
| FLOPs | `thop` | kept | — |
| YOLO `install_packages()` | pip subprocess in-cell | `requirements.txt` | removed |

**Deliberately kept (not replaced):** backbone sub-modules (SE/CBAM/CoordAtt) — to
preserve `best_GBR_model.pth` state_dict compatibility and because they are the
contribution; `compute_ap`/`calculate_ap` — already the standard VOC algorithms.

---

## 9. Status of the original Week-1 TODOs

Most of the originally-flagged issues have now been fixed (see REFACTORING_NOTES.md
§7): multi-scale backbone output, per-level anchors + correct strides, box-coord
rescaling, anchor-relative regression deltas, and the decode/eval path (anchor
decoding + per-class rotated NMS). The backbone attention modules were corrected to
match their official papers, which is why the old `best_GBR_model.pth` is retrained
from scratch (`pretrain_backbone.py`).

Still open (deliberate research/tuning work, not bugs):

1. **Naive IoU>0.5 assignment** (`obb_detector/loss.py`): upgrade to ATSS.
2. **Loss weighting**: cls/reg/obj summed with equal weight (no balancing).
3. **Anchor hyperparameters**: per-level scales/ratios/angles are a starting point
   to be tuned against baseline results.

---

## 10. Notebook cell → file mapping

| Cell(s) | Content | Destination |
|---|---|---|
| 1, 2, 23 | GPU check / imports / device setup | discarded (boilerplate) |
| 3 | OBB `DOTADataset` + collate | `obb_detector/dataset.py` |
| 4, 21, 22, 24 (top) | backbone + attention modules (identical) | `common/backbone.py` |
| 5 | FPN | `obb_detector/fpn.py` |
| 6 | `RotatedDetectionHead` | `obb_detector/head.py` |
| 7 | `RemoteDetector` | `obb_detector/detector.py` |
| 8 (degrees), 21 (radians) | rotated anchors | `obb_detector/anchors.py` (radians) |
| 9–12, 13, 17 | rotated geometry / IoU (×3) | `common/rotated_ops.py` (replaced) |
| 13, 21 | `DetectionLoss` | `obb_detector/loss.py` |
| 14 (AMP), 19 (setup) | training | `obb_detector/train.py` |
| 15 | `compute_ap` | `obb_detector/evaluate.py` |
| 16 | `decode_predictions` | `obb_detector/inference.py` |
| 18 | `evaluate_map` | `obb_detector/evaluate.py` |
| 20, 32 | `clean_*_gpu` | `common/model_utils.py` |
| 21, 22 | full duplicate OBB scripts | consolidated into `obb_detector/` |
| 24 | Faster R-CNN (RPN/ROI/FasterRCNN/dataset/metrics/train) | `faster_rcnn/*` |
| 25–31, 33 | YOLO comparison (×8; cell 33 canonical) | `yolo_compare/*` |
| 34, 35 | empty | — |

---

## 11. Environment & how to run

```bash
cd lightweight_cnn
pip install -r requirements.txt          # install ONE of mmcv (preferred) / shapely
# point config at your data (edit config.py or export env vars):
export DOTA_DATA_ROOT=/path/to/dota_dataset
export MODELS_DIR=/path/to/models
export PRETRAINED_BACKBONE=/path/to/best_GBR_model.pth

python -m obb_detector.train             # primary OBB detector
python -m faster_rcnn.train              # experimental two-stage (see §6 caveats)
python -m yolo_compare.benchmark         # YOLO comparison
```

Target hardware: 2× RTX A6000 (the OBB trainer uses `DataParallel` when
`device_count() > 1` and scales the LR by the GPU count). The repo was refactored
on a CPU/3050 laptop with no deps installed, so it was verified by `py_compile` +
static import resolution, **not** runtime-tested — expect to shake out runtime
issues on first real run, especially in the flagged areas of §9.

---

## 12. Validation performed at refactor time

- `python -m py_compile` over all 23 modules — pass.
- Static AST pass resolving every intra-project `from common/config/<pkg> import …`
  against actually-defined names — no unresolved imports.
- No execution (torch/mmcv/ultralytics not installed in the edit environment).
