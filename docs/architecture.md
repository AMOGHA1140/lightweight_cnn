# Architecture

The primary pipeline (`obb_detector/`) is a one-stage dense oriented detector:

```
image [B,3,H,W]
  → backbone.forward_features → [C3, C4, C5]      (strides 8 / 16 / 32)
  → FPN neck                  → 3 levels @ 128ch
  → RotatedDetectionHead      → (cls_outs, reg_outs, obj_outs) per level
```

Oriented boxes use the 5-parameter form `(cx, cy, w, h, θ)` with `θ` in radians.

## Backbone — `GhostTriRemoteXProPP` (`common/backbone.py`)

An efficient backbone built from Ghost bottlenecks with attention. Channel
progression (`width_mult=1.0`): 48 → 64 → 128 → 192 → 256.

| Stage | Op | Out ch | Stride | Output |
|---|---|---|---|---|
| stem | GhostModule k3 s2 | 48 | 2 | |
| stage1 | 3× GhostBottle | 64 | 4 | |
| stage2 | 4× GhostBottle | 128 | 8 | **C3** |
| stage3 | 4× GhostBottle | 192 | 16 | |
| rif | RotationInvariantFusion | 192 | 16 | **C4** |
| stage4 | 2× GhostBottle | 256 | 32 | |
| attn | MultiStripAttn → CBAM | 256 | 32 | **C5** |

`forward_features(x)` returns `[C3, C4, C5]` for the FPN. `forward(x)` adds a
classification head (global pool over C5 → dropout → linear) used only during
backbone pretraining.

### Building blocks

- **GhostModule** (GhostNet, CVPR 2020): a primary 1×1 conv produces the intrinsic
  feature maps; a strictly depthwise cheap op (`groups = init`) produces the ghost
  maps, which are concatenated with the intrinsic ones.
- **CoordAtt** (Coordinate Attention, CVPR 2021): pools along H and W separately,
  shares a 1×1 conv + BN + h-swish, then applies independent H- and W-direction
  attention via `sigmoid`. Used inside every `GhostBottle`.
- **CBAM** (ECCV 2018): `ChannelGate` (avg-pool + max-pool through a shared MLP) →
  `SpatialGate` (channel-wise avg+max → 7×7 conv + BN → sigmoid).
- **MultiStripAttn**: four asymmetric depthwise convs (1×7, 7×1, 1×15, 15×1) whose
  sum gates the input via `sigmoid`, capturing elongated structures.
- **RotationInvariantFusion (RIF)**: fuses the feature map and its 90°/180°/270°
  rotations with learnable per-channel weights `alpha[4, C, 1, 1]`.
- **ChannelShuffleFusion** (ShuffleNet): parameter-free channel shuffle (groups=4).
- **GhostBottle**: `GhostModule(expand) → depthwise conv(stride) → GhostModule(project)
  → CoordAtt → channel shuffle → + residual shortcut`.
- **SEBlock**: standalone squeeze-and-excitation (not used inside the backbone).

## Neck — `FPN` (`obb_detector/fpn.py`)

Top-down FPN over `[C3, C4, C5]`: 1×1 lateral conv → nearest-upsample + add →
3×3 smooth, producing three feature maps at 128 channels each.

## Anchors (`obb_detector/anchors.py`)

`generate_rotated_anchors` places anchors per location, each `(cx, cy, w, h, θ)`.
With an FPN each level handles a single object scale, so scales are given
per level. Defaults (wired in `obb_detector/train.py`):

- `level_scales = [[32], [64], [128]]`
- `anchor_ratios = [0.5, 1.0, 2.0]`
- `anchor_angles = [-60, 0, 60]` degrees (converted to radians)

→ **9 anchors per location**. Strides come from the real feature-map sizes
(`img_size // H`) → `[8, 16, 32]` at 1024.

## Head — `RotatedDetectionHead` (`obb_detector/head.py`)

Three branches applied to every FPN level (`A` = anchors per location):

- classification: 2×(conv3×3 + ReLU) → conv3×3 → `A·num_classes`
- box regression: 1×(conv3×3 + ReLU) → conv3×3 → `A·5`
- objectness: conv3×3 → `A`

## Box encoding (`obb_detector/loss.py`, `obb_detector/inference.py`)

The head predicts deltas relative to anchors, not absolute boxes.

- `encode_obb(gt, anchors)`: `dx, dy` normalised by anchor `w, h`; `dw, dh` as
  log-ratios; `dθ` as the angle difference.
- `decode_obb(deltas, anchors)`: the inverse.

## Loss — `DetectionLoss` (`obb_detector/loss.py`)

Per image, all levels/anchors are flattened, then:

- **assignment**: each anchor takes its best-IoU GT (rotated IoU); positive if
  IoU > 0.5.
- **classification**: sigmoid focal loss (`α=0.25, γ=2.0`), normalised by the
  positive count.
- **regression**: Smooth L1 on positive anchors, against `encode_obb` deltas.
- **objectness**: BCE-with-logits over all anchors.
- total = cls + reg + obj (equal weight).

## Inference (`obb_detector/inference.py`)

`decode_predictions` flattens all levels, applies `sigmoid` to cls/obj, computes
confidence `= obj · max-class-prob`, thresholds, decodes via `decode_obb`, and runs
per-class rotated NMS (`mmcv.ops.nms_rotated` if installed, otherwise a greedy
fallback built on `box_iou_rotated`). It returns one `(boxes, scores, labels)`
triple per image.

## Evaluation (`obb_detector/evaluate.py`)

`evaluate_map` decodes detections with `decode_predictions`, matches them to ground
truth per class using rotated IoU, and computes VOC-style all-point AP per class and
the mean (mAP).

## Rotated geometry (`common/rotated_ops.py`)

- `get_rotated_corners(boxes)`: `(cx, cy, w, h, θ)` → 4 corner points.
- `box_iou_rotated(a, b)`: pairwise IoU via `mmcv.ops.box_iou_rotated` when
  available, otherwise a `shapely` polygon-intersection fallback.
