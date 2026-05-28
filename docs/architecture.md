# Architecture

The primary pipeline (`obb_detector/`) is a one-stage dense oriented detector:

```
image [B,3,H,W]
  → backbone.forward_features → [C3, C4, C5]      (strides 8 / 16 / 32)
  → FPN neck                  → 3 levels @ out_channels (256 default)
  → RotatedDetectionHead      → (cls_outs, reg_outs, obj_outs) per level
```

Oriented boxes use the 5-parameter form `(cx, cy, w, h, θ)` with `θ` in radians.

The backbone is selected by config (`model.backbone.name`): `resnet50` (ImageNet,
the current testbed) or `custom` (`GhostTriRemoteXProPP`, needs pretraining). Both
expose `forward_features(x) -> [C3, C4, C5]`, so the neck/head/anchors are unchanged.

## Backbone — `ResNet50Backbone` (`common/backbone_resnet.py`)

The default testbed backbone: torchvision ResNet-50, ImageNet-pretrained, fine-tuned
with `frozen_stages=1` (freeze stem + layer1) and `norm_eval=True` (BatchNorm frozen)
per the mmdetection/mmrotate DOTA convention. `forward_features` returns `layer2/3/4`
outputs (C3/C4/C5) = channels `[512, 1024, 2048]` at strides `[8, 16, 32]`.

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
smooth, producing three feature maps at `out_channels` (256 by default). The smooth
stage is a standard 3×3 conv (`smooth_conv: standard`) or **GAConv**
(`smooth_conv: gaconv`, `common/gaconv.py`) — a geometric adaptive conv that predicts
per-location `(θ, σ_major, σ_minor)` and drives a depthwise deformable conv. GAConv is
identity-initialised, so it starts equivalent to a depthwise 3×3 and learns to deviate.

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

The cls/obj prediction biases use the RetinaNet prior-probability init
(`-log((1-p)/p)`, `p = model.head.prior_prob`) so focal loss starts at a sane scale.

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
per-class rotated NMS (`mmcv.ops.nms_rotated`; mmcv is required). It returns one
`(boxes, scores, labels)` triple per image.

## Evaluation (`obb_detector/evaluate.py`)

`evaluate_map` decodes detections with `decode_predictions`, matches them to ground
truth per class using rotated IoU, and computes VOC-style all-point AP per class and
the mean (mAP).

## Rotated geometry (`common/rotated_ops.py`)

- `get_rotated_corners(boxes)`: `(cx, cy, w, h, θ)` → 4 corner points.
- `box_iou_rotated(a, b)`: pairwise IoU via `mmcv.ops.box_iou_rotated` (mmcv required;
  raises a clear error if missing).
