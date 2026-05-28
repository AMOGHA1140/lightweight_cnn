# Refactoring Notes

This document records every change made while converting
`Pipeline-Object Detection.ipynb` (35 code cells) into the package, and the
reasoning behind each. The notebook itself was **not modified**.

Scope agreed with the user:
- **All three** pipelines ported, into separate top-level packages.
- **Faithful refactor**: preserve behaviour and pipeline logic; do not implement
  the planned research fixes (multi-scale backbone, ATSS, head redesign). Known
  broken pieces are kept and flagged rather than fixed.
- **Replacements**: where a component merely re-implements something a standard
  library already provides, use the library version for maintainability — even
  when the win is marginal.

---

## 1. De-duplication (the main DRY win)

The notebook defined the same code many times:

- The **backbone + all attention modules** (`GhostModule`, `CoordAtt`,
  `MultiStripAttn`, `SEBlock`, `CBAM`, `ChannelShuffleFusion`,
  `RotationInvariantFusion`, `GhostBottle`, `GhostTriRemoteXProPP`) appeared
  **identically in cells 4, 21, 22 and 24** → now a single `common/backbone.py`.
- The whole OBB pipeline was duplicated as three self-contained scripts
  (cells 3–19, 21, 22) → consolidated into the `obb_detector/` package.
- `DOTA_CLASSES` was redefined in ~6 cells → `common/classes.py`.
- The rotated-IoU geometry (`get_rotated_corners` / `polygon_area` /
  `polygon_clip`) appeared **three times** (cells 9–12, 13, 17) → one
  `common/rotated_ops.py`.
- `clean_gpu` / `clean_all_gpus` (cells 20, 32) → `common/model_utils.py`.
- The YOLO cell was duplicated **8 times** (cells 25–31, 33). The latest (cell 33)
  was taken as canonical → `yolo_compare/`.

When duplicates differed slightly, the most-refined version was chosen and the
discrepancy noted (see §3).

---

## 2. Standard-implementation replacements

| What | Was (notebook) | Now | Why |
|---|---|---|---|
| **Rotated IoU** | `O(N*M)` Python Sutherland–Hodgman polygon-clipping double loop. The documented bottleneck that stalled training (KeyboardInterrupt). | `common.rotated_ops.box_iou_rotated`, preferring `mmcv.ops.box_iou_rotated` (CUDA, field-standard) and falling back to `shapely`. | Maintainability + removes the bottleneck via a maintained op. IoU is convention-invariant under a global rotation, so the swap is behaviour-preserving. |
| **Focal loss** | Hand-written in `DetectionLoss.focal_loss`. | `torchvision.ops.sigmoid_focal_loss(..., reduction="sum")`, with the original `/max(1, num_pos)` normalisation kept. | Numerically identical, less code to maintain. |
| **Axis-aligned IoU** (Faster R-CNN) | Hand-written `box_iou`. | `torchvision.ops.box_iou`. | Direct standard equivalent. |
| **FLOPs counting** | `thop` (cell 21). | Kept (`common.model_utils.print_model_stats`). | Already standard. |
| **AMP API** | `torch.cuda.amp.autocast/GradScaler` (deprecated). | `torch.amp.autocast(device_type=...)` / `torch.amp.GradScaler(device_type)`. | Current non-deprecated API. |
| **`install_packages()` pip-subprocess hack** (YOLO cells) | In-notebook `pip install` loop. | Removed; dependencies live in `requirements.txt`. | Notebook convenience, not pipeline logic. |

### Deliberately NOT replaced

- **Backbone sub-modules** (`SEBlock`, `CBAM`, `CoordAtt`, …) have library
  equivalents (e.g. `torchvision.ops.SqueezeExcitation`) but are **kept verbatim**.
  Reason: the pretrained weights `best_GBR_model.pth` were saved against these
  exact parameter names; swapping modules would rename parameters and break
  `load_state_dict`. They are also part of the novel contribution.
- **`compute_ap` / `calculate_ap`** (VOC all-point and 11-point AP) are kept —
  they already implement the standard algorithms and a library replacement
  (pycocotools) would change semantics and add a heavy dependency.

---

## 3. Behaviour-preserving choices among conflicting duplicates

- **`forward_features` return type**: cell 4 returned a tensor, cells 21/22 a
  list. Standardised on returning a **single tensor**; the OBB `RemoteDetector`
  wraps it in a list for the neck (matches cell 7). Backbone weight compatibility
  is unaffected.
- **Anchor angles**: cell 8 stored angles in degrees; cell 21 converted to
  radians. Kept the **radians** version (`anchors.py`) — consistent with all the
  geometry code.
- **`DetectionLoss` assignment**: kept the cleaner cell-21 implementation
  (explicit `pos_idx` advanced indexing).
- **OBB training loop**: merged cell 14's clean AMP loop with cell 19's full setup
  (pretrained loading, DataParallel, cosine schedule). Debug `print` statements
  removed.
- Dropped the two **unused** collate variants in cell 24 (`pad_collate_fn`,
  `advanced_collate_fn`); only `custom_collate_fn` was wired up.

---

## 4. Known-broken pieces kept and flagged (NOT fixed)

Per the faithful-refactor scope, these are preserved with `FIXME`/`NOTE` comments:

- **Single-scale backbone** → FPN is effectively a pass-through
  (`common/backbone.py`, `obb_detector/fpn.py`).
- **`obb_detector/inference.py`** `decode_predictions`: wrong `num_classes`
  divisor, returns raw (un-decoded) regression deltas, no rotated NMS.
- **`obb_detector/evaluate.py`** `evaluate_map`: calls `decode_predictions` with
  the wrong argument structure — non-functional as written (never ran in the
  notebook because training never completed).
- **Naive IoU>0.5 assignment** in `DetectionLoss` (ATSS planned).

## 5. Reconstructed missing code

- **`parse_dota_annotation`** (`faster_rcnn/dataset.py`): cell 24 *called* this
  function but never defined it, so the cell could not run. A faithful
  reconstruction (8-point → enclosing axis-aligned box + class id) is provided so
  the pipeline is at least coherent. Flagged in the file.

## 6. Faster R-CNN caveats (cell 24)

Ported faithfully but flagged prominently (`faster_rcnn/model.py` docstring): it
is axis-aligned (contradicts the firm one-stage-OBB decision), the ROI head trains
on `torch.randn` features with synthetic targets, `forward_test` returns raw RPN
top-k without decoding/NMS, and the "multi-scale" FPN input is faked by pooling a
single feature map. Retained for completeness/comparison only.

---

## 7. Backbone cleanup + pipeline fixes (post-refactor round)

This round goes **beyond** the faithful-refactor scope: the attention modules were
corrected to match their official papers and the dead detection pipeline was made
functional. This **intentionally breaks** the old `best_GBR_model.pth` checkpoint —
the backbone is retrained from scratch via `pretrain_backbone.py`. The notes in §2
and §3 about keeping the modules verbatim and returning a single feature map are
**superseded** by this section.

**Backbone (`common/backbone.py`):**

| Module | Was | Now (official) |
|---|---|---|
| `GhostModule` | cheap op `groups=min(init,cheap)` | strictly depthwise (`groups=init`) — GhostNet, CVPR 2020 |
| `CoordAtt` | `[B,C,2,H]` cat (square-only), no BN, ReLU, `sigmoid(a+b)` | `[B,C,H+W,1]` cat, BN, h_swish, `sigmoid(a)*sigmoid(b)` — CVPR 2021 |
| `CBAM` | channel attn was `SEBlock` (avg-pool only), no BN in spatial | `ChannelGate` (avg+max shared MLP) + `SpatialGate` (with BN) — ECCV 2018 |
| backbone tail | `strip → SE → CBAM` (double channel recalibration) | `strip → CBAM` (SE removed) |
| `GhostBottle` / `MultiStripAttn` | redundant `min(in,in)` groups, dead `ch>0`/`hasattr` guards | removed |

`SEBlock` is kept as a standalone class (correct on its own; no longer used inside
the backbone).

**Multi-scale output:** `forward_features` now returns `[C3, C4, C5]` (strides
8/16/32; 128/192/256 ch), so the FPN is a real pyramid instead of a pass-through.
`RemoteDetector` no longer wraps a single tensor. (Note: this changes the shared
backbone's `forward_features` signature — `faster_rcnn/model.py`, which expects a
single tensor, must adapt accordingly.)

**Detection pipeline:**
- `dataset.py`: box coordinates are now rescaled to the resized-image frame
  (previously left in the original frame — GT did not match the image).
- `anchors.py` / `train.py`: anchors are now **per-level** (one scale per FPN
  level × 3 ratios × 3 angles = 9/location, down from a flat 54), strides derived
  from real feature-map sizes, and the head's `num_anchors` matches (was a silent
  6-vs-54 mismatch).
- `loss.py`: regression targets are anchor-relative **deltas** (`encode_obb`),
  not absolute GT coordinates.
- `inference.py`: `decode_predictions` rewritten — correct class count, anchor
  decoding (`decode_obb`), confidence = `obj * max-cls`, per-class rotated NMS
  (mmcv `nms_rotated` if available, else a greedy `box_iou_rotated` fallback);
  returns one `(boxes, scores, labels)` triple per image.
- `evaluate.py`: `evaluate_map` rewired to the fixed `decode_predictions`.
- `train.py`: loads only backbone weights (drops `fc.*`, since `strict=False`
  does not skip a size mismatch on a present key).

**Pretraining:** `pretrain_backbone.py` (new) trains the backbone on an
`ImageFolder` classification dataset (AdamW + cosine LR, AMP, best-by-val-acc) and
saves a checkpoint the detector loads with `strict=False`.
