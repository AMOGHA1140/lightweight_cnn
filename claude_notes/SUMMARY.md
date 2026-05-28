# Project Context: Lightweight Oriented Object Detection for Remote Sensing

## Resume Instructions for Claude Code
This document summarizes an extensive conversation about building a lightweight oriented object detector for remote sensing images. The user is a research intern (started ~May 18, 2026) working in a research lab, targeting publication at a decent conference. The codebase is a single Jupyter notebook (`Pipeline-Object_Detection.ipynb`). The user prefers to be asked clarifying questions before proceeding with any task.

---

## 1. Project Overview

**Goal**: Build a lightweight oriented bounding box (OBB) object detector for remote sensing (aerial/satellite) imagery, trained on the DOTA dataset (15 classes: planes, ships, vehicles, bridges, harbors, sports fields, etc.).

**Key constraint**: Lightweight — the novel backbone is ~1.4M parameters, ~5.6 GFLOPs. The overall model should remain efficient.

**Target benchmarks for comparison**:
- Strip R-CNN (arXiv 2501.03775, Nankai/Ming-Ming Cheng's group) — current SOTA on DOTA, 82.75% mAP with 30.5M params. Uses Oriented R-CNN framework (two-stage) with StripNet backbone + strip head. **Their neck is vanilla FPN — they did not innovate on the neck.**
- LMW-YOLO (Scientific Reports 2026, South China Agricultural Univ.) — lightweight HBB detector (2.6M params) for remote sensing small objects. Uses YOLO11n with CSD strategy (different modules at P3/P4/P5 levels). **HBB only, not OBB.**
- LSKNet (ICCV 2023 / IJCV 2024) — large selective kernel backbone for RS detection
- PKINet (CVPR 2024) — poly kernel inception backbone for RS detection

**Detection paradigm**: One-stage, dense prediction (NOT two-stage like Oriented R-CNN). This is a firm decision.

**Timeline**:
- Internship started: ~May 18, 2026
- Useful results needed by: ~July 4-5, 2026 (~6 weeks)
- After July 5: ablation studies and publication work

---

## 2. Current Codebase State

The code lives in a single `.ipynb` notebook. It contains:

### 2.1 What EXISTS and WORKS (backbone + utilities):

**DOTADataset**: PyTorch Dataset for DOTA. Parses 8-coordinate rotated quadrilateral labels → converts to (cx, cy, w, h, angle) format. Images resized to 1024×1024 with ImageNet normalization.

**GhostModule** (from GhostNet, CVPR 2020): Generates half channels via cheap depthwise conv. Modified: `groups = min(init, cheap)` for divisibility safety.

**CoordAtt** (CVPR 2021): Coordinate Attention — decomposes global pooling into H and W 1D operations. Preserves spatial position info. Used inside GhostBottle.

**MultiStripAttn** (inspired by Strip Pooling, CVPR 2020 / LSKA): Strip attention for elongated objects. Four asymmetric DW convs: (1×7), (7×1), (1×15), (15×1). Output: `x * sigmoid(h7 + w7 + h15 + w15)`.

**SEBlock** (CVPR 2018): Standard squeeze-and-excitation. Standalone; no longer used inside the backbone.

**CBAM** (ECCV 2018): `ChannelGate` (avg+max pool through a shared MLP) + `SpatialGate` (channel avg+max → 7×7 conv + BN → sigmoid). Matches the official implementation.

**ChannelShuffleFusion** (from ShuffleNet, CVPR 2018): Parameter-free channel shuffle. Used inside GhostBottle.

**RotationInvariantFusion (RIF)** (inspired by TI-Pooling): Creates 4 rotated copies (0°, 90°, 180°, 270°), fuses with learnable per-channel weights `alpha [4, C, 1, 1]` initialized to 1/4. Placed between stage3 and stage4 of backbone.

**GhostBottle** (custom composite): GhostModule(expand) → DW conv(stride) → BN → GhostModule(project, no ReLU) → CoordAtt → ChannelShuffle → + residual.

**GhostTriRemoteXProPP** (the backbone):
- Stem: GhostModule 3→48ch, stride 2
- Stage 1: 3× GhostBottle, 48→64ch, stride 2
- Stage 2: 4× GhostBottle, 64→128ch, stride 2 → **C3** (stride 8)
- Stage 3: 4× GhostBottle, 128→192ch, stride 2
- RIF: RotationInvariantFusion at 192ch → **C4** (stride 16)
- Stage 4: 2× GhostBottle, 192→256ch, stride 2
- Attention stack: MultiStripAttn → CBAM (at 256ch) → **C5** (stride 32)
- `forward_features()` returns `[C3, C4, C5]` for a real multi-scale FPN
- Also has a classification head (pool over C5 → dropout → FC) for pretraining

### 2.2 What EXISTS but is BROKEN/INCOMPLETE:

**FPN**: Standard top-down FPN operating on the 3 backbone levels (C3/C4/C5) as a real pyramid.

**RotatedDetectionHead**: Three-branch dense head:
- Classification subnet: 2× (conv3×3 + ReLU) → conv3×3 → A×15 classes
- Box regression subnet: 1× (conv3×3 + ReLU) → conv3×3 → A×5 (cx,cy,w,h,θ)
- Objectness branch: conv3×3 → A×1
This is a basic RetinaNet + YOLO hybrid design. Will need redesign later.

**RemoteDetector**: Simple wrapper: backbone.forward_features() → neck(feats) → head(feats)

**Rotated Anchor Generation**: Per-level anchors — one scale per FPN level × 3 ratios × 3 angles = 9 per location. Defined as (cx, cy, w, h, angle_in_radians).

**Rotated IoU Computation**: Uses Sutherland-Hodgman polygon clipping in a PYTHON DOUBLE LOOP. **THIS IS THE CRITICAL BOTTLENECK** — it caused a KeyboardInterrupt during training. O(N×M) complexity with thousands of anchors per image. **MUST be replaced** with either:
- mmrotate's CUDA-accelerated rotated IoU
- Gaussian approximation (GWD/KLD style) which is differentiable and GPU-native

**DetectionLoss**:
- Classification: Focal Loss (α=0.25, γ=2.0)
- Box regression: Smooth L1 on positive anchors only
- Objectness: Binary cross-entropy
- Anchor assignment: simple IoU > 0.5 threshold
- All three losses summed with equal weight (no balancing)

**Training loop**: Uses AMP (autocast + GradScaler), AdamW optimizer, CosineAnnealing LR scheduler. Multi-GPU DataParallel support. **Has never completed a full epoch due to the IoU bottleneck.**

### 2.3 What EXISTS but is SEPARATE (not part of the custom architecture):

The notebook also contains a completely independent YOLO comparison section that fine-tunes pretrained YOLO models (v5, v8, v9, v10, v11) using Ultralytics API on DOTA for benchmarking. This converts DOTA annotations to YOLO format (axis-aligned HBBs). **This is separate from the custom architecture and is just for comparison.**

### 2.4 Duplicate code:

The notebook has duplicated cells — the same architecture is defined twice with slight variations (one version has `forward_features` return `x`, another returns `[x]`). There are also debug print statements throughout. The code needs cleanup.

---

## 3. What the User Has Learned (Conceptual Understanding)

The user has covered these topics in depth through our conversation and separate study:

### Already understood well:
- OBB representation: 5-param (cx,cy,w,h,θ), le90 convention, COBB landscape, why standard 5-param is fine for initial work
- Anchor encoding/decoding: delta prediction vs absolute regression
- Anchor-based vs anchor-free prediction paradigms
- Classification losses: BCE, Focal Loss, QFL (Quality Focal Loss)
- Regression losses: Smooth L1, GWD (Gaussian Wasserstein Distance), KLD (KL Divergence)
- Objectness branch: why it exists in YOLO-style, why QFL removes the need for it
- Label assignment strategies: naive IoU threshold, Faster R-CNN ignore zone, RetinaNet per-GT guarantee, ATSS (adaptive threshold via mean+std), TAL (Task-Aligned Learning from TOOD)
- 8-point prediction and why the field moved away from it
- Strip R-CNN paper: StripNet backbone (sequential orthogonal strip convolutions), strip head (decoupled localization with strip module), vanilla FPN neck
- LMW-YOLO paper: CSD strategy (different modules at different pyramid levels), LKCA module (decomposed LKA for P3), MSDP module (dilated residual for P4), WIoU v3 loss

### Neck design understanding:
- FPN: top-down only, equal weight fusion, 1×1 lateral + upsample + add + 3×3 smooth
- PANet: FPN + bottom-up path (what YOLO v4/v5 use)
- BiFPN: learnable fusion weights + cross-scale skip connections + repeated stacking (from EfficientDet)
- GFPN/RepGFPN: dense queen-fusion topology + reparameterization (from DAMO-YOLO)
- Gold-YOLO: gather-and-distribute mechanism — aggregate all scales globally then redistribute (NeurIPS 2024)
- "Large neck, small head" paradigm: DAMO-YOLO showed putting computation budget in neck > backbone or head
- Rotation-aware fusion: FAA-Fusion aligns feature orientations across scales before fusion (for two-stage)
- Key distinction understood: scale-aware (different object sizes) vs aspect-ratio-aware (elongated vs square shapes) vs rotation-aware (arbitrary orientations) — these are three orthogonal properties

### Head design: NOT YET COVERED
- The user knows about TOOD (T-Head + TAL) and GFL (QFL + DFL) conceptually from the loss/assignment discussions
- But we have NOT discussed head architecture design in detail yet
- This is the next conceptual topic to cover

---

## 4. Key Design Decisions Made

### 4.1 Detection paradigm: One-stage (FIRM)
Not two-stage like Oriented R-CNN. This means dense prediction, no RPN/RoI pooling.

### 4.2 Baseline neck for benchmarking: Standard FPN
To isolate backbone contribution, use vanilla FPN (same as Strip R-CNN's neck). Any performance differences vs Strip R-CNN can then be attributed to backbone and head choices.

### 4.3 Research methodology principle agreed upon:
"If a component isn't your contribution, use the field's standard baseline for it." Don't use a fancy existing neck (BiFPN, Gold-YOLO) unless you're innovating on it — it muddies the ablation story.

### 4.4 Multi-scale backbone output: YES
The backbone will be modified to output features from stages 2, 3, 4 (approximately 128×128, 64×64, 32×32 for 1024 input). This is a prerequisite for any neck.

### 4.5 Parameter budget: Lightweight but flexible
Try to stay lightweight overall, but the neck is where parameter investment gives best returns (per DAMO-YOLO finding).

### 4.6 Novelty target: Publishable
The user is targeting publication at a decent conference. The backbone is the first novel contribution. Neck and/or head novelty is planned but not yet designed.

### 4.7 Gap statement identified for potential neck novelty:
"No existing lightweight one-stage neck design is simultaneously rotation-aware, scale-aware, AND aspect-ratio-aware for OBB detection in remote sensing."

Three potential directions were discussed:
- **Direction A (recommended)**: Strip-enhanced fusion nodes — bring strip convolutions into the neck's fusion/smoothing operations. Extends Strip R-CNN's insight to the neck (they only put strip convs in backbone + head). Safest, most ablation-friendly.
- **Direction B**: Scale-decoupled processing (CSD-style but for OBB) — different specialized modules at different pyramid levels.
- **Direction C**: Lightweight gather-and-distribute with orientation gating — adapt Gold-YOLO for rotation-aware RS detection.

**No final decision made yet on which direction to pursue.** The user will first get baseline results, then decide based on diagnosis of where the model fails.

### 4.8 Pipeline narrative (if all three components get novelty):
- **Backbone**: Rotation-**invariant** features (RIF module)
- **Neck**: Rotation-**coherent** multi-scale fusion (preserves orientation during scale fusion)
- **Head**: Rotation-**sensitive** predictions (predicts OBB parameters)

---

## 5. Immediate Next Steps (Week 1 Plan)

Priority-ordered:

### 5.1 CRITICAL: Fix rotated IoU bottleneck
The Python double-loop polygon clipping killed training. Options:
- Use `mmrotate` / `mmcv.ops` CUDA rotated IoU (preferred — most mature)
- Replace with Gaussian approximation (GWD/KLD) for loss computation
- This is engineering, not research — just needs to work

### 5.2 Modify backbone for multi-scale output — DONE
`forward_features()` returns `[C3, C4, C5]` (stages 2 / post-RIF stage 3 / stage 4).

### 5.3 Wire up standard FPN neck
Connect the multi-scale backbone outputs to the existing FPN implementation. Ensure channel projection works (stage2=128ch, stage3=192ch, stage4=256ch → all projected to 128ch by FPN laterals).

### 5.4 Wire up a standard detection head
The existing RotatedDetectionHead should work initially. Later it needs redesign, but for baseline benchmarking it's fine.

### 5.5 Fix label assignment
Current IoU > 0.5 threshold is too simplistic. Implement ATSS (adaptive threshold via mean+std of IoU) as the baseline — it's what most one-stage detectors use and is straightforward to implement.

### 5.6 Get training running
Even if results are bad, getting a full training epoch to complete is the milestone. Then iterate.

---

## 6. Broader 6-Week Plan

- **Week 1**: Get pipeline training (items 5.1-5.6 above)
- **Week 2-3**: Train properly on DOTA, get baseline mAP numbers, compare with LSKNet-S / PKINet-S, diagnose failure modes (which classes/aspect ratios/object sizes fail?)
- **Week 4-5**: Design and implement neck/head novelty based on diagnosis
- **Week 6**: Clean results, start ablation studies

---

## 7. Paper Reading List

A comprehensive reading list was generated and saved as `paper_reading_list.md`. Key papers by priority:

**Must-read for implementation**:
1. COBB (CVPR 2024) — continuous OBB representation, solves boundary discontinuity
2. GWD (ICML 2021) — Gaussian Wasserstein distance loss for OBB regression
3. GFL/DFL (NeurIPS 2020) — distribution-based box regression, used in YOLOv8
4. TOOD (ICCV 2021) — task-aligned head design + TAL assignment
5. ATSS (CVPR 2020) — adaptive training sample selection

**Must-read for context/comparison**:
6. Strip R-CNN (arXiv 2025) — primary performance benchmark
7. LMW-YOLO (Sci. Rep. 2026) — lightweight RS detection benchmark
8. Oriented R-CNN (ICCV 2021) — standard two-stage OBB baseline

**For future neck/head design**:
9. FAA (arXiv 2026) — rotation-aware neck fusion + head alignment
10. STD (AAAI 2024) — spatial transform decoupling for oriented head
11. Gold-YOLO (NeurIPS 2024) — gather-and-distribute neck mechanism
12. DAMO-YOLO (arXiv 2022) — RepGFPN neck + "large neck small head" finding

---

## 8. Important Technical Notes

### 8.1 The notebook has duplicated code
Multiple cells define the same classes with slight variations. When modifying, be careful about which version is being used. A code cleanup pass would be valuable.

### 8.2 The YOLO comparison section is independent
The last large cell (DOTADatasetProcessor + YOLOModelTrainer) is a completely separate pipeline using Ultralytics API. Don't modify it — it's for comparison benchmarking only.

### 8.3 Dataset paths are Windows-style
The code uses Windows paths (`D:/Abhi/dota_dataset`). These will need updating for whatever environment Claude Code runs in.

### 8.4 Backbone pretrained weights
The code references `best_GBR_model.pth` for backbone pretrained weights (from classification pretraining). The user has these weights available.

### 8.5 The backbone's width_mult parameter
The backbone uses `width_mult=1.0` with `make_divisible(v, divisor=8)` to ensure clean channel counts. Channel progression: 48 → 64 → 128 → 192 → 256.

---

## 9. Head Design (TO BE DISCUSSED)

This topic has NOT been covered yet in our conversation. When the user is ready, key topics to cover:

- Coupled vs decoupled heads (Strip R-CNN decouples cls+angle from localization)
- TOOD's T-Head (task-interactive features with task-aligned predictors)
- GFL's approach (merge quality estimation into classification, DFL for box regression)
- For OBB specifically: how to handle the angle prediction (separate branch? shared with cls? shared with reg?)
- Strip R-CNN's finding: classification sensitivity is concentrated at object centers, angle sensitivity at borders — they share cls+angle layers based on this
- STD's approach: decouple position prediction from rotation prediction entirely

---

## 10. User Preferences

- Prefers to be asked clarifying questions before proceeding with any task
- Making detailed notes — explanations should be clear and structured
- Research-oriented mindset — cares about clean ablation design and publishable methodology
- Currently learning object detection (started from YOLO knowledge, this is their first exposure to OBB/RS-specific detection)
- Has access to 2× NVIDIA RTX A6000 GPUs
