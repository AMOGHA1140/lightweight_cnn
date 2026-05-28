# Research & Design Reference — Lightweight OBB Detection for Remote Sensing

**Author**: C Abhineeth (IIT BHU Varanasi, intern at Changwon National University)
**Created**: May 27, 2026
**Purpose**: Exhaustive reference capturing all research discussions, design decisions,
gap analyses, and the proposed GAConv innovation. Written so that anyone (including
future-you) can pick this up months later and understand every decision.

---

## Table of Contents

1. [Project Context](#1-project-context)
2. [Backbone Architecture](#2-backbone)
3. [Detection Pipeline Configuration](#3-pipeline-config)
4. [The Multi-Scale Problem and Neck Design](#4-neck-landscape)
5. [Remote-Sensing-Specific Neck and Head Research](#5-rs-specific-research)
6. [The Unified Mathematical Insight](#6-mathematical-insight)
7. [GAConv: Geometric Adaptive Convolution](#7-gaconv)
8. [Alternative Directions (Deprioritized)](#8-alternatives)
9. [Related Work Comparison Matrix](#9-related-work)
10. [Experimental Plan](#10-experimental-plan)
11. [Papers to Read](#11-papers)
12. [Open Questions](#12-open-questions)

---

## 1. Project Context

**Goal**: Build a lightweight oriented bounding box (OBB) object detector for
remote sensing imagery. Target datasets: DOTA v1.0, DOTA v2.0, HRSC2016, DIOR-R.

**Constraint**: Lightweight — the backbone is ~1.4M params, total model target <5M.
Must be one-stage (not two-stage like Oriented R-CNN).

**Reference models**:
- Strip R-CNN (AAAI 2025): SOTA on DOTA, 82.75% mAP, ~30M params. Two-stage.
  Uses strip convolutions in backbone (StripNet) and head. Standard FPN neck.
- LO-Det (IEEE TGRS 2022): Only lightweight OBB detector (~4.5M params).
  MobileNetV2 backbone, CSA neck, DSC-Head. No orientation awareness in neck.
- LMW-YOLO (Sci. Reports 2026): Lightweight RS detector, HBB only.

**Compute**: 2x NVIDIA RTX A6000.

**Prior work on this project** (HBB only, for context):
- 39.81% mAP50 on DOTA v1.0 with 5.44M params (custom backbone + YOLO neck/head)
- Bridge: 2.92% mAP, Roundabout: 6.95% — worst classes, both geometric
- Original concept planned: Novel Backbone -> Neck -> Decoupled Head

---

## 2. Backbone Architecture

### 2.1 Architecture (GhostTriRemoteXProPP)

Channel progression (width_mult=1.0): 48 -> 64 -> 128 -> 192 -> 256.

| Stage | Op | Out Shape (1024 input) | Stride |
|---|---|---|---|
| stem | GhostModule k3 s2 | [B, 48, 512, 512] | 2 |
| stage1 | 3x GhostBottle | [B, 64, 256, 256] | 4 |
| stage2 | 4x GhostBottle | [B, 128, 128, 128] | 8 |
| stage3 | 4x GhostBottle | [B, 192, 64, 64] | 16 |
| rif | RotationInvariantFusion | [B, 192, 64, 64] | 16 |
| stage4 | 2x GhostBottle | [B, 256, 32, 32] | 32 |
| strip | MultiStripAttn | [B, 256, 32, 32] | 32 |
| cbam | CBAM | [B, 256, 32, 32] | 32 |

`forward_features()` returns [C3, C4, C5] = [stage2, stage3+rif, stage4+strip+cbam]
at strides [8, 16, 32] with channels [128, 192, 256].

### 2.2 Building Blocks

| Block | Source Paper | Novelty? |
|---|---|---|
| GhostModule | GhostNet (CVPR 2020) | NOT novel. From paper. |
| CoordAtt | Coordinate Attention (CVPR 2021) | NOT novel. From paper. |
| SEBlock | SE-Net (CVPR 2018) | NOT novel. From paper. |
| CBAM | CBAM (ECCV 2018) | NOT novel. From paper. |
| ChannelShuffleFusion | ShuffleNet (CVPR 2018) | NOT novel. From paper. |
| GhostBottle | GhostNet's GhostBottleneck + CoordAtt + Shuffle | Hybrid: base from paper, additions custom |
| MultiStripAttn | Inspired by Strip Pooling / LSKA | NOT novel. Prior: LSKA (2023), MSA-YOLO MSCAM (2023) |
| RotationInvariantFusion | Custom | NOT novel. Prior: TI-Pooling (CVPR 2016) with learned weights |

### 2.3 Note on Implementation Fidelity

Three modules (GhostModule, CoordAtt, CBAM) had deviations from their source
papers in the original codebase. These have been corrected to match the official
implementations (houqb/CoordAttention, Jongchan/attention-module). The standalone
SEBlock before CBAM was removed (redundant double channel recalibration).
Any future reference to these modules assumes the corrected versions.

### 2.4 Backbone Pretraining

Dataset: Tiny-ImageNet-200 (200 classes, matches backbone's num_classes=200).
Recipe: AdamW, cosine LR, label smoothing 0.1, 100 epochs, input 224x224.
Expected accuracy: ~55-65% top-1 (comparable to MobileNetV2 at similar scale).

---

## 3. Detection Pipeline Configuration

One-stage anchor-based OBB detector. Key design choices:

- **Neck**: Standard FPN, in_channels=[128,192,256], out=128. Top-down only
  (PANet bottom-up path to be added).
- **Head**: Separate cls/reg/obj subnets. Cls: 2x(3x3+ReLU)+3x3. Reg: 1x(3x3+ReLU)+3x3.
- **Anchors**: 9/location (1 scale per FPN level x 3 ratios x 3 angles).
  Per-level scales [32], [64], [128]. Ratios [0.5, 1, 2]. Angles [-60, 0, 60] deg.
- **Loss**: Focal (cls) + SmoothL1 with delta encoding (reg) + BCE (obj).
- **Assignment**: IoU > 0.5 threshold (ATSS planned as upgrade).
- **Training**: AdamW, cosine LR, AMP, grad clip 10.

For initial GAConv testing, a standard backbone (ResNet-50) with mmrotate may be
used to isolate the GAConv contribution from custom backbone effects.

---

## 4. The Multi-Scale Problem and Neck Design

### 4.1 Why Necks Exist

Backbone features have a tension: shallow stages (C3) have high spatial resolution
but weak semantics; deep stages (C5) have strong semantics but lose spatial detail.
The neck's job: create features that are BOTH high-res AND semantically rich at
every scale, by letting information flow between scales.

### 4.2 Standard Neck Architectures

**FPN** (Lin et al., CVPR 2017): Top-down pathway. 1x1 laterals for channel
alignment, nearest-neighbor upsample + element-wise add, 3x3 smooth conv.
Single-direction: deep->shallow. Equal-weight fusion. ~500K params for our config.

**PANet** (Liu et al., CVPR 2018): FPN + bottom-up pathway. Bidirectional
information flow. Used in YOLOv4/5/8. ~1.1M params.

**BiFPN** (Tan et al., CVPR 2020, EfficientDet): Learnable fusion weights +
cross-scale connections + repeated stacking (3-7x). More powerful but heavier.

**GFPN/RepGFPN** (DAMO-YOLO, 2022): Dense "queen-fusion" — every level connects to
every other level. Reparameterization for inference efficiency. Finding: "large
neck, small head" gives best results.

**Gold-YOLO** (NeurIPS 2024): Gather-and-distribute — aggregate all scales into one
global representation, then redistribute. Single-step global context.

### 4.3 What ALL Standard Necks Get Wrong

| Problem | FPN | PANet | BiFPN | GFPN | Gold-YOLO |
|---|---|---|---|---|---|
| One-directional flow | X | OK | OK | OK | OK |
| Equal-weight fusion | X | X | OK | X | X |
| Only-neighbor interaction | X | X | Partial | OK | OK |
| **Spatially uniform** | **X** | **X** | **X** | **X** | **X** |
| **Direction-agnostic** | **X** | **X** | **X** | **X** | **X** |

The last two rows are ALL failures. No published neck addresses spatial adaptivity
or directional awareness. This is the gap.

---

## 5. Remote-Sensing-Specific Neck and Head Research

### 5.1 Orientation-Aware Necks (all heavy, 30M+ frameworks)

**FAA-Fusion** (arXiv 2602.23790, Feb 2026, BIT/HKU):
- Uses 2D FFT to estimate dominant orientation from low-level features
- Rotates high-level features to match before FPN fusion
- Addresses "directional incoherence" in the neck
- Built on Oriented R-CNN + ResNet-50. NOT lightweight.
- Code: github.com/gcy0423/Fourier-Angle-Alignment
- KEY INSIGHT: orientation signals conflict during cross-scale fusion

**RSFPN** (Sensors, Jan 2026):
- Angle-Aware Collaborative Attention (AACA) in bidirectional FPN
- Uses orientation priors to guide feature refinement
- Built on Oriented R-CNN. NOT lightweight.

**PAMFPN RSA** (Remote Sensing, June 2025):
- Region-Sensitive Attention: vertical + horizontal strip convolutions
  as MULTIPLICATIVE ATTENTION in the neck
- Strips generate attention weights, not used as fusion operators
- NOT lightweight.
- IMPORTANT: This uses strips in the neck (our earlier DAS idea overlaps)

**SFRADNet** (Remote Sensing, May 2025):
- Couples scale and orientation: match scale first, then refine angle
- Scale Selection Matrix dynamically adjusts receptive field coverage
- One-stage, but not lightweight

### 5.2 Orientation-Aware Heads

**S2A-Net** (TGARS, 2021):
- AlignConv: samples features along anchor's predicted orientation
- Active Rotating Filters (ARF): produce orientation-sensitive features,
  max-pool across orientations for rotation-invariant features
- KEY INSIGHT: classification needs rotation-invariant features;
  regression needs rotation-sensitive features
- Single-stage but ResNet-50 based. Not lightweight.
- Code: github.com/csuhan/s2anet

**Strip R-CNN Head** (AAAI 2025):
- Strip convolutions specifically in the localization subnet (not cls)
- Shares cls + angle prediction layers
- Finding: cls sensitivity at object centers, angle sensitivity at borders
- Two-stage. Not lightweight.

**STD** (AAAI 2024):
- Spatial Transform Decoupling: predicts position, size, angle in cascaded stages
- Uses activation masks to guide attention at each stage
- Transformer-based, heavy

**ARS-DETR** (TGRS 2024):
- Aspect-Ratio-aware Circle Smooth Label: adjusts angle label smoothing
  based on object aspect ratio (slender objects need sharper angle)
- Rotated deformable attention: samples along predicted angles
- Transformer-based

### 5.3 Rotation-Sensitive Convolutions (backbone-level)

**GRA** (ECCV 2024): Group-wise Rotating and Attention. Divides conv kernel into
groups, rotates each at different angle, uses group attention to weight them.
Claims lightweight. Backbone module only.

**ARC** (ICCV 2023): Adaptive Rotated Convolution. Rotates entire conv kernels
based on input image content. One angle per IMAGE, not per location.

**ReDet** (CVPR 2021): Rotation-equivariant backbone (ReResNet). Formally
equivariant but computationally expensive.

### 5.4 The Only Lightweight OBB Detector

**LO-Det** (IEEE TGRS 2022, ~4.5M params):
- MobileNetV2 backbone
- CSA neck (Channel Separation-Aggregation): more efficient than stacked
  depthwise convs, but NOT orientation-aware
- DRF (Dynamic Receptive Field): adapts kernel size, but NOT direction
- DSC-Head: variant of gliding vertex for OBB prediction
- This is our direct comparison paper

### 5.5 Task-Decoupled Necks (general detection, not RS)

**WPDFPN** (Neurocomputing, May 2024):
- Two parallel FPN paths (top-down + bottom-up)
- Feature Decoupling Module produces separate features for cls and loc
- KEY FINDING: neck-level decoupling > head-level decoupling
- On COCO (HBB only). No rotation awareness.

---

## 6. The Unified Mathematical Insight

### 6.1 The Problem Restated

Remote sensing OBB detection requires handling three geometric properties
simultaneously:
1. **Rotation**: Objects at arbitrary angles (0-360 deg)
2. **Scale**: Objects from ~10px (small vehicle) to ~500px (bridge)
3. **Aspect ratio**: From 1:1 (storage tank) to 1:20+ (bridge)

Standard convolutions (including in FPN) use fixed, isotropic, axis-aligned kernels
that treat all three identically regardless of local content.

### 6.2 The Insight: One Matrix Captures All Three

Rotation, scale, and aspect ratio are NOT three independent problems. They are
three aspects of a SINGLE mathematical object: a 2x2 positive-definite matrix.

```
    Sigma = R(theta) * diag(sigma_major, sigma_minor) * R(theta)^T
```

where:
- R(theta) encodes rotation (eigenvectors)
- sigma_major / sigma_minor encodes aspect ratio (eigenvalue ratio)
- sqrt(sigma_major * sigma_minor) encodes scale (geometric mean of eigenvalues)

This is:
- A **2D Gaussian covariance matrix** (how GWD/KLD model OBBs)
- A **structure tensor** from classical image processing
- The **affine group** GL+(2,R) acting on 2D features

Three numbers (theta, sigma_major, sigma_minor) fully describe the local geometry
at any point in a feature map. This is the mathematical foundation for GAConv.

### 6.3 Connection to Existing Representations

The OBB representation (cx, cy, w, h, theta) already encodes this:
- (cx, cy) = position
- (w, h, theta) = the 2D Gaussian covariance parameters

GWD loss converts OBBs to 2D Gaussians. KLD computes divergence between them.
The detection field already uses this representation for BOXES. Nobody uses it
for FEATURE PROCESSING in the neck.

---

## 7. GAConv: Geometric Adaptive Convolution

### 7.1 Core Idea

GAConv is a **general-purpose module** — a drop-in replacement for any standard
convolution — that adapts its sampling pattern to the local geometry. It can be
applied in the **backbone**, **neck**, AND **head**, and the paper's contribution is
showing where it helps most (and that all three properties are needed together).

The module:
1. Predicts local geometry (3 params: theta, sigma_major, sigma_minor) at each
   spatial location from the feature map itself
2. Constructs a 2x2 affine transformation matrix from these params
3. Uses this matrix to transform the standard 3x3 sampling grid into an
   oriented, scaled, anisotropic sampling pattern
4. Applies depthwise deformable convolution with these geometric offsets

**Where it can be placed:**
- **Backbone**: Replace standard 3x3 convs in GhostBottle's depthwise layer or
  in the shortcut path. Gives the backbone orientation/aspect-ratio sensitivity
  at feature extraction time.
- **Neck**: Replace FPN's 3x3 smooth conv. Makes multi-scale fusion
  geometry-aware (the original motivation).
- **Head**: Replace 3x3 convs in the cls/reg subnets. Makes the prediction
  layers adapt to object geometry at each spatial location.

### 7.2 Mechanism Detail

**Step 1: Predict local geometry**
A 1x1 conv takes the C-channel fused feature and outputs 3 channels:
(theta, sigma_major, sigma_minor) at each spatial location.
- theta: constrained to [-pi, pi] via tanh * pi
- sigma_major: constrained to [0.5, K] via sigmoid * (K-0.5) + 0.5
- sigma_minor: constrained to [0.5, sigma_major] similarly
Cost: C * 3 parameters. For C=128: 384 params.

**Step 2: Construct affine matrix**
```
A = R(theta) * diag(sigma_major, sigma_minor)
  = [[cos(theta)*sigma_major, -sin(theta)*sigma_minor],
     [sin(theta)*sigma_major,  cos(theta)*sigma_minor]]
```
Pure math, no parameters. Differentiable w.r.t. theta, sigma_major, sigma_minor.

**Step 3: Transform sampling grid**
Base 3x3 grid: [(-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)]
For each location, for each grid point p_i:
```
offset_i = A * p_i - p_i
```
This gives 18 offset values (9 points x 2 coords) from 3 params.
Pure math, differentiable.

**Step 4: Apply deformable depthwise convolution**
Use the geometric offsets to sample features via bilinear interpolation.
Apply learned depthwise conv weights to the sampled features.
Follow with a 1x1 pointwise conv for channel mixing.

### 7.3 What Each Property Controls

For a horizontal ship (theta=0, sigma_major=3, sigma_minor=0.7):
- Sampling grid stretches horizontally (along ship's length)
- Narrow vertically (ship is thin)
- Kernel "sees" the whole ship despite being only 3x3

For a diagonal bridge (theta=45deg, sigma_major=5, sigma_minor=0.5):
- Sampling grid rotates 45 degrees
- Extreme stretch along bridge axis
- Kernel follows the bridge's direction

For a storage tank (theta=any, sigma_major~=sigma_minor~=1):
- Nearly isotropic sampling (like standard 3x3)
- Rotation doesn't matter (circular object)

### 7.4 Parameter Cost

Per GAConv module:
- Geometry predictor: 1x1 conv (C -> 3) = 384 params
- Deformable DW conv: 3x3 depthwise = 1,152 params
- Pointwise projection: 1x1 conv (C -> C) = 16,384 params
- Total per level: ~18K params

3 FPN levels: ~54K params
Standard 3x3 smooth (3 levels): ~442K params
GAConv uses ~8x fewer parameters.

### 7.5 Novelty Assessment

**What IS novel:**
- GAConv as a **general-purpose geometric-adaptive convolution** that handles
  rotation + scale + aspect ratio in a SINGLE operation via covariance-matrix
  parameterization (3 params -> affine -> structured offsets)
- The mathematical framing: treating all three geometric properties as one
  2x2 positive-definite matrix (structure tensor / Gaussian covariance)
- Systematic study of WHERE it helps: backbone vs neck vs head vs combinations
- Application to RS OBB detection (nobody has used geometric-adaptive conv
  for this task in this way)
- The geometric constraint itself: 3 structured params vs DCN's 18 free params,
  giving better regularization with fewer parameters

**What is NOT novel (must cite):**
- Deformable convolution (Dai et al., ICCV 2017) — the underlying sampling mechanism
- Adaptive kernel geometry (Cam & Tek, 2018) — Gaussian envelope with learnable
  covariance, but global per-layer, not per-location, and in backbone for classification
- ARConv (CVPR 2025) — learns kernel h,w + affine for pansharpening, not detection.
  Learns rectangular (h,w) not oriented-elliptical (theta, sigma_major, sigma_minor).
  Different parameterization, different task, different placement.
- AlignConv (S2A-Net, 2021) — anchor-predicted angle -> oriented sampling, but only
  in head, depends on anchors, only rotation (no scale/aspect)
- ARC (ICCV 2023) — rotates conv kernels, but one angle per image, not per-location
- Structure tensor (classical image processing) — the mathematical foundation
- DMAC (Brain Informatics, 2026) — oriented tubular kernels for 3D neuronal fiber
  detection. Similar geometric principle but 3D, different domain entirely.

**The gap we fill:**
Nobody has proposed a per-location, geometrically-constrained convolution that
unifies rotation + scale + aspect ratio for RS OBB detection. The closest works
each handle only a subset: ARC handles rotation only, ARConv handles scale + AR
(without explicit rotation), AlignConv handles rotation only. GAConv handles all
three in a unified covariance-matrix framework, and we systematically ablate its
placement across backbone, neck, and head.

### 7.6 Ablation Table

**Table A: Component ablation (what does GAConv need?)**

| Exp | Config | What It Tests |
|---|---|---|
| A1 | Standard model (all regular convs) | Baseline |
| A2 | Deformable conv (free 18 offsets) in neck | DCN helps but unstructured |
| A3 | GAConv theta only (fixed sigma=1) | Rotation alone insufficient |
| A4 | GAConv sigma only (theta=0) | Scale/aspect alone insufficient |
| A5 | Full GAConv (theta + sigma_major + sigma_minor) | All three needed together |

Experiments A3 and A4 are critical: they prove the paper's thesis that all three
geometric properties must be handled simultaneously.

**Table B: Placement ablation (where does GAConv help?)**

| Exp | Backbone | Neck | Head | What It Tests |
|---|---|---|---|---|
| B1 | Standard | Standard | Standard | Baseline |
| B2 | Standard | GAConv | Standard | Neck-only (original hypothesis) |
| B3 | Standard | Standard | GAConv | Head-only |
| B4 | GAConv | Standard | Standard | Backbone-only |
| B5 | Standard | GAConv | GAConv | Neck + Head |
| B6 | GAConv | GAConv | Standard | Backbone + Neck |
| B7 | GAConv | GAConv | GAConv | All three |

This table shows which placement matters most. If neck-only gives most of the
gain, the module's value is in geometry-aware FUSION. If head-only dominates,
the value is in geometry-aware PREDICTION. If backbone-only, it's in geometry-
aware EXTRACTION. Combinations may show complementary benefits.

**Table C: Comparison with existing methods**

| Method | Params | mAP50 (DOTA v1.0) | Notes |
|---|---|---|---|
| LO-Det | ~4.5M | TBD | Lightweight OBB baseline |
| YOLO-OBB-S | ~11M | TBD | Off-the-shelf comparison |
| Ours (standard FPN) | ~3M | TBD | Our backbone + standard neck/head |
| Ours (GAConv best config) | ~3M | TBD | Our backbone + GAConv |
| Strip R-CNN-S | ~30M | 82.75 | SOTA reference (not fair comparison) |

**Table D: Per-class analysis**

Focus classes: bridge (elongated, AR>10), ship (elongated, AR>3),
harbor (complex geometry), small-vehicle (tiny, dense).
These classes should show the largest improvement from GAConv.

### 7.7 The Paper Story

"Standard convolutions in object detection pipelines use fixed, isotropic, axis-
aligned sampling grids that cannot adapt to the diverse geometric properties of
remote sensing objects — varying orientations, extreme aspect ratios, and large
scale differences. We observe that these three properties are mathematically
unified by a single 2D covariance matrix (structure tensor). We propose GAConv
(Geometric Adaptive Convolution), a general-purpose module that predicts this
matrix per-location and uses it to drive geometrically-adaptive feature
processing. GAConv simultaneously handles rotation, scale, and aspect ratio in
a single lightweight operation, using only 3 geometric parameters instead of
18 free offsets (as in deformable convolution). We systematically study GAConv's
impact across backbone, neck, and head, finding that [TBD: which placement helps
most]. On DOTA v1.0/v2.0 and HRSC2016, our method achieves [TBD] mAP, with
particularly strong improvements on high-aspect-ratio classes (bridge, ship)
where standard convolutions fail."

---

## 8. Alternative Directions (Deprioritized)

### 8.1 Direction B: DAS (Directional Adaptive Smooth)

Simpler version: replace 3x3 smooth with parallel strip convolutions (1xK) + (K x1)
with CoordAtt-style adaptive gating. No rotation — only H/V direction adaptation.

Pros: Simple, easy to implement, clearly distinct from ARConv.
Cons: No rotation handling. Only two fixed directions (H/V).
Status: Deprioritized in favor of GAConv which handles rotation too.

### 8.2 Direction C: Lightweight FAA (cheap orientation alignment)

Use CoordAtt's H/W energy ratio to estimate coarse orientation (0/45/90/135),
then lightweight grid warp to align features before fusion.

Pros: Nearly zero-cost orientation estimation.
Cons: Very coarse (only 4 angles). Hard to argue vs FAA-Fusion which works well.
Status: Deprioritized.

### 8.3 Direction D: Rotation-Decoupled Neck

Produce two feature outputs per FPN level: rotation-invariant (for cls) and
rotation-sensitive (for reg). Combines WPDFPN insight (neck decoupling helps)
with S2A-Net insight (rotation-inv/sens split matters).

Pros: Strong paper narrative, clean ablation.
Cons: More complex, harder to implement in 45 days.
Status: Could be Phase 2 extension after GAConv baseline.

### 8.4 Direction E: Aspect-Ratio-Adaptive Neck

Different fusion operations for different local aspect ratios. Route features
through isotropic path (3x3) or elongated path (strips) based on estimated AR.

Pros: Addresses the bridge/ship problem directly.
Cons: Narrow scope, conditional computation is harder.
Status: Subsumed by GAConv (which handles AR via sigma_major/sigma_minor).

---

## 9. Related Work Comparison Matrix

### 9.1 Geometric-Adaptive Convolutions

| Method | What It Predicts | Per-Location? | Handles Rotation? | Handles Scale? | Handles AR? | Where Applied | Task |
|---|---|---|---|---|---|---|---|
| Standard Conv | Nothing | No | No | No | No | Everywhere | Any |
| DCN v1/v2 | 18 free offsets | Yes | Implicitly | Implicitly | Implicitly | Backbone/head | Detection |
| ARC (2023) | 1 angle per image | No (global) | Yes | No | No | Backbone | RS Detection |
| GRA (2024) | Group angles | Per-group | Yes (discrete) | No | No | Backbone | RS Detection |
| AlignConv (2021) | Angle from anchors | Yes (anchor) | Yes | No | No | Head | RS Detection |
| ARConv (2025) | h, w + affine | Yes | Via affine? | Yes | Yes | Network | Pansharpening |
| Cam & Tek (2018) | Covariance matrix | No (per-layer) | Yes | Yes | Yes | Backbone | Classification |
| DMAC (2026) | 2 orientation angles | Yes | Yes | No | Yes (tubular) | Network | Neuro imaging |
| **GAConv (ours)** | **theta, sigma_maj, sigma_min** | **Yes** | **Yes** | **Yes** | **Yes** | **FPN neck** | **RS OBB Detection** |

### 9.2 Orientation-Aware FPN Necks

| Method | Mechanism | Lightweight? | One-Stage? | OBB? | Year |
|---|---|---|---|---|---|
| Standard FPN | 1x1 lateral + upsample + 3x3 smooth | Yes | Either | N/A | 2017 |
| FAA-Fusion | FFT orientation estimation + feature rotation | No (ResNet-50) | No (Oriented R-CNN) | Yes | 2026 |
| RSFPN AACA | Angle-Aware Collaborative Attention in FPN | No | No | Yes | 2026 |
| PAMFPN RSA | Strip conv attention in FPN | No | Partial | HBB | 2025 |
| DA-FPN | Deformable conv in FPN laterals | No | Yes | HBB | 2023 |
| LO-Det CSA | Channel Separation-Aggregation (efficiency) | Yes | Yes | Yes | 2022 |
| **GAConv FPN (ours)** | **Covariance-guided deformable smooth** | **Yes** | **Yes** | **Yes** | **2026** |

---

## 10. Implementation Roadmap

### Phase 1-3: DONE (backbone fixes + pipeline wiring)

## 10. Experimental Plan

### Experiment Order (fastest feedback first)

**Exp 1: Baseline on standard backbone**
Use ResNet-50 + standard FPN + standard head via mmrotate on DOTA.
Purpose: establish a known-good baseline for GAConv comparison.

**Exp 2: GAConv in neck (standard backbone)**
Replace FPN smooth convs with GAConv. Same ResNet-50 backbone.
Purpose: isolate GAConv's contribution in neck, independent of custom backbone.
Also run component ablation (theta-only, sigma-only, full — Table A).

**Exp 3: GAConv in head (standard backbone)**
Replace 3x3 convs in detection head with GAConv.
Purpose: test whether GAConv helps more in fusion (neck) or prediction (head).
Also test neck + head combined.

**Exp 4: Custom backbone baseline**
Pre-train GhostTriRemoteXProPP on Tiny-ImageNet-200, then train detector
with standard FPN + standard head on DOTA.
Purpose: baseline for the lightweight model.

**Exp 5: GAConv in neck/head (custom backbone)**
Apply best GAConv config from Exp 2-3 to the custom backbone pipeline.
Purpose: verify gains transfer to lightweight setting.

**Exp 6: GAConv in backbone**
Replace select 3x3 convs in GhostBottle with GAConv. Requires re-pretraining.
Purpose: test backbone-level geometric adaptation.
Run full placement ablation (Table B).

**Exp 7: Multi-dataset evaluation**
Best config tested on DOTA v1.0, DOTA v2.0, HRSC2016, DIOR-R.
Per-class analysis, parameter/FLOPs comparison, geometry visualization.

---

## 11. Papers to Read

### Must-Read (for GAConv design and implementation)

1. **Deformable Convolutional Networks** (Dai et al., ICCV 2017)
   - The deformable conv mechanism GAConv builds on
   - Understand offset prediction, bilinear sampling, gradient flow

2. **S2A-Net: Align Deep Features for Oriented Object Detection** (Han et al., TGARS 2021)
   - AlignConv: closest prior art for angle-guided sampling in detection
   - Understand how they generate oriented offsets from predicted angles
   - github.com/csuhan/s2anet

3. **FAA: Fourier Angle Alignment** (Gu et al., arXiv 2602.23790, 2026)
   - FAAFusion: the only orientation-aware neck for OBB detection
   - Understand the "directional incoherence" problem (our motivation)
   - Their Fourier approach vs our covariance approach

4. **Learning Filter Scale and Orientation in CNNs** (Cam & Tek, 2018)
   - Gaussian envelope with learnable covariance matrix
   - Closest mathematical framework to our approach
   - Key difference: per-layer vs our per-location

5. **ARConv: Adaptive Rectangular Convolution** (Wang et al., CVPR 2025)
   - Learns h, w + affine for adaptive kernel shape
   - For pansharpening, not detection
   - Must differentiate clearly in related work

6. **LO-Det** (Huang et al., IEEE TGRS 2022)
   - Our direct lightweight competitor
   - Understand CSA neck and DRF mechanism

### Should-Read (for context and comparison)

7. **WPDFPN** (Han et al., Neurocomputing 2024) — neck-level task decoupling
8. **GRA** (ECCV 2024) — group-wise rotation in backbone
9. **ARS-DETR** (TGRS 2024) — aspect-ratio-aware angle prediction
10. **Strip R-CNN** (AAAI 2025) — strip convolutions, our SOTA comparison
11. **PAMFPN** (Remote Sensing 2025) — strip attention in neck

### Background (already read or skim)

12. FPN (Lin et al., CVPR 2017)
13. GhostNet (Han et al., CVPR 2020)
14. Coordinate Attention (Hou et al., CVPR 2021)
15. CBAM (Woo et al., ECCV 2018)
16. GWD (Yang et al., ICML 2021) — Gaussian modeling of OBBs
17. RetinaNet (Lin et al., ICCV 2017) — focal loss, anchor-based one-stage

---

## 12. Open Questions

### Design Questions

1. **Should GAConv use depthwise deformable conv or regular deformable conv?**
   Depthwise is much cheaper (C*9 vs C*C*9 params) but may have less capacity.
   For a lightweight model, depthwise + pointwise is preferred.

2. **How to initialize the geometry predictor?**
   Initialize to predict theta=0, sigma_major=1, sigma_minor=1 (isotropic, no
   rotation). This means GAConv starts identical to a standard 3x3 conv and
   learns to deviate only when beneficial. Safe initialization.

3. **Which convolutions in the backbone should GAConv replace?**
   Options: (a) the DW conv in GhostBottle, (b) the shortcut DW conv,
   (c) all 3x3 convs. Start with (a) — it's the main spatial processing
   operation in each block. Don't replace 1x1 convs (they're channel-only).

4. **Which convolutions in the head should GAConv replace?**
   The 3x3 convs in the cls and reg subnets. Interesting question: should
   BOTH subnets get GAConv, or only the reg subnet (since cls might benefit
   from isotropic features for classification)?

5. **What kernel size K for the strip range (sigma bounds)?**
   sigma_major should have a reasonable max (maybe 5-7) to prevent extreme
   distortion. The effective receptive field with sigma_major=5 and 3x3 base
   grid spans ~15 pixels, similar to a 15x15 kernel.

6. **Should the geometry predictor share parameters across FPN levels?**
   Probably not — different scales may need different geometric sensitivity.
   But sharing reduces params. Test both.

7. **Bottom-up path: should it also use GAConv, or standard convs?**
   Using GAConv in both top-down and bottom-up paths is the full version.
   Using it only in the smooth convs (not the downsampling convs) saves compute.

### Research Questions

6. **Does the geometric prediction actually learn meaningful orientations?**
   Visualize predicted theta maps — do they align with object orientations?
   This is a key qualitative result for the paper.

7. **Does the geometric constraint (3 params) actually outperform free offsets (18 params)?**
   Ablation C vs B in the table. If free offsets are better, the constraint
   hypothesis fails. But the constraint should help with regularization,
   especially on small datasets like DOTA.

8. **Is GAConv beneficial across ALL classes, or only elongated ones?**
   Storage tanks and roundabouts are roughly circular. GAConv should learn
   sigma_major ~= sigma_minor for these, effectively becoming isotropic.
   If it hurts isotropic classes, the 3x3 residual (from DAS design) might
   be needed.

### Practical Questions

9. **Can we use torchvision.ops.deform_conv2d for the deformable part?**
   Yes, it supports groups parameter for depthwise. The offset generation
   is custom (from predicted geometry), but the actual deformable conv op
   is standard.

10. **What if backbone pretraining gives poor accuracy on Tiny-ImageNet?**
    If <50% top-1, the backbone may not be expressive enough. Consider
    training detection end-to-end from scratch (skip pretraining).

11. **Should GAConv be validated on a standard backbone first?**
    Yes — using ResNet-50 + mmrotate isolates the GAConv contribution from
    custom backbone effects. This is standard research practice and gives
    faster initial feedback.

12. **Implementation order for placements?**
    Neck first (cheapest, no retraining) -> Head (still no retraining) ->
    Backbone (requires re-pretraining, most expensive). This order gives
    fast feedback on whether GAConv works before committing to the expensive
    backbone experiment.