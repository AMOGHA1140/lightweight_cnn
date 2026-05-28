# CLAUDE.md

Guidance for Claude Code (and humans) in this repo. Committed, so it travels with a
`git clone` — this is the authoritative entry point for picking the project up on any
machine. Detailed setup is in `claude_notes/SETUP.md`; the experiment loop is in
`claude_notes/EXPERIMENT_WORKFLOW.md`.

## What this is

A lightweight one-stage **oriented (OBB)** object detector for remote sensing (DOTA,
15 classes): backbone -> FPN neck -> rotated dense head predicting `(cx, cy, w, h, θ)`.
Active research direction is **GAConv** (geometric adaptive convolution): a drop-in 3×3
conv replacement that predicts per-location geometry and drives a depthwise deformable
conv. The current testbed is **ResNet-50 + FPN(+GAConv) + rotated head**.

Deeper context: `README.md`, `docs/`, and `RESEARCH_REFERENCE.md` (research motivation;
note this one is gitignored and stays on the dev box).

## Setup / environment

Run everything from the project root with the machine's Python environment. **Heavy
dependencies, the DOTA dataset, and model weights live only on the training machine** —
clone there, then follow `claude_notes/SETUP.md` (install `requirements.txt`, place
DOTA, train). The coding/dev box keeps only light libraries.

**mmcv is required** for rotated IoU/NMS (loss assignment, NMS, mAP); the code raises a
clear error if it is missing (no slow CPU fallback). Prebuilt mmcv wheels exist only for
a fixed matrix, so the stack is pinned: **Python 3.10, torch 2.1.0 + torchvision 0.16.0,
mmcv 2.1.0, numpy<2** (Python 3.12+ / newer torch have no mmcv wheel and force a failing
source build). Install torch first, then the prebuilt mmcv wheel — full ordered commands
in `claude_notes/SETUP.md`. Do not use `mim install mmcv` (it downgrades setuptools).

## Configuration: YAML only (no env vars)

All config is YAML under `configs/`, loaded by `common/config.py`.

- `configs/base.yaml` holds every default (data / model / anchors / train / paths /
  experiment).
- An experiment sets `_base_: ../base.yaml` and overrides only the keys it changes;
  bases are deep-merged, the experiment wins:

```yaml
# configs/exp/gaconv_neck.yaml
_base_: ../base.yaml
experiment: {name: gaconv_neck, why: "...", method: "ResNet-50 / FPN-GAConv"}
model: {neck: {smooth_conv: gaconv}}
```

- Load in code: `from common.config import load_config; cfg = load_config(path)` ->
  attribute access (`cfg.model.neck.smooth_conv`).
- Components are swapped by string via `common/registry.py`
  (`build_backbone/neck/head`). Backbones: `resnet50`, `custom`. Neck smooth conv:
  `standard`, `gaconv`.

Do **not** reintroduce `os.environ` config or a `config.py` module.

## Running an experiment

```
python -m obb_detector.train --config configs/exp/gaconv_neck.yaml   # GAConv neck
python -m obb_detector.train --config configs/exp/baseline.yaml      # standard neck
python -m obb_detector.train --resume runs/<run_dir>                 # resume
python -m obb_detector.evaluate --config <cfg> --checkpoint runs/<run>/checkpoints/best.pth
```

Validation computes **mAP** (per-class AP + mean) every `train.eval_interval` epochs and
on the last; the best checkpoint is chosen by mAP. Each run creates a self-contained
`runs/<experiment.name>_<YYYY_MM_DD-HHmm>/`:

```
config.yaml   resolved config snapshot (re-run / --resume with this)
meta.json     git commit + dirty flag, command, start time, device, resumes
metrics.csv   per-epoch lr / losses / val_mAP / best_mAP (atomic, resume-safe)
NOTES.md      why / setup / results / observations / conclusions
checkpoints/  best.pth (by mAP) + last.pth — full state (model+opt+sched+scaler)
tb/           TensorBoard logs incl. metrics/val_mAP and per-class AP/<class>
```

`runs/` is gitignored except `runs/README.md`, the curated results index
(run | accuracy=mAP | dataset | method) — a row is appended automatically on completion.
Fill accuracy/observations in `NOTES.md` by hand. Full how-to:
`claude_notes/EXPERIMENT_WORKFLOW.md`.

## Current state (2026-05-28)

Implemented and verified on synthetic tensors (`tests/`):
- GAConv (`common/gaconv.py`), ResNet-50 backbone (`common/backbone_resnet.py`),
  FPN `smooth_conv` flag, YAML config + registry + per-run dirs.
- Production trainer: **mAP-based validation**, best-by-mAP, full-state checkpoints +
  `--resume`, warmup→cosine, `metrics.csv`, per-class AP to TB, prior-bias cls/obj init.
  Shared helpers in `common/train_utils.py`; model/anchor builders in `obb_detector/build.py`.
- Standalone eval CLI: `python -m obb_detector.evaluate --config .. --checkpoint ..`.
- Tests pass (mmcv-free): `test_config_and_run`, `test_gaconv`, `test_train_utils`, and the
  forward/cost parts of `test_pipeline_synthetic` (loss/mAP parts need mmcv → run on the
  training machine).

Deferred / not yet done:
- **Real DOTA training & mAP numbers** — dataset not downloaded; run on the training machine.
- Custom backbone (`GhostTriRemoteXProPP`) is **not pretrained**; use `resnet50` for now.
- DOTA test-set submission files (tiling/merge), ATSS assignment, loss-weight tuning.

## Repo map

```
configs/         base.yaml + exp/*.yaml (the only config; YAML, no env vars)
common/          config.py, registry.py, run.py, constants.py, gaconv.py,
                 backbone.py (custom), backbone_resnet.py, rotated_ops.py, ...
obb_detector/    dataset, fpn, head, anchors, detector, loss, inference, evaluate, train
tests/           synthetic checks (no DOTA needed)
runs/            per-run outputs (gitignored except README.md index)
claude_notes/    SETUP.md, EXPERIMENT_WORKFLOW.md (committed); older notes are historical
```

## Working preferences

- Ask clarifying questions before proceeding when anything is ambiguous; knowing what
  to do before doing matters.
- Short comment lines. No long trailing-dash comments.
- Keep `common/` and `obb_detector/` reusable; experiments are config files + run dirs,
  not bespoke scripts.
