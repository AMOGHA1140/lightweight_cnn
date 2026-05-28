# Setup & run (training machine)

How to bring this repo up on a fresh machine and train the baseline. Machine-agnostic
(Linux or Windows), but the **versions are not optional**: mmcv ships prebuilt wheels
only for a fixed (Python x torch x CUDA) matrix, and there is no wheel for new torch /
Python 3.12+. Use the stack below and everything installs without compiling.

## 1. Clone + environment

Tested, lab-standard stack (OpenMMLab 2.x / mmrotate):

- **Python 3.10** — 3.8-3.11 work; **3.12+ has no prebuilt mmcv wheel**.
- **torch 2.1.0 + torchvision 0.16.0** — newest torch with prebuilt mmcv wheels.
- **mmcv 2.1.0** — prebuilt, no source build.
- **numpy < 2** — torch 2.1 is built against numpy 1.x.

```bash
git clone <repo-url> lightweight_cnn
cd lightweight_cnn
python3.10 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 1. PyTorch (CUDA 11.8 build; swap cu118 -> cu121 if your driver needs it)
pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 2. mmcv -- prebuilt wheel for torch 2.1.0 + cu118 (rotated IoU/NMS). No compilation.
pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

# 3. everything else
pip install -r requirements.txt
```

**mmcv is required** for rotated IoU/NMS (loss assignment, NMS, mAP); the code raises a
clear error if it is missing.

Why pinned and why this order: installing torch first means step 2 fetches the matching
prebuilt mmcv wheel; running `pip install -r requirements.txt` *before* step 2 (or on
torch 2.12 / Python 3.12) makes pip try to build mmcv from source, which fails. Avoid
`mim install mmcv` here — it downgrades setuptools and can break the build; the explicit
`pip install ... -f <wheel index>` above is deterministic. Confirm the CUDA build:
`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` should
print `2.1.0+cu118 True`.

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

Training is **single-GPU** (the model is small; no DataParallel/DDP). It uses `cuda:0`.
On a multi-GPU box, pick the GPU with `CUDA_VISIBLE_DEVICES` so the chosen card becomes
`cuda:0` — e.g. to train on the second GPU and leave the first alone:

```bash
# Linux/macOS
CUDA_VISIBLE_DEVICES=1 python -m obb_detector.train --config configs/exp/gaconv_neck.yaml
# Windows (cmd)
set CUDA_VISIBLE_DEVICES=1 && python -m obb_detector.train --config configs/exp/gaconv_neck.yaml
```

To run two experiments at once, launch each on its own GPU in separate shells (one
`CUDA_VISIBLE_DEVICES=0`, one `=1`) — cleaner and faster than splitting one run.

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
