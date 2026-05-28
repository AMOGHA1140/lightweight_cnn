---
name: production-train-eval-pipeline
description: "Production OBB trainer/eval with mAP, full-state ckpts, resume, mmcv-required (2026-05-28)"
metadata: 
  node_type: memory
  type: project
  originSessionId: e3445078-03de-4b92-93c7-c2fbde50b0f7
---

The OBB detector training + evaluation was upgraded to production-grade (2026-05-28).

- **mAP in validation**: `obb_detector/train.py` calls `evaluate_map` every
  `train.eval_interval` epochs (default 1) + last epoch; **best.pth chosen by mAP**.
  `runs/README.md` accuracy column is now mAP. Per-class AP + val_mAP to TensorBoard.
- **Full-state checkpoints + resume**: `checkpoints/{best,last}.pth` hold
  model+optimizer+scheduler+scaler+epoch+best+config. `python -m obb_detector.train
  --resume runs/<run_dir>` reuses that dir, loads its config.yaml snapshot + last.pth.
- **Shared infra** in `common/train_utils.py` (seed_everything, build_param_groups,
  build_scheduler = warmup→cosine, atomic_save, save/load_checkpoint, load_model_weights,
  CSVLogger). `obb_detector/build.py` has `build_model`/`build_anchors` (shared by train
  + evaluate; avoids an import cycle). metrics.csv per epoch.
- **Eval CLI**: `python -m obb_detector.evaluate --config <cfg> --checkpoint <path>
  [--split val] [--conf/iou/nms-thresh]` → prints per-class AP table + mAP, writes
  eval.json beside the checkpoint.
- **Prior-bias init**: `RotatedDetectionHead(prior_prob=0.01)` sets cls_score/obj_pred
  bias to `-log((1-p)/p)` (fixes the ~60k init focal loss). Wired via `model.head.prior_prob`.
- **mmcv is REQUIRED** (user decision): `common/rotated_ops.box_iou_rotated` and
  `inference._nms_rotated` raise a clear ImportError if mmcv missing — the shapely/greedy
  CPU fallback was removed. requirements.txt: mmcv>=2.0 (install via openmim), shapely dropped.
- **No EMA** in the detector (EMA is pretraining-only, per user).
- Config additions (configs/base.yaml): `train.{warmup_epochs:3, eta_min, eval_interval:1,
  seed:42}`, `eval.{iou_thresh:0.5, conf_thresh:0.05, nms_thresh:0.1}`, `model.head.prior_prob`.

Verification: this dev box has NO mmcv (stays light), so loss/mAP paths can't run here;
`tests/test_pipeline_synthetic.py` skips the train-step when mmcv absent. mmcv-free suites
pass here: `test_config_and_run`, `test_gaconv`, `test_train_utils`, plus forward/cost of
`test_pipeline_synthetic`. Full loss/mAP/train run happens on the training machine.

Docs updated (committed): CLAUDE.md, claude_notes/SETUP.md, docs/usage.md, docs/architecture.md,
README.md. See [[yaml-config-experiment-workflow]], [[gaconv-resnet-testbed]],
[[doc-routing-two-claudes]].