"""Checks for production training utilities (no DOTA, no mmcv needed).

Run:  python -m tests.test_train_utils
"""

import math
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from common.config import load_config
from common.train_utils import (CSVLogger, build_param_groups, build_scheduler,
                                 load_checkpoint, load_model_weights, save_checkpoint,
                                 seed_everything)
from obb_detector.build import build_model
from obb_detector.head import RotatedDetectionHead


def _tiny():
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4))


def test_param_groups():
    groups = build_param_groups(_tiny(), weight_decay=0.01)
    assert groups[0]["weight_decay"] == 0.01 and groups[1]["weight_decay"] == 0.0
    # conv weight in decay group; bias + BN params in no-decay group
    assert len(groups[0]["params"]) == 1
    assert len(groups[1]["params"]) >= 2
    print("param_groups: ok")


def test_scheduler_warmup():
    m = _tiny()
    opt = torch.optim.AdamW(m.parameters(), lr=1.0)
    sched = build_scheduler(opt, epochs=10, warmup_epochs=3, eta_min=0.0)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[2] < lrs[3] + 1e-9, lrs[:4]   # warmup ramps up
    assert lrs[-1] < lrs[3], lrs                       # then cosine decays
    print(f"scheduler_warmup: ok (lr {lrs[0]:.3f} -> {max(lrs):.3f} -> {lrs[-1]:.3f})")


def test_checkpoint_roundtrip():
    m = _tiny()
    opt = torch.optim.AdamW(m.parameters(), lr=0.1)
    sched = build_scheduler(opt, epochs=5, warmup_epochs=1)
    opt.step(); sched.step()
    cfg = load_config("configs/base.yaml")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "last.pth"
        save_checkpoint(path, m, opt, sched, None, epoch=2, best_metric=0.42, cfg=cfg)
        ckpt = load_checkpoint(path)
        assert ckpt["epoch"] == 2 and abs(ckpt["best_metric"] - 0.42) < 1e-9
        assert "optimizer" in ckpt and "scheduler" in ckpt and "config" in ckpt
        m2 = _tiny()
        load_model_weights(m2, ckpt)
        for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
            assert torch.equal(a, b)
    print("checkpoint_roundtrip: ok")


def test_csv_logger():
    fields = ["epoch", "val_mAP"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics.csv"
        log = CSVLogger(path, fields)
        log.log({"epoch": 1, "val_mAP": "0.10"})
        log.log({"epoch": 2, "val_mAP": "0.20"})
        text = path.read_text()
        assert text.count("epoch,val_mAP") == 1 and "1,0.10" in text and "2,0.20" in text
        # resume keeps only through epoch 1
        log2 = CSVLogger(path, fields, keep_through_epoch=1)
        log2.log({"epoch": 2, "val_mAP": "0.25"})
        assert "0.25" in path.read_text()
    print("csv_logger: ok (header once, resume truncation)")


def test_prior_bias_init():
    p = 0.01
    head = RotatedDetectionHead(num_classes=15, num_anchors=9, in_channels=64, prior_prob=p)
    expected = -math.log((1.0 - p) / p)
    assert torch.allclose(head.cls_score.bias, torch.full_like(head.cls_score.bias, expected), atol=1e-5)
    assert torch.allclose(head.obj_pred.bias, torch.full_like(head.obj_pred.bias, expected), atol=1e-5)
    print(f"prior_bias_init: ok (bias={expected:.3f})")


def test_build_from_cfg():
    seed_everything(0)
    cfg = load_config("configs/exp/gaconv_neck.yaml")
    cfg.model.backbone.pretrained = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feature_sizes = build_model(cfg, device, img_size=256)
    cls_outs, reg_outs, obj_outs = model(torch.randn(1, 3, 256, 256, device=device))
    assert len(cls_outs) == len(feature_sizes) == 3
    assert reg_outs[0].shape[1] == cfg.model.head.num_anchors * 5
    print("build_from_cfg: ok (model built from yaml, correct shapes)")


def main():
    test_param_groups()
    test_scheduler_warmup()
    test_checkpoint_roundtrip()
    test_csv_logger()
    test_prior_bias_init()
    test_build_from_cfg()
    print("\nall train_utils tests passed")


if __name__ == "__main__":
    main()
