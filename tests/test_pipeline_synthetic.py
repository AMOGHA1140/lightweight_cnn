"""End-to-end synthetic check: ResNet-50 + GAConv-neck + rotated head.

Proves the pipeline builds, forwards with correct shapes, and takes one real
training step (loss -> backward -> optimizer step) on synthetic targets, with no
DOTA data. Also reports params/FLOPs for the standard vs GAConv neck.

Run:  python -m tests.test_pipeline_synthetic
"""

import torch

from common.backbone_resnet import ResNet50Backbone
from common.classes import DOTA_CLASSES
from common.config import load_config
from common.gaconv import GAConv
from common.registry import build_backbone, build_head, build_neck
from obb_detector.anchors import generate_rotated_anchors
from obb_detector.detector import RemoteDetector
from obb_detector.fpn import FPN
from obb_detector.head import RotatedDetectionHead
from obb_detector.loss import DetectionLoss

IMG = 512
OUT_CH = 256
NUM_ANCHORS = 9
LEVEL_SCALES = [[32], [64], [128]]
ANCHOR_RATIOS = [0.5, 1.0, 2.0]
ANCHOR_ANGLES = [-60, 0, 60]


def build(device, smooth_conv):
    backbone = ResNet50Backbone(pretrained=False, frozen_stages=1, norm_eval=True)
    dummy = torch.randn(1, 3, IMG, IMG)
    backbone.eval()
    with torch.no_grad():
        feats = backbone.forward_features(dummy)
    in_ch = [f.shape[1] for f in feats]
    feature_sizes = [(f.shape[2], f.shape[3]) for f in feats]
    neck = FPN(in_ch, out_channels=OUT_CH, smooth_conv=smooth_conv)
    head = RotatedDetectionHead(num_classes=len(DOTA_CLASSES), num_anchors=NUM_ANCHORS, in_channels=OUT_CH)
    model = RemoteDetector(backbone, neck, head).to(device)
    return model, feature_sizes, in_ch


def synthetic_targets(batch, device):
    """A few angle-0 boxes near image center so some anchors become positive."""
    targets = []
    for _ in range(batch):
        boxes = torch.tensor([
            [256.0, 256.0, 64.0, 32.0, 0.0],
            [128.0, 384.0, 128.0, 64.0, 0.0],
        ], device=device)
        labels = torch.tensor([3, 7], device=device, dtype=torch.long)
        targets.append({"boxes": boxes, "labels": labels})
    return targets


def test_forward_and_train_step():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feature_sizes, in_ch = build(device, smooth_conv="gaconv")
    assert in_ch == [512, 1024, 2048], in_ch
    print(f"backbone C3/C4/C5 channels: {in_ch}, feature sizes: {feature_sizes}")

    batch = 2
    images = torch.randn(batch, 3, IMG, IMG, device=device)
    model.train()
    cls_outs, reg_outs, obj_outs = model(images)
    for lvl, (h, w) in enumerate(feature_sizes):
        assert cls_outs[lvl].shape == (batch, NUM_ANCHORS * len(DOTA_CLASSES), h, w)
        assert reg_outs[lvl].shape == (batch, NUM_ANCHORS * 5, h, w)
        assert obj_outs[lvl].shape == (batch, NUM_ANCHORS, h, w)
    print("forward: per-level cls/reg/obj shapes ok")

    try:
        import mmcv.ops  # noqa: F401
    except Exception:
        print("train step: SKIPPED (mmcv not installed; loss uses rotated IoU). "
              "Runs on the training machine.")
        return

    strides = [IMG // h for (h, w) in feature_sizes]
    anchors = generate_rotated_anchors(
        feature_sizes, strides, LEVEL_SCALES, ANCHOR_RATIOS, ANCHOR_ANGLES, device)
    criterion = DetectionLoss(num_classes=len(DOTA_CLASSES))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)

    targets = synthetic_targets(batch, device)
    before = model.head.cls_score.weight.detach().clone()
    optimizer.zero_grad()
    loss_dict = criterion(model(images), targets, anchors, device)
    loss = loss_dict["total_loss"]
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss), loss
    assert not torch.equal(before, model.head.cls_score.weight), "head weights did not update"
    print(f"train step: loss={loss.item():.4f} "
          f"(cls={loss_dict['cls_loss'].item():.3f}, "
          f"bbox={loss_dict['bbox_loss'].item():.3f}, "
          f"obj={loss_dict['obj_loss'].item():.3f}); weights updated")


def report_neck_cost():
    in_ch = [512, 1024, 2048]
    std = FPN(in_ch, out_channels=OUT_CH, smooth_conv="standard")
    gac = FPN(in_ch, out_channels=OUT_CH, smooth_conv="gaconv")

    def nparams(m):
        return sum(p.numel() for p in m.parameters())

    print(f"\nFPN params  standard: {nparams(std):,}   gaconv: {nparams(gac):,}")
    print(f"smooth-stage only  standard: {nparams(std.smooth):,}   "
          f"gaconv: {nparams(gac.smooth):,}")

    try:
        from thop import profile
        x = torch.randn(1, OUT_CH, 64, 64)
        std_conv = torch.nn.Conv2d(OUT_CH, OUT_CH, 3, padding=1)
        gac_conv = GAConv(OUT_CH)
        std_flops, _ = profile(std_conv, inputs=(x,), verbose=False)
        gac_flops, _ = profile(gac_conv, inputs=(x,), verbose=False)
        print(f"single smooth op FLOPs @64x64  standard 3x3: {std_flops/1e6:.1f}M   "
              f"GAConv: {gac_flops/1e6:.1f}M")
    except Exception as e:
        print(f"(thop FLOPs skipped: {e})")


def test_config_driven_build():
    # Build the model from a real experiment config via the registry, proving the
    # cfg -> registry -> model path. pretrained off to avoid a weights download.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config("configs/exp/gaconv_neck.yaml")
    cfg.model.backbone.pretrained = False
    assert cfg.model.neck.smooth_conv == "gaconv"

    backbone = build_backbone(cfg)
    dummy = torch.randn(1, 3, IMG, IMG)
    backbone.eval()
    with torch.no_grad():
        in_ch = [f.shape[1] for f in backbone.forward_features(dummy)]
    neck = build_neck(cfg, in_ch)
    head = build_head(cfg)
    model = RemoteDetector(backbone, neck, head).to(device)

    cls_outs, _, _ = model(torch.randn(1, 3, IMG, IMG, device=device))
    assert len(cls_outs) == 3
    assert any(isinstance(m, GAConv) for m in model.neck.smooth)
    print("config_driven_build: ok (registry built GAConv-neck model from yaml)")


def main():
    torch.manual_seed(0)
    test_forward_and_train_step()
    test_config_driven_build()
    report_neck_cost()
    print("\npipeline synthetic test passed")


if __name__ == "__main__":
    main()
