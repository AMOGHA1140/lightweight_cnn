"""Training entry point for the one-stage OBB detector.

Config-driven (YAML). AMP train/validate loops, optional DataParallel, AdamW + a
cosine LR schedule. Each run writes a self-contained dir under ``runs/`` (config
snapshot, NOTES.md, meta.json, checkpoints/, TensorBoard logs).

Run:  python -m obb_detector.train --config configs/exp/gaconv_neck.yaml
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common.classes import DOTA_CLASSES
from common.config import load_config
from common.model_utils import print_model_stats
from common.registry import build_backbone, build_head, build_neck
from common.run import append_results_row, create_run_dir
from .anchors import generate_rotated_anchors
from .dataset import DOTADataset, collate_fn
from .detector import RemoteDetector
from .loss import DetectionLoss


def train_epoch(model, dataloader, optimizer, criterion, device, anchors_per_level,
                epoch, total_epochs, scaler, grad_clip):
    model.train()
    totals = [0.0, 0.0, 0.0, 0.0]  # loss, cls, bbox, obj
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        optimizer.zero_grad()
        with autocast(device_type=device.type):
            predictions = model(images)
            loss_dict = criterion(predictions, targets, anchors_per_level, device)
            loss = loss_dict["total_loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        totals[0] += loss.item()
        totals[1] += loss_dict["cls_loss"].item()
        totals[2] += loss_dict["bbox_loss"].item()
        totals[3] += loss_dict["obj_loss"].item()
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Cls": f"{loss_dict['cls_loss'].item():.4f}",
            "BBox": f"{loss_dict['bbox_loss'].item():.4f}",
            "Obj": f"{loss_dict['obj_loss'].item():.4f}",
        })
    n = len(dataloader)
    return tuple(t / n for t in totals)


@torch.no_grad()
def validate(model, dataloader, criterion, device, anchors_per_level):
    model.eval()
    totals = [0.0, 0.0, 0.0, 0.0]
    for images, targets in dataloader:
        images = images.to(device)
        loss_dict = criterion(model(images), targets, anchors_per_level, device)
        totals[0] += loss_dict["total_loss"].item()
        totals[1] += loss_dict["cls_loss"].item()
        totals[2] += loss_dict["bbox_loss"].item()
        totals[3] += loss_dict["obj_loss"].item()
    n = len(dataloader)
    return tuple(t / n for t in totals)


def build_model(cfg, device, img_size):
    """Build detector from config; return (model, feature_sizes)."""
    backbone = build_backbone(cfg)

    # Probe backbone output shapes to size the FPN and anchors.
    dummy = torch.randn(1, 3, img_size, img_size)
    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        feats = backbone.forward_features(dummy)
    backbone.train(was_training)
    fpn_in_channels = [f.shape[1] for f in feats]
    feature_sizes = [(f.shape[2], f.shape[3]) for f in feats]

    neck = build_neck(cfg, fpn_in_channels)
    head = build_head(cfg)
    bcfg, ncfg = cfg.model.backbone, cfg.model.neck
    print(f"Backbone: {bcfg.name} | Neck: FPN(out={ncfg.out_channels}, "
          f"smooth={ncfg.smooth_conv})")
    model = RemoteDetector(backbone, neck, head).to(device)
    return model, feature_sizes


def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    img_size = cfg.data.img_size

    train_loader = DataLoader(
        DOTADataset(cfg.data.root, "train", img_size),
        batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers,
        collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        DOTADataset(cfg.data.root, "val", img_size),
        batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers,
        collate_fn=collate_fn, pin_memory=True,
    )
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  GPUs: {num_gpus}")

    model, feature_sizes = build_model(cfg, device, img_size)
    print_model_stats(model, input_size=(1, 3, img_size, img_size), device=device)
    if num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
        print(f"Using DataParallel over {num_gpus} GPUs")

    # Strides from actual feature-map sizes; one scale per level.
    strides = [img_size // h for (h, w) in feature_sizes]
    anchors_per_level = generate_rotated_anchors(
        feature_sizes, strides,
        level_scales=cfg.anchors.level_scales, anchor_ratios=cfg.anchors.ratios,
        anchor_angles=cfg.anchors.angles, device=device,
    )

    criterion = DetectionLoss(num_classes=len(DOTA_CLASSES))
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr * max(1, num_gpus),
                            weight_decay=cfg.train.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=1e-6)
    scaler = GradScaler(device.type)

    run_dir = create_run_dir(cfg, device=device)
    ckpt_dir = run_dir / "checkpoints"
    writer = SummaryWriter(str(run_dir / "tb"))
    print(f"Run dir: {run_dir}")

    best_val_loss = float("inf")
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss, *_ = train_epoch(model, train_loader, optimizer, criterion, device,
                                     anchors_per_level, epoch, cfg.train.epochs, scaler,
                                     cfg.train.grad_clip)
        val_loss, val_cls, val_bbox, val_obj = validate(model, val_loader, criterion,
                                                         device, anchors_per_level)
        scheduler.step()
        print(f"Epoch {epoch:03d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f} | "
              f"Cls={val_cls:.4f}, BBox={val_bbox:.4f}, Obj={val_obj:.4f}")
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("loss/val_cls", val_cls, epoch)
        writer.add_scalar("loss/val_bbox", val_bbox, epoch)
        writer.add_scalar("loss/val_obj", val_obj, epoch)

        torch.save(model.state_dict(), ckpt_dir / "last.pth")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(f"  saved new best (val_loss {val_loss:.4f})")

    writer.close()
    append_results_row(
        cfg.paths.runs_dir, run_dir.name,
        accuracy=f"val_loss {best_val_loss:.4f}",
        dataset=cfg.data.root, method=cfg.experiment.get("method", ""),
    )
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the one-stage OBB detector.")
    parser.add_argument("--config", default="configs/base.yaml",
                        help="Path to a YAML config (see configs/).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(load_config(args.config))
