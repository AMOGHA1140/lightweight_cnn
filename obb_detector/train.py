"""Training entry point for the one-stage OBB detector.

Config-driven (YAML). AMP train loop, optional DataParallel, AdamW with warmup +
cosine schedule, mAP-based validation, and full-state checkpoints. Each run writes a
self-contained dir under ``runs/`` (config snapshot, NOTES.md, meta.json, metrics.csv,
checkpoints/, TensorBoard logs). Requires mmcv (rotated IoU/NMS).

Run:    python -m obb_detector.train --config configs/exp/gaconv_neck.yaml
Resume: python -m obb_detector.train --resume runs/<run_dir>
"""

import argparse
import json
import time
from pathlib import Path

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
from common.run import append_results_row, create_run_dir
from common.train_utils import (CSVLogger, build_param_groups, build_scheduler,
                                load_checkpoint, load_model_weights, save_checkpoint,
                                seed_everything)
from .build import build_anchors, build_model
from .dataset import DOTADataset, collate_fn
from .evaluate import evaluate_map
from .loss import DetectionLoss

CSV_FIELDS = ["epoch", "lr", "train_loss", "train_cls", "train_bbox", "train_obj",
              "val_loss", "val_mAP", "best_mAP", "seconds"]


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
    total = 0.0
    for images, targets in dataloader:
        images = images.to(device)
        loss_dict = criterion(model(images), targets, anchors_per_level, device)
        total += loss_dict["total_loss"].item()
    return total / len(dataloader)


def main(cfg, resume_dir=None):
    seed_everything(cfg.train.seed)
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

    anchors_per_level = build_anchors(cfg, feature_sizes, img_size, device)

    criterion = DetectionLoss(num_classes=len(DOTA_CLASSES))
    optimizer = optim.AdamW(build_param_groups(model, cfg.train.weight_decay),
                            lr=cfg.train.lr * max(1, num_gpus))
    scheduler = build_scheduler(optimizer, cfg.train.epochs, cfg.train.warmup_epochs,
                                eta_min=cfg.train.eta_min)
    scaler = GradScaler(device.type)

    # Fresh run vs resume into an existing run dir.
    if resume_dir is not None:
        run_dir = Path(resume_dir)
        ckpt = load_checkpoint(run_dir / "checkpoints" / "last.pth")
        load_model_weights(model, ckpt)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_mAP = ckpt.get("best_metric", float("-inf"))
        csv_keep = ckpt["epoch"]
        _append_resume_note(run_dir, ckpt["epoch"])
        print(f"Resumed from {run_dir} at epoch {ckpt['epoch']} (best mAP {best_mAP:.4f})")
    else:
        run_dir = create_run_dir(cfg, device=device)
        start_epoch, best_mAP, csv_keep = 1, float("-inf"), None
        print(f"Run dir: {run_dir}")

    ckpt_dir = run_dir / "checkpoints"
    writer = SummaryWriter(str(run_dir / "tb"))
    logger = CSVLogger(run_dir / "metrics.csv", CSV_FIELDS, keep_through_epoch=csv_keep)

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_cls, train_bbox, train_obj = train_epoch(
            model, train_loader, optimizer, criterion, device, anchors_per_level,
            epoch, cfg.train.epochs, scaler, cfg.train.grad_clip)

        do_eval = (epoch % cfg.train.eval_interval == 0) or (epoch == cfg.train.epochs)
        val_mAP = None
        if do_eval:
            # Single val pass computes loss and mAP together.
            aps, val_mAP, val_loss = evaluate_map(
                model, val_loader, device, anchors_per_level, DOTA_CLASSES,
                iou_thresh=cfg.eval.iou_thresh, conf_thresh=cfg.eval.conf_thresh,
                nms_thresh=cfg.eval.nms_thresh, criterion=criterion)
            for cls, ap in aps.items():
                if ap == ap:  # skip NaN (no GT for that class)
                    writer.add_scalar(f"AP/{cls}", ap, epoch)
            writer.add_scalar("metrics/val_mAP", val_mAP, epoch)
        else:
            val_loss = validate(model, val_loader, criterion, device, anchors_per_level)

        scheduler.step()
        secs = time.time() - t0
        writer.add_scalar("lr", lr, epoch)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        map_str = f"{val_mAP:.4f}" if val_mAP is not None else "  -  "
        print(f"Epoch {epoch:03d}: train={train_loss:.4f} val={val_loss:.4f} "
              f"mAP={map_str} lr={lr:.2e} ({secs:.0f}s)")

        if val_mAP is not None and val_mAP > best_mAP:
            best_mAP = val_mAP
            save_checkpoint(ckpt_dir / "best.pth", model, optimizer, scheduler, scaler,
                            epoch, best_mAP, cfg)
            print(f"  saved new best (mAP {best_mAP:.4f})")
        save_checkpoint(ckpt_dir / "last.pth", model, optimizer, scheduler, scaler,
                        epoch, best_mAP, cfg)
        logger.log({
            "epoch": epoch, "lr": f"{lr:.6f}",
            "train_loss": f"{train_loss:.4f}", "train_cls": f"{train_cls:.4f}",
            "train_bbox": f"{train_bbox:.4f}", "train_obj": f"{train_obj:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "val_mAP": f"{val_mAP:.4f}" if val_mAP is not None else "",
            "best_mAP": f"{best_mAP:.4f}" if best_mAP > float("-inf") else "",
            "seconds": f"{secs:.1f}",
        })

    writer.close()
    acc = f"mAP {best_mAP:.4f}" if best_mAP > float("-inf") else "n/a"
    append_results_row(cfg.paths.runs_dir, run_dir.name, accuracy=acc,
                       dataset=cfg.data.root, method=cfg.experiment.get("method", ""))
    print(f"Training complete. Best mAP: {best_mAP:.4f}  (run: {run_dir})")


def _append_resume_note(run_dir, from_epoch):
    meta_path = run_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        meta = {}
    meta.setdefault("resumes", []).append({
        "from_epoch": from_epoch,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    meta_path.write_text(json.dumps(meta, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Train the one-stage OBB detector.")
    parser.add_argument("--config", default="configs/base.yaml",
                        help="Path to a YAML config (see configs/).")
    parser.add_argument("--resume", default=None,
                        help="Resume an existing run dir (uses its config.yaml snapshot).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.resume:
        cfg = load_config(Path(args.resume) / "config.yaml")
        main(cfg, resume_dir=args.resume)
    else:
        main(load_config(args.config))
