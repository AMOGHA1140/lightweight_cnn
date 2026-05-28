"""Classification pretraining for the GhostTriRemoteXProPP backbone (ImageNet-1k).

Trains the backbone on an ImageFolder classification dataset to produce a
checkpoint the detector loads as initialisation. Expects:

    <data-dir>/train/<class>/*.JPEG
    <data-dir>/val/<class>/*.JPEG

Everything for a run lives under a master directory (``--out-dir``):

    <out-dir>/
      config.json        resolved run arguments (reloaded on resume)
      metrics.csv        per-epoch metrics (crash-safe atomic rewrite)
      tb/                TensorBoard event files
      best/
        checkpoint.pth   record of the best epoch (model + metadata)
        backbone.pth     bare backbone state_dict for the detector (strict=False)
      epoch_<x>/
        checkpoint.pth   full resumable state {model, optimizer, scheduler, scaler}

A full checkpoint is written every ``--save-every`` epochs (default 1). ``best/``
holds whichever of the live or EMA model has the highest validation accuracy.

Resume with ``--resume <epoch_dir|checkpoint.pth>`` or ``--resume auto``.

Run:  python pretrain_backbone.py --data-dir /path/to/imagenet --out-dir backbone_weights
"""

import argparse
import copy
import csv
import glob
import json
import os
import random
import re
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

from common.backbone import GhostTriRemoteXProPP
from common.config import load_config
from common.constants import IMAGENET_MEAN, IMAGENET_STD


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #

def build_parser():
    p = argparse.ArgumentParser(description="Pretrain the backbone on classification.")
    p.add_argument("--data-dir", required=True,
                   help="Dataset root with train/ and val/ ImageFolder subdirs.")
    p.add_argument("--out-dir", default="./backbone_weights",
                   help="Master directory for checkpoints, logs and config.")
    p.add_argument("--num-classes", type=int, default=200)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--rand-aug-magnitude", type=int, default=9,
                   help="RandAugment magnitude; <= 0 disables RandAugment.")
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--no-ema", action="store_true", help="Disable weight EMA.")
    p.add_argument("--save-ema-in-ckpt", action="store_true",
                   help="Store EMA weights in every checkpoint (for exact EMA resume).")
    p.add_argument("--save-every", type=int, default=1,
                   help="Write a full checkpoint every N epochs.")
    p.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="bfloat16",
                   help="Mixed-precision dtype on CUDA. bfloat16 needs no GradScaler.")
    p.add_argument("--resume", default=None,
                   help="Resume from a checkpoint path / epoch dir, or 'auto'.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=None,
                   help="YAML whose 'pretrain' keys override argparse defaults (CLI still wins).")
    return p


def provided_dests(parser):
    """Set of argparse dests that were explicitly passed on the command line."""
    opt_map = {opt: a.dest for a in parser._actions for opt in a.option_strings}
    dests = set()
    for tok in sys.argv[1:]:
        dests.add(opt_map.get(tok.split("=", 1)[0]))
    dests.discard(None)
    return dests


def apply_config_file(args, parser):
    """Override argparse defaults from a YAML 'pretrain' block (CLI values win)."""
    cfg = load_config(args.config)
    pretrain = cfg.get("pretrain")
    if pretrain is None:
        return args
    keep = provided_dests(parser)
    for k, v in pretrain.to_dict().items():
        if hasattr(args, k) and k not in keep:
            setattr(args, k, v)
    return args


def apply_resumed_config(args, parser):
    """On resume, fill args from the saved config.json unless overridden on the CLI."""
    cfg_path = os.path.join(args.out_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"--resume given but no config.json in {args.out_dir}")
    with open(cfg_path) as f:
        saved = json.load(f)
    keep = provided_dests(parser)  # CLI values win over saved ones
    for k, v in saved.items():
        if hasattr(args, k) and k not in keep:
            setattr(args, k, v)
    return args


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def build_loaders(args):
    # RandAugment operates on the uint8 PIL image, so it must come before ToTensor;
    # RandomErasing operates on the normalized tensor, so it comes last.
    train_ops = [
        transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
    ]
    if args.rand_aug_magnitude > 0:
        train_ops.append(transforms.RandAugment(magnitude=args.rand_aug_magnitude))
    train_ops += [
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.1),
    ]
    train_tf = transforms.Compose(train_ops)
    val_tf = transforms.Compose([
        transforms.Resize(int(args.img_size * 256 / 224)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_set = datasets.ImageFolder(os.path.join(args.data_dir, "train"), train_tf)
    val_set = datasets.ImageFolder(os.path.join(args.data_dir, "val"), val_tf)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    return train_loader, val_loader, len(train_set.classes)


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #

def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def build_param_groups(model, weight_decay):
    """Two groups: weight decay on conv/linear weights, none on norm/bias (1-D params)."""
    decay, no_decay = [], []
    for name, p in unwrap(model).named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_scheduler(optimizer, epochs, warmup_epochs):
    cosine_epochs = max(1, epochs - warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6)
    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    return cosine


class ModelEma:
    """Exponential moving average of model weights (params + buffers).

    Uses a step-warmed decay so the value is robust to schedule length:
    ``decay_t = min(decay, (1 + t) / (10 + t))``.
    """

    def __init__(self, model, decay=0.9999, device=None):
        self.module = copy.deepcopy(unwrap(model)).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.module.to(device)
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        src = unwrap(model).state_dict()
        for k, v in self.module.state_dict().items():
            sv = src[k].detach().to(v.device)
            if v.is_floating_point():
                v.mul_(d).add_(sv, alpha=1 - d)
            else:
                v.copy_(sv)  # e.g. BatchNorm num_batches_tracked

    def reseed(self, model):
        self.module.load_state_dict(unwrap(model).state_dict())
        self.num_updates = 0


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, ema, desc,
                    amp_dtype, use_amp, writer, epoch):
    model.train()
    total_loss = correct = seen = 0
    base_step = (epoch - 1) * len(loader)
    pbar = tqdm(loader, desc=desc, leave=False)
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if ema is not None:
            ema.update(model)
        # Per-step loss (use TensorBoard smoothing for viewing).
        writer.add_scalar("train/step_loss", loss.item(), base_step + i)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += images.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{correct / seen:.3f}"})
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc, amp_dtype, use_amp):
    model.eval()
    total_loss = correct = seen = 0
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += images.size(0)
    return total_loss / seen, correct / seen


# --------------------------------------------------------------------------- #
# Checkpoint / logging I/O
# --------------------------------------------------------------------------- #

def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_full_checkpoint(ckpt_dir, model, optimizer, scheduler, scaler, epoch,
                         best_acc, args, ema=None, save_ema=False):
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_acc": best_acc,
        "args": vars(args),
    }
    if ema is not None and save_ema:
        ckpt["ema"] = ema.module.state_dict()
        ckpt["ema_num_updates"] = ema.num_updates
    atomic_save(ckpt, os.path.join(ckpt_dir, "checkpoint.pth"))


def save_best(out_dir, weights_state, epoch, val_acc, source, args):
    best_dir = os.path.join(out_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    atomic_save(weights_state, os.path.join(best_dir, "backbone.pth"))
    atomic_save(
        {"epoch": epoch, "val_acc": val_acc, "source": source,
         "model": weights_state, "args": vars(args)},
        os.path.join(best_dir, "checkpoint.pth"),
    )


def find_latest_checkpoint(out_dir):
    best_e, best_p = -1, None
    for d in glob.glob(os.path.join(out_dir, "epoch_*")):
        m = re.match(r"epoch_(\d+)$", os.path.basename(d))
        ckpt = os.path.join(d, "checkpoint.pth")
        if m and os.path.exists(ckpt) and int(m.group(1)) > best_e:
            best_e, best_p = int(m.group(1)), ckpt
    return best_p


def resolve_resume_path(resume, out_dir):
    if resume == "auto":
        return find_latest_checkpoint(out_dir)
    if os.path.isdir(resume):
        return os.path.join(resume, "checkpoint.pth")
    return resume


CSV_FIELDS = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc",
              "ema_val_loss", "ema_val_acc", "best_acc", "best_source", "seconds"]


class CSVLogger:
    """Append-only metrics log that is fully rewritten atomically each epoch."""

    def __init__(self, path, keep_through_epoch=None):
        self.path = path
        self.rows = []
        if keep_through_epoch is not None and os.path.exists(path):
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    if int(r["epoch"]) <= keep_through_epoch:
                        self.rows.append(r)

    def log(self, row):
        self.rows.append({k: row.get(k, "") for k in CSV_FIELDS})
        tmp = self.path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(self.rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.config:
        apply_config_file(args, parser)
    if args.resume:
        apply_resumed_config(args, parser)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, val_loader, n_classes = build_loaders(args)
    if args.resume and args.num_classes != n_classes:
        raise ValueError(f"Resume class count {args.num_classes} != dataset {n_classes}")
    if not args.resume and args.num_classes != n_classes:
        print(f"[warn] --num-classes={args.num_classes} but dataset has {n_classes} "
              f"classes; using {n_classes}.")
    args.num_classes = n_classes

    model = GhostTriRemoteXProPP(num_classes=n_classes).to(device)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel over {num_gpus} GPUs")

    # Mixed precision: bfloat16 needs no GradScaler; float16 does. Fall back to
    # float16 if the GPU lacks bf16 support.
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    if use_amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("[warn] bfloat16 not supported on this GPU; falling back to float16.")
        amp_dtype, args.amp_dtype = torch.float16, "float16"
    use_scaler = use_amp and amp_dtype == torch.float16

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(build_param_groups(model, args.weight_decay), lr=args.lr)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = GradScaler(enabled=(device.type == "cuda")) if use_scaler else None
    ema = None if args.no_ema else ModelEma(model, decay=args.ema_decay, device=device)

    start_epoch, best_acc = 1, 0.0
    if args.resume:
        path = resolve_resume_path(args.resume, args.out_dir)
        if path is None or not os.path.exists(path):
            print(f"[warn] no checkpoint found for --resume {args.resume}; starting fresh.")
        else:
            ckpt = torch.load(path, map_location="cpu")
            unwrap(model).load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if scaler is not None and ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            best_acc = ckpt.get("best_acc", 0.0)
            if ema is not None:
                if "ema" in ckpt:
                    ema.module.load_state_dict(ckpt["ema"])
                    ema.num_updates = ckpt.get("ema_num_updates", 0)
                else:
                    ema.reseed(model)  # EMA not stored: re-seed and re-accumulate
            print(f"Resumed from {path} at epoch {ckpt['epoch']} (best_acc {best_acc:.4f})")

    # Persist the resolved configuration for future resumes.
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb"))
    logger = CSVLogger(os.path.join(args.out_dir, "metrics.csv"),
                       keep_through_epoch=start_epoch - 1 if args.resume else None)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, ema,
            f"Epoch {epoch}/{args.epochs} [train]", amp_dtype, use_amp, writer, epoch)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, f"Epoch {epoch}/{args.epochs} [val]",
            amp_dtype, use_amp)

        ema_loss = ema_acc = None
        if ema is not None:
            ema_loss, ema_acc = evaluate(
                ema.module, val_loader, criterion, device,
                f"Epoch {epoch}/{args.epochs} [ema]", amp_dtype, use_amp)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        # Best of live vs EMA.
        candidates = [("live", val_acc, lambda: unwrap(model).state_dict())]
        if ema is not None:
            candidates.append(("ema", ema_acc, lambda: ema.module.state_dict()))
        src, epoch_acc, get_state = max(candidates, key=lambda c: c[1])
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            save_best(args.out_dir, get_state(), epoch, epoch_acc, src, args)

        # Periodic full checkpoint.
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_full_checkpoint(
                os.path.join(args.out_dir, f"epoch_{epoch}"), model, optimizer,
                scheduler, scaler, epoch, best_acc, args, ema, args.save_ema_in_ckpt)

        secs = time.time() - t0
        writer.add_scalar("lr", lr, epoch)
        writer.add_scalars("loss", {"train": tr_loss, "val": val_loss}, epoch)
        writer.add_scalars("acc", {"train": tr_acc, "val": val_acc}, epoch)
        if ema is not None:
            writer.add_scalar("loss/ema_val", ema_loss, epoch)
            writer.add_scalar("acc/ema_val", ema_acc, epoch)
        logger.log({
            "epoch": epoch, "lr": f"{lr:.6f}",
            "train_loss": f"{tr_loss:.4f}", "train_acc": f"{tr_acc:.4f}",
            "val_loss": f"{val_loss:.4f}", "val_acc": f"{val_acc:.4f}",
            "ema_val_loss": f"{ema_loss:.4f}" if ema_loss is not None else "",
            "ema_val_acc": f"{ema_acc:.4f}" if ema_acc is not None else "",
            "best_acc": f"{best_acc:.4f}", "best_source": src, "seconds": f"{secs:.1f}",
        })
        msg = (f"Epoch {epoch:03d}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
               f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if ema is not None:
            msg += f" | ema_val_acc={ema_acc:.4f}"
        msg += f" | best={best_acc:.4f}({src})"
        print(msg)

    writer.close()
    print(f"Done. Best val acc: {best_acc:.4f}. "
          f"Backbone: {os.path.join(args.out_dir, 'best', 'backbone.pth')}")


if __name__ == "__main__":
    main()