"""Shared training utilities (seeding, schedulers, checkpoints, CSV logging).

These mirror the production patterns proven in ``pretrain_backbone.py`` but are
detector-agnostic so the OBB trainer and evaluator can reuse them. No EMA here:
EMA is used only for backbone pretraining.
"""

import csv
import os
import random

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def build_param_groups(model, weight_decay):
    """Two groups: decay on conv/linear weights, none on norm/bias (1-D params)."""
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


def build_scheduler(optimizer, epochs, warmup_epochs, eta_min=1e-6):
    """Linear warmup (1% -> 100%) for ``warmup_epochs`` then cosine to ``eta_min``."""
    cosine_epochs = max(1, epochs - warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=eta_min)
    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    return cosine


def atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_metric, cfg):
    """Full resumable state written atomically."""
    atomic_save({
        "epoch": epoch,
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else cfg,
    }, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def load_model_weights(model, ckpt, strict=True):
    """Load weights from a full-state dict or a bare ``state_dict``."""
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    unwrap(model).load_state_dict(state, strict=strict)


class CSVLogger:
    """Append-only metrics log, atomically rewritten each call.

    ``fields`` defines the columns. On resume, pass ``keep_through_epoch`` to keep
    only rows up to and including that epoch.
    """

    def __init__(self, path, fields, keep_through_epoch=None):
        self.path = str(path)
        self.fields = list(fields)
        self.rows = []
        if keep_through_epoch is not None and os.path.exists(self.path):
            with open(self.path, newline="") as f:
                for r in csv.DictReader(f):
                    if int(r["epoch"]) <= keep_through_epoch:
                        self.rows.append(r)

    def log(self, row):
        self.rows.append({k: row.get(k, "") for k in self.fields})
        tmp = self.path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            w.writeheader()
            w.writerows(self.rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
