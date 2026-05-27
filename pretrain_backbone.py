"""Classification pretraining for the GhostTriRemoteXProPP backbone.

Trains the backbone on an image-classification dataset to produce a checkpoint
the detector loads as initialisation. Expects an ImageFolder layout:

    <data-dir>/train/<class>/*.jpg
    <data-dir>/val/<class>/*.jpg

The saved checkpoint is the full ``GhostTriRemoteXProPP`` state_dict; the detector
loads it with ``strict=False`` (the ``fc`` head is dropped).

Run:  python pretrain_backbone.py --data-dir /path/to/dataset --num-classes 200
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import config
from common.backbone import GhostTriRemoteXProPP


def parse_args():
    p = argparse.ArgumentParser(description="Pretrain the backbone on classification.")
    p.add_argument("--data-dir", required=True,
                   help="Dataset root with train/ and val/ ImageFolder subdirs.")
    p.add_argument("--num-classes", type=int, default=200)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=os.path.join(config.MODELS_DIR, "backbone_pretrained.pth"))
    return p.parse_args()


def build_loaders(args):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(args.img_size * 256 / 224)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
    ])
    train_set = datasets.ImageFolder(os.path.join(args.data_dir, "train"), train_tf)
    val_set = datasets.ImageFolder(os.path.join(args.data_dir, "val"), val_tf)
    if len(train_set.classes) != args.num_classes:
        print(f"[warn] --num-classes={args.num_classes} but dataset has "
              f"{len(train_set.classes)} classes; using the dataset count.")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader, len(train_set.classes)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, desc=""):
    train = optimizer is not None
    model.train(train)
    total_loss = correct = seen = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            with autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += images.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{correct / seen:.3f}"})
    return total_loss / seen, correct / seen


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, n_classes = build_loaders(args)
    model = GhostTriRemoteXProPP(num_classes=n_classes).to(device)

    ## Handle data parallel if multiple GPUs are available
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel over {num_gpus} GPUs")
    



    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr * max(1, num_gpus), weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(device.type) if device.type == "cuda" else None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                    optimizer, scaler, f"Epoch {epoch}/{args.epochs} [train]")
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device,
                                      desc=f"Epoch {epoch}/{args.epochs} [val]")
        scheduler.step()
        print(f"Epoch {epoch:03d}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(model_state, args.out)
            print(f"  saved new best (val_acc {val_acc:.4f}) -> {args.out}")

    print(f"Done. Best val acc: {best_acc:.4f}. Checkpoint: {args.out}")


if __name__ == "__main__":
    main()
