"""Training entry point for the experimental two-stage (HBB) detector.

See ``model.py`` for the caveats -- this pipeline is kept for reference only.

Run:  python -m faster_rcnn.train
"""

import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import config
from common.backbone import GhostTriRemoteXProPP
from common.classes import DOTA_CLASSES, NUM_CLASSES
from common.model_utils import count_parameters
from .dataset import DOTADataset, custom_collate_fn
from .metrics import evaluate_model
from .model import FasterRCNN


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss, num_batches = 0.0, 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        losses = model(images, targets_device)
        loss = losses["total_loss"] if isinstance(losses, dict) else losses
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Avg": f"{total_loss / num_batches:.4f}"})
    return total_loss / num_batches if num_batches else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size, lr, num_epochs, img_size = 4, 1e-3, 50, 800
    norm = transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)

    train_loader = DataLoader(
        DOTADataset(config.DATA_ROOT, "train", img_size, transforms=norm),
        batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=custom_collate_fn, pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    val_loader = DataLoader(
        DOTADataset(config.DATA_ROOT, "val", img_size, transforms=norm),
        batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=custom_collate_fn, pin_memory=torch.cuda.is_available(),
    )
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}")

    backbone = GhostTriRemoteXProPP(num_classes=200)
    if os.path.exists(config.PRETRAINED_BACKBONE):
        backbone.load_state_dict(torch.load(config.PRETRAINED_BACKBONE, map_location=device), strict=False)
        print(f"Loaded pretrained backbone from {config.PRETRAINED_BACKBONE}")
    else:
        print(f"[warn] pretrained backbone not found; random init.")

    model = FasterRCNN(backbone, num_classes=NUM_CLASSES).to(device)
    total, trainable = count_parameters(model)
    print(f"Params: total={total:,}  trainable={trainable:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    save_dir = config.ensure_models_dir()
    best_map = 0.0
    for epoch in range(num_epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch + 1)
        val_map, class_aps = evaluate_model(model, val_loader, device, NUM_CLASSES)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{num_epochs}: loss={train_loss:.4f} mAP={val_map:.4f} "
              f"({time.time() - t0:.1f}s)")

        if val_map > best_map:
            best_map = val_map
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                        "best_map": best_map, "class_aps": class_aps},
                       os.path.join(save_dir, "best_faster_rcnn.pth"))
            print(f"  new best mAP={best_map:.4f}")

    print(f"Done. Best mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()
