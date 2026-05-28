"""DOTA dataset for the one-stage oriented (OBB) detector.

Parses DOTA's 8-coordinate rotated quadrilateral labels and converts each to the
5-parameter oriented box ``(cx, cy, w, h, angle)`` (angle in radians). Images are
resized to ``img_size`` and ImageNet-normalised, and box coordinates are rescaled
by the same per-axis factors so they match the resized image.
"""

import math
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from common.classes import CLASS2ID
from common.constants import IMAGENET_MEAN, IMAGENET_STD


class DOTADataset(Dataset):
    def __init__(self, root, split="train", img_size=1024,
                 norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD):
        self.img_size = img_size
        self.split = split
        if split == "train":
            self.img_dir = os.path.join(root, "train/images")
            self.label_dir = os.path.join(root, "train/labelTxt-v1.0/labelTxt")
        elif split == "val":
            self.img_dir = os.path.join(root, "val/images")
            self.label_dir = os.path.join(root, "val/labelTxt-v1.0/labelTxt")
        else:
            self.img_dir = os.path.join(root, "test/images")
            self.label_dir = None  # No labels for test

        self.img_files = sorted(
            f for f in os.listdir(self.img_dir) if f.endswith((".png", ".jpg"))
        )
        if self.label_dir:
            self.label_files = sorted(
                f for f in os.listdir(self.label_dir) if f.endswith(".txt")
            )
            assert len(self.img_files) == len(self.label_files), \
                "Mismatch between images and labels"

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        pil_image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size
        sx = self.img_size / orig_w
        sy = self.img_size / orig_h
        image = self.transform(pil_image)

        boxes, labels = [], []
        if self.label_dir:
            label_path = os.path.join(self.label_dir, self.label_files[idx])
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue
                    coords = list(map(float, parts[:8]))
                    class_name = parts[8]
                    if class_name not in CLASS2ID:
                        continue
                    # Rescale 8-point coords to the resized image frame.
                    xs = [coords[i] * sx for i in range(0, 8, 2)]
                    ys = [coords[i] * sy for i in range(1, 8, 2)]
                    cx, cy = np.mean(xs), np.mean(ys)
                    w = math.hypot(xs[1] - xs[0], ys[1] - ys[0])
                    h = math.hypot(xs[2] - xs[1], ys[2] - ys[1])
                    angle = math.atan2(ys[1] - ys[0], xs[1] - xs[0])
                    boxes.append([cx, cy, w, h, angle])
                    labels.append(CLASS2ID[class_name])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)
        return image, {"boxes": boxes, "labels": labels}


def collate_fn(batch):
    """Stack images, keep per-image targets as a list (variable #objects)."""
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
