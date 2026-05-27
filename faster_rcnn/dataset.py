"""DOTA dataset for the (experimental) axis-aligned two-stage detector.

This pipeline works with HORIZONTAL bounding boxes ``[x1, y1, x2, y2]`` (not OBB).
Images are loaded with OpenCV, letterbox-resized to a square ``img_size`` and
padded; boxes are scaled by the same factor.
"""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common.classes import CLASS2ID


def parse_dota_annotation(ann_path):
    """Parse a DOTA labelTxt file into ``[{'bbox': [x1,y1,x2,y2], 'category': id}, ...]``.

    Skips header lines such as ``imagesource:`` / ``gsd:`` and unknown class names.
    """
    objects = []
    with open(ann_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            try:
                coords = [float(x) for x in parts[:8]]
            except ValueError:
                continue  # header line
            class_name = parts[8]
            if class_name not in CLASS2ID:
                continue
            xs, ys = coords[0::2], coords[1::2]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            objects.append({"bbox": bbox, "category": CLASS2ID[class_name]})
    return objects


class DOTADataset(Dataset):
    def __init__(self, data_root, split="train", img_size=800, transforms=None):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.transforms = transforms

        self.img_dir = os.path.join(data_root, split, "images")
        if split == "train":
            self.ann_dir = os.path.join(data_root, split, "labelTxt-v1.0", "labelTxt")
        elif split == "val":
            self.ann_dir = os.path.join(data_root, split, "labelTxt-v1.0")
        else:  # test
            self.ann_dir = None

        self.img_files = []
        if os.path.exists(self.img_dir):
            for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                self.img_files.extend(
                    f for f in os.listdir(self.img_dir) if f.lower().endswith(ext)
                )
        self.img_files.sort()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Letterbox resize (preserve aspect ratio) then pad to square.
        h, w = image.shape[:2]
        scale = self.img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
        image = np.pad(image, ((0, self.img_size - new_h), (0, self.img_size - new_w), (0, 0)),
                       mode="constant")

        targets = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)}
        if self.ann_dir and self.split != "test":
            ann_path = os.path.join(self.ann_dir, os.path.splitext(img_name)[0] + ".txt")
            if os.path.exists(ann_path):
                boxes, labels = [], []
                for obj in parse_dota_annotation(ann_path):
                    x1, y1, x2, y2 = obj["bbox"]
                    boxes.append([x1 * scale, y1 * scale, x2 * scale, y2 * scale])
                    labels.append(obj["category"])
                if boxes:
                    targets["boxes"] = torch.tensor(boxes, dtype=torch.float32)
                    targets["labels"] = torch.tensor(labels, dtype=torch.long)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if self.transforms:
            image = self.transforms(image)
        return image, targets


def custom_collate_fn(batch):
    """Stack images; keep targets as a list (variable #objects per image)."""
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)
