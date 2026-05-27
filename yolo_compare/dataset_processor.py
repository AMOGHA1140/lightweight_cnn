"""Convert the DOTA dataset into YOLO (axis-aligned HBB) format on disk.

Used only by the YOLO benchmarking pipeline. DOTA's 8-point oriented labels are
collapsed to their enclosing horizontal box and written as normalised YOLO
``<class> <xc> <yc> <w> <h>`` annotations, with images copied into the standard
``images/`` + ``labels/`` layout plus a ``dataset.yaml``.
"""

import os
import shutil
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

from common.classes import DOTA_CLASSES


class DOTADatasetProcessor:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.yolo_dataset_path = self.dataset_path / "yolo_format"
        self.class_names = list(DOTA_CLASSES)

    def convert_dota_to_yolo_format(self, txt_file, img_width, img_height):
        yolo_annotations = []
        if not os.path.exists(txt_file):
            return yolo_annotations
        with open(txt_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            try:
                coords = [float(x) for x in parts[:8]]
            except ValueError:
                continue  # header line (imagesource:/gsd:)
            class_name = parts[8]
            if class_name not in self.class_names:
                continue
            class_id = self.class_names.index(class_name)
            x_coords, y_coords = coords[::2], coords[1::2]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            # Clamp to [0, 1].
            x_center, y_center = min(max(x_center, 0), 1), min(max(y_center, 0), 1)
            width, height = min(max(width, 0), 1), min(max(height, 0), 1)
            yolo_annotations.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
        return yolo_annotations

    @staticmethod
    def get_image_dimensions(img_path):
        with Image.open(img_path) as img:
            return img.size  # (width, height)

    def process_split(self, split_name):
        print(f"Processing {split_name} split...")
        split_path = self.dataset_path / split_name
        images_path = split_path / "images"
        labels_path = split_path / "labelTxt-v1.0" / "labelTxt"
        yolo_images_path = self.yolo_dataset_path / split_name / "images"
        yolo_labels_path = self.yolo_dataset_path / split_name / "labels"
        yolo_images_path.mkdir(parents=True, exist_ok=True)
        yolo_labels_path.mkdir(parents=True, exist_ok=True)

        image_files = list(images_path.glob("*.png")) + list(images_path.glob("*.jpg"))
        for img_file in tqdm(image_files, desc=f"Processing {split_name}"):
            shutil.copy2(img_file, yolo_images_path / img_file.name)
            img_width, img_height = self.get_image_dimensions(img_file)
            txt_file = labels_path / f"{img_file.stem}.txt"
            yolo_annotations = self.convert_dota_to_yolo_format(txt_file, img_width, img_height)
            with open(yolo_labels_path / f"{img_file.stem}.txt", "w") as f:
                f.write("\n".join(yolo_annotations))

    def process_test_split(self):
        print("Processing test split...")
        test_images_path = self.dataset_path / "test" / "images"
        yolo_test_path = self.yolo_dataset_path / "test" / "images"
        yolo_test_path.mkdir(parents=True, exist_ok=True)
        image_files = list(test_images_path.glob("*.png")) + list(test_images_path.glob("*.jpg"))
        for img_file in tqdm(image_files, desc="Processing test"):
            shutil.copy2(img_file, yolo_test_path / img_file.name)

    def create_yaml_config(self):
        config = {
            "path": str(self.yolo_dataset_path.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(self.class_names),
            "names": self.class_names,
        }
        yaml_path = self.yolo_dataset_path / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        return str(yaml_path)

    def prepare_dataset(self):
        print("Preparing DOTA dataset for YOLO training...")
        if self.yolo_dataset_path.exists():
            print(f"YOLO dataset already exists at: {self.yolo_dataset_path}")
            yaml_path = self.yolo_dataset_path / "dataset.yaml"
            if yaml_path.exists():
                print("Using existing dataset configuration")
                return str(yaml_path)
            print("Dataset exists but config missing, recreating config...")
            return self.create_yaml_config()

        self.process_split("train")
        self.process_split("val")
        self.process_test_split()
        yaml_path = self.create_yaml_config()
        print(f"Dataset prepared at: {self.yolo_dataset_path}")
        return yaml_path
