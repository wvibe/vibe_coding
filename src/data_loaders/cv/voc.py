"""
Pascal VOC dataset loader for object detection
"""

import os
import random
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Dataset

# Load environment variables
load_dotenv()


def get_voc_root() -> str:
    """Get VOC dataset root path from environment variables"""
    return os.getenv("VOC_ROOT", os.path.join(os.getenv("DATA_ROOT", "data"), "VOCdevkit"))


class PascalVOCDataset(Dataset):
    """
    Pascal VOC dataset for object detection

    Loads Pascal VOC dataset from local files
    """

    def __init__(
        self,
        years: List[str] = ["2007"],
        split: str = "train",
        transform: Optional[A.Compose] = None,
        data_dir: str = get_voc_root(),
        subset_percent: Optional[float] = None,
        class_name: Optional[str] = None,
        debug_mode: bool = False,
    ):
        """
        Initialize Pascal VOC dataset

        Args:
            years: List of dataset years to load (e.g., ["2007"] or ["2007", "2012"])
            split: Dataset split ('train', 'val', 'test', 'trainval')
            transform: Optional custom data augmentation transforms
            data_dir: Root directory containing VOC datasets (defaults to VOC_ROOT from .env)
            subset_percent: Optional percentage (0.01 to 1.0) of dataset to use (for debug)
            class_name: Optional class name to load only images with a specific class
            debug_mode: Enable debug mode (logs additional information)
        """
        super().__init__()
        self.years = years
        self.split = split
        self.custom_transform = transform
        self.data_dir = data_dir
        self.subset_percent = subset_percent
        self.class_name = class_name
        self.debug_mode = debug_mode

        # Class names for Pascal VOC
        self.class_names = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
        self.num_classes = len(self.class_names)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # Validate class_name if provided
        if self.class_name is not None and self.class_name not in self.class_names:
            raise ValueError(
                f"Invalid class name: {self.class_name}. Available classes: {self.class_names}"
            )

        # Set up transforms based on split
        if self.custom_transform is not None:
            self.transform = self.custom_transform
        elif split in ["train", "trainval"]:
            # Training transforms with augmentation
            self.transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=416),
                    A.PadIfNeeded(min_height=416, min_width=416, border_mode=0),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.ColorJitter(brightness=0.1, contrast=0.1, p=0.3),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]),
            )
        else:
            # Validation/Test transforms without augmentation
            self.transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=416),
                    A.PadIfNeeded(min_height=416, min_width=416, border_mode=0),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]),
            )

        # Load dataset
        self.image_info = self._load_dataset()

    def _load_dataset(self) -> List[Dict[str, str]]:
        """
        Load the dataset from disk for all specified years

        Returns:
            List of dicts containing image info (year, id, image_path, annotation_path)
        """
        image_info = []

        for year in self.years:
            voc_dir = os.path.join(self.data_dir, f"VOC{year}")

            # Determine which file to load based on whether we're loading a class-specific dataset
            if self.class_name is not None:
                # Load class-specific split file
                split_file = os.path.join(
                    voc_dir, "ImageSets", "Main", f"{self.class_name}_{self.split}.txt"
                )
                if not os.path.exists(split_file):
                    raise FileNotFoundError(
                        f"Class-specific split file not found: {split_file}. "
                        f"Make sure you have downloaded the VOC{year} dataset "
                        f"and that the class {self.class_name} has a {self.split} split."
                    )

                # Class-specific files have format: "<image_id> <label>" where label is -1, 0, or 1
                # 1 means the object is present, -1 means the object is difficult, 0 means not present
                image_ids = []
                with open(split_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            image_id, label = parts[0], int(parts[1])
                            # Only include images where the class is present (label == 1)
                            if label == 1:
                                image_ids.append(image_id)

                if self.debug_mode:
                    print(
                        f"Loaded {len(image_ids)} images with class '{self.class_name}' from VOC{year} {self.split}"
                    )
            else:
                # Load standard split file
                split_file = os.path.join(voc_dir, "ImageSets", "Main", f"{self.split}.txt")
                if not os.path.exists(split_file):
                    raise FileNotFoundError(
                        f"Split file not found: {split_file}. "
                        f"Make sure you have downloaded the VOC{year} dataset."
                    )

                with open(split_file, "r") as f:
                    image_ids = [line.strip() for line in f.readlines()]

            # Apply subset percentage if specified
            if self.subset_percent is not None:
                if not 0.01 <= self.subset_percent <= 1.0:
                    raise ValueError("subset_percent must be between 0.01 and 1.0")

                num_samples = max(1, int(len(image_ids) * self.subset_percent))
                # Use a fixed seed for reproducibility, but allow different subsets for different years/splits
                rng = random.Random(hash(f"{year}_{self.split}_{self.subset_percent}"))
                image_ids = rng.sample(image_ids, num_samples)

                if self.debug_mode:
                    print(
                        f"Using {num_samples} images ({self.subset_percent * 100:.1f}%) from VOC{year} {self.split}"
                    )

            for image_id in image_ids:
                info = {
                    "year": year,
                    "id": image_id,
                    "image_path": os.path.join(voc_dir, "JPEGImages", f"{image_id}.jpg"),
                    "annotation_path": os.path.join(voc_dir, "Annotations", f"{image_id}.xml"),
                }
                image_info.append(info)

            if (
                self.debug_mode
                or len(self.years) > 1
                or self.class_name is not None
                or self.subset_percent is not None
            ):
                print(
                    f"Loaded {len(image_ids)} images from VOC{year} {self.split} split"
                    + (f" for class '{self.class_name}'" if self.class_name else "")
                    + (f" ({self.subset_percent * 100:.1f}% subset)" if self.subset_percent else "")
                )

        print(f"Total images loaded: {len(image_info)}")
        return image_info

    def _parse_voc_xml(self, xml_file: str) -> Tuple[List[List[float]], List[int]]:
        """
        Parse Pascal VOC XML annotation file

        Args:
            xml_file: Path to XML annotation file

        Returns:
            Tuple of (boxes, labels) where:
                boxes: List of [x_center, y_center, width, height] (normalized)
                labels: List of class indices
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get image size
        size = root.find("size")
        width = float(size.find("width").text)
        height = float(size.find("height").text)

        boxes = []
        labels = []

        # Process each object
        for obj in root.findall("object"):
            # Get class name and convert to index
            name = obj.find("name").text
            if name not in self.class_to_idx:
                continue  # Skip objects with unknown classes

            # If we're loading a class-specific dataset, skip objects that aren't of that class
            if self.class_name is not None and name != self.class_name:
                continue

            label = self.class_to_idx[name]

            # Get bounding box
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # Normalize to [0, 1]
            xmin_norm = xmin / width
            ymin_norm = ymin / height
            xmax_norm = xmax / width
            ymax_norm = ymax / height

            # Convert to [x_center, y_center, width, height]
            x_center = (xmin_norm + xmax_norm) / 2
            y_center = (ymin_norm + ymax_norm) / 2
            box_width = xmax_norm - xmin_norm
            box_height = ymax_norm - ymin_norm

            boxes.append([x_center, y_center, box_width, box_height])
            labels.append(label)

        return boxes, labels

    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.image_info)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset

        Args:
            idx: Index of the sample

        Returns:
            dict: Sample containing:
                'image': The image tensor
                'boxes': Bounding boxes in [x, y, w, h] format (normalized 0-1)
                'labels': Class labels
                'image_id': Image ID in format 'YEAR/ID' (e.g., '2007/000001')
        """
        info = self.image_info[idx]

        # Load image
        image = Image.open(info["image_path"]).convert("RGB")
        image_np = np.array(image)

        # Load annotations
        boxes, labels = self._parse_voc_xml(info["annotation_path"])

        # Handle empty boxes
        if len(boxes) == 0:
            boxes = [[0.5, 0.5, 0.1, 0.1]]
            labels = [0]  # First class

        # Apply transforms using albumentations
        transformed = self.transform(image=image_np, bboxes=boxes, labels=labels)

        # Get transformed data
        image = transformed["image"]  # Already a tensor from ToTensorV2
        boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        labels = torch.tensor(transformed["labels"], dtype=torch.int64)

        # Create image_id in format 'YEAR/ID'
        image_id = f"{info['year']}/{info['id']}"

        return {"image": image, "boxes": boxes, "labels": labels, "image_id": image_id}

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for batching samples

        Args:
            batch: List of samples

        Returns:
            dict: Batched samples
        """
        images = []
        boxes = []
        labels = []
        image_ids = []

        for sample in batch:
            images.append(sample["image"])
            boxes.append(sample["boxes"])
            labels.append(sample["labels"])
            image_ids.append(sample["image_id"])

        # Stack images into a batch
        images = torch.stack(images, dim=0)

        return {
            "images": images,
            "boxes": boxes,
            "labels": labels,
            "image_ids": image_ids,
        }
