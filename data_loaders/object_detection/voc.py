"""
Pascal VOC dataset loader for object detection
"""

import os
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
    data_root = os.getenv("DATA_ROOT", "data")
    voc_root = os.getenv("VOC_ROOT")
    if voc_root is None:
        voc_root = os.path.join(data_root, "VOCdevkit")
    return voc_root


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
    ):
        """
        Initialize Pascal VOC dataset

        Args:
            years: List of dataset years to load (e.g., ["2007"] or ["2007", "2012"])
            split: Dataset split ('train', 'val', 'test', 'trainval')
            transform: Optional custom data augmentation transforms
            data_dir: Root directory containing VOC datasets (defaults to VOC_ROOT from .env)
        """
        super().__init__()
        self.years = years
        self.split = split
        self.custom_transform = transform
        self.data_dir = data_dir

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
            split_file = os.path.join(voc_dir, "ImageSets", "Main", f"{self.split}.txt")

            if not os.path.exists(split_file):
                raise FileNotFoundError(
                    f"Split file not found: {split_file}. "
                    f"Make sure you have downloaded the VOC{year} dataset."
                )

            with open(split_file, "r") as f:
                image_ids = [line.strip() for line in f.readlines()]

            for image_id in image_ids:
                info = {
                    "year": year,
                    "id": image_id,
                    "image_path": os.path.join(
                        voc_dir, "JPEGImages", f"{image_id}.jpg"
                    ),
                    "annotation_path": os.path.join(
                        voc_dir, "Annotations", f"{image_id}.xml"
                    ),
                }
                image_info.append(info)

            print(f"Loaded {len(image_ids)} images from VOC{year} {self.split} split")

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

    def collate_fn(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
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
