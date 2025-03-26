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

    # Class names for Pascal VOC
    CLASS_NAMES = [
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

    def __init__(
        self,
        years: List[str] = ["2007"],
        split_file: str = "train.txt",
        transform: Optional[A.Compose] = None,
        data_dir: str = get_voc_root(),
        sample_pct: Optional[float] = None,
        img_size: int = 416,
    ):
        """
        Initialize Pascal VOC dataset

        Args:
            years: List of dataset years to load (e.g., ["2007"] or ["2007", "2012"])
            split_file: Filename in ImageSets/Main directory (e.g., "train.txt", "person_train.txt")
            transform: Optional custom data augmentation transforms
            data_dir: Root directory containing VOC datasets (defaults to VOC_ROOT from .env)
            sample_pct: Optional percentage (0.0-1.0) of dataset to use
            img_size: Size of output images (default: 416 for YOLOv3)
        """
        super().__init__()
        self.years = years
        self.split_file = split_file
        self.custom_transform = transform
        self.data_dir = data_dir
        self.sample_pct = sample_pct
        self.img_size = img_size

        # Setup class mapping
        self.num_classes = len(self.CLASS_NAMES)
        self.class_to_idx = {name: i for i, name in enumerate(self.CLASS_NAMES)}

        # Set up transforms
        self._setup_transforms()

        # Load dataset
        self.image_info = self._load_dataset()
        print(f"Total images loaded: {len(self.image_info)}")

    def _setup_transforms(self):
        """Setup image transformations based on split"""
        if self.custom_transform is not None:
            self.transform = self.custom_transform
        elif "train" in self.split_file:
            # Training transforms with augmentation
            self.transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=self.img_size),
                    A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=0),
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
                    A.LongestMaxSize(max_size=self.img_size),
                    A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=0),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]),
            )

    def _load_dataset(self) -> List[Dict[str, str]]:
        """
        Load the dataset from disk for all specified years

        Returns:
            List of dicts containing image info (year, id, image_path, annotation_path)
        """
        image_info = []

        for year in self.years:
            voc_dir = os.path.join(self.data_dir, f"VOC{year}")
            split_file_path = os.path.join(voc_dir, "ImageSets", "Main", self.split_file)

            if not os.path.exists(split_file_path):
                raise FileNotFoundError(
                    f"Split file not found: {split_file_path}. "
                    f"Make sure you have downloaded the VOC{year} dataset."
                )

            # Load image IDs from split file
            image_ids = self._load_image_ids_from_file(split_file_path)

            # Apply sampling if requested
            if self.sample_pct is not None:
                image_ids = self._sample_ids(image_ids, year)

            # Log dataset size
            print(f"Loaded {len(image_ids)} images from VOC{year} split file {self.split_file}")

            # Create image info entries
            for image_id in image_ids:
                info = {
                    "year": year,
                    "id": image_id,
                    "image_path": os.path.join(voc_dir, "JPEGImages", f"{image_id}.jpg"),
                    "annotation_path": os.path.join(voc_dir, "Annotations", f"{image_id}.xml"),
                }
                image_info.append(info)

        return image_info

    def _load_image_ids_from_file(self, split_file_path: str) -> List[str]:
        """
        Load image IDs from a split file, handling different formats

        Args:
            split_file_path: Path to the split file

        Returns:
            List of image IDs
        """
        with open(split_file_path, "r") as f:
            lines = f.readlines()

        # Different file formats
        if len(lines) > 0 and len(lines[0].strip().split()) > 1:
            # Format: "<image_id> <label>" (e.g., class-specific files)
            return [
                line.strip().split()[0]
                for line in lines
                if len(line.strip().split()) > 1 and int(line.strip().split()[1]) == 1
            ]
        else:
            # Format: "<image_id>" (standard split files)
            return [line.strip() for line in lines]

    def _sample_ids(self, image_ids: List[str], year: str) -> List[str]:
        """
        Sample a percentage of image IDs

        Args:
            image_ids: List of all image IDs
            year: Dataset year (used for reproducible sampling)

        Returns:
            Sampled list of image IDs
        """
        # Validate percentage is between 0 and 1
        pct = self.sample_pct
        if pct is not None and (pct < 0.0 or pct > 1.0):
            raise ValueError("sample_pct must be between 0.0 and 1.0")

        # Calculate number of samples
        num_samples = max(1, int(len(image_ids) * pct))

        # Use a fixed seed for reproducibility, but allow different subsets for different years
        rng = random.Random(hash(f"{year}_{self.split_file}_{pct}"))
        sampled_ids = rng.sample(image_ids, num_samples)

        print(f"Sampled {num_samples} images ({pct * 100:.1f}%) from {len(image_ids)} total")
        return sampled_ids

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
            name_elem = obj.find("name")
            # For compatibility with tests that might use <n> instead of <name>
            if name_elem is None:
                name_elem = obj.find("n")

            if name_elem is None:
                continue  # Skip objects without a name tag

            name = name_elem.text
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
