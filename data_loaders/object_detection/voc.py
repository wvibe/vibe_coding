"""
Pascal VOC dataset loader for object detection
"""

import os
import xml.etree.ElementTree as ET

import torch
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Load environment variables
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT", "/Users/wmu/vibe/data")


class PascalVOCDataset(Dataset):
    """
    Pascal VOC dataset for object detection

    Loads Pascal VOC dataset from local files
    """

    def __init__(self, split="train", transform=None, data_dir=None, year="2012"):
        """
        Initialize Pascal VOC dataset

        Args:
            split: Dataset split ('train', 'val', 'test')
            transform: Optional data augmentation transforms
            data_dir: Directory containing VOC dataset (if None, uses default path)
            year: Dataset year ('2007' or '2012')
        """
        super().__init__()
        self.split = split
        self.transform = transform
        self.year = year

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

        # Class name to index mapping
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # Set default data directory if not provided
        if data_dir is None:
            # Use the correct path for the VOC dataset from environment variable
            self.data_dir = os.path.join(DATA_ROOT, "VOCdevkit", f"VOC{year}")
        else:
            self.data_dir = data_dir

        # Default transformations if none provided
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((416, 416)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        # Load dataset
        self._load_dataset()

    def _load_dataset(self):
        """Load the dataset from disk"""
        # Get list of image IDs from the ImageSets directory
        split_file = os.path.join(
            self.data_dir, "ImageSets", "Main", f"{self.split}.txt"
        )

        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Split file not found: {split_file}. "
                f"Make sure you have downloaded the VOC{self.year} dataset."
            )

        with open(split_file, "r") as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        print(
            f"Loaded {len(self.image_ids)} images for {self.split} split from VOC{self.year}"
        )

    def _parse_voc_xml(self, xml_file):
        """Parse Pascal VOC XML annotation file"""
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
            # Skip difficult objects if desired
            difficult = (
                int(obj.find("difficult").text)
                if obj.find("difficult") is not None
                else 0
            )
            # if difficult == 1 and not self.keep_difficult:
            #     continue

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

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset

        Args:
            idx: Index of the sample

        Returns:
            dict: Sample containing:
                'image': The image tensor
                'boxes': Bounding boxes in [x, y, w, h] format (normalized 0-1)
                'labels': Class labels
                'image_id': Image ID
        """
        image_id = self.image_ids[idx]

        # Load image
        image_path = os.path.join(self.data_dir, "JPEGImages", f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")

        # Load annotations
        annotation_path = os.path.join(self.data_dir, "Annotations", f"{image_id}.xml")
        boxes, labels = self._parse_voc_xml(annotation_path)

        # Handle empty boxes
        if len(boxes) == 0:
            # Create a dummy box to avoid errors
            boxes = [[0.5, 0.5, 0.1, 0.1]]
            labels = [0]  # First class

        # Convert to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Apply transforms to the image
        if self.transform:
            image = self.transform(image)

        return {"image": image, "boxes": boxes, "labels": labels, "image_id": image_id}

    def collate_fn(self, batch):
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
