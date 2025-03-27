"""
Dummy dataset for object detection model training/testing

This module provides a dummy dataset that generates random images and bounding boxes
for testing detection models without requiring actual datasets.
"""

import random

import torch
from torch.utils.data import Dataset


class DummyDetectionDataset(Dataset):
    """
    A dummy dataset for object detection that generates random data
    for testing model training pipeline without requiring the actual VOC dataset.
    """

    def __init__(self, num_samples=100, img_size=416, num_classes=20):
        """
        Initialize dummy dataset

        Args:
            num_samples: Number of samples to generate
            img_size: Size of generated images
            num_classes: Number of object classes
        """
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        # For compatibility with real datasets
        self.class_names = [f"class_{i}" for i in range(num_classes)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Generate a random image with random boxes and labels"""
        # Create a random RGB image
        img = torch.rand(3, self.img_size, self.img_size)

        # Create random number of boxes (0 to 5)
        num_boxes = random.randint(0, 5)

        if num_boxes == 0:
            # No objects
            boxes = torch.zeros((0, 4))
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            # Generate random boxes in normalized coordinates [0, 1]
            boxes = []
            for _ in range(num_boxes):
                # Ensure minimum size to avoid degenerate boxes
                w = random.uniform(0.1, 0.3)
                h = random.uniform(0.1, 0.3)

                # Random center position
                cx = random.uniform(w / 2, 1 - w / 2)
                cy = random.uniform(h / 2, 1 - h / 2)

                # Convert to [x1, y1, x2, y2] format
                x1 = max(0, cx - w / 2)
                y1 = max(0, cy - h / 2)
                x2 = min(1, cx + w / 2)
                y2 = min(1, cy + h / 2)

                boxes.append([x1, y1, x2, y2])

            boxes = torch.tensor(boxes, dtype=torch.float32)

            # Generate random labels
            labels = torch.randint(0, self.num_classes, (num_boxes,))

        return img, boxes, labels

    def collate_fn(self, batch):
        """Custom collate function to handle variable number of objects"""
        images = []
        boxes = []
        labels = []

        for img, box, label in batch:
            images.append(img)
            boxes.append(box)
            labels.append(label)

        images = torch.stack(images, dim=0)

        return {
            "images": images,
            "boxes": boxes,
            "labels": labels,
            "image_ids": [str(i) for i in range(len(images))],  # Add image_ids for compatibility
        }
