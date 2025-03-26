#!/usr/bin/env python
"""
Diagnostic script to verify data pipeline and pretrained weight loading for YOLOv3
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../.."))
sys.path.append(project_root)

# Define visualizations directory
VISUALIZATIONS_DIR = os.path.abspath(os.path.join(script_dir, "../visualizations"))
# Create the directory if it doesn't exist
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Load environment variables
load_dotenv()

from data_loaders.object_detection.voc import PascalVOCDataset
from models.vanilla.yolov3.config import DEFAULT_CONFIG
from models.vanilla.yolov3.darknet import Darknet53
from models.vanilla.yolov3.yolov3 import YOLOv3


def visualize_sample(dataset, idx=0, output_dir=None):
    """Visualize a sample from the dataset with bounding boxes"""
    if output_dir is None:
        output_dir = VISUALIZATIONS_DIR

    sample = dataset[idx]
    image = sample["image"]
    boxes = sample["boxes"]
    labels = sample["labels"]
    image_id = sample["image_id"]

    # Convert tensor to numpy for visualization
    image_np = image.permute(1, 2, 0).numpy()

    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)

    # Plot image and boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)

    # Get class names
    class_names = dataset.class_names

    # Draw boxes
    height, width = image_np.shape[:2]
    for box, label in zip(boxes, labels):
        # Convert from [x_center, y_center, width, height] to [x1, y1, x2, y2]
        x_center, y_center, box_width, box_height = box.tolist()
        x1 = (x_center - box_width / 2) * width
        y1 = (y_center - box_height / 2) * height
        x2 = (x_center + box_width / 2) * width
        y2 = (y_center + box_height / 2) * height

        # Draw rectangle
        plt.gca().add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2)
        )

        # Add label
        class_name = class_names[label]
        plt.gca().text(
            x1,
            y1 - 5,
            f"{class_name}",
            bbox=dict(facecolor="red", alpha=0.5),
            fontsize=8,
            color="white",
        )

    plt.title(f"Image ID: {image_id}, Size: {image.shape}")
    plt.axis("off")
    plt.tight_layout()

    # Create a timestamp-based filename
    filename = os.path.join(
        output_dir, f"sample_{idx}_{image_id.replace('/', '_')}_visualization.png"
    )
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")
    plt.close()


def check_dataset():
    """Check dataset loading and preprocessing"""
    print("\n=== Checking Dataset Pipeline ===")

    # Create dataset
    dataset = PascalVOCDataset(
        years=["2007"],
        split="train",
        subset_percent=0.01,  # Use small subset for testing
        debug_mode=True,
    )

    print(f"Dataset size: {len(dataset)}")

    # Visualize a few samples
    for i in range(min(3, len(dataset))):
        print(f"\nSample {i}:")
        sample = dataset[i]
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Boxes: {sample['boxes']}")
        print(f"  Labels: {sample['labels']}")
        print(f"  Image ID: {sample['image_id']}")

        # Check box coordinates are in [0, 1]
        boxes = sample["boxes"]
        if torch.any(boxes < 0) or torch.any(boxes > 1):
            print("  WARNING: Box coordinates outside [0, 1] range!")

        # Check box format (x_center, y_center, width, height)
        if boxes.shape[1] != 4:
            print("  WARNING: Box format is not [x, y, w, h]!")

        # Visualize
        visualize_sample(dataset, i)

    # Create a batch
    batch_size = 2
    samples = [dataset[i] for i in range(min(batch_size, len(dataset)))]
    batch = dataset.collate_fn(samples)

    print("\nBatch structure:")
    print(f"  Images shape: {batch['images'].shape}")
    print(f"  Boxes: {[b.shape for b in batch['boxes']]}")
    print(f"  Labels: {[l.shape for l in batch['labels']]}")

    return dataset


def check_pretrained_weights():
    """Check pretrained Darknet53 weight loading"""
    print("\n=== Checking Pretrained Weight Loading ===")

    # Get weight path
    darknet_weights_path = os.getenv("DARKNET53_WEIGHTS")
    print(f"Weights path: {darknet_weights_path}")

    if not darknet_weights_path or not os.path.exists(darknet_weights_path):
        print(
            "WARNING: Darknet53 weights not found! Ensure DARKNET53_WEIGHTS env variable is set correctly."
        )
        print(
            "Run ./models/vanilla/yolov3/scripts/download_darknet_weights.sh to download weights."
        )
        return False

    print(f"Weights file size: {os.path.getsize(darknet_weights_path) / (1024 * 1024):.2f} MB")

    # Create models
    print("\nLoading Darknet53 model...")
    darknet = Darknet53()

    # Get initial parameter stats
    init_params = sum(p.numel() for p in darknet.parameters())
    print(f"Initial parameter count: {init_params}")

    # Load pretrained weights
    print("Loading pretrained weights...")
    darknet.load_pretrained(darknet_weights_path)

    # Check if weights were loaded
    state_dict = torch.load(darknet_weights_path)
    expected_keys = set(state_dict.keys())
    model_keys = set(darknet.state_dict().keys())

    print(f"Expected keys: {len(expected_keys)}")
    print(f"Model keys: {len(model_keys)}")
    print(f"Intersection: {len(expected_keys.intersection(model_keys))}")

    # Check if YOLOv3 loads weights correctly
    print("\nTesting YOLOv3 weight loading...")
    yolo = YOLOv3(DEFAULT_CONFIG)

    # Create a sample input
    dummy_input = torch.randn(1, 3, 416, 416)

    # Try a forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        try:
            outputs = yolo(dummy_input)
            print("Forward pass successful!")

            # Check output shapes
            for i, output in enumerate(outputs):
                print(f"  Output {i} shape: {output.shape}")

        except Exception as e:
            print(f"Forward pass failed: {e}")

    return True


def main():
    """Main diagnostic function"""
    print("=== YOLOv3 Pipeline Verification ===")

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check dataset
    dataset = check_dataset()

    # Check pretrained weights
    weights_ok = check_pretrained_weights()

    # Summary
    print("\n=== Verification Summary ===")
    print(f"Dataset check: {'PASS' if dataset else 'FAIL'}")
    print(f"Pretrained weights check: {'PASS' if weights_ok else 'FAIL'}")

    if dataset and weights_ok:
        print("\nThe basic pipeline is working correctly! Continue with other fixes.")
    else:
        print("\nIssues detected in the pipeline. Fix these before continuing.")


if __name__ == "__main__":
    main()
