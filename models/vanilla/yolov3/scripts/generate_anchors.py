#!/usr/bin/env python
"""
Generate custom anchor boxes for YOLOv3 using k-means clustering on VOC2007 dataset
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from tqdm import tqdm

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../.."))
sys.path.append(project_root)

# Define the anchors directory
ANCHORS_DIR = os.path.abspath(os.path.join(script_dir, "../anchors"))
# Create the directory if it doesn't exist
os.makedirs(ANCHORS_DIR, exist_ok=True)

# Load environment variables
load_dotenv()

from data_loaders.object_detection.voc import PascalVOCDataset


def iou(box, clusters):
    """
    Calculate IoU between a box and clusters

    Args:
        box: Single box [w, h]
        clusters: Array of clusters [N, 2] where N is number of clusters

    Returns:
        Array of IoUs with each cluster
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou = intersection / (box_area + cluster_area - intersection + 1e-10)
    return iou


def avg_iou(boxes, clusters):
    """
    Calculate average IoU between boxes and their closest clusters

    Args:
        boxes: Array of boxes [N, 2]
        clusters: Array of clusters [k, 2]

    Returns:
        Average IoU
    """
    sum_iou = 0.0
    for box in boxes:
        sum_iou += np.max(iou(box, clusters))
    return sum_iou / boxes.shape[0]


def kmeans_anchors(boxes, k, iterations=100, method="kmeans"):
    """
    Run k-means clustering on box dimensions to find anchor boxes

    Args:
        boxes: Array of boxes [N, 2] (width, height)
        k: Number of clusters (anchors)
        iterations: Maximum iterations for k-means
        method: 'kmeans' for standard sklearn, 'iou' for IoU-based distance

    Returns:
        Array of anchor boxes [k, 2]
    """
    # Convert to numpy array if not already
    boxes = np.array(boxes)

    # Get number of boxes
    n_boxes = boxes.shape[0]

    if method == "kmeans":
        # Use sklearn's KMeans implementation (faster)
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=iterations)
        kmeans.fit(boxes)
        clusters = kmeans.cluster_centers_
    else:
        # Initialize clusters randomly
        clusters = boxes[np.random.choice(n_boxes, k, replace=False)]

        # Store old clusters for convergence check
        old_clusters = np.zeros((k, 2))

        # Run k-means with IoU as distance metric
        for _ in tqdm(range(iterations), desc="K-means clustering"):
            # Calculate IoU between boxes and clusters
            distances = []
            for box in boxes:
                distances.append(1 - iou(box, clusters))
            distances = np.array(distances)

            # Assign boxes to closest cluster
            assignments = np.argmin(distances, axis=1)

            # Save old clusters for convergence check
            old_clusters = np.copy(clusters)

            # Update clusters
            for j in range(k):
                cluster_boxes = boxes[assignments == j]
                if len(cluster_boxes) > 0:
                    clusters[j] = np.mean(cluster_boxes, axis=0)

            # Check for convergence
            if np.sum(np.abs(old_clusters - clusters)) < 1e-6:
                break

    # Sort clusters by area
    areas = clusters[:, 0] * clusters[:, 1]
    sorted_indices = np.argsort(areas)
    clusters = clusters[sorted_indices]

    return clusters


def collect_boxes_from_dataset(dataset):
    """
    Collect all bounding boxes from a dataset

    Args:
        dataset: PascalVOCDataset object

    Returns:
        Array of boxes [N, 2] (width, height) normalized to 416x416
    """
    all_boxes = []

    print(f"Collecting boxes from {len(dataset)} images...")
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        boxes = sample["boxes"]

        # Skip dummy boxes [0.5, 0.5, 0.1, 0.1]
        for box in boxes:
            # Extract width and height (normalized)
            _, _, w, h = box.tolist()

            # Convert to absolute pixels in 416x416 image
            w_abs = w * 416
            h_abs = h * 416

            # Skip very small boxes (likely errors or difficult objects)
            if w_abs > 1 and h_abs > 1:
                all_boxes.append([w_abs, h_abs])

    return np.array(all_boxes)


def visualize_anchors(boxes, anchors, dataset_name="voc", output_path=None):
    """
    Visualize original boxes and generated anchors

    Args:
        boxes: Array of boxes [N, 2] (width, height)
        anchors: Array of anchor boxes [k, 2]
        dataset_name: Name of the dataset (for file naming)
        output_path: Path to save visualization (if None, use ANCHORS_DIR)
    """
    if output_path is None:
        output_path = os.path.join(ANCHORS_DIR, f"{dataset_name}_anchor_visualization.png")

    plt.figure(figsize=(10, 10))

    # Plot all boxes
    plt.scatter(boxes[:, 0], boxes[:, 1], s=1, c="blue", alpha=0.1, label="Ground truth boxes")

    # Plot anchors
    plt.scatter(anchors[:, 0], anchors[:, 1], s=200, c="red", marker="x", label="Anchors")

    # Add box around each anchor
    for i, anchor in enumerate(anchors):
        w, h = anchor
        plt.gca().add_patch(
            plt.Rectangle((w - w / 2, h - h / 2), w, h, fill=False, edgecolor="red", linewidth=2)
        )
        plt.text(w, h + 10, f"({w:.1f}, {h:.1f})", ha="center")

    # Add labels and title
    plt.xlabel("Width (pixels in 416x416 image)")
    plt.ylabel("Height (pixels in 416x416 image)")
    plt.title(f"YOLOv3 Anchor Boxes for {dataset_name.upper()} Dataset")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path)
    print(f"Saved anchor visualization to {output_path}")

    # Additional visualization: width vs height distribution
    dist_output_path = os.path.join(ANCHORS_DIR, f"{dataset_name}_box_distribution.png")
    plt.figure(figsize=(10, 10))

    # Plot distribution
    plt.hexbin(boxes[:, 0], boxes[:, 1], gridsize=50, cmap="Blues", bins="log")

    # Plot anchors
    plt.scatter(anchors[:, 0], anchors[:, 1], s=200, c="red", marker="x")

    # Add labels
    for i, anchor in enumerate(anchors):
        w, h = anchor
        plt.text(w, h + 10, f"{i + 1}: ({w:.1f}, {h:.1f})", ha="center")

    plt.colorbar(label="log10(N)")
    plt.xlabel("Width (pixels in 416x416 image)")
    plt.ylabel("Height (pixels in 416x416 image)")
    plt.title(f"{dataset_name.upper()} Bounding Box Distribution and Anchors")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig(dist_output_path)
    print(f"Saved box distribution to {dist_output_path}")


def group_anchors_by_scale(anchors, num_scales=3):
    """
    Group anchors by scale for YOLOv3's multi-scale detection

    Args:
        anchors: Array of anchor boxes [9, 2]
        num_scales: Number of scales (3 for YOLOv3)

    Returns:
        List of grouped anchors by scale
    """
    # Sort anchors by area
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_indices = np.argsort(areas)
    sorted_anchors = anchors[sorted_indices]

    # Group anchors by scale (YOLOv3 uses 3 anchors per scale)
    anchors_per_scale = len(anchors) // num_scales
    grouped_anchors = []

    for i in range(num_scales):
        start_idx = i * anchors_per_scale
        end_idx = start_idx + anchors_per_scale
        scale_anchors = sorted_anchors[start_idx:end_idx]
        grouped_anchors.append(scale_anchors)

    return grouped_anchors


def format_for_config(anchors):
    """
    Format anchors for YOLOv3 config (tuple of tuples)

    Args:
        anchors: Array of anchor boxes [9, 2]

    Returns:
        Formatted string to paste into config.py
    """
    anchors_tuple = tuple(tuple(map(lambda x: round(float(x), 1), anchor)) for anchor in anchors)

    # Format for config.py
    config_str = "anchors: List[Tuple[float, float]] = (\n"

    # Group by scale (3 anchors per scale for YOLOv3)
    grouped_anchors = group_anchors_by_scale(anchors)

    for i, scale_anchors in enumerate(grouped_anchors):
        config_str += "    # Scale {} ({}x{})\n".format(i + 1, [13, 26, 52][i], [13, 26, 52][i])
        for anchor in scale_anchors:
            config_str += f"    ({anchor[0]:.1f}, {anchor[1]:.1f}),\n"

    config_str += ")"

    return config_str


def main():
    """Main function to generate anchors"""
    # Get dataset name from command line args if provided
    dataset_name = "voc"
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1].lower()

    timestamp = "latest"
    if len(sys.argv) > 2:
        timestamp = sys.argv[2]

    print(f"=== Generating Custom Anchors for YOLOv3 on {dataset_name.upper()} Dataset ===")

    # Create dataset
    print(f"Loading {dataset_name.upper()} dataset...")
    dataset = PascalVOCDataset(years=["2007"], split="train", debug_mode=True)

    # Collect all boxes
    boxes = collect_boxes_from_dataset(dataset)
    print(f"Collected {len(boxes)} valid bounding boxes")

    # Run k-means to generate anchors
    print("Running k-means clustering to generate anchors...")
    k = 9  # YOLOv3 uses 9 anchors (3 per scale)
    anchors = kmeans_anchors(boxes, k, method="kmeans")

    # Calculate average IoU
    avg_iou_score = avg_iou(boxes, anchors)
    print(f"Average IoU with closest anchor: {avg_iou_score:.4f}")

    # Print anchors
    print("\nGenerated Anchors (width, height):")
    for i, anchor in enumerate(anchors):
        print(f"Anchor {i + 1}: ({anchor[0]:.1f}, {anchor[1]:.1f})")

    # Print formatted for config.py
    print("\nFormatted for config.py:")
    print(format_for_config(anchors))

    # Save anchors to file
    anchors_file = os.path.join(ANCHORS_DIR, f"{dataset_name}_{timestamp}_anchors.txt")
    np.savetxt(anchors_file, anchors, fmt="%.1f")
    print(f"Saved anchors to {anchors_file}")

    # Visualize anchors
    visualize_anchors(boxes, anchors, dataset_name)

    # Also save the config-formatted string to a file
    config_file = os.path.join(ANCHORS_DIR, f"{dataset_name}_{timestamp}_config.txt")
    with open(config_file, "w") as f:
        f.write(format_for_config(anchors))
    print(f"Saved config-formatted anchors to {config_file}")


if __name__ == "__main__":
    main()
