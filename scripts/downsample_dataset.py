#!/usr/bin/env python3
"""
Script to downsample a dataset (e.g., COV_SEGM) to reduce size for faster training or validation.
Ensures images and corresponding labels remain in sync.
Handles dataset structure with subfolders under images/ and labels/.
Creates new folders for downsampled data using a specified output tag.

Usage:
    python scripts/downsample_dataset.py --root <dataset_root> \
        --tag <folder_tag> --rate <downsample_rate> --output-tag <custom_tag>

Example:
    python scripts/downsample_dataset.py \
        --root /home/ubuntu/vibe/hub/datasets/COV_SEGM/visible \
        --tag validation --rate 0.1 --output-tag val_subset_10percent
"""

import argparse
import random
import shutil
from pathlib import Path


def parse_arguments():
    """Parse command-line arguments for downsampling configuration."""
    parser = argparse.ArgumentParser(
        description="Downsample a dataset while keeping images and labels in sync."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory of the dataset (e.g., "
        "/home/ubuntu/vibe/hub/datasets/COV_SEGM/visible)",
    )
    parser.add_argument(
        "--tag", type=str, required=True, help="Folder tag to downsample (e.g., validation, test)"
    )
    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="Downsample rate (0.0 to 1.0), e.g., 0.1 to sample 10% of the data",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        required=True,
        help="Custom tag for the output folders (e.g., val_subset_10percent)",
    )
    return parser.parse_args()


def downsample_dataset(root_dir: str, folder_tag: str, downsample_rate: float, output_tag: str):
    """Downsample the dataset based on the specified rate and copy sampled
    images and labels to new folders."""
    root_path = Path(root_dir)
    img_dir = root_path / "images" / folder_tag
    lbl_dir = root_path / "labels" / folder_tag

    # Define output directories for downsampled data
    output_img_dir = root_path / "images" / output_tag
    output_lbl_dir = root_path / "labels" / output_tag

    # Validate input directories
    if not img_dir.is_dir():
        print(f"Error: Image directory not found: {img_dir}")
        exit(1)
    if not lbl_dir.is_dir():
        print(f"Error: Label directory not found: {lbl_dir}")
        exit(1)

    # Create output directories if they don't exist
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directories: {output_img_dir} and {output_lbl_dir}")

    # Scan for image files (support common extensions)
    print(f"Scanning image directory: {img_dir}")
    image_extensions = ["*.jpg", "*.png", "*.jpeg"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(img_dir.glob(ext)))

    num_total_images = len(image_files)
    print(f"Found {num_total_images} images in the directory.")

    if num_total_images == 0:
        print("Error: No images found. Check the directory path and image extensions.")
        exit(1)

    # Calculate number of samples based on rate
    if downsample_rate <= 0 or downsample_rate > 1.0:
        print(f"Error: Downsample rate must be between 0.0 and 1.0, got {downsample_rate}")
        exit(1)
    num_samples = max(1, int(num_total_images * downsample_rate))
    print(f"Randomly sampling {num_samples} images ({downsample_rate * 100:.1f}% of total)...")

    # Randomly sample images
    sampled_image_paths = random.sample(image_files, num_samples)

    # Verify corresponding label files exist and copy files to output directories
    missing_labels = []
    valid_sampled_paths = []
    for img_path in sampled_image_paths:
        label_filename = img_path.with_suffix(".txt").name
        expected_label_path = lbl_dir / label_filename
        if expected_label_path.is_file():
            valid_sampled_paths.append(img_path)
            # Copy image to output directory
            output_img_path = output_img_dir / img_path.name
            shutil.copy2(img_path, output_img_path)
            # Copy label to output directory
            output_lbl_path = output_lbl_dir / label_filename
            shutil.copy2(expected_label_path, output_lbl_path)
        else:
            missing_labels.append(img_path.name)
            print(
                f"Warning: Corresponding label file not found for "
                f"{img_path.name} at {expected_label_path}"
            )

    if missing_labels:
        print(f"Warning: {len(missing_labels)} images were excluded due to missing labels.")
        if not valid_sampled_paths:
            print("Error: No images have corresponding labels. Cannot create subset.")
            exit(1)
        print(f"Proceeding with {len(valid_sampled_paths)} valid samples.")
    else:
        print("All sampled images have corresponding labels.")

    # Summary
    print(
        f"Downsampling complete. Total images: {num_total_images}, "
        f"Sampled and copied: {len(valid_sampled_paths)}"
    )
    print(f"Downsampled images saved to: {output_img_dir}")
    print(f"Downsampled labels saved to: {output_lbl_dir}")
    print(
        f"To use this subset, update your dataset YAML to point to the new "
        f"folders with tag '{output_tag}'."
    )


if __name__ == "__main__":
    args = parse_arguments()
    downsample_dataset(args.root, args.tag, args.rate, args.output_tag)
