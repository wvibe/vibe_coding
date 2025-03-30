"""Utility functions for the detection benchmark."""

import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

# Assuming this is run relative to the benchmark directory or project root
from .config import DatasetConfig

SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def find_dataset_files(config: DatasetConfig) -> List[Tuple[Path, Optional[Path]]]:
    """
    Finds image files and their corresponding label files based on the dataset config.

    Args:
        config: The dataset configuration object.

    Returns:
        A list of tuples, where each tuple is (image_path, label_path).
        label_path will be None if the corresponding label file is not found,
        or if annotation_format is not supported for finding labels this way.
    """
    image_dir = config.test_images_dir
    label_dir = config.annotations_dir
    annotation_format = config.annotation_format

    logging.info(f"Searching for images in: {image_dir}")
    image_files = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        image_files.extend(sorted(image_dir.glob(f"*{ext}")))
        image_files.extend(sorted(image_dir.glob(f"*{ext.upper()}")))

    if not image_files:
        logging.warning(f"No image files found in {image_dir}")
        return []

    logging.info(f"Found {len(image_files)} potential image files.")

    dataset_pairs: List[Tuple[Path, Optional[Path]]] = []

    if annotation_format == "yolo_txt":
        logging.info(f"Searching for corresponding YOLO labels in: {label_dir}")
        found_labels = 0
        missing_labels = 0
        for img_path in image_files:
            label_filename = img_path.stem + ".txt"
            label_path = label_dir / label_filename
            if label_path.is_file():
                dataset_pairs.append((img_path, label_path))
                found_labels += 1
            else:
                # Still include the image even if the label is missing,
                # evaluation step will need to handle this (or user can filter later)
                dataset_pairs.append((img_path, None))
                logging.debug(f"Label file not found for image {img_path.name} at {label_path}")
                missing_labels += 1
        logging.info(f"Found {found_labels} matching label files. {missing_labels} labels missing.")
    elif annotation_format == "voc_xml":
        logging.info(f"Searching for corresponding VOC XML labels in: {label_dir}")
        found_labels = 0
        missing_labels = 0
        for img_path in image_files:
            label_filename = img_path.stem + ".xml"
            label_path = label_dir / label_filename
            if label_path.is_file():
                dataset_pairs.append((img_path, label_path))
                found_labels += 1
            else:
                dataset_pairs.append((img_path, None))
                logging.debug(f"Label file not found for image {img_path.name} at {label_path}")
                missing_labels += 1
        logging.info(f"Found {found_labels} matching label files. {missing_labels} labels missing.")
    else:
        # For other formats (like coco_json), we expect a single annotation file.
        # The loading logic might need to be different (e.g., load JSON first, then match images).
        # For now, just pair images with None label path.
        logging.warning(
            f"Annotation format '{annotation_format}' might require specific loading. Pairing images without specific labels for now."
        )
        for img_path in image_files:
            dataset_pairs.append((img_path, None))

    return dataset_pairs


def select_subset(
    all_files: List[Tuple[Path, Optional[Path]]], config: DatasetConfig
) -> List[Tuple[Path, Optional[Path]]]:
    """
    Selects a subset of files based on the configuration.

    Args:
        all_files: The full list of (image_path, label_path) tuples.
        config: The dataset configuration object.

    Returns:
        The selected subset of file pairs.
    """
    method = config.subset_method
    size = config.subset_size
    num_available = len(all_files)

    if method == "all":
        logging.info(f"Using all {num_available} files.")
        return all_files
    elif method == "first_n":
        if size >= num_available:
            logging.warning(
                f"Subset size ({size}) >= available files ({num_available}). Using all files."
            )
            return all_files
        else:
            logging.info(f"Selecting first {size} files.")
            return all_files[:size]
    elif method == "random":
        if size >= num_available:
            logging.warning(
                f"Subset size ({size}) >= available files ({num_available}). Using all files (shuffled)."
            )
            random.shuffle(all_files)
            return all_files
        else:
            logging.info(f"Selecting random {size} files from {num_available}.")
            return random.sample(all_files, size)
    else:
        # Should be caught by Pydantic, but good to have a fallback
        logging.error(f"Invalid subset_method: {method}. Returning all files.")
        return all_files


# TODO: Add function to parse yolo_txt labels
# TODO: Add function to parse voc_xml labels
