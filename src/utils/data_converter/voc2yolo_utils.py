#!/usr/bin/env python3
"""Utility functions and constants for VOC data converters."""

import logging
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Constants ---

# --- Standard VOC Directory Names ---
ANNOTATIONS_DIR = "Annotations"
JPEG_IMAGES_DIR = "JPEGImages"
SEGMENTATION_CLASS_DIR = "SegmentationClass"
SEGMENTATION_OBJECT_DIR = "SegmentationObject"
IMAGESETS_DIR = "ImageSets"
IMAGESETS_MAIN_DIR = Path("ImageSets") / "Main"
IMAGESETS_SEGMENTATION_DIR = Path("ImageSets") / "Segmentation"
IMAGESETS_ACTION_DIR = f"{IMAGESETS_DIR}/Action"
IMAGESETS_LAYOUT_DIR = f"{IMAGESETS_DIR}/Layout"

# Define subdirs relative to ImageSets
MAIN_DIR = "Main"
SEGMENTATION_DIR = "Segmentation"

# --- Output Directory Names ---
OUTPUT_IMAGES_DIR = "images"
OUTPUT_LABELS_DETECT_DIR = "labels_detect"
OUTPUT_LABELS_SEGMENT_DIR = "labels_segment"

# --- VOC Classes ---
VOC_CLASSES = [
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

VOC_CLASS_TO_ID = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

# --- Deprecated YEAR_SPLITS --- (No longer used by refactored scripts)
# # Standard VOC splits (Year, ImageSet file name segment)
# YEAR_SPLITS = (
#     ("2007", "train"),
#     ("2007", "val"),
#     ("2007", "test"),
#     ("2012", "train"),
#     ("2012", "val"),
# )

# --- Functions ---


def parse_voc_xml(
    xml_path: Path,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Tuple[int, int]]]:
    """
    Parse VOC XML annotation file to extract object info and image size.

    Args:
        xml_path: Path to the XML annotation file.

    Returns:
        Tuple: (List of object dictionaries, (image_width, image_height)) or (None, None) if parsing fails.
               Object dictionary keys: 'name', 'bbox' ([xmin, ymin, xmax, ymax] absolute coords),
                                     'difficult' (int, 0 or 1).
               Only includes objects whose class is in VOC_CLASSES.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML: {xml_path}, Error: {e}")
        return None, None
    except FileNotFoundError:
        logger.error(f"XML file not found: {xml_path}")
        return None, None

    size_elem = root.find("size")
    if size_elem is None:
        logger.error(f"Missing 'size' tag in {xml_path}")
        return None, None
    width_elem = size_elem.find("width")
    height_elem = size_elem.find("height")

    if (
        width_elem is None
        or height_elem is None
        or width_elem.text is None
        or height_elem.text is None
    ):
        logger.error(f"Missing or empty 'width' or 'height' tag in {xml_path}")
        return None, None

    try:
        img_width = int(width_elem.text)
        img_height = int(height_elem.text)
        if img_width <= 0 or img_height <= 0:
            raise ValueError("Non-positive image dimensions")
    except ValueError as e:
        logger.error(f"Invalid 'width' or 'height' value in {xml_path}: {e}")
        return None, None

    objects = []
    for obj in root.findall("object"):
        cls_name_elem = obj.find("name")
        if cls_name_elem is None or not cls_name_elem.text:
            logger.warning(f"Skipping object with missing name in {xml_path}")
            continue
        cls_name = cls_name_elem.text

        # Check if class is known before proceeding
        if cls_name not in VOC_CLASS_TO_ID:
            logger.warning(f"Skipping object with unknown class '{cls_name}' in {xml_path}")
            continue

        bbox_elem = obj.find("bndbox")
        if bbox_elem is None:
            logger.warning(f"Skipping object '{cls_name}' with missing bndbox in {xml_path}")
            continue

        try:
            xmin = float(bbox_elem.findtext("xmin", -1))
            ymin = float(bbox_elem.findtext("ymin", -1))
            xmax = float(bbox_elem.findtext("xmax", -1))
            ymax = float(bbox_elem.findtext("ymax", -1))
            # Basic validation against image dimensions
            if not (0 <= xmin < xmax <= img_width and 0 <= ymin < ymax <= img_height):
                raise ValueError("Invalid bbox coordinates relative to image size")
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(
                f"Skipping object '{cls_name}' with invalid/missing bbox coordinates in {xml_path}: {e}"
            )
            continue

        # Get difficult flag (default to 0 if not present)
        difficult_text = obj.findtext("difficult", "0")
        try:
            difficult = int(difficult_text)
            if difficult not in [0, 1]:
                logger.warning(
                    f"Invalid difficult flag '{difficult_text}' for object '{cls_name}' in {xml_path}. Assuming 0."
                )
                difficult = 0
        except ValueError:
            logger.warning(
                f"Non-integer difficult flag '{difficult_text}' for object '{cls_name}' in {xml_path}. Assuming 0."
            )
            difficult = 0

        objects.append({"name": cls_name, "bbox": [xmin, ymin, xmax, ymax], "difficult": difficult})

    if not objects:
        logger.debug(f"No valid objects belonging to known classes found in {xml_path}")

    return objects, (img_width, img_height)


# --- Environment and Path Utilities ---

# Removed load_voc_root_from_env - Handled in individual scripts


def get_voc_dir(voc_root: Path, year: str) -> Path:
    """Get the path to the specific VOC year directory (e.g., VOC_ROOT/VOCdevkit/VOC2012)."""
    # return voc_root / f"VOC{year}"
    return voc_root / "VOCdevkit" / f"VOC{year}"


def get_image_set_path(
    voc_dir: Path, set_type: str, tag: str
) -> Path:
    """Construct the path to an ImageSet file within a specific year's directory.

    Args:
        voc_dir: Path to the specific VOC year directory (e.g., /path/to/VOCdevkit/VOC2012).
        set_type: The type of image set ('detect' or 'segment'). Corresponds to subdirs 'Main' or 'Segmentation'.
        tag: The dataset tag (e.g., 'train', 'val', 'test').

    Returns:
        Path to the imageset file.

    Raises:
        ValueError: If set_type is invalid.
    """
    if set_type == "detect" or set_type == "main":
        subdir = MAIN_DIR
    elif set_type == "segment":
        subdir = SEGMENTATION_DIR
    else:
        raise ValueError(f"Invalid set_type '{set_type}'. Must be 'detect' or 'segment'.")

    # Construct path relative to the year-specific voc_dir
    return voc_dir / IMAGESETS_DIR / subdir / f"{tag}.txt"


def get_image_path(voc_dir: Path, image_id: str) -> Path:
    """Get the path to a JPEG image within the VOC structure."""
    return voc_dir / JPEG_IMAGES_DIR / f"{image_id}.jpg"


def get_annotation_path(voc_dir: Path, image_id: str) -> Path:
    """Get the path to an XML annotation file within the VOC structure."""
    return voc_dir / ANNOTATIONS_DIR / f"{image_id}.xml"


def get_segmentation_mask_path(voc_dir: Path, image_id: str) -> Path:
    """Get the path to a segmentation mask PNG file within the VOC structure."""
    return voc_dir / SEGMENTATION_OBJECT_DIR / f"{image_id}.png"


def get_output_image_dir(output_root: Path, year: str, tag: str) -> Path:
    """Get the output directory path for images for a given year and tag."""
    return output_root / OUTPUT_IMAGES_DIR / f"{tag}{year}"


def get_output_detect_label_dir(output_root: Path, year: str, tag: str) -> Path:
    """Get the output directory path for detection labels for a given year and tag."""
    return output_root / OUTPUT_LABELS_DETECT_DIR / f"{tag}{year}"


def get_output_segment_label_dir(output_root: Path, year: str, tag: str) -> Path:
    """Get the output directory path for segmentation labels for a given year and tag."""
    return output_root / OUTPUT_LABELS_SEGMENT_DIR / f"{tag}{year}"


# --- Image Set Reading Utilities ---


def read_image_ids(
    imageset_path: Path, num_samples: Optional[int] = None, seed: Optional[int] = 42
) -> List[str]:
    """Reads image IDs from an ImageSet file, with optional random sampling.

    Args:
        imageset_path (Path): Path to the ImageSet .txt file.
        num_samples (Optional[int], optional): Number of samples to randomly select.
                                             If None, reads all IDs. Defaults to None.
        seed (Optional[int], optional): Random seed for sampling. Defaults to 42.

    Returns:
        List[str]: List of image IDs (potentially sampled).

    Raises:
        FileNotFoundError: If the imageset_path does not exist.
        IOError: If the file cannot be read.
    """
    if not imageset_path.exists():
        raise FileNotFoundError(f"ImageSet file not found: {imageset_path}")

    try:
        with open(imageset_path, "r") as f:
            # Reads lines, strips whitespace, takes first element if line has multiple cols
            # (some VOC sets have a second column like -1 or 1)
            all_ids = [line.strip().split()[0] for line in f if line.strip()]
    except IOError as e:
        logger.error(f"Could not read ImageSet file {imageset_path}: {e}")
        raise  # Re-raise the exception

    if not all_ids:
        logger.warning(f"No image IDs found in {imageset_path}.")
        return []

    if num_samples is not None and num_samples > 0:
        if num_samples >= len(all_ids):
            logger.info(
                f"Requested sample size ({num_samples}) >= total IDs ({len(all_ids)}). Returning all IDs."
            )
            return all_ids
        else:
            if seed is not None:
                random.seed(seed)
            sampled_ids = random.sample(all_ids, num_samples)
            logger.info(f"Sampled {len(sampled_ids)} IDs from {imageset_path} (seed={seed}).")
            return sampled_ids
    else:
        return all_ids
