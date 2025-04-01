#!/usr/bin/env python3
"""Utility functions and constants for VOC data converters."""

import logging
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
IMAGESETS_MAIN_DIR = f"{IMAGESETS_DIR}/Main"
IMAGESETS_SEGMENTATION_DIR = f"{IMAGESETS_DIR}/Segmentation"
IMAGESETS_ACTION_DIR = f"{IMAGESETS_DIR}/Action"
IMAGESETS_LAYOUT_DIR = f"{IMAGESETS_DIR}/Layout"

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
               Object dictionary keys: 'name', 'bbox' ([xmin, ymin, xmax, ymax] absolute coords).
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

        objects.append({"name": cls_name, "bbox": [xmin, ymin, xmax, ymax]})

    if not objects:
        logger.debug(f"No valid objects belonging to known classes found in {xml_path}")

    return objects, (img_width, img_height)
