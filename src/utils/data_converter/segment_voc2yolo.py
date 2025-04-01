#!/usr/bin/env python3
"""
Convert Pascal VOC Segmentation Annotations to YOLO Format for a Specific ImageSet.

Reads segmentation masks (PNG) and XML annotations from the specified VOCdevkit
structure based on a given year and ImageSet tag. Outputs YOLO segmentation
format labels (.txt) with normalized polygon coordinates directly into the
specified output directory.

Usage:
    python src/utils/data_converter/segment_voc2yolo.py \
        --devkit-path /path/to/VOCdevkit \
        --year 2012 \
        --tag trainval \
        --output-dir /path/to/output/labels_segment \
        --iou-threshold 0.5
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Import common utilities
from src.utils.data_converter.converter_utils import (
    VOC_CLASS_TO_ID,
    parse_voc_xml,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Constants for Segmentation Logic ---
# Minimum contour area to consider for polygon conversion
MIN_CONTOUR_AREA = 1.0
# Tolerance factor for polygon approximation (relative to arc length)
POLYGON_APPROX_TOLERANCE = 0.005


class VOC2YOLOConverter:
    """Convert VOC dataset annotations (masks) to YOLO segmentation format for a specific tag."""

    def __init__(
        self, devkit_path: str, year: str, output_segment_dir: str, iou_threshold: float = 0.5
    ):
        """
        Initialize the converter.

        Args:
            devkit_path: Path to the VOCdevkit directory.
            year: Year of the dataset (e.g., '2007', '2012').
            output_segment_dir: Directory to save YOLO segmentation format labels (flat structure).
            iou_threshold: IoU threshold for matching mask instances to XML boxes.
        """
        self.devkit_path = Path(devkit_path)
        self.year = year
        self.output_segment_dir = Path(output_segment_dir)
        self.iou_threshold = iou_threshold
        self.voc_year_path = self.devkit_path / f"VOC{self.year}"

        # Verify paths
        if not self.devkit_path.exists() or not self.devkit_path.is_dir():
            raise ValueError(
                f"VOCdevkit path does not exist or is not a directory: {self.devkit_path}"
            )
        if not self.voc_year_path.exists():
            raise ValueError(
                f"VOC year directory does not exist within devkit: {self.voc_year_path}"
            )

        # Ensure output directory exists
        self.output_segment_dir.mkdir(parents=True, exist_ok=True)

        # Use imported constants
        self.class_to_id = VOC_CLASS_TO_ID
        # self.classes = VOC_CLASSES # Not strictly needed if only using IDs

        logger.info(f"Initialized converter for VOCdevkit: {self.devkit_path}, Year: {self.year}")
        logger.info(f"Outputting segmentation labels to: {self.output_segment_dir}")

    def _get_mask_instances(self, mask_path: Path) -> Optional[Dict[int, np.ndarray]]:
        """
        Read a segmentation mask and extract binary masks for each unique instance.

        Args:
            mask_path: Path to the segmentation mask PNG file.

        Returns:
            Dictionary {instance_id: binary_mask} or None if read fails.
            Excludes background (0) and boundary (255).
        """
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise IOError("cv2.imread returned None")
        except (IOError, cv2.error) as e:
            logger.error(f"Failed to read or decode mask: {mask_path}, Error: {e}")
            return None

        instance_ids = np.unique(mask)
        instance_masks = {}
        for instance_id in instance_ids:
            if instance_id == 0 or instance_id == 255:  # Skip background and boundary
                continue
            instance_masks[instance_id] = (mask == instance_id).astype(np.uint8)

        return instance_masks

    def _mask_to_polygons(
        self, binary_mask: np.ndarray, tolerance: float = POLYGON_APPROX_TOLERANCE
    ) -> List[List[float]]:
        """
        Convert a binary instance mask to normalized polygon coordinates.

        Args:
            binary_mask: A numpy array representing the binary mask for one instance.
            tolerance: Approximation tolerance factor for cv2.approxPolyDP.

        Returns:
            List of polygons, where each polygon is a flat list of normalized [x1, y1, x2, y2, ...].
            Returns empty list if no valid contours/polygons are found.
        """
        polygons = []
        contours, hierarchy = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None:  # Handle cases with no contours
            return []

        h, w = binary_mask.shape[:2]
        if w <= 0 or h <= 0:
            logger.warning("Cannot normalize polygons for zero-dimension mask.")
            return []

        for i, contour in enumerate(contours):
            # Check if contour is likely significant using defined constant
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue

            # Approximate contour using tolerance argument (defaults to constant)
            epsilon = tolerance * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Need at least 3 points for a valid polygon
            if len(approx) >= 3:
                # Normalize and flatten coordinates
                # approx shape is (N, 1, 2), needs reshape to (N, 2) for iteration
                normalized_polygon = []
                for point in approx.reshape(-1, 2):
                    px, py = point
                    # Clamp coordinates strictly within [0, 1] after normalization
                    norm_x = np.clip(px / w, 0.0, 1.0)
                    norm_y = np.clip(py / h, 0.0, 1.0)
                    normalized_polygon.extend([norm_x, norm_y])

                # Check again if polygon still has >= 3 unique points after normalization/clipping
                if len(normalized_polygon) >= 6:
                    polygons.append(normalized_polygon)
            # else: logger.debug(f"Skipping approximated contour with < 3 points.")

        return polygons

    def _get_mask_bbox(self, binary_mask: np.ndarray) -> Optional[List[float]]:
        """
        Calculate the normalized bounding box [xmin, ymin, xmax, ymax] from a binary mask.

        Args:
            binary_mask: A numpy array representing the binary mask for one instance.

        Returns:
            Normalized [xmin, ymin, xmax, ymax] or None if mask is empty.
        """
        y_indices, x_indices = np.where(binary_mask > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return None  # No foreground pixels

        h, w = binary_mask.shape[:2]
        if w <= 0 or h <= 0:
            return None

        xmin = np.min(x_indices)
        xmax = np.max(x_indices)
        ymin = np.min(y_indices)
        ymax = np.max(y_indices)

        # Normalize (add 1 to max coords because they are inclusive indices)
        norm_xmin = xmin / w
        norm_ymin = ymin / h
        norm_xmax = (xmax + 1) / w
        norm_ymax = (ymax + 1) / h

        # Clamp values to be safe
        return [
            np.clip(norm_xmin, 0.0, 1.0),
            np.clip(norm_ymin, 0.0, 1.0),
            np.clip(norm_xmax, 0.0, 1.0),
            np.clip(norm_ymax, 0.0, 1.0),
        ]

    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Compute Intersection over Union (IoU) between two bounding boxes.
        Assumes boxes are in [xmin, ymin, xmax, ymax] format (absolute or normalized).

        Args:
            box1: First bounding box [xmin, ymin, xmax, ymax].
            box2: Second bounding box [xmin, ymin, xmax, ymax].

        Returns:
            IoU score (float between 0.0 and 1.0).
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Calculate intersection coordinates
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        # Calculate intersection area
        inter_width = max(0, inter_xmax - inter_xmin)
        inter_height = max(0, inter_ymax - inter_ymin)
        intersection_area = inter_width * inter_height

        if intersection_area == 0:
            return 0.0

        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersection_area

        if union_area == 0:
            # Avoid division by zero; should only happen if both boxes have zero area
            return 0.0

        iou = intersection_area / union_area

        # Clamp IoU to [0, 1] due to potential floating point inaccuracies
        return np.clip(iou, 0.0, 1.0)

    def _match_instance_to_class(
        self,
        instance_mask: np.ndarray,
        xml_objects: List[Dict[str, Any]],
        img_dims: Tuple[int, int],
    ) -> Optional[str]:
        """
        Match a mask instance to an object class from XML annotations using IoU.

        Args:
            instance_mask: Binary mask for the specific instance.
            xml_objects: List of objects parsed from XML (dicts with 'name', 'bbox').
            img_dims: Tuple of (image_width, image_height) from XML.

        Returns:
            The matched class name (str) or None if no suitable match is found.
        """
        if not xml_objects:
            logger.warning("Cannot match instance: No XML objects provided.")
            return None

        mask_bbox_norm = self._get_mask_bbox(instance_mask)
        if mask_bbox_norm is None:
            logger.warning("Cannot match instance: Failed to get bounding box from mask.")
            return None

        best_iou = -1.0
        best_match_class = None
        # Keep track of used XML objects to handle multiple instances of same class
        # This basic version doesn't prevent assigning multiple masks to the same XML box,
        # A more complex assignment (e.g., Hungarian algorithm) could be used if needed.
        img_width, img_height = img_dims

        for xml_obj in xml_objects:
            xml_bbox_abs = xml_obj["bbox"]
            # Normalize XML bbox for IoU calculation
            xml_bbox_norm = [
                xml_bbox_abs[0] / img_width,
                xml_bbox_abs[1] / img_height,  # xmin, ymin
                xml_bbox_abs[2] / img_width,
                xml_bbox_abs[3] / img_height,  # xmax, ymax
            ]

            iou = self._compute_iou(mask_bbox_norm, xml_bbox_norm)

            if iou > best_iou:
                best_iou = iou
                best_match_class = xml_obj["name"]

        if best_iou >= self.iou_threshold:
            logger.debug(
                f"Matched instance mask (bbox: {mask_bbox_norm}) to XML class '{best_match_class}' (IoU: {best_iou:.4f})"
            )
            return best_match_class
        else:
            logger.warning(
                f"Could not match instance mask (bbox: {mask_bbox_norm}). Best IoU ({best_iou:.4f}) below threshold ({self.iou_threshold})."
            )
            return None

    def _process_segmentation_file(self, img_id: str, output_dir: Path):
        """
        Process a single image's segmentation mask and XML to generate YOLO label file.
        Writes output directly to the specified flat output_dir.

        Args:
            img_id: The image identifier (filename without extension).
            output_dir: The flat output directory to save the label file.
        """
        # Construct paths using self.voc_year_path initialized in __init__
        mask_path = self.voc_year_path / "SegmentationObject" / f"{img_id}.png"
        xml_path = self.voc_year_path / "Annotations" / f"{img_id}.xml"

        # 1. Check if mask exists
        if not mask_path.exists():
            logger.info(f"Segmentation mask not found: {mask_path}, skipping image {img_id}.")
            return False  # Indicate skipped

        # 2. Parse XML using utility function
        xml_objects, img_dims = parse_voc_xml(xml_path)
        if xml_objects is None or img_dims is None:
            logger.warning(f"Skipping image {img_id} due to XML parsing error or missing info.")
            return False

        if not xml_objects:
            logger.warning(
                f"No valid objects found in XML {xml_path} for {img_id}, skipping mask processing."
            )
            return False

        # 3. Read mask and extract instances
        instance_masks = self._get_mask_instances(mask_path)
        if instance_masks is None:
            logger.warning(f"Skipping image {img_id} due to mask reading error.")
            return False

        if not instance_masks:
            logger.info(f"No instances found in mask {mask_path} for {img_id}.")
            return False

        # 4. Process each instance
        output_lines = []
        for instance_id, binary_mask in instance_masks.items():
            polygons = self._mask_to_polygons(binary_mask)
            if not polygons:
                logger.debug(f"No valid polygons for instance {instance_id} in {img_id}.")
                continue

            matched_class = self._match_instance_to_class(binary_mask, xml_objects, img_dims)
            if matched_class is None:
                logger.warning(
                    f"Skipping instance {instance_id} in {img_id} due to failed matching."
                )
                continue

            try:
                class_id = self.class_to_id[matched_class]
            except KeyError:
                logger.error(
                    f"Internal error: Matched unknown class '{matched_class}' for {img_id}."
                )
                continue

            for poly in polygons:
                poly_str = " ".join(map(lambda x: f"{x:.6f}", poly))
                output_lines.append(f"{class_id} {poly_str}")

        # 5. Write output file (directly to output_dir)
        if output_lines:
            output_path = output_dir / f"{img_id}.txt"  # Flat output
            try:
                with open(output_path, "w") as f:
                    f.write("\n".join(output_lines) + "\n")
                logger.debug(f"Successfully wrote {len(output_lines)} lines for {img_id}.")
                return True
            except IOError as e:
                logger.error(f"Failed to write output file {output_path}: {e}")
                return False
        else:
            logger.info(f"No valid instances processed/matched for {img_id}, no output written.")
            return False

    # Replace the old convert method with one that processes a single tag
    def convert_single_tag(self, tag: str):
        """Convert a specific ImageSet tag (e.g., train, val) for the initialized year."""
        logger.info(f"Processing tag: {tag} for year {self.year}")

        imageset_file = self.voc_year_path / f"ImageSets/Main/{tag}.txt"
        segmentation_dir = self.voc_year_path / "SegmentationObject"
        annotations_dir = self.voc_year_path / "Annotations"

        # Check necessary files/dirs
        if not imageset_file.exists():
            logger.error(f"ImageSet file not found: {imageset_file}. Cannot process tag '{tag}'.")
            return
        # Segmentation objects might not exist for all images/tags (e.g., test set)
        if not segmentation_dir.exists():
            logger.warning(
                f"SegmentationObject directory not found: {segmentation_dir}. Output may be incomplete for tag '{tag}'."
            )
        if not annotations_dir.exists():
            logger.error(
                f"Annotations directory not found: {annotations_dir}. Cannot process tag '{tag}'."
            )
            return

        # Read image IDs
        try:
            with open(imageset_file, "r") as f:
                img_ids = [line.strip().split()[0] for line in f if line.strip()]
            if not img_ids:
                logger.warning(f"No image IDs found in {imageset_file}.")
                return
        except IOError as e:
            logger.error(f"Could not read ImageSet file {imageset_file}: {e}")
            return

        logger.info(f"Found {len(img_ids)} image IDs for tag '{tag}'. Converting...")

        success_count = 0
        fail_count = 0
        # Process each image ID
        for img_id in tqdm(img_ids, desc=f"Converting {self.year}/{tag}"):
            try:
                # Process file, outputting directly to self.output_segment_dir
                success = self._process_segmentation_file(img_id, self.output_segment_dir)
                if success:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.error(
                    f"Unexpected error processing image ID {img_id} for tag {tag}: {e}",
                    exc_info=True,
                )
                fail_count += 1

        logger.info(f"Finished processing tag '{tag}':")
        logger.info(f"  Successfully converted: {success_count}")
        logger.info(f"  Failed/Skipped: {fail_count}")
        logger.info(f"Output labels saved to: {self.output_segment_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC segmentation masks for a specific ImageSet tag to YOLO format."
    )
    parser.add_argument(
        "--devkit-path",
        type=str,
        required=True,
        help="Path to the VOCdevkit directory (e.g., /path/to/VOC/VOCdevkit).",
    )
    parser.add_argument(
        "--year",
        type=str,
        required=True,
        choices=["2007", "2012"],  # Add more years if needed
        help="Year of the VOC dataset (e.g., 2007, 2012)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="ImageSet tag to process (e.g., train, val, trainval, person_trainval)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the flat output directory for segmentation label files",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching mask instances to XML bounding boxes (default: 0.5).",
    )
    args = parser.parse_args()

    # Instantiate converter
    converter = VOC2YOLOConverter(
        devkit_path=args.devkit_path,
        year=args.year,
        output_segment_dir=args.output_dir,
        iou_threshold=args.iou_threshold,
    )
    # Convert the specified tag
    converter.convert_single_tag(args.tag)
