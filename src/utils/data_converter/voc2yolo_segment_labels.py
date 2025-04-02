#!/usr/bin/env python3
"""
Convert Pascal VOC Segmentation Annotations to YOLO Format for Specific ImageSets.

Reads segmentation masks (PNG) and XML annotations from the specified VOCdevkit
structure based on given years and ImageSet tags (from ImageSets/Segmentation).
Outputs YOLO segmentation format labels (.txt) with normalized polygon coordinates
directly into the specified output directory under labels_segment/<tag+year>.

Usage:
    python src/utils/data_converter/voc2yolo_segment_labels.py \\
        --voc-root /path/to/VOC \\
        --output-root /path/to/output \\
        --years 2012 \\
        --tags train,val
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from scipy.stats import mode
from tqdm import tqdm

# Import common utilities from voc2yolo_utils (moved here)
from src.utils.data_converter.voc2yolo_utils import (
    ANNOTATIONS_DIR,
    SEGMENTATION_OBJECT_DIR,
    VOC_CLASS_TO_ID,
    VOC_CLASSES,
    get_image_set_path,
    get_output_segment_label_dir,
    get_segm_cls_mask_path,
    get_segm_inst_mask_path,
    get_voc_dir,
    read_image_ids,
)

load_dotenv()  # Load .env variables

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
        self,
        voc_root: Path,
        output_root: Path,
        year: str,
        tag: str,
    ):
        """
        Initialize the converter.

        Args:
            voc_root: Path to the VOC dataset root directory.
            output_root: Path to the root directory for output.
            year: Year of the dataset (e.g., '2007', '2012').
            tag: ImageSet tag to process (e.g., 'train').
        """
        self.voc_root = voc_root
        self.output_root = output_root
        self.year = year
        self.tag = tag

        # Use utility functions to get paths
        self.voc_year_path = get_voc_dir(self.voc_root, self.year)
        self.output_segment_dir = get_output_segment_label_dir(
            self.output_root, self.year, self.tag
        )

        # Verify input path for the year exists
        if not self.voc_year_path.exists() or not self.voc_year_path.is_dir():
            raise ValueError(
                f"VOC year directory does not exist or is not a directory: {self.voc_year_path}"
            )

        # Ensure output directory exists
        self.output_segment_dir.mkdir(parents=True, exist_ok=True)

        self.class_to_id = VOC_CLASS_TO_ID

        logger.info(
            "Initialized converter for VOC Root: "
            f"{self.voc_root}, Year: {self.year}, Tag: {self.tag}"
        )
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
            # Read the palette-mode PNG using PIL
            mask_img = Image.open(str(mask_path))
            # Convert to numpy array - this will give us the palette indices
            mask = np.array(mask_img)
            logger.debug(f"Successfully loaded instance mask with PIL: {mask_path}")
            logger.debug(f"Instance mask unique values: {np.unique(mask)}")
        except Exception as e:
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

        for contour in contours:  # Removed unused 'i' variable
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

    def _match_instance_to_class(
        self,
        binary_mask: np.ndarray,
        class_mask: np.ndarray,
        instance_id: int,
        img_id: str,
    ) -> Optional[str]:
        """
        Determine the class name for a given instance mask by finding the
        most frequent valid class ID within the instance region in the class mask.

        Args:
            binary_mask: Binary mask (0/1) for the specific instance.
            class_mask: The full SegmentationClass mask containing class IDs.
            instance_id: The ID of the instance being matched.
            img_id: Image ID for logging.

        Returns:
            The matched class name (str) or None if no valid class is found.
        """
        if binary_mask.shape != class_mask.shape:
            logger.error(
                f"Mismatch in shapes between instance mask {binary_mask.shape} and "
                f"class mask {class_mask.shape} for img {img_id}, instance {instance_id}."
            )
            return None

        # Get class mask pixels corresponding to the instance mask pixels
        instance_pixels = class_mask[binary_mask == 1]

        if instance_pixels.size == 0:
            logger.warning(
                f"Instance {instance_id} in {img_id} has no corresponding pixels"
                f" in the binary mask."
            )
            return None

        # Filter out background (0) and boundary (255)
        valid_class_pixels = instance_pixels[(instance_pixels > 0) & (instance_pixels != 255)]

        if valid_class_pixels.size == 0:
            logger.warning(
                f"Instance {instance_id} in {img_id} only overlaps with background/boundary pixels."
            )
            return None

        # Find the most frequent valid class ID
        try:
            mode_result = mode(valid_class_pixels)
            # Handle scalar vs array output from mode depending on scipy version
            if np.isscalar(mode_result.mode):
                most_common_id = int(mode_result.mode)
            elif mode_result.mode.size > 0:
                most_common_id = int(mode_result.mode[0])  # Use the first mode if multiple
            else:
                logger.warning(
                    f"Could not determine mode class ID for instance {instance_id} in {img_id}. "
                    f"Valid pixels: {valid_class_pixels}"
                )
                return None
        except Exception as e:
            logger.error(
                f"Error finding mode class ID for instance {instance_id} in {img_id}: {e}. "
                f"Valid pixels: {valid_class_pixels}",
                exc_info=True,
            )
            return None

        # Convert class ID to class name
        try:
            # Find the class name corresponding to the ID (VOC IDs are 1-based in list)
            # Need to find the key (class name) for the value (ID)
            # Rebuild VOC_ID_TO_CLASS if needed, or iterate VOC_CLASSES
            # Assuming VOC_CLASSES is 0-indexed list, ID 1 = VOC_CLASSES[0]
            # But VOC standard uses 1-20. Let's use the imported VOC_CLASSES list directly.
            if 1 <= most_common_id < len(VOC_CLASSES) + 1:
                class_name = VOC_CLASSES[most_common_id - 1]  # Adjust ID to 0-based index
                logger.debug(
                    f"Matched instance {instance_id} in {img_id} to class '{class_name}'"
                    f" (ID: {most_common_id})"
                )
                return class_name
            else:
                logger.warning(
                    f"Most common class ID {most_common_id} for instance {instance_id} "
                    f"in {img_id} is out of range for VOC_CLASSES."
                )
                return None
        except IndexError:
            logger.error(
                f"Internal error: Class ID {most_common_id} out of bounds for VOC_CLASSES list.",
                exc_info=True,
            )
            return None

    def _process_instance(
        self, instance_id: int, binary_mask: np.ndarray, class_mask: np.ndarray, img_id: str
    ) -> Optional[List[str]]:
        """Process a single instance and return its YOLO format lines."""
        polygons = self._mask_to_polygons(binary_mask)
        if not polygons:
            logger.debug(f"No valid polygons for instance {instance_id} in {img_id}.")
            return None

        matched_class = self._match_instance_to_class(binary_mask, class_mask, instance_id, img_id)
        if matched_class is None:
            logger.warning(f"Skipping instance {instance_id} in {img_id} due to failed matching.")
            return None

        try:
            class_id = self.class_to_id[matched_class]
        except KeyError:
            logger.error(f"Internal error: Matched unknown class '{matched_class}' for {img_id}.")
            return None

        output_lines = []
        for poly in polygons:
            poly_str = " ".join(map(lambda x: f"{x:.6f}", poly))
            output_lines.append(f"{class_id} {poly_str}")
        return output_lines

    def _process_segmentation_file(self, img_id: str) -> bool:
        """
        Process a single image's segmentation mask and XML to generate YOLO label file.
        Writes output directly to the class instance's output_segment_dir.

        Args:
            img_id: The image identifier (filename without extension).

        Returns:
            bool: True if processing was successful, False otherwise.
        """
        # Construct paths using self.voc_year_path and utilities
        mask_path = get_segm_inst_mask_path(self.voc_year_path, img_id)
        class_mask_path = get_segm_cls_mask_path(self.voc_year_path, img_id)

        # 1. Check if mask exists
        if not mask_path.exists():
            logger.info(f"Segmentation mask not found: {mask_path}, skipping image {img_id}.")
            return False

        # 2. Load class mask
        try:
            class_mask = np.array(Image.open(str(class_mask_path)))
            logger.debug(f"Successfully loaded class mask with PIL: {class_mask_path}")
            logger.debug(f"Class mask unique values: {np.unique(class_mask)}")
        except Exception as e:
            logger.warning(
                f"Failed to load class segmentation mask with PIL: {class_mask_path}. "
                f"Error: {e}. Class names may be 'Unknown'."
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
            instance_lines = self._process_instance(instance_id, binary_mask, class_mask, img_id)
            if instance_lines:
                output_lines.extend(instance_lines)

        # 5. Write output file
        if output_lines:
            output_path = self.output_segment_dir / f"{img_id}.txt"
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

    def _validate_directories(self) -> bool:
        """Validate required directories exist."""
        annotations_dir = self.voc_year_path / ANNOTATIONS_DIR
        segmentation_dir = self.voc_year_path / SEGMENTATION_OBJECT_DIR

        if not segmentation_dir.exists():
            logger.warning(
                f"SegmentationObject directory not found: {segmentation_dir}. "
                f"Output may be incomplete for tag '{self.tag}'."
            )
        if not annotations_dir.exists():
            logger.error(
                f"Annotations directory not found: {annotations_dir}. "
                f"Cannot process tag '{self.tag}'."
            )
            return False
        return True

    def _get_image_ids_for_tag(self) -> Optional[List[str]]:
        """Reads image IDs for the current tag, handling file errors."""
        try:
            # Get image set file
            imageset_file = get_image_set_path(self.voc_year_path, set_type="segment", tag=self.tag)

            # Read image IDs
            img_ids = read_image_ids(imageset_file)
            if not img_ids:
                logger.warning(f"No image IDs found in {imageset_file}. Returning empty list.")
                return []  # Return empty list instead of None

            logger.info(f"Found {len(img_ids)} image IDs for tag '{self.tag}'.")
            return img_ids

        except FileNotFoundError as e:
            logger.error(f"Error reading image set file: {e}")
            # Try checking the 'Main' directory as a hint
            try:
                imageset_main_file = get_image_set_path(
                    self.voc_year_path, set_type="detect", tag=self.tag
                )
                if imageset_main_file.exists():
                    logger.warning(
                        f"Note: ImageSet file *does* exist in Main directory: {imageset_main_file}"
                    )
            except ValueError:  # Should not happen if set_type='detect'
                pass
            return None  # Indicate failure to read IDs
        except (ValueError, IOError) as e:  # Catch errors from read_image_ids
            logger.error(f"Error reading image IDs from {imageset_file}: {e}")
            return None  # Indicate failure

    def _process_image_list(self, img_ids: List[str]) -> Tuple[int, int]:
        """Processes a list of image IDs and returns success/fail counts."""
        success_count = 0
        fail_count = 0
        for img_id in tqdm(img_ids, desc=f"Converting {self.year}/{self.tag}"):
            try:
                success = self._process_segmentation_file(img_id)
                if success:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.error(
                    f"Unexpected error processing image ID {img_id} for tag {self.tag}: {e}",
                    exc_info=True,
                )
                fail_count += 1
        return success_count, fail_count

    def convert(self) -> Tuple[int, int]:
        """
        Convert the specific ImageSet tag for the initialized year.

        Returns:
            Tuple[int, int]: (success_count, fail_count) for this tag.
        """
        logger.info(f"--- Processing Year: {self.year}, Tag: {self.tag} ---")
        initial_fail_count = 0  # Count failures before processing images

        try:
            # Validate directories
            if not self._validate_directories():
                return 0, 1  # Return 1 failure if dirs invalid

            # Get image IDs
            img_ids = self._get_image_ids_for_tag()
            if img_ids is None:
                # Error logged in _get_image_ids_for_tag
                return 0, 1  # Return 1 failure if IDs couldn't be read
            if not img_ids:
                # Warning logged, but not a failure for the overall process
                return 0, 0  # No images to process

            # Process images
            logger.info(f"Converting {len(img_ids)} images...")
            success_count, fail_count = self._process_image_list(img_ids)

            logger.info(f"Finished processing tag '{self.tag}':")
            logger.info(f"  Successfully converted: {success_count}")
            logger.info(f"  Failed/Skipped: {fail_count}")

            return success_count, fail_count + initial_fail_count

        except Exception as e:  # Catch unexpected errors during setup/validation
            logger.error(
                f"Unexpected error during setup for {self.year}/{self.tag}: {e}", exc_info=True
            )
            return 0, 1  # Indicate failure


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert Pascal VOC segmentation masks for specific ImageSet tags to YOLO format."
        )
    )
    parser.add_argument(
        "--voc-root",
        type=str,
        default=None,
        help=(
            "Path to the VOC dataset root directory (containing VOCdevkit). "
            "If not set, uses VOC_ROOT from .env."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Path to the root directory for output labels. Defaults to --voc-root if not specified."
        ),
    )
    parser.add_argument(
        "--years",
        type=str,
        required=True,
        # choices=["2007", "2012"], # Removed choices constraint
        help="Comma-separated list of years of the VOC dataset (e.g., 2007,2012)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        required=True,
        help=(
            "Comma-separated list of ImageSet tags to process "
            "(from ImageSets/Segmentation, e.g., train,val,trainval)"
        ),
    )
    return parser.parse_args()


def _determine_paths(args: argparse.Namespace) -> Tuple[Optional[Path], Optional[Path]]:
    """Determine and validate VOC root and output root paths."""
    # Determine VOC Root Path
    if args.voc_root:
        voc_root_str = args.voc_root
        logger.info(f"Using specified --voc-root: {voc_root_str}")
    else:
        voc_root_str = os.getenv("VOC_ROOT")
        if not voc_root_str:
            logger.error(
                "VOC root not specified via --voc-root and VOC_ROOT not found in environment."
            )
            return None, None
        logger.info(f"Using VOC_ROOT from environment: {voc_root_str}")

    try:
        voc_root = Path(voc_root_str).expanduser()
        if not voc_root.exists() or not voc_root.is_dir():
            raise ValueError(f"Determined VOC root path is invalid: {voc_root}")
    except ValueError as e:
        logger.error(f"Error validating VOC root path: {e}")
        return None, None

    # Determine Output Root Path
    if args.output_root:
        output_root_str = args.output_root
        logger.info(f"Using specified --output-root: {output_root_str}")
    else:
        output_root_str = voc_root_str  # Default to voc_root if not specified
        logger.info(f"--output-root not specified, defaulting to VOC root: {output_root_str}")

    try:
        output_root = Path(output_root_str).expanduser()
    except Exception as e:
        logger.error(f"Error processing output root path '{output_root_str}': {e}")
        return voc_root, None

    return voc_root, output_root


def main():
    args = parse_args()
    # load_dotenv() # Already called globally

    voc_root, output_root = _determine_paths(args)
    if not voc_root or not output_root:
        return  # Error logged in helper

    # Parse comma-separated lists
    try:
        years = [y.strip() for y in args.years.split(",") if y.strip()]
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        if not years or not tags:
            raise ValueError("Years and Tags arguments cannot be empty.")
    except Exception as e:
        logger.error(f"Error parsing --years or --tags argument: {e}")
        return

    logger.info("Starting VOC to YOLO segmentation conversion.")
    logger.info(f"Years to process: {years}")
    logger.info(f"Tags to process: {tags}")

    total_success = 0
    total_fail = 0

    # Process each year and tag combination
    for year in years:
        for tag in tags:
            try:
                # Instantiate and run converter for this specific combo
                converter = VOC2YOLOConverter(
                    voc_root=voc_root,
                    output_root=output_root,
                    year=year,
                    tag=tag,
                )
                s, f = converter.convert()
                total_success += s
                total_fail += f
            except ValueError as e:
                logger.error(f"Initialization error for {year}/{tag}: {e}")
                # Decide how to count failures here, maybe increment total_fail?
                # For now, just log and continue. Failure count handled inside convert.
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred during conversion for {year}/{tag}: {e}",
                    exc_info=True,
                )
                # Increment fail count if an unexpected error occurs outside convert()
                total_fail += 1  # Or estimate based on expected number of images?

    logger.info("\nConversion completed!")
    logger.info(f"Overall success count: {total_success}")
    logger.info(f"Overall failed/skipped count: {total_fail}")


if __name__ == "__main__":
    main()
