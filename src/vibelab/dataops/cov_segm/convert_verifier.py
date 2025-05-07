"""Core logic for verifying YOLO dataset conversion against original HF dataset."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Local imports
from vibelab.dataops.cov_segm.datamodel import ClsSegment, SegmSample
from vibelab.utils.common.bbox import calculate_iou as calculate_bbox_iou
from vibelab.utils.common.label_match import match_instances
from vibelab.utils.common.mask import calculate_mask_iou, polygons_to_mask

logger = logging.getLogger(__name__)


@dataclass
class YoloInstanceRecord:
    """Represents a single instance parsed from a YOLO label file."""

    sample_id: str
    class_id: int
    polygon_abs: List[Tuple[int, int]]  # Polygon coordinates in absolute pixels
    derived_mask: np.ndarray = field(repr=False)  # Binary mask derived from polygon
    bbox: Tuple[int, int, int, int]  # Bbox derived from mask (xmin, ymin, xmax, ymax)


@dataclass
class OriginalInstanceRecord:
    """Represents an expected instance derived from the original SegmSample."""

    sample_id: str
    segment_idx: int  # Index of the source segment within the SegmSample
    mask_idx: int  # Index of the specific mask within the segment's list
    class_id: int
    original_mask: np.ndarray = field(repr=False)  # Original binary mask
    bbox: Tuple[int, int, int, int]  # Original bbox (xmin, ymin, xmax, ymax)


@dataclass
class VerificationResult:
    """Holds the results of verifying a single sample."""

    sample_id: str
    # Common results
    processing_error: Optional[str] = None

    # Mask-based matching results
    mask_matched_pairs: List[Dict[str, Any]] = field(default_factory=list)
    mask_lost_instances: List[OriginalInstanceRecord] = field(default_factory=list)
    mask_extra_instances: List[YoloInstanceRecord] = field(default_factory=list)

    # Bbox-based matching results
    bbox_matched_pairs: List[Dict[str, Any]] = field(default_factory=list)
    bbox_lost_instances: List[OriginalInstanceRecord] = field(default_factory=list)
    bbox_extra_instances: List[YoloInstanceRecord] = field(default_factory=list)

    # Legacy fields for backward compatibility - will be populated from mask-based results
    @property
    def matched_pairs(self) -> List[Dict[str, Any]]:
        """For backward compatibility."""
        return self.mask_matched_pairs

    @property
    def lost_instances(self) -> List[OriginalInstanceRecord]:
        """For backward compatibility."""
        return self.mask_lost_instances

    @property
    def extra_instances(self) -> List[YoloInstanceRecord]:
        """For backward compatibility."""
        return self.mask_extra_instances


def _get_mapping_info(
    segment: ClsSegment,
    phrase_map: Dict[str, Dict[str, Any]],
) -> Optional[Tuple[Dict[str, Any], str]]:
    """Checks segment phrases against the map.

    Args:
        segment: The segment containing phrases to check against the mapping.
        phrase_map: Mapping of phrases to class information.

    Returns:
        Tuple[Dict, str] (mapping_info, matched_phrase) if a match is found,
        otherwise None.
    """
    matched_phrase = None
    mapping_info = None

    for phrase in segment.phrases:
        phrase_text = phrase.text.strip()
        if not phrase_text:
            logger.debug("Skipping empty phrase in segment.")  # Use debug
            continue

        current_mapping = phrase_map.get(phrase_text)
        if current_mapping:
            mapping_info = current_mapping
            matched_phrase = phrase_text
            break

    if mapping_info is None or matched_phrase is None:
        logger.debug(f"Segment phrases {[p.text for p in segment.phrases]} not found in mapping.")
        return None

    # Return the mapping info without any sampling
    return mapping_info, matched_phrase


def _calculate_bbox_from_mask(binary_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Calculate bounding box (xmin, ymin, xmax, ymax) from a binary mask."""
    rows, cols = np.where(binary_mask)
    if rows.size == 0 or cols.size == 0:
        return None  # No foreground pixels
    x_min, y_min = int(cols.min()), int(rows.min())
    x_max, y_max = int(cols.max()), int(rows.max())
    return x_min, y_min, x_max, y_max


def _load_yolo_instances(
    sample_id: str, yolo_label_path: Path, yolo_image_path: Path
) -> Tuple[List[YoloInstanceRecord], Optional[str]]:
    """Load and parse instances from a YOLO label file and corresponding image."""
    yolo_instances: List[YoloInstanceRecord] = []

    # 1. Load Image to get dimensions
    try:
        with Image.open(yolo_image_path) as img:
            img.load()  # Ensure image data is loaded
            width, height = img.size
    except FileNotFoundError:
        return [], f"YOLO image not found: {yolo_image_path}"
    except Exception as e:
        return [], f"Error loading YOLO image {yolo_image_path}: {e}"

    if height <= 0 or width <= 0:
        return [], f"Invalid image dimensions ({height}x{width}) for {yolo_image_path}"

    # 2. Read YOLO label file
    try:
        with open(yolo_label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return [], f"YOLO label file not found: {yolo_label_path}"
    except Exception as e:
        return [], f"Error reading YOLO label file {yolo_label_path}: {e}"

    # 3. Parse each line
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 7 or len(parts) % 2 != 1:
            error_msg = (
                f"Invalid format in {yolo_label_path}, line {i + 1}: "
                f"Expected class_id followed by pairs of coords."
            )
            return ([], error_msg)

        try:
            class_id = int(parts[0])
            norm_coords = [float(p) for p in parts[1:]]
        except ValueError:
            return [], f"Invalid numeric value in {yolo_label_path}, line {i + 1}."

        # Convert normalized polygon to absolute pixel coordinates
        poly_abs: List[Tuple[int, int]] = []

        # Initialize min/max x,y values for direct bounding box calculation
        x_min, y_min = width, height  # Initialize to max possible values
        x_max, y_max = 0, 0  # Initialize to min possible values

        for j in range(0, len(norm_coords), 2):
            norm_x, norm_y = norm_coords[j], norm_coords[j + 1]
            abs_x = int(round(norm_x * width))
            abs_y = int(round(norm_y * height))
            # Clip to ensure coords are within image bounds for mask generation
            abs_x = max(0, min(width - 1, abs_x))
            abs_y = max(0, min(height - 1, abs_y))
            poly_abs.append((abs_x, abs_y))

            # Update bbox coordinates directly from polygon points
            x_min = min(x_min, abs_x)
            y_min = min(y_min, abs_y)
            x_max = max(x_max, abs_x)
            y_max = max(y_max, abs_y)

        if len(poly_abs) < 3:
            log_msg = (
                f"Sample {sample_id}, Line {i + 1}: Degenerate polygon after "
                f"conversion (< 3 points), skipping instance."
            )
            return [], log_msg

        # Generate mask from absolute polygon (still needed for IoU calculations)
        derived_mask = polygons_to_mask(poly_abs, (height, width), normalized=False)
        derived_bbox = _calculate_bbox_from_mask(derived_mask)
        if derived_mask.sum() == 0 or derived_bbox is None:
            log_msg = (
                f"Sample {sample_id}, Line {i + 1}: Polygon resulted in empty "
                f"mask or bbox, skipping instance."
            )
            return [], log_msg

        # Use directly calculated bbox
        direct_bbox = (x_min, y_min, x_max, y_max)

        # Check if differences exceed margin
        bbox_margin = 2
        if (
            abs(direct_bbox[0] - derived_bbox[0]) > bbox_margin
            or abs(direct_bbox[1] - derived_bbox[1]) > bbox_margin
            or abs(direct_bbox[2] - derived_bbox[2]) > bbox_margin
            or abs(direct_bbox[3] - derived_bbox[3]) > bbox_margin
        ):
            log_msg = (
                f"Sample {sample_id}, Line {i + 1}: Significant difference between "
                f"direct bbox {direct_bbox} and mask-derived bbox {derived_bbox}."
            )
            return [], log_msg

        yolo_instances.append(
            YoloInstanceRecord(
                sample_id=sample_id,
                class_id=class_id,
                polygon_abs=poly_abs,
                derived_mask=derived_mask,
                bbox=direct_bbox,  # Use the direct calculation
            )
        )

    return yolo_instances, None


def _process_original_sample(
    original_sample: SegmSample,
    phrase_map: Dict[str, Dict[str, Any]],
    mask_type: str,
) -> Tuple[List[OriginalInstanceRecord], Optional[str]]:
    """Process the original SegmSample to extract expected instances."""
    expected_instances: List[OriginalInstanceRecord] = []

    for seg_idx, segment in enumerate(original_sample.segments):
        # Check if this segment corresponds to a class we care about
        mapping_result = _get_mapping_info(segment, phrase_map)

        if mapping_result is None:
            continue  # Skip segment (not mapped)

        mapping_info, _ = mapping_result
        class_id = mapping_info["class_id"]

        # Select the correct mask list (visible or full)
        masks_to_process = segment.visible_masks if mask_type == "visible" else segment.full_masks

        for mask_idx, mask in enumerate(masks_to_process):
            if mask.is_valid and mask.bbox is not None:  # Need bbox for record
                expected_instances.append(
                    OriginalInstanceRecord(
                        sample_id=original_sample.id,
                        segment_idx=seg_idx,
                        mask_idx=mask_idx,
                        class_id=class_id,
                        original_mask=mask.binary_mask,  # Store the actual mask
                        bbox=mask.bbox,  # Store the original bbox
                    )
                )
            elif not mask.is_valid:
                log_msg = (
                    f"Sample {original_sample.id}, Seg {seg_idx}, Mask {mask_idx}: "
                    f"Skipping invalid original mask."
                )
                return [], log_msg
            elif mask.bbox is None:
                log_msg = (
                    f"Sample {original_sample.id}, Seg {seg_idx}, Mask {mask_idx}: "
                    f"Original mask is valid but has no bbox? Skipping."
                )
                return [], log_msg

    return expected_instances, None


def _group_instances_by_class(instances: List[Any]) -> Dict[int, List[Any]]:
    """Groups instances by their class_id.

    Args:
        instances: List of instances (either OriginalInstanceRecord or YoloInstanceRecord)

    Returns:
        Dict mapping class_id to lists of instances of that class
    """
    instances_by_class: Dict[int, List[Any]] = {}

    for instance in instances:
        class_id = instance.class_id
        if class_id not in instances_by_class:
            instances_by_class[class_id] = []
        instances_by_class[class_id].append(instance)

    return instances_by_class


def _match_instances_for_class(
    expected_instances: List[OriginalInstanceRecord],
    yolo_instances: List[YoloInstanceRecord],
    iou_cutoff: float,
) -> Dict[str, Any]:
    """Match instances of a single class using both mask and bbox IoU.

    Extracts relevant data (masks, bboxes) before calling the generic
    match_instances function with the direct IoU calculation methods.

    Args:
        expected_instances: List of expected instances for a single class
        yolo_instances: List of YOLO instances for the same class
        iou_cutoff: Minimum IoU threshold for matching

    Returns:
        Dict containing match results for both methods, including IoU values in matched pairs
    """
    match_results = {}

    # 1. Extract data needed for matching
    original_masks = [inst.original_mask for inst in expected_instances]
    yolo_masks = [inst.derived_mask for inst in yolo_instances]
    original_bboxes = [np.array(inst.bbox) for inst in expected_instances]
    yolo_bboxes = [np.array(inst.bbox) for inst in yolo_instances]

    # 2. Remove intermediate helper functions
    # def _compute_mask_iou_for_match(...):
    # def _compute_bbox_iou_for_match(...):

    # 3. Modify _perform_matching to use extracted data and direct IoU funcs
    def _perform_matching(match_type):
        """Helper to perform matching with error handling for both match types."""
        try:
            if match_type == "mask":
                data_a = original_masks
                data_b = yolo_masks
                iou_func = calculate_mask_iou
            elif match_type == "bbox":
                data_a = original_bboxes
                data_b = yolo_bboxes
                iou_func = calculate_bbox_iou
            else:
                # Should not happen, but good practice
                raise ValueError(f"Unknown match_type: {match_type}")

            matched_indices, lost_indices, extra_indices = match_instances(
                dataset_a=data_a,
                dataset_b=data_b,
                compute_iou_fn=iou_func,  # Use direct IoU function
                iou_cutoff=iou_cutoff,
                use_hungarian=True,
            )
            match_results[f"{match_type}_matched"] = matched_indices  # includes IoU values
            match_results[f"{match_type}_lost"] = lost_indices
            match_results[f"{match_type}_extra"] = extra_indices
            match_results[f"{match_type}_error"] = None
        except Exception as e:
            error_msg = f"Error during {match_type}-based matching: {e}"
            logger.warning(error_msg)
            match_results[f"{match_type}_matched"] = []
            match_results[f"{match_type}_lost"] = list(range(len(expected_instances)))
            match_results[f"{match_type}_extra"] = list(range(len(yolo_instances)))
            match_results[f"{match_type}_error"] = error_msg

    # Perform mask-based matching
    _perform_matching("mask")

    # Perform bbox-based matching
    _perform_matching("bbox")

    return match_results


def _process_matched_pairs(
    expected_instances: List[OriginalInstanceRecord],
    mask_matched_indices: List[Tuple[int, int, float]],
    bbox_matched_indices: List[Tuple[int, int, float]],
    iou_top: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process both mask and bbox matched pairs using a unified approach.

    Takes advantage of the IoU values already calculated during matching.
    Avoids unnecessary cross-type IoU calculations with a generic processing function.

    Args:
        expected_instances: List of expected instances
        mask_matched_indices: List of (expected_idx, yolo_idx, mask_iou) tuples
        bbox_matched_indices: List of (expected_idx, yolo_idx, bbox_iou) tuples
        iou_top: High quality threshold for IoU

    Returns:
        Tuple of (mask_matched_pairs, bbox_matched_pairs)
    """

    def _process_match(
        orig_idx: int, yolo_idx: int, iou_value: float, match_type: str
    ) -> Dict[str, Any]:
        """Process a single match generically for any match type."""
        orig_inst = expected_instances[orig_idx]

        match_info = {
            "original_segment_idx": orig_inst.segment_idx,
            "original_mask_idx": orig_inst.mask_idx,
            "yolo_instance_index": yolo_idx,
            "class_id": orig_inst.class_id,
            "iou": iou_value,
            "threshold_passed": iou_value >= iou_top,
            "match_type": match_type,
        }
        return match_info

    # Process mask-based matches first
    mask_matched_pairs = [
        _process_match(orig_idx, yolo_idx, iou_value, "mask")
        for orig_idx, yolo_idx, iou_value in mask_matched_indices
    ]

    # Process bbox-based matches unconditionally using list comprehension
    bbox_matched_pairs = [
        _process_match(orig_idx, yolo_idx, iou_value, "bbox")
        for orig_idx, yolo_idx, iou_value in bbox_matched_indices
    ]

    return mask_matched_pairs, bbox_matched_pairs


def _process_lost_and_extra_instances(
    expected_instances: List[OriginalInstanceRecord],
    yolo_instances: List[YoloInstanceRecord],
    lost_indices: List[int],
    extra_indices: List[int],
    existing_lost: List[OriginalInstanceRecord],
    existing_extra: List[YoloInstanceRecord],
) -> Tuple[List[OriginalInstanceRecord], List[YoloInstanceRecord]]:
    """Process lost and extra instances, avoiding duplicates with existing ones.

    Args:
        expected_instances: List of expected instances
        yolo_instances: List of YOLO instances
        lost_indices: Indices of lost instances in expected_instances
        extra_indices: Indices of extra instances in yolo_instances
        existing_lost: Already identified lost instances to check against
        existing_extra: Already identified extra instances to check against

    Returns:
        Tuple of (new_lost_instances, new_extra_instances)
    """
    # Set of IDs for existing lost instances
    existing_lost_ids = {(lost.segment_idx, lost.mask_idx) for lost in existing_lost}

    # Set of IDs for existing extra instances
    existing_extra_ids = {
        (extra.sample_id, extra.class_id, tuple(extra.bbox)) for extra in existing_extra
    }

    # Filter lost instances
    new_lost = []
    for idx in lost_indices:
        lost_inst = expected_instances[idx]
        if (lost_inst.segment_idx, lost_inst.mask_idx) not in existing_lost_ids:
            new_lost.append(lost_inst)

    # Filter extra instances
    new_extra = []
    for idx in extra_indices:
        extra_inst = yolo_instances[idx]
        if (
            extra_inst.sample_id,
            extra_inst.class_id,
            tuple(extra_inst.bbox),
        ) not in existing_extra_ids:
            new_extra.append(extra_inst)

    return new_lost, new_extra


def verify_sample_conversion(
    sample_id: str,
    yolo_label_path: Path,
    yolo_image_path: Path,
    original_sample: Optional[SegmSample],
    phrase_map: Dict[str, Dict[str, Any]],
    mask_type: str,
    iou_cutoff: float,
    iou_top: float,
) -> VerificationResult:
    """Verifies the conversion for a single sample ID."""

    result = VerificationResult(sample_id=sample_id)

    # 1. Load and parse YOLO data
    yolo_mcls_instances, yolo_load_error = _load_yolo_instances(
        sample_id, yolo_label_path, yolo_image_path
    )
    if yolo_load_error:
        logger.error(f"Failed to load YOLO data for {sample_id}: {yolo_load_error}")
        result.processing_error = f"YOLO Load Error: {yolo_load_error}"
        return result

    # 2. Check if original_sample is available (loaded by main loop)
    if original_sample is None:
        logger.error(f"Original SegmSample for {sample_id} was not loaded. Cannot verify.")
        # If YOLO instances exist, they are all considered 'extra'
        result.mask_extra_instances.extend(yolo_mcls_instances)
        result.bbox_extra_instances.extend(yolo_mcls_instances)
        result.processing_error = "Original HF sample not loaded"
        return result

    # 3. Process original_sample -> expected instances
    expected_mcls_instances, original_process_error = _process_original_sample(
        original_sample, phrase_map, mask_type
    )
    if original_process_error:
        logger.error(f"Error processing original sample for {sample_id}: {original_process_error}")
        result.processing_error = f"Original Process Error: {original_process_error}"
        return result

    # 4. Group instances by class for both expected and YOLO instances
    expected_by_class = _group_instances_by_class(expected_mcls_instances)
    yolo_by_class = _group_instances_by_class(yolo_mcls_instances)

    # Get all unique class IDs from both sets
    all_class_ids = set(expected_by_class.keys()) | set(yolo_by_class.keys())
    logger.debug(f"Found {len(all_class_ids)} unique class IDs across expected and YOLO")

    # 5. Process each class separately
    for class_id in all_class_ids:
        expected_instances = expected_by_class.get(class_id, [])
        yolo_instances = yolo_by_class.get(class_id, [])

        # Log counts for this class
        logger.debug(
            f"Class {class_id}: {len(expected_instances)} expected, {len(yolo_instances)} YOLO"
        )

        # Skip if either list is empty
        if not expected_instances or not yolo_instances:
            # If expected instances exist but no YOLO instances, they're all lost
            if expected_instances:
                result.mask_lost_instances.extend(expected_instances)
                result.bbox_lost_instances.extend(expected_instances)

            # If YOLO instances exist but no expected instances, they're all extra
            if yolo_instances:
                result.mask_extra_instances.extend(yolo_instances)
                result.bbox_extra_instances.extend(yolo_instances)

            continue

        # Perform matching for this class
        match_results = _match_instances_for_class(expected_instances, yolo_instances, iou_cutoff)

        # Process mask-based matches
        mask_pairs, bbox_pairs = _process_matched_pairs(
            expected_instances,
            match_results["mask_matched"],
            match_results["bbox_matched"],
            iou_top,
        )
        result.mask_matched_pairs.extend(mask_pairs)
        result.bbox_matched_pairs.extend(bbox_pairs)

        # Process lost and extra instances for mask-based matching
        mask_lost, mask_extra = _process_lost_and_extra_instances(
            expected_instances,
            yolo_instances,
            match_results["mask_lost"],
            match_results["mask_extra"],
            result.mask_lost_instances,
            result.mask_extra_instances,
        )
        result.mask_lost_instances.extend(mask_lost)
        result.mask_extra_instances.extend(mask_extra)

        # Process lost and extra instances for bbox-based matching
        bbox_lost, bbox_extra = _process_lost_and_extra_instances(
            expected_instances,
            yolo_instances,
            match_results["bbox_lost"],
            match_results["bbox_extra"],
            result.bbox_lost_instances,
            result.bbox_extra_instances,
        )
        result.bbox_lost_instances.extend(bbox_lost)
        result.bbox_extra_instances.extend(bbox_extra)

    # Mark as completed (no error)
    result.processing_error = None

    # Log summary
    logger.debug(
        f"Verification completed for {sample_id}: "
        f"Mask-based: {len(result.mask_matched_pairs)} matches, "
        f"{len(result.mask_lost_instances)} lost, {len(result.mask_extra_instances)} extra. "
        f"Bbox-based: {len(result.bbox_matched_pairs)} matches, "
        f"{len(result.bbox_lost_instances)} lost, {len(result.bbox_extra_instances)} extra."
    )

    return result
