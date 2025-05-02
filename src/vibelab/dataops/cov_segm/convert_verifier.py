"""Core logic for verifying YOLO dataset conversion against original HF dataset."""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Local imports
from vibelab.dataops.cov_segm.datamodel import ClsSegment, SegmSample
from vibelab.utils.common.bbox import calculate_iou as calculate_bbox_iou
from vibelab.utils.common.label_match import match_instances
from vibelab.utils.common.mask import calculate_mask_iou, polygon_to_mask

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
    matched_pairs: List[Dict[str, Any]] = field(default_factory=list)
    lost_instances: List[OriginalInstanceRecord] = field(default_factory=list)
    extra_instances: List[YoloInstanceRecord] = field(default_factory=list)
    processing_error: Optional[str] = None


def _get_sampled_mapping_info(
    segment: ClsSegment,
    phrase_map: Dict[str, Dict[str, Any]],
    global_sample_ratio: float,
) -> Optional[Tuple[Dict[str, Any], str]]:
    """Checks segment phrases against the map and applies sampling.

    Args:
        segment: The segment containing phrases to check against the mapping.
        phrase_map: Mapping of phrases to class information.
        global_sample_ratio: Global sampling ratio (0.0-1.0) to apply.

    Returns:
        Tuple[Dict, str] (mapping_info, matched_phrase) if a sampled match is found,
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
        # stats_counters["skipped_segments_no_mapping"] += 1 # Removed stats
        return None

    # Only apply sampling if a ratio > 0 is specified
    if global_sample_ratio > 0.0:
        local_sampling_ratio = mapping_info.get("sampling_ratio", 1.0)
        effective_ratio = global_sample_ratio * local_sampling_ratio

        # Use the *same* random logic as converter: skip if random() > ratio
        if random.random() > effective_ratio:
            log_msg = (
                f"Segment for phrase '{matched_phrase}' skipped due to sampling "
                f"(ratio={effective_ratio:.3f})."
            )
            logger.debug(log_msg)
            # stats_counters["skipped_segments_sampling"] += 1 # Removed stats
            return None
        else:
            log_msg = (
                f"Segment for phrase '{matched_phrase}' kept after sampling "
                f"(ratio={effective_ratio:.3f})."
            )
            logger.debug(log_msg)

    # If no sampling or sampling passed, return the info
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
        # If the label file doesn't exist, it means no annotations were generated.
        # This is a valid case (e.g., skipped sample), not necessarily an error here.
        logger.debug(f"YOLO label file not found (no annotations generated?): {yolo_label_path}")
        return [], None
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
        for j in range(0, len(norm_coords), 2):
            norm_x, norm_y = norm_coords[j], norm_coords[j + 1]
            abs_x = int(round(norm_x * width))
            abs_y = int(round(norm_y * height))
            # Clip to ensure coords are within image bounds for mask generation
            abs_x = max(0, min(width - 1, abs_x))
            abs_y = max(0, min(height - 1, abs_y))
            poly_abs.append((abs_x, abs_y))

        if len(poly_abs) < 3:
            log_msg = (
                f"Sample {sample_id}, Line {i + 1}: Degenerate polygon after "
                f"conversion (< 3 points), skipping instance."
            )
            logger.warning(log_msg)
            continue

        # Generate mask from absolute polygon
        derived_mask = polygon_to_mask(poly_abs, height, width)
        if derived_mask.sum() == 0:
            log_msg = (
                f"Sample {sample_id}, Line {i + 1}: Polygon resulted in empty "
                f"mask, skipping instance."
            )
            logger.warning(log_msg)
            continue

        # Calculate bounding box from the derived mask
        bbox = _calculate_bbox_from_mask(derived_mask)
        if bbox is None:
            log_msg = (
                f"Sample {sample_id}, Line {i + 1}: Could not calculate bbox from "
                f"derived mask, skipping instance."
            )
            logger.warning(log_msg)
            continue

        yolo_instances.append(
            YoloInstanceRecord(
                sample_id=sample_id,
                class_id=class_id,
                polygon_abs=poly_abs,
                derived_mask=derived_mask,
                bbox=bbox,
            )
        )

    return yolo_instances, None


def _process_original_sample(
    original_sample: SegmSample,
    phrase_map: Dict[str, Dict[str, Any]],
    mask_type: str,
    global_sample_ratio: float,
) -> List[OriginalInstanceRecord]:
    """Process the original SegmSample to extract expected instances."""
    expected_instances: List[OriginalInstanceRecord] = []

    for seg_idx, segment in enumerate(original_sample.segments):
        # Check if this segment corresponds to a class we care about and passes sampling
        mapping_result = _get_sampled_mapping_info(segment, phrase_map, global_sample_ratio)

        if mapping_result is None:
            continue  # Skip segment (not mapped or sampled out)

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
                logger.debug(log_msg)
            elif mask.bbox is None:
                log_msg = (
                    f"Sample {original_sample.id}, Seg {seg_idx}, Mask {mask_idx}: "
                    f"Original mask is valid but has no bbox? Skipping."
                )
                logger.warning(log_msg)

    return expected_instances


def _compute_mask_iou_for_match(
    record_a: OriginalInstanceRecord, record_b: YoloInstanceRecord
) -> float:
    """Helper function to compute mask IoU for the match_instances call."""
    try:
        return calculate_mask_iou(record_a.original_mask, record_b.derived_mask)
    except ValueError as e:
        logger.warning(
            f"Error computing mask IoU for matching between orig "
            f"(sample {record_a.sample_id}, seg {record_a.segment_idx}, mask {record_a.mask_idx}) "
            f"and yolo (sample {record_b.sample_id}): {e}. Returning 0.0 IoU."
        )
        return 0.0


def verify_sample_conversion(
    sample_id: str,
    yolo_label_path: Path,
    yolo_image_path: Path,
    original_sample: Optional[SegmSample],
    phrase_map: Dict[str, Dict[str, Any]],
    mask_type: str,
    iou_cutoff: float,
    iou_top: float,
    global_sample_ratio: float = 1.0,
) -> VerificationResult:
    """Verifies the conversion for a single sample ID."""

    result = VerificationResult(sample_id=sample_id)

    # 1. Load and parse YOLO data
    yolo_instances, yolo_load_error = _load_yolo_instances(
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
        result.extra_instances.extend(yolo_instances)
        result.processing_error = "Original HF sample not loaded"
        return result

    # 3. Process original_sample -> expected instances
    original_instances_expected = _process_original_sample(
        original_sample, phrase_map, mask_type, global_sample_ratio
    )

    # 4. Match instances using generic matcher
    # Note: We match based on mask IoU >= iou_cutoff
    matched_indices, lost_indices, extra_indices = match_instances(
        dataset_a=original_instances_expected,
        dataset_b=yolo_instances,
        compute_iou_fn=_compute_mask_iou_for_match,
        iou_cutoff=iou_cutoff,
        use_hungarian=True,
    )

    # 5. Process matches and calculate quality metrics
    for orig_idx, yolo_idx in matched_indices:
        orig_inst = original_instances_expected[orig_idx]
        yolo_inst = yolo_instances[yolo_idx]

        # Calculate mask IoU (might be slightly different from match helper due to error handling)
        try:
            mask_iou = calculate_mask_iou(orig_inst.original_mask, yolo_inst.derived_mask)
        except ValueError:
            mask_iou = 0.0

        # Calculate bbox IoU
        try:
            bbox_iou = calculate_bbox_iou(np.array(orig_inst.bbox), np.array(yolo_inst.bbox))
        except Exception as e:
            log_msg = (
                f"Error calculating bbox IoU for sample {sample_id}, "
                f"orig_idx {orig_idx}, yolo_idx {yolo_idx}: {e}"
            )
            logger.warning(log_msg)
            bbox_iou = 0.0

        # Check against the high IoU threshold (iou_top)
        mask_threshold_passed = mask_iou >= iou_top
        bbox_threshold_passed = bbox_iou >= iou_top

        match_info = {
            "original_segment_idx": orig_inst.segment_idx,
            "original_mask_idx": orig_inst.mask_idx,
            "yolo_instance_index": yolo_idx,
            "class_id": orig_inst.class_id,
            "mask_iou": mask_iou,
            "bbox_iou": bbox_iou,
            "mask_threshold_passed": mask_threshold_passed,
            "bbox_threshold_passed": bbox_threshold_passed,
        }
        result.matched_pairs.append(match_info)

    # 6. Populate lost and extra instances
    result.lost_instances = [original_instances_expected[i] for i in lost_indices]
    result.extra_instances = [yolo_instances[i] for i in extra_indices]

    # Mark as completed (no error)
    result.processing_error = None

    logger.debug(
        f"Verification completed for {sample_id}: {len(result.matched_pairs)} matches, "
        f"{len(result.lost_instances)} lost, {len(result.extra_instances)} extra."
    )

    return result
