"""Core logic for verifying YOLO dataset conversion against original HF dataset."""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

# Local imports
from vibelab.dataops.cov_segm.datamodel import ClsSegment, SegmSample
from vibelab.utils.common.bbox import calculate_iou as compute_bbox_iou
from vibelab.utils.common.geometry import compute_mask_iou, polygon_to_mask

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
    bbox_iou_failures: List[Dict[str, Any]] = field(default_factory=list)
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
            logger.debug(
                f"Segment for phrase '{matched_phrase}' skipped due to sampling (ratio={effective_ratio:.3f})."
            )
            # stats_counters["skipped_segments_sampling"] += 1 # Removed stats
            return None
        else:
            logger.debug(
                f"Segment for phrase '{matched_phrase}' kept after sampling (ratio={effective_ratio:.3f})."
            )

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
            return (
                [],
                f"Invalid format in {yolo_label_path}, line {i + 1}: Expected class_id followed by pairs of coords.",
            )

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
            logger.warning(
                f"Sample {sample_id}, Line {i + 1}: Degenerate polygon after conversion (< 3 points), skipping instance."
            )
            continue

        # Generate mask from absolute polygon
        derived_mask = polygon_to_mask(poly_abs, height, width)
        if derived_mask.sum() == 0:
            logger.warning(
                f"Sample {sample_id}, Line {i + 1}: Polygon resulted in empty mask, skipping instance."
            )
            continue

        # Calculate bounding box from the derived mask
        bbox = _calculate_bbox_from_mask(derived_mask)
        if bbox is None:
            logger.warning(
                f"Sample {sample_id}, Line {i + 1}: Could not calculate bbox from derived mask, skipping instance."
            )
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
                logger.debug(
                    f"Sample {original_sample.id}, Seg {seg_idx}, Mask {mask_idx}: Skipping invalid original mask."
                )
            elif mask.bbox is None:
                logger.warning(
                    f"Sample {original_sample.id}, Seg {seg_idx}, Mask {mask_idx}: Original mask is valid but has no bbox? Skipping."
                )

    return expected_instances


def _group_instances_by_class(
    original_instances: List[OriginalInstanceRecord], yolo_instances: List[YoloInstanceRecord]
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Set[int]]:
    """Group original and YOLO instances by class ID.

    Args:
        original_instances: List of expected instances from the original sample.
        yolo_instances: List of instances parsed from the YOLO label.

    Returns:
        A tuple containing:
        - orig_by_class: Dictionary mapping class IDs to lists of indices in original_instances
        - yolo_by_class: Dictionary mapping class IDs to lists of indices in yolo_instances
        - all_class_ids: Set of all unique class IDs from both original and YOLO instances
    """
    orig_by_class: Dict[int, List[int]] = {}
    yolo_by_class: Dict[int, List[int]] = {}

    for i, inst in enumerate(original_instances):
        orig_by_class.setdefault(inst.class_id, []).append(i)

    for i, inst in enumerate(yolo_instances):
        yolo_by_class.setdefault(inst.class_id, []).append(i)

    all_class_ids = set(orig_by_class.keys()) | set(yolo_by_class.keys())

    return orig_by_class, yolo_by_class, all_class_ids


def _build_cost_matrix(
    original_instances: List[OriginalInstanceRecord],
    yolo_instances: List[YoloInstanceRecord],
    orig_indices: List[int],
    yolo_indices: List[int],
    mask_min_iou: float,
    class_id: int,
) -> np.ndarray:
    """Build the cost matrix for matching instances of a given class.

    Args:
        original_instances: List of expected instances from the original sample.
        yolo_instances: List of instances parsed from the YOLO label.
        orig_indices: Indices in original_instances for instances of this class.
        yolo_indices: Indices in yolo_instances for instances of this class.
        mask_min_iou: The minimum mask IoU threshold for a match.
        class_id: The class ID being processed (for logging).

    Returns:
        A cost matrix where cost_matrix[i, j] contains -IoU or a large negative value
        (negative IoU used for minimization algorithm).
    """
    num_orig = len(orig_indices)
    num_yolo = len(yolo_indices)

    # Use a large negative finite value instead of -inf to avoid numerical issues
    # with linear_sum_assignment while still heavily penalizing non-matches
    VERY_LARGE_COST = -1e9  # Large negative value instead of -inf

    # Initialize with large negative value (will be excluded from matching)
    cost_matrix = np.full((num_orig, num_yolo), VERY_LARGE_COST)

    for r_idx, orig_idx in enumerate(orig_indices):
        for c_idx, yolo_idx in enumerate(yolo_indices):
            try:
                iou = compute_mask_iou(
                    original_instances[orig_idx].original_mask,
                    yolo_instances[yolo_idx].derived_mask,
                )
                if iou >= mask_min_iou:
                    # Store negative IoU because linear_sum_assignment finds minimum cost
                    cost_matrix[r_idx, c_idx] = -iou
                # If iou < threshold, cost remains VERY_LARGE_COST (preventing match)
            except ValueError as e:
                logger.warning(
                    f"Error computing mask IoU for class {class_id} between "
                    f"orig[{orig_idx}] and yolo[{yolo_idx}]: {e}. Skipping pair."
                )

    return cost_matrix


def _match_class_instances(
    original_instances: List[OriginalInstanceRecord],
    yolo_instances: List[YoloInstanceRecord],
    mask_min_iou: float,
    class_id: int,
    orig_indices: List[int],
    yolo_indices: List[int],
    orig_matched_mask: np.ndarray,
    yolo_matched_mask: np.ndarray,
) -> List[Tuple[int, int]]:
    """Match instances of a specific class using the Hungarian algorithm.

    Args:
        original_instances: List of expected instances from the original sample.
        yolo_instances: List of instances parsed from the YOLO label.
        mask_min_iou: The minimum mask IoU threshold for a match.
        class_id: The class ID being processed.
        orig_indices: Indices in original_instances for instances of this class.
        yolo_indices: Indices in yolo_instances for instances of this class.
        orig_matched_mask: Boolean mask of already matched original instances.
        yolo_matched_mask: Boolean mask of already matched YOLO instances.

    Returns:
        List of tuples (orig_idx, yolo_idx) of matched instances.
    """
    if not orig_indices or not yolo_indices:
        return []  # No instances of this class in one of the lists

    cost_matrix = _build_cost_matrix(
        original_instances, yolo_instances, orig_indices, yolo_indices, mask_min_iou, class_id
    )

    # Define the threshold for considering a value as essentially -inf
    VERY_LARGE_COST_THRESHOLD = -1e8

    # If cost_matrix contains only very large negative values, assignment will be meaningless
    if np.all(cost_matrix < VERY_LARGE_COST_THRESHOLD):
        logger.debug(f"No pairs met IoU threshold for class {class_id}. Skipping assignment.")
        return []

    # Handle cost matrix with valid costs
    matched_pairs = []
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Process matches from assignment
        for r, c in zip(row_ind, col_ind):
            # Check if the assigned cost is valid (not extremely negative, meaning IoU >= threshold)
            if cost_matrix[r, c] > VERY_LARGE_COST_THRESHOLD:
                orig_actual_idx = orig_indices[r]
                yolo_actual_idx = yolo_indices[c]
                matched_pairs.append((orig_actual_idx, yolo_actual_idx))
                orig_matched_mask[orig_actual_idx] = True
                yolo_matched_mask[yolo_actual_idx] = True

    except ValueError as e:
        # This should no longer happen with finite values, but keep as a safeguard
        logger.error(
            f"linear_sum_assignment failed for class {class_id} despite using finite values: {e}. "
            f"Cost matrix shape: {cost_matrix.shape}"
        )
        # Fallback to greedy matching
        logger.info(f"Falling back to greedy matching for class {class_id}.")

        # Create flattened list of (row, col, cost) tuples
        valid_costs = []
        for r, orig_idx in enumerate(orig_indices):
            for c, yolo_idx in enumerate(yolo_indices):
                cost = cost_matrix[r, c]
                if cost > VERY_LARGE_COST_THRESHOLD:
                    valid_costs.append((r, c, cost))

        # Sort by cost (ascending, since we have negative IoUs)
        valid_costs.sort(key=lambda x: x[2])

        # Greedily assign matches
        matched_r = set()
        matched_c = set()

        for r, c, _ in valid_costs:
            if r not in matched_r and c not in matched_c:
                orig_actual_idx = orig_indices[r]
                yolo_actual_idx = yolo_indices[c]
                matched_pairs.append((orig_actual_idx, yolo_actual_idx))
                orig_matched_mask[orig_actual_idx] = True
                yolo_matched_mask[yolo_actual_idx] = True
                matched_r.add(r)
                matched_c.add(c)

    return matched_pairs


def _match_instances(
    original_instances: List[OriginalInstanceRecord],
    yolo_instances: List[YoloInstanceRecord],
    mask_min_iou: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Matches original and YOLO instances based on class ID and mask IoU.

    Uses the Hungarian algorithm for optimal assignment within each class.

    Args:
        original_instances: List of expected instances from the original sample.
        yolo_instances: List of instances parsed from the YOLO label.
        mask_min_iou: The minimum mask IoU threshold for a match.

    Returns:
        A tuple containing:
        - matched_indices: List of tuples (orig_idx, yolo_idx) of matched instances.
        - lost_indices: List of indices into original_instances that were not matched.
        - extra_indices: List of indices into yolo_instances that were not matched.
    """
    matched_indices: List[Tuple[int, int]] = []
    orig_matched_mask = np.zeros(len(original_instances), dtype=bool)
    yolo_matched_mask = np.zeros(len(yolo_instances), dtype=bool)

    # Group instances by class ID
    orig_by_class, yolo_by_class, all_class_ids = _group_instances_by_class(
        original_instances, yolo_instances
    )

    # Match within each class
    for class_id in all_class_ids:
        orig_indices = orig_by_class.get(class_id, [])
        yolo_indices = yolo_by_class.get(class_id, [])

        class_matches = _match_class_instances(
            original_instances,
            yolo_instances,
            mask_min_iou,
            class_id,
            orig_indices,
            yolo_indices,
            orig_matched_mask,
            yolo_matched_mask,
        )

        matched_indices.extend(class_matches)

    # Identify unmatched instances
    lost_indices = [i for i, matched in enumerate(orig_matched_mask) if not matched]
    extra_indices = [i for i, matched in enumerate(yolo_matched_mask) if not matched]

    return matched_indices, lost_indices, extra_indices


def verify_sample_conversion(
    sample_id: str,
    yolo_label_path: Path,
    yolo_image_path: Path,
    original_sample: Optional[SegmSample],
    phrase_map: Dict[str, Dict[str, Any]],
    mask_type: str,
    mask_min_iou: float,
    bbox_min_iou: float,
    global_sample_ratio: float = 1.0,  # Add this if reusing _get_sampled_mapping_info
) -> VerificationResult:
    """Verifies the conversion for a single sample ID."""

    result = VerificationResult(sample_id=sample_id)

    # --- TODO: Implement core logic --- #
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

    # 4. Match instances
    matched_indices, lost_indices, extra_indices = _match_instances(
        original_instances_expected, yolo_instances, mask_min_iou
    )

    # 5. Process matches and check bbox IoU
    for orig_idx, yolo_idx in matched_indices:
        orig_inst = original_instances_expected[orig_idx]
        yolo_inst = yolo_instances[yolo_idx]

        # Re-calculate mask IoU for reporting (linear_sum_assignment gives cost)
        mask_iou = compute_mask_iou(orig_inst.original_mask, yolo_inst.derived_mask)

        # Calculate bbox IoU
        bbox_iou = compute_bbox_iou(np.array(orig_inst.bbox), np.array(yolo_inst.bbox))

        match_info = {
            "original_segment_idx": orig_inst.segment_idx,
            "original_mask_idx": orig_inst.mask_idx,
            "yolo_instance_index": yolo_idx,  # Index within the parsed yolo list
            "class_id": orig_inst.class_id,
            "mask_iou": mask_iou,
            "bbox_iou": bbox_iou,
            "bbox_threshold_passed": bbox_iou >= bbox_min_iou,
        }
        result.matched_pairs.append(match_info)

        # Record bbox IoU failures separately
        if not match_info["bbox_threshold_passed"]:
            result.bbox_iou_failures.append(match_info)

    # 6. Populate lost and extra instances
    result.lost_instances = [original_instances_expected[i] for i in lost_indices]
    result.extra_instances = [yolo_instances[i] for i in extra_indices]

    # Mark as completed (no error)
    result.processing_error = None

    logger.debug(
        f"Verification completed for {sample_id}: {len(result.matched_pairs)} matches, "
        f"{len(result.lost_instances)} lost, {len(result.extra_instances)} extra, "
        f"{len(result.bbox_iou_failures)} bbox fails."
    )

    return result
