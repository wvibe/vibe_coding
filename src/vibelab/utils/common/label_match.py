"""Generic instance matching utilities based on Hungarian algorithm."""

import logging
from typing import Any, Callable, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# Constant used in the IoU matrix for pairs below the threshold.
# Since valid IoUs are [0, 1], -1.0 is a safe indicator for the maximizer to ignore.
_INVALID_IOU_PLACEHOLDER = -1.0


def _build_iou_matrix(
    dataset_a: List[Any],
    dataset_b: List[Any],
    compute_iou_fn: Callable[[Any, Any], float],
    iou_threshold: float,
) -> np.ndarray:
    """Build the IoU matrix for maximization.

    Args:
        dataset_a: First list of objects to match.
        dataset_b: Second list of objects to match.
        compute_iou_fn: Function that computes IoU between two objects.
        iou_threshold: The minimum IoU threshold for a match.

    Returns:
        An IoU matrix where matrix[i, j] contains IoU if it's >= threshold,
        otherwise _INVALID_IOU_PLACEHOLDER.
    """
    num_a = len(dataset_a)
    num_b = len(dataset_b)

    # Initialize with the placeholder value
    iou_matrix = np.full((num_a, num_b), _INVALID_IOU_PLACEHOLDER, dtype=np.float64)

    for i in range(num_a):
        for j in range(num_b):
            try:
                iou = compute_iou_fn(dataset_a[i], dataset_b[j])
                if iou >= iou_threshold:
                    iou_matrix[i, j] = iou
                # If iou < threshold, value remains _INVALID_IOU_PLACEHOLDER
            except ValueError as e:
                logger.warning(
                    f"Error computing IoU between a[{i}] and b[{j}]: {e}. Skipping pair."
                )

    logger.debug(f"IoU Matrix (for maximization) AFTER build:\n{iou_matrix}")  # DEBUG PRINT
    return iou_matrix


def match_instances(
    dataset_a: List[Any],
    dataset_b: List[Any],
    compute_iou_fn: Callable[[Any, Any], float],
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match instances between two datasets using the Hungarian algorithm based on IoU.

    Uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment) to find
    the optimal assignment that maximizes the sum of IoUs between matched pairs,
    considering only pairs with IoU >= iou_threshold.

    Args:
        dataset_a: First list of objects to match.
        dataset_b: Second list of objects to match.
        compute_iou_fn: Function that computes IoU between two objects. This function
                        should take two objects (one from each dataset) and return a
                        float representing the IoU value (expected range [0, 1]).
        iou_threshold: Minimum IoU value to consider a valid match.

    Returns:
        A tuple containing:
        - matched_pairs: List of tuples (a_idx, b_idx) of matched instances.
        - unmatched_a: List of indices in dataset_a that weren't matched.
        - unmatched_b: List of indices in dataset_b that weren't matched.
    """
    matched_pairs: List[Tuple[int, int]] = []
    a_matched_mask = np.zeros(len(dataset_a), dtype=bool)
    b_matched_mask = np.zeros(len(dataset_b), dtype=bool)

    # No instances to match in one of the datasets
    if len(dataset_a) == 0 or len(dataset_b) == 0:
        unmatched_a = list(range(len(dataset_a)))
        unmatched_b = list(range(len(dataset_b)))
        return matched_pairs, unmatched_a, unmatched_b

    # Build IoU matrix (using actual IoU values for maximization)
    iou_matrix = _build_iou_matrix(dataset_a, dataset_b, compute_iou_fn, iou_threshold)
    # logger.debug(f"IoU Matrix (for maximization):\n{iou_matrix}") # DEBUG PRINT (Moved to end of build)

    # If iou_matrix contains only placeholder values, no matches above threshold
    if np.all(iou_matrix <= _INVALID_IOU_PLACEHOLDER):
        logger.debug(f"No pairs met IoU threshold ({iou_threshold}). No matches found.")
        unmatched_a = list(range(len(dataset_a)))
        unmatched_b = list(range(len(dataset_b)))
        return matched_pairs, unmatched_a, unmatched_b

    # Process matches using Hungarian algorithm (maximizing IoU)
    try:
        # maximize=True finds the assignment that maximizes the sum of iou_matrix values
        row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)
        logger.debug(
            f"Assignment indices (row_ind, col_ind): ({row_ind}, {col_ind})"
        )  # DEBUG PRINT

        # Process matches from assignment
        for r, c in zip(row_ind, col_ind, strict=False):
            # Check if the assigned pair had a valid IoU (i.e., not the placeholder)
            # This ensures we only pair items meeting the threshold.
            current_iou = iou_matrix[r, c]
            is_valid = current_iou >= iou_threshold
            logger.debug(  # DEBUG PRINT
                f"Checking pair ({r}, {c}): IoU={current_iou}, Threshold={iou_threshold}, Valid={is_valid}"
            )
            if is_valid:
                matched_pairs.append((r, c))
                a_matched_mask[r] = True
                b_matched_mask[c] = True

    except ValueError as e:
        # Keep fallback just in case, although less likely needed now
        logger.error(f"linear_sum_assignment failed: {e}. IoU matrix shape: {iou_matrix.shape}")
        logger.info("Falling back to greedy matching.")

        # Create flattened list of (row, col, iou) tuples with valid IoU
        valid_ious = []
        for r in range(len(dataset_a)):
            for c in range(len(dataset_b)):
                iou = iou_matrix[r, c]
                # Check against threshold, not placeholder
                if iou >= iou_threshold:
                    valid_ious.append((r, c, iou))

        # Sort by IoU (descending for greedy best match)
        valid_ious.sort(key=lambda x: x[2], reverse=True)

        # Greedily assign matches
        # Re-initialize masks for the greedy part
        a_matched_mask.fill(False)
        b_matched_mask.fill(False)
        matched_pairs = []  # Reset matches
        matched_r = set()
        matched_c = set()

        for r, c, _ in valid_ious:
            if r not in matched_r and c not in matched_c:
                matched_pairs.append((r, c))
                a_matched_mask[r] = True
                b_matched_mask[c] = True
                matched_r.add(r)
                matched_c.add(c)

    # Identify unmatched instances
    unmatched_a = [i for i, matched in enumerate(a_matched_mask) if not matched]
    unmatched_b = [i for i, matched in enumerate(b_matched_mask) if not matched]

    return matched_pairs, unmatched_a, unmatched_b
  