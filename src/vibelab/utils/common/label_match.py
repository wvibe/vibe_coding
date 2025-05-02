"""Generic instance matching utilities based on Hungarian algorithm."""

import logging
from typing import Any, Callable, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# Constant used in the IoU matrix for pairs below the threshold.
# Using a large negative value to ensure pairs below threshold are never selected
_INVALID_IOU_PLACEHOLDER = -1e6


def _build_iou_matrix(
    dataset_a: List[Any],
    dataset_b: List[Any],
    compute_iou_fn: Callable[[Any, Any], float],
    iou_cutoff: float,
) -> np.ndarray:
    """Build the IoU matrix for matching.

    Args:
        dataset_a: First list of objects to match.
        dataset_b: Second list of objects to match.
        compute_iou_fn: Function that computes IoU between two objects.
        iou_cutoff: The minimum IoU threshold for a match.

    Returns:
        An IoU matrix where matrix[i, j] contains IoU if it's >= cutoff,
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
                if not (0 <= iou <= 1):
                    raise ValueError(f"IoU value out of range: {iou}")

                if iou >= iou_cutoff:  # if iou is above the cutoff, set to iou
                    iou_matrix[i, j] = iou
                # If iou < cutoff, value remains _INVALID_IOU_PLACEHOLDER
            except ValueError as e:
                logger.warning(
                    f"Error computing IoU between a[{i}] and b[{j}]: {e}. Skipping pair."
                )

    return iou_matrix


def _match_instances_hungarian(
    iou_matrix: np.ndarray,
    iou_cutoff: float,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Perform optimal matching with the Hungarian algorithm (linear_sum_assignment).

    This version pads the IoU matrix with “dummy” rows / columns so that:
    • Every real row/column can be assigned either to a real partner **or** to a dummy.
    • Dummy cells have cost = 0 ( < iou_cutoff ) so they’ll never be accepted as a
      *valid* match, but they stop the algorithm from being forced to pair two invalid
      real elements simply to satisfy the square‑matrix requirement.

    Args
    ----
    iou_matrix : 2‑D ndarray
        IoU scores for every (row, col) pair. Invalid pairs must already be set to
        _INVALID_IOU_PLACEHOLDER.
    iou_cutoff : float
        Minimum IoU considered a valid match.

    Returns
    -------
    matched_pairs : List[Tuple[int,int]]
        List of (row_idx_in_A, col_idx_in_B) for accepted matches (IoU ≥ cutoff).
    a_matched_mask : ndarray[bool]
        True for rows that were matched to a *valid* column.
    b_matched_mask : ndarray[bool]
        True for columns that were matched to a *valid* row.
    """
    n_rows, n_cols = iou_matrix.shape
    size = max(n_rows, n_cols)  # final square size after padding

    # --- Pad matrix with dummy rows / columns ---------------------------------
    padded = np.full((size, size), _INVALID_IOU_PLACEHOLDER, dtype=iou_matrix.dtype)
    padded[:n_rows, :n_cols] = iou_matrix

    #   • Real‑to‑dummy and dummy‑to‑real cells use cost 0
    #     (maximizer treats 0 < any real IoU ≥ iou_cutoff)
    padded[n_rows:, :] = 0.0  # dummy rows
    padded[:, n_cols:] = 0.0  # dummy cols
    # --------------------------------------------------------------------------

    # Hungarian algorithm – SciPy ≥ 1.11 supports `maximize=True`
    row_ind, col_ind = linear_sum_assignment(padded, maximize=True)

    matched_pairs: List[Tuple[int, int]] = []
    a_matched_mask = np.zeros(n_rows, dtype=bool)
    b_matched_mask = np.zeros(n_cols, dtype=bool)

    for r, c in zip(row_ind, col_ind):
        # Skip anything that involves a dummy row/col
        if r >= n_rows or c >= n_cols:
            continue
        # Accept only if IoU meets the cutoff
        if padded[r, c] >= iou_cutoff:
            matched_pairs.append((r, c))
            a_matched_mask[r] = True
            b_matched_mask[c] = True

    return matched_pairs, a_matched_mask, b_matched_mask


def _match_instances_greedy(
    iou_matrix: np.ndarray, iou_cutoff: float
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """Perform greedy matching based on highest IoU first.

    Note: Assumes iou_matrix is a valid 2D ndarray.
    """
    n_rows, n_cols = iou_matrix.shape
    matched_pairs: List[Tuple[int, int]] = []
    a_matched_mask = np.zeros(n_rows, dtype=bool)
    b_matched_mask = np.zeros(n_cols, dtype=bool)

    # Create flattened list of (row, col, iou) tuples with valid IoU
    valid_ious = []
    for r in range(n_rows):
        for c in range(n_cols):
            iou = iou_matrix[r, c]
            # Check against cutoff
            if iou >= iou_cutoff:
                valid_ious.append((r, c, iou))

    # Sort by IoU (descending for greedy best match)
    valid_ious.sort(key=lambda x: x[2], reverse=True)

    # Greedily assign matches
    matched_r = set()
    matched_c = set()

    for r, c, _ in valid_ious:
        if r not in matched_r and c not in matched_c:
            matched_pairs.append((r, c))
            a_matched_mask[r] = True
            b_matched_mask[c] = True
            matched_r.add(r)
            matched_c.add(c)

    return matched_pairs, a_matched_mask, b_matched_mask


def match_instances(
    dataset_a: List[Any],
    dataset_b: List[Any],
    compute_iou_fn: Callable[[Any, Any], float],
    iou_cutoff: float = 0.5,
    use_hungarian: bool = True,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match instances between two datasets using Hungarian or Greedy algorithm.

    Builds an IoU matrix and then uses the specified algorithm to find matches.

    Args:
        dataset_a: First list of objects to match.
        dataset_b: Second list of objects to match.
        compute_iou_fn: Function that computes IoU (range [0, 1]).
        iou_cutoff: Minimum IoU value to consider a valid match.
        use_hungarian: If True, use Hungarian algorithm (optimal), otherwise use Greedy.

    Returns:
        Tuple: (matched_pairs, unmatched_a_indices, unmatched_b_indices).
    """
    # Handle empty datasets early
    if len(dataset_a) == 0 or len(dataset_b) == 0:
        return [], list(range(len(dataset_a))), list(range(len(dataset_b)))

    # Build IoU matrix
    iou_matrix = _build_iou_matrix(dataset_a, dataset_b, compute_iou_fn, iou_cutoff)

    # Check if any valid IoUs were found
    if np.all(iou_matrix <= _INVALID_IOU_PLACEHOLDER + 1):
        # Note: Use +1 for floating point safety when comparing with _INVALID_IOU_PLACEHOLDER
        logger.debug(f"No pairs met IoU cutoff ({iou_cutoff}). No matches found.")
        return [], list(range(len(dataset_a))), list(range(len(dataset_b)))

    # Perform matching using the selected algorithm
    if use_hungarian:
        matched_pairs, a_matched_mask, b_matched_mask = _match_instances_hungarian(
            iou_matrix, iou_cutoff
        )
    else:
        matched_pairs, a_matched_mask, b_matched_mask = _match_instances_greedy(
            iou_matrix, iou_cutoff
        )

    # Identify unmatched instances
    unmatched_a = [i for i, matched in enumerate(a_matched_mask) if not matched]
    unmatched_b = [i for i, matched in enumerate(b_matched_mask) if not matched]

    return matched_pairs, unmatched_a, unmatched_b
