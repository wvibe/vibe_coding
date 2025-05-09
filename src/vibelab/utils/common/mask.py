"""Geometric utility functions for mask operations."""

import logging
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default minimum contour area to consider for polygon conversion
DEFAULT_MIN_CONTOUR_AREA = 1.0
# Default tolerance factor for polygon approximation (relative to arc length)
DEFAULT_POLYGON_APPROX_TOLERANCE = 0.01

_debug_save_counter = 0
DEBUG_MASK_DIR = "tmp/mask_debug"


def _preprocess_binary_mask(binary_mask: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """Validate and preprocess binary mask for contour finding.

    Args:
        binary_mask: Input mask to validate and preprocess
        img_shape: Target (height, width) for validation

    Returns:
        Processed mask as uint8 with values 0/255

    Raises:
        ValueError: For invalid image shape or mask dimensions
    """
    # Check image shape validity
    h, w = img_shape
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape: {img_shape}")

    # Check mask dimensions
    if binary_mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got {binary_mask.ndim}D")

    if binary_mask.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # Normalize to uint8 with values 0/255
    if binary_mask.dtype != np.uint8:
        max_val = binary_mask.max()
        if max_val == 1:
            mask_uint8 = (binary_mask * 255).astype(np.uint8)
        elif max_val == 0:  # Empty mask
            mask_uint8 = binary_mask.astype(np.uint8)
        elif max_val == 255:
            mask_uint8 = binary_mask.astype(np.uint8)  # Already uint8 0/255
        else:
            # Convert non-zero values to 255
            mask_uint8 = np.where(binary_mask > 0, 255, 0).astype(np.uint8)
    else:
        mask_uint8 = binary_mask

    return mask_uint8


def _find_and_simplify_contours(
    binary_mask: np.ndarray,
    min_contour_area: float,
    polygon_approx_tolerance: float,
) -> List[np.ndarray]:
    """Find, filter, and optionally simplify contours from a binary mask.

    Operates on a copy of the input mask to avoid modifying the original.
    Filtering and simplification are applied conditionally based on parameters.

    Args:
        binary_mask: A 2D binary image (dtype uint8) where non-zero pixels
                    represent the object(s) of interest.
        min_contour_area: Minimum contour area (in pixels) to consider. If <= 0,
                        no area filtering is applied.
        polygon_approx_tolerance: Approximation tolerance factor for simplification
                                (relative to arc length). If <= 0, no
                                simplification is applied.

    Returns:
        A list of contours that pass the filtering criteria, where each contour
        is a numpy array of shape (N, 1, 2) representing the vertices.
    """
    # Create a copy to avoid modifying the original mask
    mask_copy = binary_mask.copy()

    # Find contours
    contours, _ = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    valid_contours = []
    for contour in contours:
        processed_contour = contour  # Start with the original contour

        # Conditional Area Filtering
        if min_contour_area > 0:
            area = cv2.contourArea(processed_contour)
            if area < min_contour_area:
                continue

        # Conditional Polygon Approximation (Simplification)
        if polygon_approx_tolerance > 0:
            epsilon = polygon_approx_tolerance * cv2.arcLength(processed_contour, True)
            processed_contour = cv2.approxPolyDP(processed_contour, epsilon, True)

        # Need at least 3 points for a valid polygon
        if len(processed_contour) >= 3:
            valid_contours.append(processed_contour)

    return valid_contours


def _connect_contours_stitched(
    simplified_contours: List[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Connect multiple contours using stitching logic.

    Args:
        simplified_contours: List of simplified contours to connect

    Returns:
        Tuple of (result, error) where:
        - result: Single (M, 2) polygon array (pixel coords) or None if connection fails
        - error: None on success, or error message explaining why connection failed
    """
    # Calculate centroids and sort (Y then X)
    centroids = []
    for contour in simplified_contours:
        # Ensure contour is float32 for moments
        contour_float = contour.astype(np.float32)
        M = cv2.moments(contour_float)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else np.mean(contour[:, 0, 0])
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else np.mean(contour[:, 0, 1])
        centroids.append((cx, cy))

    # Sort contours based on centroids (top-to-bottom, then left-to-right)
    sorted_indices = sorted(
        range(len(simplified_contours)), key=lambda k: (centroids[k][1], centroids[k][0])
    )
    # Store sorted polygons as (N, 2) arrays for easier processing
    sorted_polygons = [simplified_contours[i].reshape(-1, 2) for i in sorted_indices]
    num_polygons = len(sorted_polygons)

    # Find Closest Connection Points between consecutive polygons
    connection_pairs = []
    for i in range(num_polygons):
        poly_i = sorted_polygons[i]
        poly_next = sorted_polygons[(i + 1) % num_polygons]  # Wrap around

        min_dist_sq = float("inf")
        best_pair = (None, None)  # (vertex_coord_on_poly_i, vertex_coord_on_poly_next)

        for pt_i in poly_i:
            for pt_next in poly_next:
                dist_sq = np.sum((pt_i - pt_next) ** 2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_pair = (pt_i, pt_next)

        if best_pair[0] is not None:
            connection_pairs.append(best_pair)
        else:
            return None, "no_connection_points"

    # Stitch Vertices
    stitched_vertices_list = []
    for i in range(num_polygons):
        current_poly = sorted_polygons[i]
        num_vertices_current = len(current_poly)

        # Entry point connects from previous polygon (i-1)
        # Exit point connects to next polygon (i)
        entry_point_on_current = connection_pairs[(i - 1 + num_polygons) % num_polygons][1]
        exit_point_on_current = connection_pairs[i][0]

        # Find the *indices* of these points
        try:
            entry_idx_arr = np.where(np.all(current_poly == entry_point_on_current, axis=1))[0]
            exit_idx_arr = np.where(np.all(current_poly == exit_point_on_current, axis=1))[0]
            if len(entry_idx_arr) == 0 or len(exit_idx_arr) == 0:
                raise IndexError
            entry_idx = entry_idx_arr[0]
            exit_idx = exit_idx_arr[0]
        except IndexError:
            dist_entry = np.sum((current_poly - entry_point_on_current) ** 2, axis=1)
            dist_exit = np.sum((current_poly - exit_point_on_current) ** 2, axis=1)
            entry_idx = np.argmin(dist_entry)
            exit_idx = np.argmin(dist_exit)
            if entry_idx == exit_idx:
                return None, "entry_exit_same_point"

        # Traverse vertices from entry_idx to exit_idx
        current_v_idx = entry_idx
        traversed_indices = 0
        while traversed_indices < num_vertices_current * 2:
            stitched_vertices_list.append(current_poly[current_v_idx].tolist())
            if current_v_idx == exit_idx:
                break
            current_v_idx = (current_v_idx + 1) % num_vertices_current
            traversed_indices += 1
        else:
            return None, "traversal_failed"

        # Add the bridge point (entry point of the *next* polygon)
        next_entry_point = connection_pairs[i][1]
        stitched_vertices_list.append(next_entry_point.tolist())

    # Final processing of stitched vertices
    if not stitched_vertices_list:
        return None, "empty_stitching_result"

    # Remove potential duplicate point at the very end
    if len(stitched_vertices_list) > 1 and np.allclose(
        stitched_vertices_list[0], stitched_vertices_list[-1]
    ):
        final_vertices = np.array(stitched_vertices_list[:-1])
    else:
        final_vertices = np.array(stitched_vertices_list)

    # Return the single combined polygon if it's valid
    if len(final_vertices) >= 3:
        return final_vertices, None
    else:
        return None, "degenerate_polygon"


def _normalize_and_flatten_polygons(
    polygons_pixels: List[np.ndarray], img_shape: Tuple[int, int]
) -> List[List[float]]:
    """Normalize, clamp, clip, and flatten multiple polygon coordinates.

    Args:
        polygons_pixels: A list of numpy arrays, where each array represents a
                         polygon in pixel coordinates (shape (N, 1, 2) or (N, 2)).
        img_shape: The original image shape (height, width).

    Returns:
        A list where each sublist contains the flattened, normalized coordinates
        [x1, y1, x2, y2, ...] for a polygon.

    Raises:
        ValueError: For invalid image shape
    """
    h, w = img_shape
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape: {img_shape}")

    normalized_polygons_flat_list = []
    for polygon_pixels in polygons_pixels:
        if polygon_pixels is None or len(polygon_pixels) < 3:
            continue  # Skip invalid polygons silently, they can be filtered by the caller

        # Reshape to (N, 2) if needed and ensure float32
        poly = np.array(polygon_pixels).reshape(-1, 2).astype(np.float32)

        # Normalize coordinates
        poly[:, 0] /= w  # Normalize x
        poly[:, 1] /= h  # Normalize y

        # Clip normalized coordinates to [0.0, 1.0] to handle any out-of-bounds points
        poly = np.clip(poly, 0.0, 1.0)

        # Flatten to [x1, y1, x2, y2, ...]
        normalized_flat = poly.flatten().tolist()

        # Final check: Ensure at least 3 points (6 coordinates) after normalization
        if len(normalized_flat) >= 6:
            normalized_polygons_flat_list.append(normalized_flat)

    return normalized_polygons_flat_list


def polygons_to_mask(
    polygons: Union[List[List[float]], List[np.ndarray], List[Tuple[int, int]]],
    img_shape: Tuple[int, int],
    normalized: bool = False,
) -> np.ndarray:
    """Convert polygons to a binary mask.

    Supports multiple input formats: YOLO normalized coordinates, pixel coordinates
    as arrays, or simple list of (x,y) tuples.

    Args:
        polygons: Polygon coordinates in one of three formats:
                 1. List of normalized YOLO coordinates [x1,y1,x2,y2,...] (if normalized=True)
                 2. List of numpy arrays with pixel coordinates (shape (N,1,2) or (N,2))
                 3. Single polygon as List of (x,y) pixel coordinate tuples
        img_shape: The target mask shape (height, width)
        normalized: Whether the coordinates are normalized [0.0-1.0] (YOLO format)
                    or already in pixel coordinates

    Returns:
        A binary mask as numpy array (dtype=bool)

    Raises:
        ValueError: For invalid image shape
    """
    h, w = img_shape
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape: {img_shape}")

    mask = np.zeros(img_shape, dtype=np.uint8)

    if not polygons:
        return mask.astype(bool)

    # Handle input format variations
    if normalized:
        # Assume we have normalized YOLO format [x1,y1,x2,y2,...]
        for poly in polygons:  # type: ignore
            # Reshape from flat list to array of points
            try:
                points = np.array(poly).reshape(-1, 2)

                # Denormalize by multiplying by image dimensions
                points_pixels = (points * np.array([w, h])).astype(np.int32)

                # Reshape for fillPoly: (N, 1, 2)
                points_pixels = points_pixels.reshape((-1, 1, 2))

                # Skip invalid polygons (less than 3 points)
                if len(points_pixels) < 3:
                    continue

                # Draw the polygon on the mask
                cv2.fillPoly(mask, [points_pixels], 1)
            except (ValueError, TypeError):
                # Skip malformed polygons
                continue
    else:
        # Handle non-normalized polygons (already in pixel coordinates)
        pts_list = []

        # Check if we have a single polygon given as list of tuples
        if isinstance(polygons[0], tuple) and len(polygons[0]) == 2:
            # Single polygon as list of coordinate tuples [(x1,y1), (x2,y2), ...]
            if len(polygons) < 3:  # type: ignore
                return mask.astype(bool)

            pts_np = np.array(polygons, dtype=np.int32).reshape((-1, 1, 2))  # type: ignore
            pts_list = [pts_np]
        else:
            # Multiple polygons, each as numpy array
            for poly in polygons:
                if poly is None or len(poly) < 3:
                    continue

                # Reshape if necessary and ensure int32
                try:
                    pts_np = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
                    pts_list.append(pts_np)
                except (ValueError, TypeError):
                    continue

        if pts_list:
            try:
                cv2.fillPoly(mask, pts_list, 1)
            except Exception:
                # Return empty mask on error
                return np.zeros(img_shape, dtype=bool)

    return mask.astype(bool)


def mask_to_yolo_polygons(
    binary_mask: np.ndarray,
    img_shape: Tuple[int, int],
    connect_parts: bool = False,
    min_contour_area: float = DEFAULT_MIN_CONTOUR_AREA,
    polygon_approx_tolerance: float = DEFAULT_POLYGON_APPROX_TOLERANCE,
) -> Tuple[List[List[float]], Optional[str]]:
    """
    Convert a binary instance mask to normalized YOLO polygon coordinates.

    Handles optional connection of disconnected parts into a single polygon.

    Args:
        binary_mask: A 2D numpy array representing the binary mask (0/1 or 0/255).
        img_shape: The original image shape (height, width).
        connect_parts: If True and multiple contours found, attempt to connect them
                       into a single polygon list. Otherwise, return separate polygons.
        min_contour_area: Minimum contour area (in pixels) to consider.
        polygon_approx_tolerance: Approximation tolerance factor for cv2.approxPolyDP
                                 (relative to arc length).

    Returns:
        Tuple containing:
        - List of polygons, where each sublist is a separate polygon [x1, y1, x2, y2, ...].
          If connect_parts is True and successful, returns a list containing a single
          sublist for the combined polygon. Empty list if no valid polygons found.
        - Error code (string) or None if successful.

        Coordinates are normalized to [0.0, 1.0].
    """
    try:
        # Preprocess and validate the mask
        mask_uint8 = _preprocess_binary_mask(binary_mask, img_shape)
    except ValueError as e:
        return [], str(e)

    # Find and simplify contours
    simplified_contours = _find_and_simplify_contours(
        mask_uint8, min_contour_area, polygon_approx_tolerance
    )

    if not simplified_contours:
        return [], "no_contours"

    # --- Polygon Generation ---
    final_polygons_pixels = []  # List to store polygons in pixel coordinates (N, 2)

    # Handle connection or separate processing
    if connect_parts and len(simplified_contours) > 1:
        connected_polygon, error = _connect_contours_stitched(simplified_contours)
        if connected_polygon is not None:
            # Store the single connected polygon
            final_polygons_pixels.append(connected_polygon)
        else:
            # Fallback: connection failed, process separately
            for contour in simplified_contours:
                final_polygons_pixels.append(contour.reshape(-1, 2))
    else:
        # Process each contour separately (connect_parts=False or only 1 contour)
        for contour in simplified_contours:
            final_polygons_pixels.append(contour.reshape(-1, 2))

    # Normalize and flatten results
    try:
        normalized_polygons = _normalize_and_flatten_polygons(final_polygons_pixels, img_shape)
        if not normalized_polygons:
            return [], "normalization_produced_empty_result"
        return normalized_polygons, None
    except ValueError as e:
        return [], str(e)


def save_mask_and_polygon_debug_images(
    original_binary_mask: np.ndarray,
    yolo_polygon_normalized: List[float],  # Expecting a single polygon's flat coordinate list
    img_shape: Tuple[int, int],
) -> None:
    """Saves the original binary mask and a mask reconstructed from a YOLO polygon for debugging."""
    global _debug_save_counter
    _debug_save_counter += 1

    try:
        if not os.path.exists(DEBUG_MASK_DIR):
            os.makedirs(DEBUG_MASK_DIR)

        # Save original mask
        original_mask_to_save = (original_binary_mask.astype(np.uint8)) * 255
        original_file_path = os.path.join(
            DEBUG_MASK_DIR, f"{_debug_save_counter}_original_mask.png"
        )
        cv2.imwrite(original_file_path, original_mask_to_save)
        logger.info(f"Saved debug original mask to: {original_file_path}")

        # Reconstruct and save polygon mask
        if yolo_polygon_normalized and len(yolo_polygon_normalized) >= 6:
            # polygons_to_mask expects a list of polygons
            reconstructed_mask = polygons_to_mask(
                [yolo_polygon_normalized], img_shape, normalized=True
            )
            reconstructed_mask_to_save = (reconstructed_mask.astype(np.uint8)) * 255
            polygon_file_path = os.path.join(
                DEBUG_MASK_DIR, f"{_debug_save_counter}_polygon_mask.png"
            )
            cv2.imwrite(polygon_file_path, reconstructed_mask_to_save)
            logger.info(f"Saved debug polygon mask to: {polygon_file_path}")
        else:
            logger.warning(
                f"Debug save: Invalid or empty polygon for counter {_debug_save_counter}, not saving polygon mask."
            )

    except Exception as e:
        logger.error(f"Failed to save debug masks for counter {_debug_save_counter}: {e}")


def mask_to_yolo_polygons_verified(
    binary_mask: np.ndarray,
    img_shape: Tuple[int, int],
    min_contour_area: float,  # Parameter passed from caller
    polygon_approx_tolerance: float,  # Parameter passed from caller
) -> Tuple[List[List[float]], float, Optional[str]]:
    """
    Convert a binary instance mask to normalized YOLO polygon coordinates.
    Calculates and returns the IoU between the original mask and the mask
    reconstructed from the generated polygons.

    Args:
        binary_mask: A 2D numpy array representing the binary mask (0/1 or 0/255).
        img_shape: The original image shape (height, width).
        min_contour_area: Minimum contour area (pixels) to keep.
        polygon_approx_tolerance: Approximation tolerance factor for simplification.

    Returns:
        Tuple containing:
        - List of polygons [[x1,y1,x2,y2,...], ...]. Empty if no valid polygons.
        - IoU value (float) between original and reconstructed. 0.0 if error before IoU.
        - Error code (string) or None if successful.
    """
    try:
        mask_uint8 = _preprocess_binary_mask(binary_mask, img_shape)
    except ValueError as e:
        return [], 0.0, str(e)

    valid_contours = _find_and_simplify_contours(
        mask_uint8, min_contour_area, polygon_approx_tolerance
    )

    if not valid_contours:
        return [], 0.0, "no_contours" if mask_uint8.any() else None

    try:
        normalized_polygons = _normalize_and_flatten_polygons(valid_contours, img_shape)
    except ValueError as e:
        return [], 0.0, str(e)

    if not normalized_polygons:
        return [], 0.0, "normalization_failed"

    reconstructed_mask = polygons_to_mask(normalized_polygons, img_shape, normalized=True)
    input_mask_bool = mask_uint8.astype(bool)

    try:
        iou = calculate_mask_iou(
            input_mask_bool, reconstructed_mask
        )  # reconstructed_mask is already bool
    except ValueError as e:
        return normalized_polygons, 0.0, f"iou_calculation_error:{str(e)}"

    # Return polygons, IoU, and no error if successful up to this point
    return normalized_polygons, iou, None


def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculates the Intersection over Union (IoU) between two binary masks.

    Args:
        mask1: The first boolean mask (np.ndarray, dtype=bool).
        mask2: The second boolean mask (np.ndarray, dtype=bool).

    Returns:
        The IoU value (float) between 0.0 and 1.0.

    Raises:
        ValueError: If masks have different shapes or are not boolean.
    """
    if mask1.shape != mask2.shape:
        raise ValueError(f"Mask shapes must match. Got {mask1.shape} and {mask2.shape}")
    if mask1.dtype != bool or mask2.dtype != bool:
        raise ValueError("Masks must be boolean arrays.")

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        # If both masks are empty, IoU is 1 if intersection is also 0 (which it must be)
        # If union is 0 but intersection somehow isn't (shouldn't happen), return 0.
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / union
    return float(np.clip(iou, 0.0, 1.0))
