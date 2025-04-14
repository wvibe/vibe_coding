# src/utils/common/geometry.py

"""Geometric utility functions."""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default minimum contour area to consider for polygon conversion
DEFAULT_MIN_CONTOUR_AREA = 1.0
# Default tolerance factor for polygon approximation (relative to arc length)
DEFAULT_POLYGON_APPROX_TOLERANCE = 0.005


def _find_and_simplify_contours(
    binary_mask: np.ndarray,
    min_contour_area: float,
    polygon_approx_tolerance: float,
) -> List[np.ndarray]:
    """Find, filter, and simplify contours from a binary mask."""
    # Find contours
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    valid_simplified_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue

        # Approximate contour
        epsilon = polygon_approx_tolerance * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Need at least 3 points for a valid polygon
        if len(approx) >= 3:
            valid_simplified_contours.append(approx)

    return valid_simplified_contours


def _connect_contours_stitched(simplified_contours: List[np.ndarray]) -> Optional[np.ndarray]:
    """Connect multiple contours using Gemini's stitching logic.

    Returns single (M, 2) polygon array (pixel coords) or None if connection fails.
    """
    logger.debug(f"Attempting to connect {len(simplified_contours)} contours.")

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
            logger.error("Could not find connection points between polygons. Cannot connect parts.")
            return None  # Indicate connection failure

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
            logger.warning(
                f"Connection points ({entry_point_on_current.tolist()}, "
                f"{exit_point_on_current.tolist()}) not found exactly in "
                f"polygon vertices after approximation. Using fallback."
            )
            dist_entry = np.sum((current_poly - entry_point_on_current) ** 2, axis=1)
            dist_exit = np.sum((current_poly - exit_point_on_current) ** 2, axis=1)
            entry_idx = np.argmin(dist_entry)
            exit_idx = np.argmin(dist_exit)
            if entry_idx == exit_idx:
                logger.error("Fallback failed: Entry/Exit indices same. Cannot connect.")
                return None  # Indicate connection failure

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
            logger.error("Stitching traversal failed (infinite loop?). Cannot connect.")
            return None  # Indicate connection failure

        # Add the bridge point (entry point of the *next* polygon)
        next_entry_point = connection_pairs[i][1]
        stitched_vertices_list.append(next_entry_point.tolist())

    # Final processing of stitched vertices
    if not stitched_vertices_list:
        return None  # Should not happen if loops completed

    # Remove potential duplicate point at the very end
    if len(stitched_vertices_list) > 1 and np.allclose(
        stitched_vertices_list[0], stitched_vertices_list[-1]
    ):
        final_vertices = np.array(stitched_vertices_list[:-1])
    else:
        final_vertices = np.array(stitched_vertices_list)

    # Return the single combined polygon if it's valid
    if len(final_vertices) >= 3:
        return final_vertices
    else:
        logger.warning("Combined polygon is degenerate (< 3 points). Connection failed.")
        return None


def _normalize_and_flatten_polygons(
    polygons_pixels: List[np.ndarray], img_shape: Tuple[int, int]
) -> List[List[float]]:
    """Normalize, clamp, clip, and flatten polygon coordinates."""
    h, w = img_shape
    if h <= 0 or w <= 0:
        return []  # Should be checked earlier, but safeguard

    normalized_polygons = []
    for polygon_pixels in polygons_pixels:
        if len(polygon_pixels) < 3:
            continue

        # Ensure numpy array for operations
        poly = np.array(polygon_pixels).astype(np.float32)

        # Clamp pixel coordinates to image bounds before normalization
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)  # x-coordinates
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)  # y-coordinates

        # Normalize coordinates
        poly[:, 0] /= w  # Normalize x
        poly[:, 1] /= h  # Normalize y

        # Clip normalized coordinates strictly to [0.0, 1.0] as final safeguard
        poly = np.clip(poly, 0.0, 1.0)

        # Flatten to [x1, y1, x2, y2, ...]
        normalized_flat = poly.flatten().tolist()

        # Final check: Ensure at least 3 points (6 coordinates) after normalization
        if len(normalized_flat) >= 6:
            normalized_polygons.append(normalized_flat)
        else:
            logger.debug("Polygon degenerate after normalization/clipping, skipping.")

    return normalized_polygons


def mask_to_yolo_polygons(
    binary_mask: np.ndarray,
    img_shape: Tuple[int, int],
    connect_parts: bool = False,
    min_contour_area: float = DEFAULT_MIN_CONTOUR_AREA,
    polygon_approx_tolerance: float = DEFAULT_POLYGON_APPROX_TOLERANCE,
) -> List[List[float]]:
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
        List of polygons. If connect_parts is False, each sublist is a separate
        polygon [x1, y1, x2, y2, ...]. If connect_parts is True and successful,
        returns a list containing a single sublist for the combined polygon.
        Returns an empty list if no valid contours/polygons are found.
        Coordinates are normalized to [0.0, 1.0].
    """
    h, w = img_shape
    if h <= 0 or w <= 0:
        logger.warning("Invalid image shape provided (height or width is zero).")
        return []

    if binary_mask.ndim != 2:
        logger.error(f"Input mask must be 2D, but got shape {binary_mask.shape}")
        return []

    # Ensure mask is uint8 for findContours
    if binary_mask.dtype != np.uint8:
        mask_uint8 = binary_mask.astype(np.uint8)
        if mask_uint8.max() == 1:
            mask_uint8 *= 255
    else:
        mask_uint8 = binary_mask

    # 1. Find and simplify contours
    simplified_contours = _find_and_simplify_contours(
        mask_uint8, min_contour_area, polygon_approx_tolerance
    )

    if not simplified_contours:
        logger.warning("No valid contours found in mask.")
        return []

    # --- Polygon Generation ---
    final_polygons_pixels = []  # List to store polygons in pixel coordinates (N, 2)

    # 2. Handle connection or separate processing
    if connect_parts and len(simplified_contours) > 1:
        connected_polygon = _connect_contours_stitched(simplified_contours)
        if connected_polygon is not None:
            # Store the single connected polygon
            final_polygons_pixels.append(connected_polygon)
        else:
            # Fallback: connection failed, process separately
            logger.warning("Contour connection failed, processing contours separately.")
            for contour in simplified_contours:
                final_polygons_pixels.append(contour.reshape(-1, 2))
    else:
        # Process each contour separately (connect_parts=False or only 1 contour)
        for contour in simplified_contours:
            final_polygons_pixels.append(contour.reshape(-1, 2))

    # 3. Normalize and flatten results
    return _normalize_and_flatten_polygons(final_polygons_pixels, img_shape)
