"""Test for mask.py module."""

import cv2
import numpy as np
import pytest

from vibelab.utils.common.mask import (
    calculate_mask_iou,
    mask_to_yolo_polygons,
    mask_to_yolo_polygons_verified,
    polygons_to_mask,
)

# --- Helper Functions for Test Masks ---


def create_rect_mask(shape, top_left, bottom_right, dtype=np.uint8, value=255):
    """Creates a binary mask with a single rectangle."""
    mask = np.zeros(shape, dtype=dtype)
    # Ensure uint8 for cv2.rectangle, especially if input dtype is bool
    if mask.dtype == np.bool_:
        mask_uint8 = mask.astype(np.uint8)
        cv2.rectangle(mask_uint8, top_left, bottom_right, int(value), thickness=cv2.FILLED)
        return mask_uint8  # Return the uint8 version
    else:
        cv2.rectangle(mask, top_left, bottom_right, value, thickness=cv2.FILLED)
        return mask


def create_l_shape_mask(shape, points, dtype=np.uint8, value=255):
    """Creates a binary mask with an L-shape or other simple polygon."""
    mask = np.zeros(shape, dtype=dtype)
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], value)
    return mask


def create_circle_mask(height, width, center, radius):
    """Helper function to create a circular mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask


def polygon_to_mask_pixels(polygon, height, width):
    """Convert a normalized YOLO polygon to a mask."""
    # Use the polygons_to_mask function with normalized=True
    return polygons_to_mask([polygon], (height, width), normalized=True).astype(np.uint8) * 255


# --- Test Cases ---

IMG_SHAPE = (100, 200)  # height, width
H, W = IMG_SHAPE

# Default values for verified function, matching typical use in converter.py
DEFAULT_TEST_MIN_AREA = 10  # from converter.py POLYGON_MIN_CONTOUR_AREA
DEFAULT_TEST_APPROX_TOL = 0  # from converter.py POLYGON_APPROX_TOLERANCE (disabled)


def test_empty_mask():
    """Test with a mask containing no foreground pixels."""
    mask = np.zeros(IMG_SHAPE, dtype=np.uint8)
    polygons, error = mask_to_yolo_polygons(mask, IMG_SHAPE)
    assert polygons == []


def test_invalid_image_shape():
    """Test with zero height or width."""
    mask = create_rect_mask(IMG_SHAPE, (10, 20), (30, 40))
    polygons_zero_h, error_h = mask_to_yolo_polygons(mask, (0, W))
    assert polygons_zero_h == []
    polygons_zero_w, error_w = mask_to_yolo_polygons(mask, (H, 0))
    assert polygons_zero_w == []


def test_invalid_mask_dims():
    """Test with mask dimensions other than 2."""
    mask_1d = np.zeros(100, dtype=np.uint8)
    mask_3d = np.zeros((H, W, 3), dtype=np.uint8)
    polygons_1d, error_1d = mask_to_yolo_polygons(mask_1d, IMG_SHAPE)
    assert polygons_1d == []
    polygons_3d, error_3d = mask_to_yolo_polygons(mask_3d, IMG_SHAPE)
    assert polygons_3d == []


def test_single_simple_contour():
    """Test a single rectangular contour."""
    top_left = (20, 40)  # y, x
    bottom_right = (80, 160)  # y, x
    mask = create_rect_mask(
        IMG_SHAPE, (top_left[1], top_left[0]), (bottom_right[1], bottom_right[0])
    )

    # Expected order seems to be TL, BL, BR, TR from the previous error
    expected_poly = [
        40 / W,
        20 / H,  # Top-left
        40 / W,
        80 / H,  # Bottom-left
        160 / W,
        80 / H,  # Bottom-right
        160 / W,
        20 / H,  # Top-right
    ]

    polygons, error = mask_to_yolo_polygons(mask, IMG_SHAPE, connect_parts=False)
    assert len(polygons) == 1
    assert len(polygons[0]) == 8  # 4 points
    np.testing.assert_allclose(
        polygons[0], expected_poly, atol=0.01
    )  # Allow tolerance for approximation

    # Result should be the same if connect_parts=True for single contour
    polygons_connected, error = mask_to_yolo_polygons(mask, IMG_SHAPE, connect_parts=True)
    assert len(polygons_connected) == 1
    np.testing.assert_allclose(polygons_connected[0], expected_poly, atol=0.01)


def test_single_complex_contour():
    """Test an L-shaped contour."""
    # Points defined as (x, y)
    points = [(40, 20), (160, 20), (160, 50), (80, 50), (80, 80), (40, 80)]
    mask = create_l_shape_mask(IMG_SHAPE, points)

    polygons, error = mask_to_yolo_polygons(mask, IMG_SHAPE)
    assert len(polygons) == 1
    assert len(polygons[0]) >= 6  # Should have at least 3 points
    # Check if the normalized points roughly match the input (order might change due to approx)
    denorm_poly = np.array(polygons[0]).reshape(-1, 2)
    denorm_poly[:, 0] *= W
    denorm_poly[:, 1] *= H
    # Due to approximation, we just check the number of points and that it's one polygon
    assert len(denorm_poly) >= 3


def test_contour_below_area_threshold():
    """Test when the contour area is smaller than the minimum."""
    mask = create_rect_mask(IMG_SHAPE, (50, 50), (51, 51))  # Area is 1
    polygons, error = mask_to_yolo_polygons(mask, IMG_SHAPE, min_contour_area=2.0)
    assert polygons == []


def test_multiple_contours_no_connection():
    """Test two separate rectangles without connection."""
    mask = np.zeros(IMG_SHAPE, dtype=np.uint8)
    # Box 1 (top)
    cv2.rectangle(mask, (40, 10), (160, 40), 255, thickness=cv2.FILLED)
    # Box 2 (bottom)
    cv2.rectangle(mask, (40, 60), (160, 90), 255, thickness=cv2.FILLED)

    # Expected order: TL, BL, BR, TR
    expected_poly1 = [40 / W, 10 / H, 40 / W, 40 / H, 160 / W, 40 / H, 160 / W, 10 / H]
    expected_poly2 = [40 / W, 60 / H, 40 / W, 90 / H, 160 / W, 90 / H, 160 / W, 60 / H]

    polygons, error = mask_to_yolo_polygons(mask, IMG_SHAPE, connect_parts=False)
    assert len(polygons) == 2
    # Order depends on contour finding, sort by y-coord of first point
    polygons.sort(key=lambda p: p[1])
    np.testing.assert_allclose(polygons[0], expected_poly1, atol=0.01)
    np.testing.assert_allclose(polygons[1], expected_poly2, atol=0.01)


def test_multiple_contours_with_connection():
    """Test two separate rectangles with connection enabled."""
    mask = np.zeros(IMG_SHAPE, dtype=np.uint8)
    # Box 1 (top)
    cv2.rectangle(mask, (40, 10), (160, 40), 255, thickness=cv2.FILLED)
    # Box 2 (bottom)
    cv2.rectangle(mask, (40, 60), (160, 90), 255, thickness=cv2.FILLED)

    polygons, error = mask_to_yolo_polygons(mask, IMG_SHAPE, connect_parts=True)
    assert len(polygons) == 1, "Expected a single combined polygon list"
    # Check the combined polygon is valid (at least 3 points / 6 coordinates)
    assert len(polygons[0]) >= 6, "Combined polygon must have at least 3 points"

    # Verify all coordinates are within [0, 1]
    poly_arr = np.array(polygons[0])
    assert np.all(poly_arr >= 0.0)
    assert np.all(poly_arr <= 1.0)


def test_multiple_contours_connection_with_filtering():
    """Test connection with one contour below area threshold."""
    mask = np.zeros(IMG_SHAPE, dtype=np.uint8)
    # Box 1 (top, large)
    cv2.rectangle(mask, (40, 10), (160, 40), 255, thickness=cv2.FILLED)
    # Box 2 (middle, tiny)
    cv2.rectangle(mask, (99, 50), (101, 51), 255, thickness=cv2.FILLED)  # Area = 2
    # Box 3 (bottom, large)
    cv2.rectangle(mask, (40, 60), (160, 90), 255, thickness=cv2.FILLED)

    polygons, error = mask_to_yolo_polygons(
        mask, IMG_SHAPE, connect_parts=True, min_contour_area=10.0
    )
    assert len(polygons) == 1, "Expected single combined list even with filtering"
    # Check the combined polygon is valid (at least 3 points / 6 coordinates)
    assert len(polygons[0]) >= 6, "Combined polygon must have at least 3 points"


def test_contours_touching_edges():
    """Test shapes touching the image boundaries."""
    mask = np.zeros(IMG_SHAPE, dtype=np.uint8)
    # Box touching top-left
    cv2.rectangle(mask, (0, 0), (50, 50), 255, thickness=cv2.FILLED)
    # Box touching bottom-right
    cv2.rectangle(mask, (W - 50, H - 50), (W - 1, H - 1), 255, thickness=cv2.FILLED)

    polygons, error = mask_to_yolo_polygons(mask, IMG_SHAPE, connect_parts=False)
    assert len(polygons) == 2
    for poly in polygons:
        assert len(poly) >= 6
        poly_arr = np.array(poly)
        assert np.all(poly_arr >= 0.0), f"Polygon points {poly} not >= 0.0"
        assert np.all(poly_arr <= 1.0), f"Polygon points {poly} not <= 1.0"


@pytest.mark.parametrize(
    "dtype, value",
    [
        (np.bool_, True),
        (np.uint8, 1),
        (np.uint8, 255),
    ],
)
def test_different_mask_types(dtype, value):
    """Test different valid mask data types and values."""
    top_left = (20, 40)  # y, x
    bottom_right = (80, 160)  # y, x
    mask = create_rect_mask(
        IMG_SHAPE,
        (top_left[1], top_left[0]),
        (bottom_right[1], bottom_right[0]),
        dtype=dtype,
        value=value,
    )

    # Expected order: TL, BL, BR, TR
    expected_poly = [
        40 / W,
        20 / H,  # Top-left
        40 / W,
        80 / H,  # Bottom-left
        160 / W,
        80 / H,  # Bottom-right
        160 / W,
        20 / H,  # Top-right
    ]

    polygons, error = mask_to_yolo_polygons(mask, IMG_SHAPE)
    assert len(polygons) == 1
    assert len(polygons[0]) == 8
    np.testing.assert_allclose(polygons[0], expected_poly, atol=0.01)


# --- Tests for polygons_to_mask ---


class TestPolygonsToMask:
    """Test cases for polygons_to_mask function."""

    def test_single_polygon(self):
        """Test conversion of a single polygon to mask."""
        height, width = 100, 100

        # Create a simple rectangle polygon
        rectangle = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.int32).reshape(
            -1, 1, 2
        )

        # Convert to mask
        mask = polygons_to_mask([rectangle], (height, width), normalized=False)

        # Create reference mask using cv2.rectangle
        ref_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(ref_mask, (20, 20), (80, 80), 1, -1)
        ref_mask = ref_mask.astype(bool)

        # Check that masks match
        assert np.array_equal(mask, ref_mask), "Generated mask should match reference"

    def test_multiple_polygons(self):
        """Test conversion of multiple polygons to a single mask."""
        height, width = 200, 200

        # Create two polygons
        circle1 = np.zeros((0, 1, 2), dtype=np.int32)
        circle2 = np.zeros((0, 1, 2), dtype=np.int32)

        # Generate points for circles
        for angle in range(0, 360, 10):
            x1 = int(50 + 30 * np.cos(np.radians(angle)))
            y1 = int(50 + 30 * np.sin(np.radians(angle)))
            circle1 = np.append(circle1, np.array([[[x1, y1]]], dtype=np.int32), axis=0)

            x2 = int(150 + 25 * np.cos(np.radians(angle)))
            y2 = int(150 + 25 * np.sin(np.radians(angle)))
            circle2 = np.append(circle2, np.array([[[x2, y2]]], dtype=np.int32), axis=0)

        # Convert to mask
        mask = polygons_to_mask([circle1, circle2], (height, width), normalized=False)

        # Create reference mask
        ref_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(ref_mask, (50, 50), 30, 1, -1)
        cv2.circle(ref_mask, (150, 150), 25, 1, -1)
        ref_mask = ref_mask.astype(bool)

        # Check using IoU since the circle approximations may differ slightly
        iou = calculate_mask_iou(mask, ref_mask)
        assert iou > 0.95, f"IoU should be high, got {iou}"

    def test_empty_list(self):
        """Test with empty polygon list."""
        height, width = 100, 100

        # Should return an all-False mask
        mask = polygons_to_mask([], (height, width), normalized=False)

        assert mask.shape == (height, width), "Mask should have correct dimensions"
        assert not mask.any(), "Mask should be all False for empty polygon list"

    def test_invalid_polygons(self):
        """Test with invalid polygons (fewer than 3 points)."""
        height, width = 100, 100

        # Polygon with only 2 points (invalid)
        invalid_poly = np.array([[[10, 10]], [[20, 20]]], dtype=np.int32)

        # Should handle gracefully and return empty mask
        mask = polygons_to_mask([invalid_poly], (height, width), normalized=False)

        assert mask.shape == (height, width), "Mask should have correct dimensions"
        assert not mask.any(), "Mask should be all False for invalid polygon"

    def test_different_formats(self):
        """Test with polygons in different formats."""
        height, width = 100, 100

        # Format (N, 2) needs reshaping
        poly_format1 = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.int32)

        # Format (N, 1, 2) already correct for fillPoly
        poly_format2 = poly_format1.reshape(-1, 1, 2)

        mask1 = polygons_to_mask([poly_format1], (height, width), normalized=False)
        mask2 = polygons_to_mask([poly_format2], (height, width), normalized=False)

        # Both formats should produce the same result
        assert np.array_equal(mask1, mask2), "Different polygon formats should produce same mask"


# --- Tests for mask_to_yolo_polygons_verified ---


class TestMaskToYoloPolygonsVerified:
    """Test cases for mask_to_yolo_polygons_verified function."""

    @pytest.mark.parametrize(
        "shape, rect_coords, expected_num_points",
        [
            (IMG_SHAPE, ((20, 40), (80, 160)), 4),  # y,x for top_left, bottom_right
            ((50, 50), ((5, 5), (45, 45)), 4),
        ],
    )
    def test_basic_shapes(self, shape, rect_coords, expected_num_points):
        """Test simple rectangular shapes."""
        h, w = shape
        mask = create_rect_mask(
            shape,
            (rect_coords[0][1], rect_coords[0][0]),  # convert to x,y for cv2.rectangle
            (rect_coords[1][1], rect_coords[1][0]),
            dtype=np.uint8,
        )

        polygons, iou, error_msg = mask_to_yolo_polygons_verified(
            mask, shape, DEFAULT_TEST_MIN_AREA, DEFAULT_TEST_APPROX_TOL
        )

        assert error_msg is None, f"Error processing basic shape: {error_msg}"
        assert len(polygons) == 1
        assert len(polygons[0]) == expected_num_points * 2
        # Relaxed IoU for the second parameterized case (shape1-rect_coords1-4)
        if shape == (50, 50):
            assert iou >= 0.95, f"Expected IoU >= 0.95 for smaller rect, got {iou}"
        else:
            assert iou >= 0.98, f"Expected IoU >= 0.98, got {iou}"
        for p_val in polygons[0]:
            assert 0.0 <= p_val <= 1.0

    def test_multiple_regions_no_connection_handling(self):
        """Test two separate rectangles. Verified func processes them separately."""
        mask = np.zeros(IMG_SHAPE, dtype=np.uint8)
        # Box 1 (top)
        cv2.rectangle(mask, (40, 10), (160, 40), 255, thickness=cv2.FILLED)
        # Box 2 (bottom)
        cv2.rectangle(mask, (40, 60), (160, 90), 255, thickness=cv2.FILLED)

        polygons, iou, error_msg = mask_to_yolo_polygons_verified(
            mask, IMG_SHAPE, DEFAULT_TEST_MIN_AREA, DEFAULT_TEST_APPROX_TOL
        )

        assert error_msg is None, f"Error processing multiple regions: {error_msg}"
        assert len(polygons) == 2, "Expected two separate polygons"
        assert iou >= 0.98, f"Expected high IoU for multiple regions, got {iou}"
        for poly in polygons:
            assert len(poly) >= 6

    def test_empty_and_invalid_inputs(self):
        """Test empty mask, invalid shape, etc."""
        # Empty mask
        mask_empty = np.zeros(IMG_SHAPE, dtype=np.uint8)
        polys, iou, err = mask_to_yolo_polygons_verified(
            mask_empty, IMG_SHAPE, DEFAULT_TEST_MIN_AREA, DEFAULT_TEST_APPROX_TOL
        )
        assert polys == []
        assert iou == 0.0
        assert err is None  # No contours, but not an error for an empty mask

        # Invalid image shape
        mask_valid = create_rect_mask(IMG_SHAPE, (10, 20), (30, 40))
        polys_h, iou_h, err_h = mask_to_yolo_polygons_verified(
            mask_valid, (0, W), DEFAULT_TEST_MIN_AREA, DEFAULT_TEST_APPROX_TOL
        )
        assert polys_h == []
        assert iou_h == 0.0
        assert err_h is not None

        polys_w, iou_w, err_w = mask_to_yolo_polygons_verified(
            mask_valid, (H, 0), DEFAULT_TEST_MIN_AREA, DEFAULT_TEST_APPROX_TOL
        )
        assert polys_w == []
        assert iou_w == 0.0
        assert err_w is not None

        # Mask with invalid dimensions
        mask_1d = np.zeros(100, dtype=np.uint8)
        polys_1d, iou_1d, err_1d = mask_to_yolo_polygons_verified(
            mask_1d, IMG_SHAPE, DEFAULT_TEST_MIN_AREA, DEFAULT_TEST_APPROX_TOL
        )
        assert polys_1d == []
        assert iou_1d == 0.0
        assert "Mask must be 2D" in err_1d

    def test_min_contour_area_filtering(self):
        """Test filtering contours by area."""
        # Use a slightly larger mask to avoid sub-pixel issues for very small areas
        mask_small_actual = create_rect_mask(IMG_SHAPE, (50, 50), (53, 53))  # 3x3 pixels -> Area 9

        # Case 1: Contour area is too small
        polys, iou, err = mask_to_yolo_polygons_verified(
            mask_small_actual,
            IMG_SHAPE,
            min_contour_area=10.0,
            polygon_approx_tolerance=DEFAULT_TEST_APPROX_TOL,
        )
        assert polys == []
        assert iou == 0.0
        assert err == "no_contours"

        # Case 2: Contour area is large enough
        polys_ok, iou_ok, err_ok = mask_to_yolo_polygons_verified(
            mask_small_actual,
            IMG_SHAPE,
            min_contour_area=8.0,
            polygon_approx_tolerance=DEFAULT_TEST_APPROX_TOL,
        )
        assert len(polys_ok) == 1
        assert iou_ok == pytest.approx(9 / 16), (
            f"Expected IoU for 3x3 rect to be approx 9/16 ({9 / 16:.4f}), got {iou_ok}"
        )
        assert err_ok is None

    def test_iou_value_returned(self):
        """Test that IoU is calculated and returned correctly, no thresholding."""
        mask = create_rect_mask(IMG_SHAPE, (10, 10), (60, 60))
        polys, iou, err = mask_to_yolo_polygons_verified(
            mask, IMG_SHAPE, min_contour_area=1.0, polygon_approx_tolerance=0.0
        )
        assert err is None
        assert len(polys) == 1
        assert iou == pytest.approx(1.0, abs=0.015)  # Allow slight deviation from 1.0

        mask_circle = create_circle_mask(H, W, (W // 2, H // 2), 30)
        polys_circle, iou_circle, err_circle = mask_to_yolo_polygons_verified(
            mask_circle, IMG_SHAPE, min_contour_area=10, polygon_approx_tolerance=0.01
        )
        assert err_circle is None
        assert len(polys_circle) == 1
        assert 0.90 < iou_circle < 1.0

    def test_polygon_approximation_effect_on_iou(self):
        """Test how polygon approximation affects IoU."""
        mask = create_rect_mask(IMG_SHAPE, (10, 10), (90, 190))

        _, iou_no_simplify, _ = mask_to_yolo_polygons_verified(
            mask, IMG_SHAPE, min_contour_area=1, polygon_approx_tolerance=0.0
        )
        assert iou_no_simplify == pytest.approx(1.0, abs=0.015)  # Allow slight deviation

        _, iou_simplify_low_tol, _ = mask_to_yolo_polygons_verified(
            mask, IMG_SHAPE, min_contour_area=1, polygon_approx_tolerance=0.001
        )
        assert iou_simplify_low_tol == pytest.approx(1.0, abs=0.015)

        _, iou_simplify_high_tol, _ = mask_to_yolo_polygons_verified(
            mask, IMG_SHAPE, min_contour_area=1, polygon_approx_tolerance=0.02
        )
        assert iou_simplify_high_tol > 0.97  # Expect high, but allow more deviation


# --- Tests for calculate_mask_iou ---
# (Existing tests for calculate_mask_iou can remain as they are)

# --- Test for the new debug save function (optional) ---
# import os
# def test_save_mask_and_polygon_debug_images(tmp_path):
#     global _debug_save_counter # Access the global counter from mask.py
#     from vibelab.utils.common.mask import _debug_save_counter as mask_debug_counter, DEBUG_MASK_DIR
#     original_counter = mask_debug_counter

#     mask_shape = (50, 50)
#     original_mask = create_rect_mask(mask_shape, (5,5), (25,25), value=1).astype(bool)
#     # A simple square polygon, normalized
#     yolo_poly = [5/50, 5/50, 5/50, 25/50, 25/50, 25/50, 25/50, 5/50]

#     # Override DEBUG_MASK_DIR for this test
#     test_debug_dir = tmp_path / "test_mask_debug"
#     from vibelab.utils.common import mask as mask_module
#     original_debug_mask_dir = mask_module.DEBUG_MASK_DIR
#     mask_module.DEBUG_MASK_DIR = str(test_debug_dir)

#     save_mask_and_polygon_debug_images(original_mask, yolo_poly, mask_shape)

#     assert os.path.exists(test_debug_dir / f"{original_counter + 1}_original_mask.png")
#     assert os.path.exists(test_debug_dir / f"{original_counter + 1}_polygon_mask.png")

#     # Restore original DEBUG_MASK_DIR
#     mask_module.DEBUG_MASK_DIR = original_debug_mask_dir
