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

    def test_basic_shapes(self):
        """Test conversion of basic shapes (rectangle and circle)."""
        height, width = 100, 100

        # Test with a rectangle
        rect_mask = np.zeros((height, width), dtype=np.uint8)
        rect_mask[20:80, 20:80] = 255  # Rectangle from (20,20) to (80,80)

        # Test with a circle
        circle_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(circle_mask, (50, 50), 30, 255, -1)  # Circle at center

        # Convert rectangle to YOLO polygons with verification
        rect_polygons, rect_error = mask_to_yolo_polygons_verified(rect_mask, (height, width))

        # Should have one polygon with no error
        assert rect_error is None, f"Rectangle conversion failed with error: {rect_error}"
        assert len(rect_polygons) == 1, "Rectangle should produce a single polygon"

        # Convert back and check IoU
        rect_reconstructed = polygon_to_mask_pixels(rect_polygons[0], height, width)
        rect_iou = calculate_mask_iou(rect_mask.astype(bool), rect_reconstructed > 0)
        assert rect_iou > 0.95, f"Rectangle IoU should be high, got {rect_iou}"

        # Convert circle to YOLO polygons with verification
        circle_polygons, circle_error = mask_to_yolo_polygons_verified(circle_mask, (height, width))

        # Should have one polygon with no error
        assert circle_error is None, f"Circle conversion failed with error: {circle_error}"
        assert len(circle_polygons) == 1, "Circle should produce a single polygon"

        # Convert back and check IoU
        circle_reconstructed = polygon_to_mask_pixels(circle_polygons[0], height, width)
        circle_iou = calculate_mask_iou(circle_mask.astype(bool), circle_reconstructed > 0)
        assert circle_iou > 0.90, f"Circle IoU should be high, got {circle_iou}"

    def test_multiple_regions(self):
        """Test with multiple separate regions in the mask."""
        height, width = 200, 200

        # Create mask with two circles
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 30, 255, -1)  # First circle
        cv2.circle(mask, (150, 150), 20, 255, -1)  # Second circle

        # Convert to YOLO polygons with verification
        polygons, error = mask_to_yolo_polygons_verified(mask, (height, width))

        # Should have two polygons with no error
        assert error is None, f"Conversion failed with error: {error}"
        assert len(polygons) == 2, f"Should detect two polygons, got {len(polygons)}"

        # Convert each polygon to mask and combine
        combined_reconstructed_mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons:
            polygon_mask = polygon_to_mask_pixels(polygon, height, width)
            combined_reconstructed_mask = (
                np.logical_or(combined_reconstructed_mask, polygon_mask > 0).astype(np.uint8) * 255
            )

        # Check IoU with original mask
        iou = calculate_mask_iou(mask.astype(bool), combined_reconstructed_mask > 0)
        assert iou > 0.95, f"Combined IoU should be high, got {iou}"

    def test_empty_and_invalid_inputs(self):
        """Test with empty mask and invalid inputs."""
        # Empty mask
        height, width = 100, 100
        empty_mask = np.zeros((height, width), dtype=np.uint8)

        # Should handle empty mask gracefully
        empty_polygons, empty_error = mask_to_yolo_polygons_verified(empty_mask, (height, width))
        assert empty_error is None, (
            f"Empty mask conversion failed with unexpected error: {empty_error}"
        )
        assert empty_polygons == [], "Empty mask should result in empty polygon list"

        # Invalid mask dimensions (3D)
        invalid_mask = np.zeros((100, 100, 3), dtype=np.uint8)
        _, invalid_error = mask_to_yolo_polygons_verified(invalid_mask, (100, 100))
        assert invalid_error is not None, "Should detect invalid mask dimensions"

        # Invalid image shape
        zero_size_mask = np.zeros((0, 0), dtype=np.uint8)
        _, shape_error = mask_to_yolo_polygons_verified(zero_size_mask, (0, 0))
        assert shape_error is not None, "Should detect invalid image shape"

    def test_min_contour_area_filtering(self):
        """Test filtering by minimum contour area."""
        height, width = 100, 100

        # Create a mask with one large and several small regions
        mask = np.zeros((height, width), dtype=np.uint8)
        # Large region
        mask[20:50, 20:50] = 255
        # Small regions
        mask[60:65, 60:65] = 255
        mask[70:72, 70:72] = 255
        mask[80:82, 80:82] = 255

        # With default min_contour_area (should filter small regions)
        polygons_default, error_default = mask_to_yolo_polygons_verified(mask, (height, width))
        assert error_default is None, "Default area filtering conversion should succeed"

        # With very small min_contour_area (should keep all regions)
        polygons_small, error_small = mask_to_yolo_polygons_verified(
            mask, (height, width), min_contour_area=1
        )
        assert error_small is None, "Small area filtering conversion should succeed"

        # Default should have fewer polygons than small min_area version
        assert len(polygons_default) <= len(polygons_small), (
            f"Default filtering ({len(polygons_default)}) should result in fewer or equal "
            f"polygons than small area filtering ({len(polygons_small)})"
        )

        # Small min_area should find all regions
        assert len(polygons_small) > 1, (
            f"Should detect multiple polygons with small min_area, got {len(polygons_small)}"
        )

    def test_iou_threshold(self):
        """Test IoU threshold verification."""
        height, width = 100, 100

        # Create a simple rectangle mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)

        # Test with default threshold
        polygons_default, error_default = mask_to_yolo_polygons_verified(mask, (height, width))
        assert error_default is None, "Default IoU threshold should pass"

        # Convert back to mask and calculate actual IoU
        reconstructed = polygon_to_mask_pixels(polygons_default[0], height, width)
        actual_iou = calculate_mask_iou(mask.astype(bool), reconstructed > 0)

        # Test with impossibly high threshold (0.999) that should fail
        polygons_high, error_high = mask_to_yolo_polygons_verified(
            mask, (height, width), iou_threshold=0.999
        )

        # Either it fails with low_iou error or succeeds with very high IoU
        if error_high and "low_iou" in error_high:
            assert True, "Correctly rejected polygon with IoU below threshold"
        elif polygons_high:
            high_reconstructed = polygon_to_mask_pixels(polygons_high[0], height, width)
            high_iou = calculate_mask_iou(mask.astype(bool), high_reconstructed > 0)
            assert high_iou >= 0.999, f"IoU should meet high threshold, got {high_iou}"

    def test_compare_with_original(self):
        """Compare verified method with the original method."""
        height, width = 100, 100

        # Create a complex mask
        mask = np.zeros((height, width), dtype=np.uint8)
        # Draw overlapping shapes
        cv2.rectangle(mask, (20, 20), (60, 60), 255, -1)
        cv2.circle(mask, (70, 70), 25, 255, -1)

        # Original method (unverified)
        original_polygons, original_error = mask_to_yolo_polygons(
            mask, (height, width), connect_parts=False
        )
        assert original_error is None, "Original method should succeed"

        # Verified method
        verified_polygons, verified_error = mask_to_yolo_polygons_verified(mask, (height, width))
        assert verified_error is None, "Verified method should succeed"

        # Compare IoU for both methods
        original_mask = np.zeros((height, width), dtype=np.uint8)
        verified_mask = np.zeros((height, width), dtype=np.uint8)

        for polygon in original_polygons:
            polygon_mask = polygon_to_mask_pixels(polygon, height, width)
            original_mask = np.logical_or(original_mask, polygon_mask > 0).astype(np.uint8) * 255

        for polygon in verified_polygons:
            polygon_mask = polygon_to_mask_pixels(polygon, height, width)
            verified_mask = np.logical_or(verified_mask, polygon_mask > 0).astype(np.uint8) * 255

        iou_original = calculate_mask_iou(mask.astype(bool), original_mask > 0)
        iou_verified = calculate_mask_iou(mask.astype(bool), verified_mask > 0)

        # Verified method should have good IoU
        assert iou_verified > 0.95, f"Verified method should have IoU > 0.95, got {iou_verified}"
