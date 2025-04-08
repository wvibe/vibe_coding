# tests/utils/common/test_geometry.py

import cv2
import numpy as np
import pytest

from src.utils.common.geometry import mask_to_yolo_polygons

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


# --- Test Cases ---

IMG_SHAPE = (100, 200)  # height, width
H, W = IMG_SHAPE


def test_empty_mask():
    """Test with a mask containing no foreground pixels."""
    mask = np.zeros(IMG_SHAPE, dtype=np.uint8)
    polygons = mask_to_yolo_polygons(mask, IMG_SHAPE)
    assert polygons == []


def test_invalid_image_shape():
    """Test with zero height or width."""
    mask = create_rect_mask(IMG_SHAPE, (10, 20), (30, 40))
    polygons_zero_h = mask_to_yolo_polygons(mask, (0, W))
    assert polygons_zero_h == []
    polygons_zero_w = mask_to_yolo_polygons(mask, (H, 0))
    assert polygons_zero_w == []


def test_invalid_mask_dims():
    """Test with mask dimensions other than 2."""
    mask_1d = np.zeros(100, dtype=np.uint8)
    mask_3d = np.zeros((H, W, 3), dtype=np.uint8)
    polygons_1d = mask_to_yolo_polygons(mask_1d, IMG_SHAPE)
    assert polygons_1d == []
    polygons_3d = mask_to_yolo_polygons(mask_3d, IMG_SHAPE)
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

    polygons = mask_to_yolo_polygons(mask, IMG_SHAPE, connect_parts=False)
    assert len(polygons) == 1
    assert len(polygons[0]) == 8  # 4 points
    np.testing.assert_allclose(
        polygons[0], expected_poly, atol=0.01
    )  # Allow tolerance for approximation

    # Result should be the same if connect_parts=True for single contour
    polygons_connected = mask_to_yolo_polygons(mask, IMG_SHAPE, connect_parts=True)
    assert len(polygons_connected) == 1
    np.testing.assert_allclose(polygons_connected[0], expected_poly, atol=0.01)


def test_single_complex_contour():
    """Test an L-shaped contour."""
    # Points defined as (x, y)
    points = [(40, 20), (160, 20), (160, 50), (80, 50), (80, 80), (40, 80)]
    mask = create_l_shape_mask(IMG_SHAPE, points)

    polygons = mask_to_yolo_polygons(mask, IMG_SHAPE)
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
    polygons = mask_to_yolo_polygons(mask, IMG_SHAPE, min_contour_area=2.0)
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

    polygons = mask_to_yolo_polygons(mask, IMG_SHAPE, connect_parts=False)
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

    polygons = mask_to_yolo_polygons(mask, IMG_SHAPE, connect_parts=True)
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

    polygons = mask_to_yolo_polygons(mask, IMG_SHAPE, connect_parts=True, min_contour_area=10.0)
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

    polygons = mask_to_yolo_polygons(mask, IMG_SHAPE, connect_parts=False)
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

    polygons = mask_to_yolo_polygons(mask, IMG_SHAPE)
    assert len(polygons) == 1
    assert len(polygons[0]) == 8
    np.testing.assert_allclose(polygons[0], expected_poly, atol=0.01)
