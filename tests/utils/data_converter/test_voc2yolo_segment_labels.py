#!/usr/bin/env python3
"""Unit tests for the VOC to YOLO segmentation label converter.

Focuses on testing the core logic components:
- Instance ID parsing from palette masks
- Instance-to-class matching using class masks
- Binary mask to polygon conversion
- Processing of a single instance
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.utils.data_converter.voc2yolo_segment_labels import (
    VOC2YOLOConverter,
)

# --- Test Data Fixtures ---


@pytest.fixture
def mock_converter() -> VOC2YOLOConverter:
    """Create a mock converter instance for testing methods.

    Note: We don't need real paths as we'll mock file loading.
    """
    # Use dummy paths as they are not directly used in the tested methods
    mock_voc_root = MagicMock()
    mock_output_root = MagicMock()
    return VOC2YOLOConverter(mock_voc_root, mock_output_root, "2012", "test")


@pytest.fixture
def sample_instance_mask_array() -> np.ndarray:
    """4x4 mask with instance IDs 1 (top-left) and 2 (bottom-right)."""
    return np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]], dtype=np.uint8)


@pytest.fixture
def sample_class_mask_array() -> np.ndarray:
    """4x4 class mask: Person (15) top-left, Car (7) bottom-right."""
    return np.array([[15, 15, 0, 0], [15, 15, 0, 0], [0, 0, 7, 7], [0, 0, 7, 7]], dtype=np.uint8)


# --- Core Logic Tests ---


@patch("PIL.Image.open")
def test_get_mask_instances(mock_pil_open, mock_converter, sample_instance_mask_array):
    """Test parsing instance IDs from a mock palette PNG."""
    # Mock PIL Image loading
    mock_image = MagicMock()
    mock_image.__array__ = MagicMock(return_value=sample_instance_mask_array)
    mock_pil_open.return_value = mock_image

    instance_masks = mock_converter._get_mask_instances(MagicMock())  # Path doesn't matter

    assert instance_masks is not None
    assert 1 in instance_masks
    assert 2 in instance_masks
    assert 0 not in instance_masks  # Background excluded
    assert 255 not in instance_masks  # Boundary excluded

    # Check content of one instance mask
    expected_mask_1 = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
    )
    np.testing.assert_array_equal(instance_masks[1], expected_mask_1)


def test_mask_to_polygons(mock_converter):
    """Test converting a simple binary mask to normalized polygons."""
    # Simple square mask
    binary_mask = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)

    polygons = mock_converter._mask_to_polygons(binary_mask)

    assert polygons is not None
    assert len(polygons) >= 1  # Should find at least one contour
    poly = polygons[0]
    assert isinstance(poly, list)
    assert len(poly) >= 6  # Polygon needs at least 3 points (6 coordinates)
    assert all(isinstance(coord, float) for coord in poly)  # Check coordinates are floats
    assert all(0.0 <= coord <= 1.0 for coord in poly)  # Check coordinates are normalized


def test_match_instance_to_class(mock_converter, sample_class_mask_array):
    """Test matching an instance mask region to a class in the class mask."""
    # Instance 1 mask (top-left)
    instance_mask_1 = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
    )

    # Instance 2 mask (bottom-right)
    instance_mask_2 = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.uint8
    )

    # Match instance 1 -> should be Person (15)
    class_name_1 = mock_converter._match_instance_to_class(
        instance_mask_1, sample_class_mask_array, 1, "test_img"
    )
    assert class_name_1 == "person"

    # Match instance 2 -> should be Car (7)
    class_name_2 = mock_converter._match_instance_to_class(
        instance_mask_2, sample_class_mask_array, 2, "test_img"
    )
    assert class_name_2 == "car"

    # Test boundary case: Mask with only background/boundary pixels in class mask
    class_mask_boundary = np.zeros_like(sample_class_mask_array)
    class_mask_boundary[0, 0] = 255
    class_name_boundary = mock_converter._match_instance_to_class(
        instance_mask_1, class_mask_boundary, 1, "test_img"
    )
    assert class_name_boundary is None  # No valid class found


@patch("src.utils.data_converter.voc2yolo_segment_labels.VOC2YOLOConverter._mask_to_polygons")
@patch(
    "src.utils.data_converter.voc2yolo_segment_labels.VOC2YOLOConverter._match_instance_to_class"
)
def test_process_instance(mock_match_class, mock_polygons, mock_converter, sample_class_mask_array):
    """Test the processing of a single instance (polygon + class matching)."""
    # Setup instance data
    instance_id = 1
    binary_mask = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    img_id = "test_img"

    # Mock dependencies
    mock_polygons.return_value = [[0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4]]  # Single polygon
    mock_match_class.return_value = "person"

    # Execute the method
    output_lines = mock_converter._process_instance(
        instance_id, binary_mask, sample_class_mask_array, img_id
    )

    # Verify calls
    mock_polygons.assert_called_once_with(binary_mask)
    mock_match_class.assert_called_once_with(
        binary_mask, sample_class_mask_array, instance_id, img_id
    )

    # Verify output
    assert output_lines is not None
    assert len(output_lines) == 1
    expected_line = "14 0.100000 0.100000 0.400000 0.100000 0.400000 0.400000 0.100000 0.400000"
    assert output_lines[0] == expected_line

    # Test case: No polygons found
    mock_polygons.reset_mock()
    mock_match_class.reset_mock()
    mock_polygons.return_value = []
    output_lines_no_poly = mock_converter._process_instance(
        instance_id, binary_mask, sample_class_mask_array, img_id
    )
    assert output_lines_no_poly is None
    mock_match_class.assert_not_called()  # Should not attempt matching if no polygons

    # Test case: Class matching fails
    mock_polygons.reset_mock()
    mock_match_class.reset_mock()
    mock_polygons.return_value = [[0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4]]
    mock_match_class.return_value = None  # Simulate matching failure
    output_lines_no_match = mock_converter._process_instance(
        instance_id, binary_mask, sample_class_mask_array, img_id
    )
    assert output_lines_no_match is None
