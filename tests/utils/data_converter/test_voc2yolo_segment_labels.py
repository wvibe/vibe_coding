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
    apply_sampling_across_splits,
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
    return VOC2YOLOConverter(mock_voc_root, mock_output_root, "2012", "test", connect_parts=True)


@pytest.fixture
def mock_converter_without_connect_parts() -> VOC2YOLOConverter:
    """Create a mock converter with connect_parts=False for testing."""
    mock_voc_root = MagicMock()
    mock_output_root = MagicMock()
    return VOC2YOLOConverter(
        mock_voc_root, mock_output_root, "2012", "test", connect_parts=False, min_contour_area=1.0
    )


@pytest.fixture
def mock_converter_with_image_ids() -> VOC2YOLOConverter:
    """Create a mock converter with predefined image IDs."""
    mock_voc_root = MagicMock()
    mock_output_root = MagicMock()
    return VOC2YOLOConverter(mock_voc_root, mock_output_root, "2012", "test", ["img001", "img002"])


@pytest.fixture
def sample_instance_mask_array() -> np.ndarray:
    """4x4 mask with instance IDs 1 (top-left) and 2 (bottom-right)."""
    return np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]], dtype=np.uint8)


@pytest.fixture
def sample_class_mask_array() -> np.ndarray:
    """4x4 class mask: Person (15) top-left, Car (7) bottom-right."""
    return np.array([[15, 15, 0, 0], [15, 15, 0, 0], [0, 0, 7, 7], [0, 0, 7, 7]], dtype=np.uint8)


@pytest.fixture
def disconnected_mask() -> np.ndarray:
    """8x8 mask with two disconnected parts of the same instance."""
    mask = np.zeros((8, 8), dtype=np.uint8)
    # Top-left part
    mask[1:3, 1:3] = 1
    # Bottom-right part
    mask[5:7, 5:7] = 1
    return mask


# --- Test Constructor with Image IDs ---


def test_converter_constructor_with_image_ids(mock_converter_with_image_ids):
    """Test the converter constructor with the new image_ids parameter."""
    assert mock_converter_with_image_ids.image_ids == ["img001", "img002"]
    assert len(mock_converter_with_image_ids.image_ids) == 2

    # Test without image_ids (default)
    mock_voc_root = MagicMock()
    mock_output_root = MagicMock()
    converter = VOC2YOLOConverter(mock_voc_root, mock_output_root, "2012", "test")
    assert converter.image_ids is None


# --- Test Connect Parts Constructor ---


def test_converter_constructor_with_connect_parts():
    """Test the constructor with the connect_parts parameter."""
    mock_voc_root = MagicMock()
    mock_output_root = MagicMock()

    # Test with connect_parts=True (default)
    converter = VOC2YOLOConverter(mock_voc_root, mock_output_root, "2012", "test")
    assert converter.connect_parts is True

    # Test with connect_parts=False
    converter = VOC2YOLOConverter(
        mock_voc_root, mock_output_root, "2012", "test", connect_parts=False
    )
    assert converter.connect_parts is False

    # Test with custom min_contour_area
    custom_area = 5.0
    converter = VOC2YOLOConverter(
        mock_voc_root, mock_output_root, "2012", "test", min_contour_area=custom_area
    )
    assert converter.min_contour_area == custom_area


# --- Test Sampling Function ---


def test_apply_sampling_across_splits():
    """Test the sampling function that selects a subset of image IDs."""
    # Setup test data
    all_image_ids_map = {
        ("2007", "train"): ["img1", "img2", "img3", "img4", "img5"],
        ("2007", "val"): ["img6", "img7", "img8"],
        ("2012", "train"): ["img9", "img10"],
    }
    total_ids = 10

    # Test no sampling (sample_count=None)
    result = apply_sampling_across_splits(all_image_ids_map, None, total_ids, seed=42)
    assert result == all_image_ids_map

    # Test sampling more than available (should return all)
    result = apply_sampling_across_splits(all_image_ids_map, 20, total_ids, seed=42)
    assert result == all_image_ids_map

    # Test sampling a subset
    result = apply_sampling_across_splits(all_image_ids_map, 5, total_ids, seed=42)
    assert sum(len(ids) for ids in result.values()) == 5

    # Test sampling with consistent seed
    result1 = apply_sampling_across_splits(all_image_ids_map, 3, total_ids, seed=42)
    result2 = apply_sampling_across_splits(all_image_ids_map, 3, total_ids, seed=42)
    # Both results should be the same with the same seed
    assert result1 == result2


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


def test_match_instance_to_class(mock_converter, sample_class_mask_array):
    """Test matching an instance binary mask to its class name."""
    # Instance 1 (top-left) should match Person (ID 15)
    instance_mask_1 = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
    )
    class_name_1 = mock_converter._match_instance_to_class(
        instance_mask_1, sample_class_mask_array, 1, "test_img"
    )
    assert class_name_1 == "person"

    # Instance 2 (bottom-right) should match Car (ID 7)
    instance_mask_2 = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.uint8
    )
    class_name_2 = mock_converter._match_instance_to_class(
        instance_mask_2, sample_class_mask_array, 2, "test_img"
    )
    assert class_name_2 == "car"

    # Test case with no overlap with valid class pixels (only background)
    instance_mask_bg = np.array(
        [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
    )  # Overlaps only with background in class mask
    class_name_bg = mock_converter._match_instance_to_class(
        instance_mask_bg, sample_class_mask_array, 3, "test_img"
    )
    assert class_name_bg is None


@patch("src.utils.data_converter.voc2yolo_segment_labels.mask_to_yolo_polygons")
@patch(
    "src.utils.data_converter.voc2yolo_segment_labels.VOC2YOLOConverter._match_instance_to_class"
)
def test_process_instance(mock_match_class, mock_polygons, mock_converter, sample_class_mask_array):
    """Test processing a single instance: polygon generation and class matching."""
    instance_id = 1
    binary_mask_1 = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
    )

    # Mock return values
    mock_polygons.return_value = [
        [0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.0]
    ]  # Mock normalized polygon
    mock_match_class.return_value = "person"

    result = mock_converter._process_instance(
        instance_id, binary_mask_1, sample_class_mask_array, "img001"
    )

    assert result is not None
    assert len(result) == 1  # One line for one polygon
    # Person class ID is 14 (0-based)
    expected_line = "14 0.000000 0.000000 0.000000 0.250000 0.250000 0.250000 0.250000 0.000000"
    assert result[0] == expected_line

    # Check that mock_polygons was called correctly
    h, w = sample_class_mask_array.shape[:2]
    mock_polygons.assert_called_once_with(
        binary_mask=binary_mask_1,
        img_shape=(h, w),
        connect_parts=mock_converter.connect_parts,  # Should be False for this fixture
        min_contour_area=mock_converter.min_contour_area,
    )
    mock_match_class.assert_called_once_with(
        binary_mask_1, sample_class_mask_array, instance_id, "img001"
    )

    # Test case where polygon generation fails
    mock_polygons.reset_mock()
    mock_match_class.reset_mock()
    mock_polygons.return_value = []  # Simulate no polygons found
    result_fail_poly = mock_converter._process_instance(
        instance_id, binary_mask_1, sample_class_mask_array, "img001"
    )
    assert result_fail_poly is None
    mock_polygons.assert_called_once()
    mock_match_class.assert_not_called()  # Should not match class if polygons fail

    # Test case where class matching fails
    mock_polygons.reset_mock()
    mock_match_class.reset_mock()
    mock_polygons.return_value = [[0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.0]]  # Polygons succeed
    mock_match_class.return_value = None  # Simulate class match failure
    result_fail_match = mock_converter._process_instance(
        instance_id, binary_mask_1, sample_class_mask_array, "img001"
    )
    assert result_fail_match is None
    mock_polygons.assert_called_once()
    mock_match_class.assert_called_once()


@patch("src.utils.data_converter.voc2yolo_segment_labels.mask_to_yolo_polygons")
@patch(
    "src.utils.data_converter.voc2yolo_segment_labels.VOC2YOLOConverter._match_instance_to_class"
)
def test_process_instance_default_connect_parts(
    mock_match_class, mock_polygons, mock_converter, sample_class_mask_array
):
    """Test processing instance with default connect_parts=True."""
    instance_id = 1
    binary_mask_1 = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
    )

    # Mock return values
    mock_polygons.return_value = [[0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.0]]  # Mock polygon
    mock_match_class.return_value = "person"

    result = mock_converter._process_instance(
        instance_id, binary_mask_1, sample_class_mask_array, "img001"
    )

    assert result is not None
    assert len(result) == 1
    expected_line = "14 0.000000 0.000000 0.000000 0.250000 0.250000 0.250000 0.250000 0.000000"
    assert result[0] == expected_line

    # Check that mock_polygons was called correctly with connect_parts=True
    h, w = sample_class_mask_array.shape[:2]
    mock_polygons.assert_called_once_with(
        binary_mask=binary_mask_1,
        img_shape=(h, w),
        connect_parts=True,  # Key check for this test
        min_contour_area=mock_converter.min_contour_area,
    )
    mock_match_class.assert_called_once_with(
        binary_mask_1, sample_class_mask_array, instance_id, "img001"
    )


@patch("src.utils.data_converter.voc2yolo_segment_labels.mask_to_yolo_polygons")
@patch(
    "src.utils.data_converter.voc2yolo_segment_labels.VOC2YOLOConverter._match_instance_to_class"
)
def test_process_instance_without_connect_parts(
    mock_match_class, mock_polygons, mock_converter_without_connect_parts, sample_class_mask_array
):
    """Test processing instance with connect_parts=False."""
    instance_id = 1
    binary_mask_1 = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
    )

    # Mock return values
    mock_polygons.return_value = [[0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.0]]  # Mock polygon
    mock_match_class.return_value = "person"

    result = mock_converter_without_connect_parts._process_instance(
        instance_id, binary_mask_1, sample_class_mask_array, "img001"
    )

    assert result is not None
    assert len(result) == 1
    expected_line = "14 0.000000 0.000000 0.000000 0.250000 0.250000 0.250000 0.250000 0.000000"
    assert result[0] == expected_line

    # Check that mock_polygons was called correctly with connect_parts=False
    h, w = sample_class_mask_array.shape[:2]
    mock_polygons.assert_called_once_with(
        binary_mask=binary_mask_1,
        img_shape=(h, w),
        connect_parts=False,  # Key check for this test
        min_contour_area=mock_converter_without_connect_parts.min_contour_area,
    )
    mock_match_class.assert_called_once_with(
        binary_mask_1, sample_class_mask_array, instance_id, "img001"
    )


# --- Integration/End-to-End Style Tests ---
# These tests use temporary directories and files
