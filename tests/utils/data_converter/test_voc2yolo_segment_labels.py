#!/usr/bin/env python3
"""Unit tests for the VOC to YOLO segmentation label converter.

Focuses on testing the core logic components:
- Instance ID parsing from palette masks
- Instance-to-class matching using class masks
- Binary mask to polygon conversion
- Processing of a single instance
"""

import tempfile
from pathlib import Path
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
    return VOC2YOLOConverter(mock_voc_root, mock_output_root, "2012", "test")


@pytest.fixture
def mock_converter_with_connect_parts() -> VOC2YOLOConverter:
    """Create a mock converter with connect_parts=True for testing."""
    mock_voc_root = MagicMock()
    mock_output_root = MagicMock()
    return VOC2YOLOConverter(
        mock_voc_root, mock_output_root, "2012", "test", connect_parts=True, min_contour_area=1.0
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

    # Test with connect_parts=True
    converter = VOC2YOLOConverter(
        mock_voc_root, mock_output_root, "2012", "test", connect_parts=True
    )
    assert converter.connect_parts is True

    # Test with connect_parts=False (default)
    converter = VOC2YOLOConverter(mock_voc_root, mock_output_root, "2012", "test")
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


def test_mask_to_polygons_with_connect_parts(mock_converter_with_connect_parts, disconnected_mask):
    """Test converting a mask with disconnected parts with connect_parts=True."""
    # This test verifies that when connect_parts=True, we get a single polygon for disconnected parts

    polygons = mock_converter_with_connect_parts._mask_to_polygons(disconnected_mask)

    # Check we have exactly one polygon representing both disconnected regions
    assert len(polygons) == 1, "When connect_parts=True, should produce exactly one polygon"

    # The polygon should have at least 8 points (16 coordinates) to represent the complex shape
    # (minimum 3 points per part plus connecting points)
    assert len(polygons[0]) >= 16, (
        "Connected polygon should have enough points to represent both regions"
    )

    # All coordinates should be normalized between 0 and 1
    assert all(0.0 <= coord <= 1.0 for coord in polygons[0])

    # When connect_parts=False, we should get multiple polygons
    mock_converter = VOC2YOLOConverter(
        MagicMock(), MagicMock(), "2012", "test", connect_parts=False
    )
    separate_polygons = mock_converter._mask_to_polygons(disconnected_mask)

    # Should find two separate polygons
    assert len(separate_polygons) == 2, (
        "When connect_parts=False, should produce two separate polygons"
    )


@patch("cv2.findContours")
@patch("cv2.approxPolyDP")
def test_connect_disconnected_parts(
    mock_approx, mock_find_contours, mock_converter_with_connect_parts
):
    """Test the _connect_disconnected_parts method directly."""
    # Create two synthetic contours
    contour1 = np.array([[[1, 1]], [[2, 1]], [[2, 2]], [[1, 2]]], dtype=np.int32)
    contour2 = np.array([[[5, 5]], [[6, 5]], [[6, 6]], [[5, 6]]], dtype=np.int32)
    contours = [contour1, contour2]

    # Set up approxPolyDP to return the contours unchanged for simplicity
    mock_approx.side_effect = lambda contour, epsilon, closed: contour

    # Call the method directly
    img_shape = (8, 8)  # 8x8 image
    result = mock_converter_with_connect_parts._connect_disconnected_parts(contours, img_shape)

    # Should produce a single polygon with normalized coordinates
    assert len(result) > 0, "Should produce a non-empty polygon"
    assert all(0.0 <= coord <= 1.0 for coord in result), "All coordinates should be normalized"

    # The result should have at least the points from both contours plus connecting points
    # Each contour has 4 points (8 coords) plus connecting points
    assert len(result) >= 16, "Should contain points from both contours plus connecting points"

    # Test with a single contour - should just normalize and return it
    result_single = mock_converter_with_connect_parts._connect_disconnected_parts(
        [contour1], img_shape
    )
    assert len(result_single) == 8, "Single contour should have 8 coordinates (4 points)"

    # Test with no contours
    result_empty = mock_converter_with_connect_parts._connect_disconnected_parts([], img_shape)
    assert result_empty == [], "Empty contours should return empty result"

    # Test with tiny contours below the area threshold
    tiny_contour = np.array(
        [[[1, 1]], [[1, 2]], [[2, 1]]], dtype=np.int32
    )  # Triangle with area 0.5
    mock_converter_with_min_area = VOC2YOLOConverter(
        MagicMock(), MagicMock(), "2012", "test", connect_parts=True, min_contour_area=2.0
    )
    result_filtered = mock_converter_with_min_area._connect_disconnected_parts(
        [tiny_contour], img_shape
    )
    assert result_filtered == [], "Contours below min_contour_area should be filtered out"


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


@patch("src.utils.data_converter.voc2yolo_segment_labels.VOC2YOLOConverter._mask_to_polygons")
@patch(
    "src.utils.data_converter.voc2yolo_segment_labels.VOC2YOLOConverter._match_instance_to_class"
)
def test_process_instance_with_connect_parts(
    mock_match_class, mock_polygons, mock_converter_with_connect_parts, sample_class_mask_array
):
    """Test processing an instance with connect_parts=True produces correct output."""
    # Setup
    instance_id = 1
    binary_mask = np.zeros((8, 8), dtype=np.uint8)
    binary_mask[1:3, 1:3] = 1  # Top-left part
    binary_mask[5:7, 5:7] = 1  # Bottom-right part
    img_id = "test_img"

    # Connected parts should produce a single polygon
    connected_polygon = [
        0.1,
        0.1,
        0.2,
        0.1,
        0.2,
        0.2,
        0.1,
        0.2,
        0.6,
        0.6,
        0.8,
        0.6,
        0.8,
        0.8,
        0.6,
        0.8,
    ]
    mock_polygons.return_value = [connected_polygon]  # Single polygon with multiple parts
    mock_match_class.return_value = "person"

    # Execute
    output_lines = mock_converter_with_connect_parts._process_instance(
        instance_id, binary_mask, sample_class_mask_array, img_id
    )

    # Verify
    assert output_lines is not None
    assert len(output_lines) == 1  # Should produce exactly one line (one polygon)

    # Parse the output line
    parts = output_lines[0].split()
    assert parts[0] == "14"  # Class ID for person

    # The coordinates should match our mocked polygon
    for i, coord in enumerate(connected_polygon):
        assert float(parts[i + 1]) == coord

    # With connect_parts disabled, multiple polygons would be separate lines
    mock_polygons.reset_mock()
    mock_match_class.reset_mock()

    # Mock returning multiple polygons
    separate_polygons = [
        [0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2],  # First part
        [0.6, 0.6, 0.8, 0.6, 0.8, 0.8, 0.6, 0.8],  # Second part
    ]
    mock_polygons.return_value = separate_polygons
    mock_match_class.return_value = "person"

    # Create a new converter with connect_parts=False
    regular_converter = VOC2YOLOConverter(
        MagicMock(), MagicMock(), "2012", "test", connect_parts=False
    )

    # Use regular converter (connect_parts=False)
    output_lines_separate = regular_converter._process_instance(
        instance_id, binary_mask, sample_class_mask_array, img_id
    )

    # Should have two lines (two polygons)
    assert output_lines_separate is not None
    assert len(output_lines_separate) == 2


# --- Test Process Segmentation File with File Skipping ---


@patch("src.utils.data_converter.voc2yolo_segment_labels.VOC2YOLOConverter._validate_directories")
@patch("src.utils.data_converter.voc2yolo_utils.get_segm_inst_mask_path")
@patch("src.utils.data_converter.voc2yolo_utils.get_segm_cls_mask_path")
def test_process_segmentation_file_with_existing_file(mock_cls_path, mock_inst_path, mock_validate):
    """Test that the _process_segmentation_file method returns 'skipped' for existing files."""
    # Mock the directory validation
    mock_validate.return_value = True

    # Mock the path getters
    mock_inst_path.return_value = Path("fake_inst_path.png")
    mock_cls_path.return_value = Path("fake_cls_path.png")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        # Create a converter with our temp directory
        with patch(
            "src.utils.data_converter.voc2yolo_segment_labels.get_voc_dir"
        ) as mock_get_voc_dir:
            # Make get_voc_dir return our temp dir
            mock_get_voc_dir.return_value = temp_dir

            converter = VOC2YOLOConverter(temp_dir, temp_dir, "2012", "test")

            # Directly set the output directory
            converter.output_segment_dir = temp_dir

            # Create a pre-existing output file
            img_id = "test001"
            output_path = temp_dir / f"{img_id}.txt"
            output_path.touch()  # Create an empty file

            # Test the method returns "skipped"
            result, _, _ = converter._process_segmentation_file(img_id)
            assert result == "skipped"
