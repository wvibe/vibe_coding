"""Unit tests for the conversion verifier implementation."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from vibelab.dataops.cov_segm.convert_verifier import (
    _match_instances,
    _calculate_bbox_from_mask,
    OriginalInstanceRecord,
    YoloInstanceRecord,
)


def test_calculate_bbox_from_mask():
    """Test the _calculate_bbox_from_mask function."""
    # Create a simple 10x10 mask with a square in the middle
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 2:8] = True  # Square from (2,3) to (7,6)

    # The expected bounding box is (xmin, ymin, xmax, ymax)
    expected_bbox = (2, 3, 7, 6)
    result_bbox = _calculate_bbox_from_mask(mask)

    assert result_bbox == expected_bbox, f"Expected {expected_bbox}, got {result_bbox}"

    # Test with empty mask
    empty_mask = np.zeros((5, 5), dtype=bool)
    assert _calculate_bbox_from_mask(empty_mask) is None, "Empty mask should return None"


def test_match_instances_exact_match():
    """Test _match_instances with instances that have exact mask matches."""
    # Create two original instances with simple masks
    mask1 = np.zeros((10, 10), dtype=bool)
    mask1[0:5, 0:5] = True  # Square in top-left (0,0) to (4,4)

    mask2 = np.zeros((10, 10), dtype=bool)
    mask2[5:10, 5:10] = True  # Square in bottom-right (5,5) to (9,9)

    # Create original instances (different classes)
    original_instances = [
        OriginalInstanceRecord(
            sample_id="sample1",
            segment_idx=0,
            mask_idx=0,
            class_id=1,
            original_mask=mask1,
            bbox=(0, 0, 4, 4),
        ),
        OriginalInstanceRecord(
            sample_id="sample1",
            segment_idx=1,
            mask_idx=0,
            class_id=2,
            original_mask=mask2,
            bbox=(5, 5, 9, 9),
        ),
    ]

    # Create matching YOLO instances (same masks but reverse order)
    yolo_instances = [
        YoloInstanceRecord(
            sample_id="sample1",
            class_id=2,
            polygon_abs=[(5, 5), (9, 5), (9, 9), (5, 9)],  # Not used in test
            derived_mask=mask2,
            bbox=(5, 5, 9, 9),
        ),
        YoloInstanceRecord(
            sample_id="sample1",
            class_id=1,
            polygon_abs=[(0, 0), (4, 0), (4, 4), (0, 4)],  # Not used in test
            derived_mask=mask1,
            bbox=(0, 0, 4, 4),
        ),
    ]

    # Match with 100% minimum IoU (exact match required)
    matched, lost, extra = _match_instances(original_instances, yolo_instances, mask_min_iou=1.0)

    # We expect both instances to match correctly
    assert len(matched) == 2, f"Expected 2 matches, got {len(matched)}"
    assert len(lost) == 0, f"Expected 0 lost instances, got {len(lost)}"
    assert len(extra) == 0, f"Expected 0 extra instances, got {len(extra)}"

    # The first original should match the second YOLO (class 1)
    # The second original should match the first YOLO (class 2)
    expected_matches = {(0, 1), (1, 0)}  # Set of (orig_idx, yolo_idx) tuples
    actual_matches = set(matched)

    assert actual_matches == expected_matches, f"Expected matches {expected_matches}, got {actual_matches}"


def test_match_instances_partial_overlap():
    """Test _match_instances with instances that partially overlap."""
    # Create masks with partial overlap
    mask1 = np.zeros((10, 10), dtype=bool)
    mask1[0:6, 0:6] = True  # Larger square in top-left (0,0) to (5,5)

    mask2 = np.zeros((10, 10), dtype=bool)
    mask2[3:8, 3:8] = True  # Overlapping square (3,3) to (7,7)

    # The overlap area is (3,3) to (5,5), which is 3x3 = 9 pixels
    # Area of mask1 is 6x6 = 36 pixels
    # Area of mask2 is 5x5 = 25 pixels
    # Union is 36 + 25 - 9 = 52 pixels
    # IoU = 9/52 ≈ 0.173

    original_instances = [
        OriginalInstanceRecord(
            sample_id="sample1",
            segment_idx=0,
            mask_idx=0,
            class_id=1,
            original_mask=mask1,
            bbox=(0, 0, 5, 5),
        ),
    ]

    yolo_instances = [
        YoloInstanceRecord(
            sample_id="sample1",
            class_id=1,
            polygon_abs=[(3, 3), (7, 3), (7, 7), (3, 7)],  # Not used in test
            derived_mask=mask2,
            bbox=(3, 3, 7, 7),
        ),
    ]

    # Test with IoU threshold just below the expected IoU
    matched, lost, extra = _match_instances(original_instances, yolo_instances, mask_min_iou=0.17)

    # We expect a match
    assert len(matched) == 1, f"Expected 1 match, got {len(matched)}"
    assert len(lost) == 0, f"Expected 0 lost instances, got {len(lost)}"
    assert len(extra) == 0, f"Expected 0 extra instances, got {len(extra)}"

    # Test with IoU threshold just above the expected IoU
    matched, lost, extra = _match_instances(original_instances, yolo_instances, mask_min_iou=0.18)

    # We expect no matches
    assert len(matched) == 0, f"Expected 0 matches, got {len(matched)}"
    assert len(lost) == 1, f"Expected 1 lost instance, got {len(lost)}"
    assert len(extra) == 1, f"Expected 1 extra instance, got {len(extra)}"


def test_match_instances_multiple_classes():
    """Test _match_instances with multiple class instances."""
    # Create sample masks for three classes
    mask_class1a = np.zeros((10, 10), dtype=bool)
    mask_class1a[0:3, 0:3] = True  # Class 1, instance 1

    mask_class1b = np.zeros((10, 10), dtype=bool)
    mask_class1b[0:3, 3:6] = True  # Class 1, instance 2 (distinct from 1a)

    mask_class2 = np.zeros((10, 10), dtype=bool)
    mask_class2[3:6, 0:3] = True  # Class 2

    mask_class3 = np.zeros((10, 10), dtype=bool)
    mask_class3[6:9, 6:9] = True  # Class 3

    # Create original instances
    original_instances = [
        OriginalInstanceRecord(
            sample_id="sample1", segment_idx=0, mask_idx=0, class_id=1,
            original_mask=mask_class1a, bbox=(0, 0, 2, 2),
        ),
        OriginalInstanceRecord(
            sample_id="sample1", segment_idx=1, mask_idx=0, class_id=1,
            original_mask=mask_class1b, bbox=(3, 0, 5, 2),
        ),
        OriginalInstanceRecord(
            sample_id="sample1", segment_idx=2, mask_idx=0, class_id=2,
            original_mask=mask_class2, bbox=(0, 3, 2, 5),
        ),
        OriginalInstanceRecord(
            sample_id="sample1", segment_idx=3, mask_idx=0, class_id=3,
            original_mask=mask_class3, bbox=(6, 6, 8, 8),
        ),
    ]

    # Create YOLO instances - match class 2, class 1a, and provide another instance of class 1
    # to ensure the class 1 matching works properly
    yolo_instances = [
        YoloInstanceRecord(
            sample_id="sample1", class_id=1,
            polygon_abs=[(0, 0), (2, 0), (2, 2), (0, 2)],
            derived_mask=mask_class1a, bbox=(0, 0, 2, 2),
        ),
        YoloInstanceRecord(
            sample_id="sample1", class_id=1,
            polygon_abs=[(0, 3), (2, 3), (6, 5), (3, 6)],
            derived_mask=mask_class1b, bbox=(3, 0, 5, 2),
        ),
        YoloInstanceRecord(
            sample_id="sample1", class_id=2,
            polygon_abs=[(0, 3), (2, 3), (2, 5), (0, 5)],
            derived_mask=mask_class2, bbox=(0, 3, 2, 5),
        ),
        # Class 3 is missing from YOLO
    ]

    # Match with perfect IoU required
    matched, lost, extra = _match_instances(original_instances, yolo_instances, mask_min_iou=1.0)

    # We expect 3 matches (class 1a, class 1b, and class 2), 1 lost (class 3)
    assert len(matched) == 3, f"Expected 3 matches, got {len(matched)}"
    assert len(lost) == 1, f"Expected 1 lost instance, got {len(lost)}"
    assert len(extra) == 0, f"Expected 0 extra instances, got {len(extra)}"

    # Check the specific matches - order may vary based on implementation
    matched_pairs = set(matched)
    assert (0, 0) in matched_pairs, "Class 1a should match first YOLO instance"
    assert (1, 1) in matched_pairs, "Class 1b should match second YOLO instance"
    assert (2, 2) in matched_pairs, "Class 2 should match third YOLO instance"

    # Check which instance was lost
    assert set(lost) == {3}, f"Expected lost instance [3], got {lost}"  # Original class 3