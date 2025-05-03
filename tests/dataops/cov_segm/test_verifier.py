"""Unit tests for the conversion verifier implementation."""

import numpy as np

from vibelab.dataops.cov_segm.convert_verifier import (
    OriginalInstanceRecord,
    YoloInstanceRecord,
    _calculate_bbox_from_mask,
)
from vibelab.utils.common.label_match import match_instances
from vibelab.utils.common.mask import calculate_mask_iou, polygon_to_mask


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
    """Test _match_instances with perfect matches."""
    # Create two masks with no overlap and different classes
    mask1 = np.zeros((10, 10), dtype=bool)
    mask1[0:5, 0:5] = True  # Class 1: 5x5 square in top-left

    mask2 = np.zeros((10, 10), dtype=bool)
    mask2[5:10, 5:10] = True  # Class 2: 5x5 square in bottom-right

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
    matched, lost, extra = match_instances(
        original_instances,
        yolo_instances,
        compute_iou_fn=_mask_iou_wrapper,
        iou_cutoff=1.0,
    )

    # We expect both instances to match correctly
    assert len(matched) == 2, f"Expected 2 matches, got {len(matched)}"
    assert len(lost) == 0, f"Expected 0 lost instances, got {len(lost)}"
    assert len(extra) == 0, f"Expected 0 extra instances, got {len(extra)}"

    # The first original should match the second YOLO (class 1)
    # The second original should match the first YOLO (class 2)
    # Matches now include IoU values (idx_a, idx_b, iou)
    # Extract just the indices for comparison
    match_indices = {(a, b) for a, b, _ in matched}
    expected_indices = {(0, 1), (1, 0)}  # Set of (orig_idx, yolo_idx) tuples

    assert match_indices == expected_indices, (
        f"Expected match indices {expected_indices}, got {match_indices}"
    )

    # Verify IoU values are 1.0 for perfect matches
    for _, _, iou in matched:
        assert iou == 1.0, f"Expected IoU of 1.0 for exact match, got {iou}"


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
    matched, lost, extra = match_instances(
        original_instances,
        yolo_instances,
        compute_iou_fn=_mask_iou_wrapper,
        iou_cutoff=0.17,
    )

    # We expect a match
    assert len(matched) == 1, f"Expected 1 match, got {len(matched)}"
    assert len(lost) == 0, f"Expected 0 lost instances, got {len(lost)}"
    assert len(extra) == 0, f"Expected 0 extra instances, got {len(extra)}"

    # Verify the IoU value is included
    assert len(matched[0]) == 3, f"Expected match tuple to have 3 elements, got {len(matched[0])}"
    _, _, iou = matched[0]
    # IoU should be around 0.173
    assert 0.17 <= iou <= 0.18, f"Expected IoU around 0.173, got {iou}"

    # Test with IoU threshold just above the expected IoU
    matched, lost, extra = match_instances(
        original_instances,
        yolo_instances,
        compute_iou_fn=_mask_iou_wrapper,
        iou_cutoff=0.18,
    )

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
            sample_id="sample1",
            segment_idx=0,
            mask_idx=0,
            class_id=1,
            original_mask=mask_class1a,
            bbox=(0, 0, 2, 2),
        ),
        OriginalInstanceRecord(
            sample_id="sample1",
            segment_idx=1,
            mask_idx=0,
            class_id=1,
            original_mask=mask_class1b,
            bbox=(3, 0, 5, 2),
        ),
        OriginalInstanceRecord(
            sample_id="sample1",
            segment_idx=2,
            mask_idx=0,
            class_id=2,
            original_mask=mask_class2,
            bbox=(0, 3, 2, 5),
        ),
        OriginalInstanceRecord(
            sample_id="sample1",
            segment_idx=3,
            mask_idx=0,
            class_id=3,
            original_mask=mask_class3,
            bbox=(6, 6, 8, 8),
        ),
    ]

    # Create YOLO instances - match class 2, class 1a, and provide another instance of class 1
    # to ensure the class 1 matching works properly
    yolo_instances = [
        YoloInstanceRecord(
            sample_id="sample1",
            class_id=1,
            polygon_abs=[(0, 0), (2, 0), (2, 2), (0, 2)],
            derived_mask=mask_class1a,
            bbox=(0, 0, 2, 2),
        ),
        YoloInstanceRecord(
            sample_id="sample1",
            class_id=1,
            polygon_abs=[(0, 3), (2, 3), (6, 5), (3, 6)],
            derived_mask=mask_class1b,
            bbox=(3, 0, 5, 2),
        ),
        YoloInstanceRecord(
            sample_id="sample1",
            class_id=2,
            polygon_abs=[(0, 3), (2, 3), (2, 5), (0, 5)],
            derived_mask=mask_class2,
            bbox=(0, 3, 2, 5),
        ),
        # Class 3 is missing from YOLO
    ]

    # Match with perfect IoU required
    matched, lost, extra = match_instances(
        original_instances,
        yolo_instances,
        compute_iou_fn=_mask_iou_wrapper,
        iou_cutoff=1.0,
    )

    # We expect 3 matches (class 1a, class 1b, and class 2), 1 lost (class 3)
    assert len(matched) == 3, f"Expected 3 matches, got {len(matched)}"
    assert len(lost) == 1, f"Expected 1 lost instance, got {len(lost)}"
    assert len(extra) == 0, f"Expected 0 extra instances, got {len(extra)}"

    # Check the specific matches - extract index pairs for comparison
    match_indices = {(a, b) for a, b, _ in matched}
    expected_indices = {(0, 0), (1, 1), (2, 2)}
    assert match_indices == expected_indices, (
        f"Expected match indices {expected_indices}, got {match_indices}"
    )

    # Verify IoU values are 1.0 for perfect matches
    for _, _, iou in matched:
        assert iou == 1.0, f"Expected IoU of 1.0 for exact match, got {iou}"

    # Check which instance was lost
    assert set(lost) == {3}, f"Expected lost instance [3], got {lost}"  # Original class 3


# Helper function for match_instances
def _mask_iou_wrapper(a: OriginalInstanceRecord, b: YoloInstanceRecord) -> float:  # type: ignore
    """Compute IoU between the stored masks in two instance records."""
    return calculate_mask_iou(a.original_mask, b.derived_mask)


DEFAULT_IOU_CUTOFF = 0.5  # generic cutoff for matching


def test_bbox_calculation_methods():
    """Test that direct bbox calculation and mask-based calculation yield similar results."""
    # Test case 1: Simple square polygon
    width, height = 100, 100
    poly_abs = [(25, 25), (75, 25), (75, 75), (25, 75)]  # Simple square

    # Direct calculation
    x_min = min(x for x, _ in poly_abs)
    y_min = min(y for _, y in poly_abs)
    x_max = max(x for x, _ in poly_abs)
    y_max = max(y for _, y in poly_abs)
    direct_bbox = (x_min, y_min, x_max, y_max)  # Should be (25, 25, 75, 75)

    # Mask-based calculation
    mask = polygon_to_mask(poly_abs, height, width)
    mask_bbox = _calculate_bbox_from_mask(mask)

    # Compare results for simple polygon
    assert direct_bbox == mask_bbox, f"Simple polygon: Direct {direct_bbox} != Mask {mask_bbox}"

    # Test case 2: Complex polygon
    complex_poly = [(30, 20), (70, 10), (90, 40), (80, 80), (40, 70), (10, 50)]

    # Direct calculation
    complex_x_min = min(x for x, _ in complex_poly)
    complex_y_min = min(y for _, y in complex_poly)
    complex_x_max = max(x for x, _ in complex_poly)
    complex_y_max = max(y for _, y in complex_poly)
    complex_direct_bbox = (complex_x_min, complex_y_min, complex_x_max, complex_y_max)

    # Mask-based calculation
    complex_mask = polygon_to_mask(complex_poly, height, width)
    complex_mask_bbox = _calculate_bbox_from_mask(complex_mask)

    # Compare results for complex polygon with allowed margin
    allowed_margin = 1  # Allow 1 pixel difference
    assert all(
        abs(a - b) <= allowed_margin for a, b in zip(complex_direct_bbox, complex_mask_bbox)
    ), (
        f"Complex polygon: Direct {complex_direct_bbox} differs significantly from Mask {complex_mask_bbox}"
    )
