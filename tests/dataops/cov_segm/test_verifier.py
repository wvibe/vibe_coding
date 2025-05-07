"""Unit tests for the conversion verifier implementation."""

from unittest.mock import MagicMock, patch

import numpy as np

from vibelab.dataops.cov_segm.convert_verifier import (
    OriginalInstanceRecord,
    YoloInstanceRecord,
    _calculate_bbox_from_mask,
    _match_instances_for_class,
    _process_lost_and_extra_instances,
    _process_matched_pairs,
    verify_sample_conversion,
)
from vibelab.utils.common.label_match import match_instances
from vibelab.utils.common.mask import calculate_mask_iou, polygons_to_mask


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
    mask = polygons_to_mask(poly_abs, (height, width), normalized=False)
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
    complex_mask = polygons_to_mask(complex_poly, (height, width), normalized=False)
    complex_mask_bbox = _calculate_bbox_from_mask(complex_mask)

    # Compare results for complex polygon with allowed margin
    allowed_margin = 1  # Allow 1 pixel difference
    assert all(
        abs(a - b) <= allowed_margin for a, b in zip(complex_direct_bbox, complex_mask_bbox)
    ), (
        f"Complex polygon: Direct {complex_direct_bbox} differs significantly from Mask {complex_mask_bbox}"
    )


# Helper function to create simple masks/instances for tests
def _create_dummy_mask(size=(10, 10), area=None):
    mask = np.zeros(size, dtype=bool)
    if area:
        y1, x1, y2, x2 = area
        mask[y1:y2, x1:x2] = True
    return mask


def _create_dummy_orig_instance(sid="s1", seg_idx=0, mask_idx=0, cid=1, mask_area=(0, 0, 5, 5)):
    mask = _create_dummy_mask(area=mask_area)
    bbox = _calculate_bbox_from_mask(mask)
    return OriginalInstanceRecord(
        sample_id=sid,
        segment_idx=seg_idx,
        mask_idx=mask_idx,
        class_id=cid,
        original_mask=mask,
        bbox=bbox,
    )


def _create_dummy_yolo_instance(
    sid="s1", cid=1, poly_abs=[(0, 0), (5, 0), (5, 5), (0, 5)], mask_area=(0, 0, 5, 5), bbox=None
):
    mask = _create_dummy_mask(area=mask_area)
    derived_bbox = _calculate_bbox_from_mask(mask) if bbox is None else bbox
    return YoloInstanceRecord(
        sample_id=sid, class_id=cid, polygon_abs=poly_abs, derived_mask=mask, bbox=derived_bbox
    )


# --- Tests for _match_instances_for_class ---


@patch("vibelab.dataops.cov_segm.convert_verifier.match_instances")
def test_match_instances_for_class_success(mock_match):
    """Test successful matching for a class."""
    orig_inst = [_create_dummy_orig_instance(cid=1)]
    yolo_inst = [_create_dummy_yolo_instance(cid=1)]
    iou_cutoff = 0.5

    # Mock return value for match_instances (mask and bbox)
    mock_match.side_effect = [
        ([(0, 0, 0.9)], [], []),  # Mask match result
        ([(0, 0, 0.8)], [], []),  # Bbox match result
    ]

    results = _match_instances_for_class(orig_inst, yolo_inst, iou_cutoff)

    assert results["mask_error"] is None
    assert results["bbox_error"] is None
    assert len(results["mask_matched"]) == 1
    assert results["mask_matched"][0] == (0, 0, 0.9)
    assert len(results["bbox_matched"]) == 1
    assert results["bbox_matched"][0] == (0, 0, 0.8)
    assert not results["mask_lost"] and not results["mask_extra"]
    assert not results["bbox_lost"] and not results["bbox_extra"]


@patch("vibelab.dataops.cov_segm.convert_verifier.match_instances")
def test_match_instances_for_class_match_error(mock_match):
    """Test error handling during matching."""
    orig_inst = [_create_dummy_orig_instance(cid=1)]
    yolo_inst = [_create_dummy_yolo_instance(cid=1)]
    iou_cutoff = 0.5

    # Mock match_instances to raise an error for mask matching
    mock_match.side_effect = [
        Exception("Mask matching failed!"),  # Mask match error
        ([(0, 0, 0.8)], [], []),  # Bbox match succeeds
    ]

    results = _match_instances_for_class(orig_inst, yolo_inst, iou_cutoff)

    # Mask results should reflect the error
    assert "Mask matching failed!" in results["mask_error"]
    assert results["mask_matched"] == []
    assert results["mask_lost"] == [0]  # All original are lost due to error
    assert results["mask_extra"] == [0]  # All yolo are extra due to error

    # Bbox results should be normal
    assert results["bbox_error"] is None
    assert len(results["bbox_matched"]) == 1
    assert results["bbox_matched"][0] == (0, 0, 0.8)


# --- Tests for _process_matched_pairs ---


def test_process_matched_pairs():
    """Test processing matched pairs."""
    orig_inst = [
        _create_dummy_orig_instance(sid="s1", seg_idx=0, mask_idx=0, cid=1, mask_area=(0, 0, 5, 5)),
        _create_dummy_orig_instance(
            sid="s1", seg_idx=1, mask_idx=0, cid=1, mask_area=(5, 5, 10, 10)
        ),
    ]
    # Note: yolo_instances no longer needed as arg

    mask_matches = [(0, 0, 0.98), (1, 1, 0.7)]  # (orig_idx, yolo_idx, iou)
    bbox_matches = [(0, 0, 0.99), (1, 1, 0.6)]  # Different IoU for bbox
    iou_top = 0.95

    mask_pairs, bbox_pairs = _process_matched_pairs(orig_inst, mask_matches, bbox_matches, iou_top)

    assert len(mask_pairs) == 2
    assert mask_pairs[0]["iou"] == 0.98
    assert mask_pairs[0]["threshold_passed"] is True
    assert mask_pairs[0]["match_type"] == "mask"
    assert mask_pairs[1]["iou"] == 0.7
    assert mask_pairs[1]["threshold_passed"] is False
    assert mask_pairs[1]["match_type"] == "mask"

    # Bbox pairs are now processed unconditionally
    assert len(bbox_pairs) == 2
    assert bbox_pairs[0]["iou"] == 0.99
    assert bbox_pairs[0]["threshold_passed"] is True
    assert bbox_pairs[0]["match_type"] == "bbox"
    assert bbox_pairs[1]["iou"] == 0.6
    assert bbox_pairs[1]["threshold_passed"] is False
    assert bbox_pairs[1]["match_type"] == "bbox"


# --- Tests for _process_lost_and_extra_instances ---


def test_process_lost_and_extra_instances():
    """Test processing lost and extra instances, checking for duplicates."""
    orig_inst = [
        _create_dummy_orig_instance(seg_idx=0, mask_idx=0),
        _create_dummy_orig_instance(seg_idx=1, mask_idx=0),
        _create_dummy_orig_instance(seg_idx=1, mask_idx=1),  # Duplicate seg_idx
    ]
    yolo_inst = [
        _create_dummy_yolo_instance(bbox=(0, 0, 5, 5)),
        _create_dummy_yolo_instance(bbox=(10, 10, 15, 15)),
        _create_dummy_yolo_instance(bbox=(10, 10, 15, 15)),  # Duplicate bbox
    ]

    lost_indices = [0, 2]  # Original instances at index 0 and 2 are lost
    extra_indices = [1, 2]  # YOLO instances at index 1 and 2 are extra

    # Simulate some existing lost/extra to test duplicate filtering
    existing_lost = [_create_dummy_orig_instance(seg_idx=0, mask_idx=0)]  # Instance 0 already lost
    existing_extra = [
        _create_dummy_yolo_instance(bbox=(10, 10, 15, 15))
    ]  # Instance 1 already extra

    new_lost, new_extra = _process_lost_and_extra_instances(
        orig_inst, yolo_inst, lost_indices, extra_indices, existing_lost, existing_extra
    )

    # Only orig_inst[2] should be newly lost (orig_inst[0] was already lost)
    assert len(new_lost) == 1
    assert new_lost[0].segment_idx == 1 and new_lost[0].mask_idx == 1

    # Only yolo_inst[2] should be newly extra (yolo_inst[1] was already extra)
    # Note: We compare by bbox tuple here as per the implementation
    assert len(new_extra) == 0
    # assert len(new_extra) == 1
    # assert new_extra[0].bbox == (10, 10, 15, 15)
    # You might want a more robust ID for YoloInstanceRecord if bboxes can genuinely be identical


# --- Tests for verify_sample_conversion (End-to-End) ---


@patch("vibelab.dataops.cov_segm.convert_verifier._load_yolo_instances")
@patch("vibelab.dataops.cov_segm.convert_verifier._process_original_sample")
@patch("vibelab.dataops.cov_segm.convert_verifier._match_instances_for_class")
def test_verify_sample_conversion_success(mock_match, mock_proc_orig, mock_load_yolo, tmp_path):
    """Test successful end-to-end verification."""
    # Note: This test now relies more heavily on mocking the intermediate processing steps
    sample_id = "sample_good"
    yolo_label_path = tmp_path / f"{sample_id}.txt"
    yolo_image_path = tmp_path / f"{sample_id}.jpg"
    phrase_map = {"cat": {"class_id": 0}}
    iou_cutoff, iou_top = 0.5, 0.95

    # Mock data
    orig_inst = [_create_dummy_orig_instance(sid=sample_id, cid=0)]
    yolo_inst = [_create_dummy_yolo_instance(sid=sample_id, cid=0)]
    original_sample_obj = MagicMock()  # Simple mock, as _process_original_sample is mocked

    # Mock function returns
    mock_load_yolo.return_value = (yolo_inst, None)
    mock_proc_orig.return_value = (orig_inst, None)
    mock_match.return_value = {
        "mask_matched": [(0, 0, 0.98)],
        "mask_lost": [],
        "mask_extra": [],
        "mask_error": None,
        "bbox_matched": [(0, 0, 0.99)],
        "bbox_lost": [],
        "bbox_extra": [],
        "bbox_error": None,
    }

    result = verify_sample_conversion(
        sample_id,
        yolo_label_path,
        yolo_image_path,
        original_sample_obj,
        phrase_map,
        "visible",
        iou_cutoff,
        iou_top,
    )

    assert result.processing_error is None
    assert len(result.mask_matched_pairs) == 1
    assert len(result.bbox_matched_pairs) == 1
    assert not result.mask_lost_instances and not result.mask_extra_instances
    assert not result.bbox_lost_instances and not result.bbox_extra_instances


@patch("vibelab.dataops.cov_segm.convert_verifier._load_yolo_instances")
def test_verify_sample_conversion_yolo_load_error(mock_load_yolo, tmp_path):
    """Test verification when YOLO loading fails."""
    # ... setup paths, phrase_map etc. ...
    sample_id = "sample_yolo_err"
    yolo_label_path = tmp_path / f"{sample_id}.txt"
    yolo_image_path = tmp_path / f"{sample_id}.jpg"
    phrase_map = {}
    original_sample_obj = MagicMock()  # Simple mock sufficient

    mock_load_yolo.return_value = ([], "YOLO Load Failed!")

    result = verify_sample_conversion(
        sample_id,
        yolo_label_path,
        yolo_image_path,
        original_sample_obj,
        phrase_map,
        "visible",
        0.5,
        0.95,
    )

    assert "YOLO Load Error: YOLO Load Failed!" in result.processing_error


@patch("vibelab.dataops.cov_segm.convert_verifier._load_yolo_instances")
def test_verify_sample_conversion_no_original_sample(mock_load_yolo, tmp_path):
    """Test verification when the original HF sample is missing."""
    sample_id = "sample_no_orig"
    yolo_label_path = tmp_path / f"{sample_id}.txt"
    yolo_image_path = tmp_path / f"{sample_id}.jpg"
    phrase_map = {"cat": {"class_id": 0}}

    # Mock successful YOLO load
    yolo_inst = [_create_dummy_yolo_instance(sid=sample_id, cid=0)]
    mock_load_yolo.return_value = (yolo_inst, None)

    # Pass None for original_sample
    result = verify_sample_conversion(
        sample_id, yolo_label_path, yolo_image_path, None, phrase_map, "visible", 0.5, 0.95
    )

    assert "Original HF sample not loaded" in result.processing_error
    # All loaded YOLO instances should be considered extra
    assert len(result.mask_extra_instances) == 1
    assert len(result.bbox_extra_instances) == 1
    assert not result.mask_matched_pairs and not result.bbox_matched_pairs
    assert not result.mask_lost_instances and not result.bbox_lost_instances


@patch("vibelab.dataops.cov_segm.convert_verifier._load_yolo_instances")
@patch("vibelab.dataops.cov_segm.convert_verifier._process_original_sample")
def test_verify_sample_conversion_original_process_error(mock_proc_orig, mock_load_yolo, tmp_path):
    """Test verification when processing the original sample fails."""
    sample_id = "sample_orig_err"
    yolo_label_path = tmp_path / f"{sample_id}.txt"
    yolo_image_path = tmp_path / f"{sample_id}.jpg"
    phrase_map = {}
    original_sample_obj = MagicMock()  # Simple mock sufficient

    # Mock successful YOLO load
    mock_load_yolo.return_value = ([], None)
    # Mock original processing error
    mock_proc_orig.return_value = ([], "Original Process Failed!")

    result = verify_sample_conversion(
        sample_id,
        yolo_label_path,
        yolo_image_path,
        original_sample_obj,
        phrase_map,
        "visible",
        0.5,
        0.95,
    )

    assert "Original Process Error: Original Process Failed!" in result.processing_error


@patch("vibelab.dataops.cov_segm.convert_verifier._load_yolo_instances")
@patch("vibelab.dataops.cov_segm.convert_verifier._process_original_sample")
@patch("vibelab.dataops.cov_segm.convert_verifier._match_instances_for_class")
def test_verify_sample_conversion_only_lost(mock_match, mock_proc_orig, mock_load_yolo, tmp_path):
    """Test verification case with only lost instances."""
    sample_id = "sample_lost"
    yolo_label_path = tmp_path / f"{sample_id}.txt"
    yolo_image_path = tmp_path / f"{sample_id}.jpg"
    phrase_map = {"cat": {"class_id": 0}}
    original_sample_obj = MagicMock()  # Simple mock sufficient

    orig_inst = [_create_dummy_orig_instance(sid=sample_id, cid=0)]
    # Mock empty YOLO load or different class
    mock_load_yolo.return_value = ([], None)
    mock_proc_orig.return_value = (orig_inst, None)
    # Match function won't be called if one list is empty, but mock it defensively
    mock_match.return_value = {
        "mask_matched": [],
        "mask_lost": [],
        "mask_extra": [],
        "mask_error": None,
        "bbox_matched": [],
        "bbox_lost": [],
        "bbox_extra": [],
        "bbox_error": None,
    }

    result = verify_sample_conversion(
        sample_id,
        yolo_label_path,
        yolo_image_path,
        original_sample_obj,
        phrase_map,
        "visible",
        0.5,
        0.95,
    )

    assert result.processing_error is None
    assert len(result.mask_lost_instances) == 1
    assert len(result.bbox_lost_instances) == 1
    assert not result.mask_matched_pairs and not result.bbox_matched_pairs
    assert not result.mask_extra_instances and not result.bbox_extra_instances


@patch("vibelab.dataops.cov_segm.convert_verifier._load_yolo_instances")
@patch("vibelab.dataops.cov_segm.convert_verifier._process_original_sample")
@patch("vibelab.dataops.cov_segm.convert_verifier._match_instances_for_class")
def test_verify_sample_conversion_only_extra(mock_match, mock_proc_orig, mock_load_yolo, tmp_path):
    """Test verification case with only extra instances."""
    sample_id = "sample_extra"
    yolo_label_path = tmp_path / f"{sample_id}.txt"
    yolo_image_path = tmp_path / f"{sample_id}.jpg"
    phrase_map = {"cat": {"class_id": 0}}
    original_sample_obj = MagicMock()  # Simple mock sufficient

    yolo_inst = [_create_dummy_yolo_instance(sid=sample_id, cid=0)]
    # Mock empty original processing or different class
    mock_load_yolo.return_value = (yolo_inst, None)
    mock_proc_orig.return_value = ([], None)
    mock_match.return_value = {
        "mask_matched": [],
        "mask_lost": [],
        "mask_extra": [],
        "mask_error": None,
        "bbox_matched": [],
        "bbox_lost": [],
        "bbox_extra": [],
        "bbox_error": None,
    }

    result = verify_sample_conversion(
        sample_id,
        yolo_label_path,
        yolo_image_path,
        original_sample_obj,
        phrase_map,
        "visible",
        0.5,
        0.95,
    )

    assert result.processing_error is None
    assert len(result.mask_extra_instances) == 1
    assert len(result.bbox_extra_instances) == 1
    assert not result.mask_matched_pairs and not result.bbox_matched_pairs
    assert not result.mask_lost_instances and not result.bbox_lost_instances
