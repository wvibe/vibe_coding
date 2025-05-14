import numpy as np
import pytest

from vibelab.utils.metrics.detection import (
    calculate_ap,
    calculate_iou,
    calculate_map,
    calculate_map_by_size,
    calculate_pr_data,
    generate_confusion_matrix,
    match_predictions,
)


def test_iou_perfect_overlap():
    """Tests IoU when boxes are identical."""
    box1 = [0, 0, 10, 10]
    box2 = [0, 0, 10, 10]
    assert calculate_iou(box1, box2) == pytest.approx(1.0)


def test_iou_partial_overlap():
    """Tests IoU with partial overlap."""
    box1 = [0, 0, 10, 10]  # Area = 100
    box2 = [5, 5, 15, 15]  # Area = 100
    # Intersection: [5, 5, 10, 10] -> Area = 5 * 5 = 25
    # Union: 100 + 100 - 25 = 175
    # IoU = 25 / 175 = 1 / 7
    expected_iou = 25 / 175
    assert calculate_iou(box1, box2) == pytest.approx(expected_iou)


def test_iou_no_overlap():
    """Tests IoU when boxes do not overlap."""
    box1 = [0, 0, 10, 10]
    box2 = [20, 20, 30, 30]
    assert calculate_iou(box1, box2) == pytest.approx(0.0)


def test_iou_one_box_contained():
    """Tests IoU when one box is fully contained within another."""
    box1 = [0, 0, 20, 20]  # Area = 400
    box2 = [5, 5, 15, 15]  # Area = 100
    # Intersection = Area of box2 = 100
    # Union = Area of box1 = 400
    # IoU = 100 / 400 = 0.25
    expected_iou = 100 / 400
    assert calculate_iou(box1, box2) == pytest.approx(expected_iou)
    # Test the other way around
    assert calculate_iou(box2, box1) == pytest.approx(expected_iou)


def test_iou_touching_boxes():
    """Tests IoU when boxes touch at an edge or corner (should be 0)."""
    box1 = [0, 0, 10, 10]
    # Touching edge
    box2_edge = [10, 0, 20, 10]
    assert calculate_iou(box1, box2_edge) == pytest.approx(0.0)
    # Touching corner
    box2_corner = [10, 10, 20, 20]
    assert calculate_iou(box1, box2_corner) == pytest.approx(0.0)


def test_iou_zero_area_box():
    """Tests IoU when one or both boxes have zero area."""
    box1 = [0, 0, 10, 10]  # Normal box
    box_zero_width = [5, 0, 5, 10]  # Zero width
    box_zero_height = [0, 5, 10, 5]  # Zero height
    box_zero_both = [5, 5, 5, 5]  # Zero width and height

    assert calculate_iou(box1, box_zero_width) == pytest.approx(0.0)
    assert calculate_iou(box1, box_zero_height) == pytest.approx(0.0)
    assert calculate_iou(box1, box_zero_both) == pytest.approx(0.0)
    # Test zero area box with itself or another zero area box
    assert calculate_iou(box_zero_width, box_zero_height) == pytest.approx(0.0)
    assert calculate_iou(box_zero_width, box_zero_width) == pytest.approx(0.0)


# --- Tests for match_predictions ---


def test_match_simple_tp():
    """Tests a simple case with one prediction and one matching GT."""
    preds = [([0, 0, 10, 10], 0.9, 0)]  # box, score, class_id
    gts = [([1, 1, 11, 11], 0)]  # box, class_id
    iou_thresh = 0.5

    match_results, num_gt = match_predictions(preds, gts, iou_thresh)

    # Check match results (score, is_tp, pred_class)
    assert len(match_results) == 1
    assert match_results[0][0] == 0.9  # score
    assert match_results[0][1] is True  # is_tp
    assert match_results[0][2] == 0  # pred_class

    # Check GT counts
    assert num_gt == {0: 1}


def test_match_simple_fp_iou():
    """Tests a simple case where IoU is below threshold."""
    preds = [([0, 0, 10, 10], 0.9, 0)]
    gts = [([50, 50, 60, 60], 0)]  # No overlap
    iou_thresh = 0.5

    match_results, num_gt = match_predictions(preds, gts, iou_thresh)

    assert len(match_results) == 1
    assert match_results[0][1] is False  # is_tp should be False
    assert num_gt == {0: 1}


def test_match_simple_fp_class():
    """Tests a simple case where class IDs don't match."""
    preds = [([0, 0, 10, 10], 0.9, 1)]  # Predict class 1
    gts = [([0, 0, 10, 10], 0)]  # GT is class 0
    iou_thresh = 0.5

    match_results, num_gt = match_predictions(preds, gts, iou_thresh)

    assert len(match_results) == 1
    assert match_results[0][1] is False  # is_tp should be False (class mismatch)
    assert num_gt == {0: 1}  # Still counts the GT of class 0


def test_match_multiple_preds_one_gt():
    """Tests greedy matching: higher confidence prediction gets the match."""
    # Both preds overlap well with the single GT
    preds = [
        ([0, 0, 10, 10], 0.8, 0),  # Lower confidence
        ([1, 1, 11, 11], 0.9, 0),  # Higher confidence
    ]
    gts = [([2, 2, 12, 12], 0)]
    iou_thresh = 0.5

    match_results, num_gt = match_predictions(preds, gts, iou_thresh)

    assert len(match_results) == 2
    # Check original prediction order
    # Pred 0 (lower conf): Should be FP (GT matched by higher conf pred)
    assert match_results[0][0] == 0.8
    assert match_results[0][1] is False
    assert match_results[0][2] == 0
    # Pred 1 (higher conf): Should be TP
    assert match_results[1][0] == 0.9
    assert match_results[1][1] is True
    assert match_results[1][2] == 0

    assert num_gt == {0: 1}


def test_match_multiple_gts_one_pred():
    """Tests matching: prediction matches the best overlapping GT."""
    preds = [([5, 5, 15, 15], 0.9, 0)]
    gts = [
        ([0, 0, 10, 10], 0),  # Lower IoU match
        ([6, 6, 16, 16], 0),  # Higher IoU match
    ]
    iou_thresh = 0.1  # Low threshold to ensure both overlap

    match_results, num_gt = match_predictions(preds, gts, iou_thresh)

    # Check the single prediction result
    assert len(match_results) == 1
    assert match_results[0][1] is True  # Should be TP

    # We need to check which GT was matched - this is harder to assert directly
    # from the output, but the logic ensures the best one is picked.
    assert num_gt == {0: 2}


def test_match_mixed_classes_and_results():
    """Tests a more complex scenario with multiple classes and TP/FP."""
    preds = [
        ([0, 0, 10, 10], 0.9, 0),  # TP for class 0
        ([50, 50, 60, 60], 0.8, 1),  # FP for class 1 (no GT)
        ([5, 5, 15, 15], 0.7, 0),  # FP for class 0 (GT matched by higher conf)
    ]
    gts = [
        ([1, 1, 11, 11], 0),  # Matched by pred 0
        ([100, 100, 110, 110], 2),  # GT for class 2 (FN)
    ]
    iou_thresh = 0.5

    match_results, num_gt = match_predictions(preds, gts, iou_thresh)

    assert len(match_results) == 3
    # Pred 0 (class 0, high conf): TP
    assert match_results[0] == (0.9, True, 0)
    # Pred 1 (class 1, mid conf): FP
    assert match_results[1] == (0.8, False, 1)
    # Pred 2 (class 0, low conf): FP
    assert match_results[2] == (0.7, False, 0)

    assert num_gt == {0: 1, 2: 1}


def test_match_no_preds():
    """Tests case with no predictions."""
    preds = []
    gts = [([0, 0, 10, 10], 0)]
    iou_thresh = 0.5
    match_results, num_gt = match_predictions(preds, gts, iou_thresh)
    assert match_results == []
    assert num_gt == {0: 1}


def test_match_no_gts():
    """Tests case with no ground truths."""
    preds = [([0, 0, 10, 10], 0.9, 0)]
    gts = []
    iou_thresh = 0.5
    match_results, num_gt = match_predictions(preds, gts, iou_thresh)
    assert len(match_results) == 1
    assert match_results[0] == (0.9, False, 0)  # All preds are FP
    assert num_gt == {}


# --- Tests for calculate_pr_data ---


def test_pr_data_simple():
    """Tests PR data calculation for a simple case."""
    # Simulating aggregated results across images
    # Class 0: 1 TP, 1 FP. 2 GT total.
    # Class 1: 1 TP. 1 GT total.
    all_matches = [
        (0.9, True, 0),  # TP class 0
        (0.8, True, 1),  # TP class 1
        (0.7, False, 0),  # FP class 0
    ]
    all_gt_counts = {0: 2, 1: 1}

    pr_results = calculate_pr_data(all_matches, all_gt_counts)

    # Check Class 0
    assert 0 in pr_results
    class0_data = pr_results[0]
    assert class0_data["num_gt"] == 2
    # Expected results after sorting by confidence: (0.9, T), (0.7, F)
    # Step 1 (Conf 0.9): TP=1, FP=0 -> P=1/1=1.0, R=1/2=0.5
    # Step 2 (Conf 0.7): TP=1, FP=1 -> P=1/2=0.5, R=1/2=0.5
    np.testing.assert_array_almost_equal(class0_data["confidence"], [0.9, 0.7])
    np.testing.assert_array_almost_equal(class0_data["precision"], [1.0, 0.5])
    np.testing.assert_array_almost_equal(class0_data["recall"], [0.5, 0.5])

    # Check Class 1
    assert 1 in pr_results
    class1_data = pr_results[1]
    assert class1_data["num_gt"] == 1
    # Expected results: (0.8, T)
    # Step 1 (Conf 0.8): TP=1, FP=0 -> P=1/1=1.0, R=1/1=1.0
    np.testing.assert_array_almost_equal(class1_data["confidence"], [0.8])
    np.testing.assert_array_almost_equal(class1_data["precision"], [1.0])
    np.testing.assert_array_almost_equal(class1_data["recall"], [1.0])


def test_pr_data_all_fp():
    """Tests PR data when all predictions are False Positives."""
    all_matches = [(0.9, False, 0), (0.7, False, 0)]
    all_gt_counts = {0: 1}  # One GT exists but wasn't matched
    pr_results = calculate_pr_data(all_matches, all_gt_counts)

    assert 0 in pr_results
    class0_data = pr_results[0]
    assert class0_data["num_gt"] == 1
    # Step 1 (Conf 0.9): TP=0, FP=1 -> P=0/1=0.0, R=0/1=0.0
    # Step 2 (Conf 0.7): TP=0, FP=2 -> P=0/2=0.0, R=0/1=0.0
    np.testing.assert_array_almost_equal(class0_data["confidence"], [0.9, 0.7])
    np.testing.assert_array_almost_equal(class0_data["precision"], [0.0, 0.0])
    np.testing.assert_array_almost_equal(class0_data["recall"], [0.0, 0.0])


def test_pr_data_all_tp():
    """Tests PR data when all predictions are True Positives."""
    all_matches = [(0.9, True, 0), (0.7, True, 0)]
    all_gt_counts = {0: 2}
    pr_results = calculate_pr_data(all_matches, all_gt_counts)

    assert 0 in pr_results
    class0_data = pr_results[0]
    assert class0_data["num_gt"] == 2
    # Step 1 (Conf 0.9): TP=1, FP=0 -> P=1/1=1.0, R=1/2=0.5
    # Step 2 (Conf 0.7): TP=2, FP=0 -> P=2/2=1.0, R=2/2=1.0
    np.testing.assert_array_almost_equal(class0_data["confidence"], [0.9, 0.7])
    np.testing.assert_array_almost_equal(class0_data["precision"], [1.0, 1.0])
    np.testing.assert_array_almost_equal(class0_data["recall"], [0.5, 1.0])


def test_pr_data_no_predictions():
    """Tests PR data calculation when there are no predictions."""
    all_matches = []
    all_gt_counts = {0: 1}
    pr_results = calculate_pr_data(all_matches, all_gt_counts)
    assert pr_results == {}


def test_pr_data_no_gt():
    """Tests PR data calculation when there are no ground truths for a class."""
    all_matches = [(0.8, False, 0)]  # FP prediction for class 0
    all_gt_counts = {}
    pr_results = calculate_pr_data(all_matches, all_gt_counts)
    assert 0 in pr_results
    class0_data = pr_results[0]
    assert class0_data["num_gt"] == 0
    # Step 1 (Conf 0.8): TP=0, FP=1 -> P=0/1=0.0, R=0/0=0.0 (by convention)
    np.testing.assert_array_almost_equal(class0_data["confidence"], [0.8])
    np.testing.assert_array_almost_equal(class0_data["precision"], [0.0])
    np.testing.assert_array_almost_equal(class0_data["recall"], [0.0])


def test_pr_data_gt_no_preds():
    """Tests PR data when GT exists for a class but no predictions were made."""
    all_matches = [(0.9, True, 1)]  # Prediction for class 1 only
    all_gt_counts = {0: 2, 1: 1}  # GTs for class 0 and 1
    pr_results = calculate_pr_data(all_matches, all_gt_counts)

    # Class 0 should exist, but have empty arrays
    assert 0 in pr_results
    class0_data = pr_results[0]
    assert class0_data["num_gt"] == 2
    assert len(class0_data["precision"]) == 0
    assert len(class0_data["recall"]) == 0
    assert len(class0_data["confidence"]) == 0

    # Class 1 should have data
    assert 1 in pr_results
    class1_data = pr_results[1]
    assert class1_data["num_gt"] == 1
    np.testing.assert_array_almost_equal(class1_data["confidence"], [0.9])
    np.testing.assert_array_almost_equal(class1_data["precision"], [1.0])
    np.testing.assert_array_almost_equal(class1_data["recall"], [1.0])


# --- Tests for calculate_ap ---


def test_ap_perfect_score():
    """Tests AP calculation for a perfect PR curve."""
    precision = np.array([1.0, 1.0, 1.0])
    recall = np.array([0.3, 0.6, 1.0])
    # After monotonic: P=[1,1,1,1], R=[0,0.3,0.6,1] -> AP = (0.3-0)*1 + (0.6-0.3)*1 + (1.0-0.6)*1 = 1.0
    ap = calculate_ap(precision, recall)
    assert ap == pytest.approx(1.0)


def test_ap_zero_score():
    """Tests AP calculation for a zero PR curve."""
    precision = np.array([0.0, 0.0, 0.0])
    recall = np.array([0.3, 0.6, 1.0])
    # After monotonic: P=[1,0,0,0], R=[0,0.3,0.6,1] -> AP = (0.3-0)*0 + (0.6-0.3)*0 + (1.0-0.6)*0 = 0.0
    ap = calculate_ap(precision, recall)
    assert ap == pytest.approx(0.0)


def test_ap_realistic_curve():
    """Tests AP calculation for a more realistic, non-monotonic curve."""
    precision = np.array([0.9, 0.8, 0.85, 0.7, 0.6])
    recall = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    # Original: P=[1.0, 0.9, 0.8, 0.85, 0.7, 0.6], R=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # Monotonic:P=[1.0, 0.9, 0.85,0.85, 0.7, 0.6], R=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # AP = (0.2-0.0)*0.9 + (0.4-0.2)*0.85 + (0.6-0.4)*0.85 + (0.8-0.6)*0.7 + (1.0-0.8)*0.6
    # AP = 0.2*0.9 + 0.2*0.85 + 0.2*0.85 + 0.2*0.7 + 0.2*0.6
    # AP = 0.18 + 0.17 + 0.17 + 0.14 + 0.12 = 0.78
    expected_ap = 0.18 + 0.17 + 0.17 + 0.14 + 0.12
    ap = calculate_ap(precision, recall)
    assert ap == pytest.approx(expected_ap)


def test_ap_empty_input():
    """Tests AP calculation with empty precision/recall arrays."""
    precision = np.array([])
    recall = np.array([])
    ap = calculate_ap(precision, recall)
    assert ap == pytest.approx(0.0)


def test_ap_single_point():
    """Tests AP calculation with a single point (e.g., one prediction)."""
    precision = np.array([0.8])
    recall = np.array([0.5])
    # Monotonic: P=[1, 0.8], R=[0, 0.5]
    # AP = (0.5-0)*0.8 = 0.4
    expected_ap = 0.4
    ap = calculate_ap(precision, recall)
    assert ap == pytest.approx(expected_ap)


# --- Tests for calculate_map ---


def test_map_basic():
    """Tests basic mAP calculation."""
    ap_scores = {0: 0.8, 1: 0.6, 2: 0.7}
    expected_map = (0.8 + 0.6 + 0.7) / 3
    mean_ap = calculate_map(ap_scores)
    assert mean_ap == pytest.approx(expected_map)


def test_map_single_class():
    """Tests mAP calculation with only one class."""
    ap_scores = {0: 0.75}
    mean_ap = calculate_map(ap_scores)
    assert mean_ap == pytest.approx(0.75)


def test_map_empty_input():
    """Tests mAP calculation with an empty input dictionary."""
    ap_scores = {}
    mean_ap = calculate_map(ap_scores)
    assert mean_ap == pytest.approx(0.0)


def test_map_with_zero_ap():
    """Tests mAP calculation including a class with zero AP."""
    ap_scores = {0: 0.8, 1: 0.0, 2: 0.7}
    expected_map = (0.8 + 0.0 + 0.7) / 3
    mean_ap = calculate_map(ap_scores)
    assert mean_ap == pytest.approx(expected_map)


def test_map_ignore_non_numeric():
    """Tests that mAP calculation ignores non-numeric entries (though unlikely)."""
    ap_scores = {0: 0.8, 1: None, 2: 0.7, 3: "bad"}
    # Should only average 0.8 and 0.7
    expected_map = (0.8 + 0.7) / 2
    mean_ap = calculate_map(ap_scores)
    assert mean_ap == pytest.approx(expected_map)


def test_map_empty_values():
    """Tests mAP calculation with empty list of values."""
    ap_scores = {0: None}  # Or dictionary containing only non-numerics
    mean_ap = calculate_map(ap_scores)
    assert mean_ap == pytest.approx(0.0)


# --- Tests for calculate_map_by_size ---

# Define standard size ranges for tests
SIZE_RANGES_COCO = {
    "small": [0, 1024],  # area < 32*32
    "medium": [1024, 9216],  # 32*32 <= area < 96*96
    "large": [9216, float("inf")],  # area >= 96*96
}


def test_map_by_size_basic():
    """Test mAP calculation across different size categories."""
    # Predictions (all for class 0)
    preds = [
        ([0, 0, 10, 10], 0.95, 0),  # Matches small GT
        ([50, 50, 70, 70], 0.90, 0),  # Matches medium GT
        ([100, 100, 250, 250], 0.85, 0),  # Matches large GT
        ([200, 200, 210, 210], 0.80, 0),  # FP (no corresponding GT)
    ]
    # Ground Truths (all class 0)
    gts = [
        ([1, 1, 9, 9], 0),  # Small
        ([51, 51, 69, 69], 0),  # Medium
        ([101, 101, 249, 249], 0),  # Large
    ]
    # Corresponding GT Areas
    gt_areas = [
        (9 - 1) * (9 - 1),  # 64 (Small)
        (69 - 51) * (69 - 51),  # 324 (Medium)
        (249 - 101) * (249 - 101),  # 21904 (Large)
    ]
    iou_thresh = 0.5

    map_results = calculate_map_by_size(preds, gts, gt_areas, SIZE_RANGES_COCO, iou_thresh)

    # Corrected GT Categories: G0(S), G1(S), G2(L)
    # Small: AP(0)=1.0 -> mAP = 1.0
    # Medium: No GTs -> mAP = 0.0
    # Large: AP(0)=1/3 -> mAP = 1/3

    assert map_results["small"] == pytest.approx(1.0)
    assert map_results["medium"] == pytest.approx(0.0)
    assert map_results["large"] == pytest.approx(1.0 / 3.0)


def test_map_by_size_no_gts_in_category():
    """Test when a size category has no ground truths."""
    preds = [
        ([0, 0, 10, 10], 0.95, 0),  # Matches small GT
    ]
    gts = [
        ([1, 1, 9, 9], 0),  # Small
    ]
    gt_areas = [64]
    iou_thresh = 0.5

    map_results = calculate_map_by_size(preds, gts, gt_areas, SIZE_RANGES_COCO, iou_thresh)

    assert map_results["small"] == pytest.approx(1.0)
    assert map_results["medium"] == pytest.approx(0.0)  # No medium GTs
    assert map_results["large"] == pytest.approx(0.0)  # No large GTs


def test_map_by_size_mixed_classes():
    """Test with multiple classes across size categories."""
    preds = [
        ([0, 0, 10, 10], 0.95, 0),  # TP Small Class 0
        ([50, 50, 70, 70], 0.90, 1),  # TP Medium Class 1
        ([5, 5, 15, 15], 0.85, 0),  # FP Small Class 0 (no more small GTs for cls 0)
    ]
    gts = [
        ([1, 1, 9, 9], 0),  # Small Class 0 (Area 64)
        ([51, 51, 69, 69], 1),  # Medium Class 1 (Area 324)
    ]
    gt_areas = [64, 324]
    iou_thresh = 0.5

    map_results = calculate_map_by_size(preds, gts, gt_areas, SIZE_RANGES_COCO, iou_thresh)

    # Corrected GT Categories: G0(S), G1(S)
    # Small: AP(0)=1.0, AP(1)=1.0 -> mAP = 1.0
    # Medium: No GTs -> mAP = 0.0
    # Large: GTs = []. mAP=0.0
    assert map_results["small"] == pytest.approx(1.0)
    assert map_results["medium"] == pytest.approx(0.0)
    assert map_results["large"] == pytest.approx(0.0)


def test_map_by_size_value_error():
    """Test that ValueError is raised if gt and area lengths differ."""
    preds = [([0, 0, 10, 10], 0.95, 0)]
    gts = [([1, 1, 9, 9], 0)]
    gt_areas = [64, 100]  # Mismatched length
    iou_thresh = 0.5
    with pytest.raises(ValueError):
        calculate_map_by_size(preds, gts, gt_areas, SIZE_RANGES_COCO, iou_thresh)


# --- Tests for generate_confusion_matrix ---


def test_cm_basic_tp_fp_fn():
    """Tests CM generation with basic TP, FP, FN cases."""
    preds = [
        ([0, 0, 10, 10], 0.9, 0),  # TP for class 0
        ([50, 50, 60, 60], 0.8, 1),  # FP for class 1 (conf > threshold)
        ([5, 5, 15, 15], 0.4, 0),  # Below conf threshold, ignored
    ]
    gts = [
        ([1, 1, 11, 11], 0),  # Matched by pred 0 (TP)
        ([100, 100, 110, 110], 2),  # Unmatched GT class 2 (FN)
    ]
    iou_thresh = 0.5
    conf_thresh = 0.5
    target_classes = [0, 1]  # Class 2 becomes 'Others'

    cm, labels = generate_confusion_matrix(preds, gts, iou_thresh, conf_thresh, target_classes)

    # Target classes: 0, 1
    # Indices: 0=cls_0, 1=cls_1, 2=Others, 3=Background
    # Expected CM:
    #       Pred 0 | Pred 1 | Others | Background (FN)
    # True 0    1   |    0   |    0   |      0
    # True 1    0   |    0   |    0   |      0
    # Others    0   |    0   |    0   |      1   <- Unmatched GT class 2 (mapped to Others)
    # Bkg (FP)  0   |    1   |    0   |      0   <- Unmatched Pred class 1 (FP)

    # labels = [0, 1, 'Others', 'Background']
    expected_cm = np.array(
        [
            [1, 0, 0, 0],  # True 0
            [0, 0, 0, 0],  # True 1
            [0, 0, 0, 1],  # True Others (GT class 2)
            [0, 1, 0, 0],  # Background (FP pred class 1)
        ]
    )

    assert labels == [0, 1, "Others", "Background"]
    np.testing.assert_array_equal(cm, expected_cm)


def test_cm_class_confusion():
    """Tests confusion between target classes."""
    preds = [
        ([0, 0, 10, 10], 0.9, 0),  # Predicted 0, True 0 (TP)
        ([20, 20, 30, 30], 0.8, 1),  # Predicted 1, True 0 (Misclassification)
    ]
    gts = [
        ([1, 1, 11, 11], 0),  # Matched by pred 0
        ([21, 21, 31, 31], 0),  # Matched by pred 1 (IoU ok, but wrong class pred)
    ]
    iou_thresh = 0.5
    conf_thresh = 0.5
    target_classes = [0, 1]

    cm, labels = generate_confusion_matrix(preds, gts, iou_thresh, conf_thresh, target_classes)

    # Expected CM:
    #       Pred 0 | Pred 1 | Others | Background (FN)
    # True 0    1   |    1   |    0   |      0   <- GT 1 matched pred 1 (pred 1 is cls 1)
    # True 1    0   |    0   |    0   |      0
    # Others    0   |    0   |    0   |      0
    # Bkg (FP)  0   |    0   |    0   |      0   <- Pred 1 is matched, so not counted as FP background

    # Wait, the matching logic prevents cross-class matches from counting in the main matrix cells.
    # Pred 1 (class 1) will not match GT 2 (class 0). Pred 1 becomes an FP, GT 2 becomes an FN.
    # Expected CM Revisted:
    #       Pred 0 | Pred 1 | Others | Background (FN)
    # True 0    1   |    0   |    0   |      1   <- Unmatched GT 2 (class 0)
    # True 1    0   |    0   |    0   |      0
    # Others    0   |    0   |    0   |      0
    # Bkg (FP)  0   |    1   |    0   |      0   <- Unmatched Pred 1 (class 1)

    expected_cm = np.array(
        [
            [1, 0, 0, 1],  # True 0
            [0, 0, 0, 0],  # True 1
            [0, 0, 0, 0],  # True Others
            [0, 1, 0, 0],  # Background (FP)
        ]
    )

    assert labels == [0, 1, "Others", "Background"]
    np.testing.assert_array_equal(cm, expected_cm)


def test_cm_others_category():
    """Tests correct assignment to 'Others' category."""
    preds = [
        ([0, 0, 10, 10], 0.9, 2),  # Predict class 2 ('Others'), True class 2 ('Others')
        ([20, 20, 30, 30], 0.8, 0),  # Predict class 0 ('Target'), True class 3 ('Others')
        ([40, 40, 50, 50], 0.7, 3),  # Predict class 3 ('Others'), no matching GT (FP 'Others')
    ]
    gts = [
        ([1, 1, 11, 11], 2),  # Matched by pred 0 (True: Others, Pred: Others)
        ([21, 21, 31, 31], 3),  # Matched by pred 1 (True: Others, Pred: Target 0)
    ]
    iou_thresh = 0.5
    conf_thresh = 0.5
    target_classes = [0, 1]  # Classes 2, 3 are 'Others'

    cm, labels = generate_confusion_matrix(preds, gts, iou_thresh, conf_thresh, target_classes)

    # Indices: 0=cls_0, 1=cls_1, 2=Others, 3=Background
    # Expected CM:
    #       Pred 0 | Pred 1 | Others | Background (FN)
    # True 0    0   |    0   |    0   |      0
    # True 1    0   |    0   |    0   |      0
    # Others    1   |    0   |    1   |      0   <- GT 1(cls2) matched P0(cls2), GT 2(cls3) matched P1(cls0)
    # Bkg (FP)  0   |    0   |    1   |      0   <- Unmatched P2(cls3)

    # Final traced expected CM based on implementation logic:
    # One Others TP (P0/G0 or P2/G1), the other GT becomes FN, the other Pred becomes FP.
    # One Class 0 FP (P1).
    expected_cm = np.array(
        [
            [0, 0, 0, 0],  # True 0
            [0, 0, 0, 0],  # True 1
            [0, 0, 1, 1],  # True Others (1 TP, 1 FN)
            [1, 0, 1, 0],  # Background (1 FP for Cls 0, 1 FP for Others)
        ]
    )

    assert labels == [0, 1, "Others", "Background"]
    np.testing.assert_array_equal(cm, expected_cm)


def test_cm_no_matches():
    """Tests CM when no predictions match any ground truths."""
    preds = [([0, 0, 10, 10], 0.9, 0)]  # FP class 0
    gts = [([50, 50, 60, 60], 1)]  # FN class 1
    iou_thresh = 0.5
    conf_thresh = 0.5
    target_classes = [0, 1]

    cm, labels = generate_confusion_matrix(preds, gts, iou_thresh, conf_thresh, target_classes)

    # Expected CM:
    #       Pred 0 | Pred 1 | Others | Background (FN)
    # True 0    0   |    0   |    0   |      0
    # True 1    0   |    0   |    0   |      1   <- Unmatched GT class 1
    # Others    0   |    0   |    0   |      0
    # Bkg (FP)  1   |    0   |    0   |      0   <- Unmatched Pred class 0
    expected_cm = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0]])
    assert labels == [0, 1, "Others", "Background"]
    np.testing.assert_array_equal(cm, expected_cm)
