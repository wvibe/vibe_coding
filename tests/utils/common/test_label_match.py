"""Tests for label_match.py module."""

import logging

import numpy as np

from vibelab.utils.common.bbox import calculate_iou
from vibelab.utils.common.label_match import match_instances
from vibelab.utils.common.mask import calculate_mask_iou


class MockBBox:
    """Mock bounding box class for testing."""

    def __init__(self, coords):
        """Initialize with [xmin, ymin, xmax, ymax] coordinates."""
        self.coords = np.array(coords)


class MockMask:
    """Mock mask class for testing."""

    def __init__(self, binary_data):
        """Initialize with a binary mask array."""
        self.data = binary_data


def bbox_iou_fn(box1, box2):
    """IoU function for testing with MockBBox."""
    return calculate_iou(box1.coords, box2.coords)


def mask_iou_fn(mask1, mask2):
    """IoU function for testing with MockMask."""
    return calculate_mask_iou(mask1.data, mask2.data)


class TestLabelMatch:
    """Test label_match.py functionality."""

    def test_empty_datasets(self):
        """Test matching with empty datasets."""
        dataset_a = []
        dataset_b = [MockBBox([10, 10, 20, 20])]

        matches, unmatched_a, unmatched_b = match_instances(dataset_a, dataset_b, bbox_iou_fn)

        assert len(matches) == 0
        assert len(unmatched_a) == 0
        assert len(unmatched_b) == 1
        assert unmatched_b[0] == 0

    def test_bbox_matching_exact(self):
        """Test matching of identical bounding boxes."""
        dataset_a = [MockBBox([10, 10, 20, 20]), MockBBox([30, 30, 40, 40])]
        dataset_b = [MockBBox([10, 10, 20, 20]), MockBBox([30, 30, 40, 40])]

        matches, unmatched_a, unmatched_b = match_instances(
            dataset_a, dataset_b, bbox_iou_fn, iou_threshold=0.99
        )

        assert len(matches) == 2
        match_pairs = {(a, b) for a, b in matches}
        assert (0, 0) in match_pairs
        assert (1, 1) in match_pairs
        assert len(unmatched_a) == 0
        assert len(unmatched_b) == 0

    def test_bbox_matching_partial(self):
        """Test matching of partially overlapping bounding boxes."""
        dataset_a = [
            MockBBox([10, 10, 20, 20]),
            MockBBox([30, 30, 40, 40]),
            MockBBox([50, 50, 60, 60]),
        ]
        dataset_b = [
            MockBBox([12, 12, 22, 22]),  # Overlaps with first box
            MockBBox([45, 45, 55, 55]),  # Overlaps with third box
            MockBBox([70, 70, 80, 80]),  # No overlap with any box
        ]

        matches, unmatched_a, unmatched_b = match_instances(
            dataset_a, dataset_b, bbox_iou_fn, iou_threshold=0.1
        )

        assert len(matches) == 2
        match_pairs = {(a, b) for a, b in matches}
        assert (0, 0) in match_pairs
        assert (2, 1) in match_pairs
        assert len(unmatched_a) == 1
        assert 1 in unmatched_a  # Second box in dataset_a has no match
        assert len(unmatched_b) == 1
        assert 2 in unmatched_b  # Third box in dataset_b has no match

    def test_bbox_matching_iou_threshold(self):
        """Test that IoU threshold properly excludes low-IoU matches."""
        dataset_a = [MockBBox([10, 10, 20, 20])]
        dataset_b = [MockBBox([15, 15, 25, 25])]  # Low IoU

        # With high threshold
        matches, unmatched_a, unmatched_b = match_instances(
            dataset_a, dataset_b, bbox_iou_fn, iou_threshold=0.5
        )

        # IoU should be below 0.5
        assert len(matches) == 0
        assert len(unmatched_a) == 1
        assert len(unmatched_b) == 1

        # With lower threshold
        matches, unmatched_a, unmatched_b = match_instances(
            dataset_a, dataset_b, bbox_iou_fn, iou_threshold=0.1
        )

        # Now the boxes should match
        assert len(matches) == 1
        assert matches[0] == (0, 0)

    def test_mask_matching(self, caplog):
        """Test matching with binary masks."""
        caplog.set_level(logging.DEBUG)

        # Create test masks (5x5)
        mask1 = np.zeros((5, 5), dtype=bool)
        mask1[1:4, 1:4] = True  # 3x3 square

        mask2 = np.zeros((5, 5), dtype=bool)
        mask2[2:5, 2:5] = True  # 3x3 square, shifted

        mask3 = np.zeros((5, 5), dtype=bool)
        mask3[0:3, 0:3] = True  # 3x3 square, different position

        dataset_a = [MockMask(mask1), MockMask(mask3)]
        dataset_b = [MockMask(mask2), MockMask(mask1.copy())]

        matches, unmatched_a, unmatched_b = match_instances(
            dataset_a, dataset_b, mask_iou_fn, iou_threshold=0.2
        )

        # Based on correct IoU calculations and maximization:
        # IoU Matrix = [[0.2857, 1.0], [-1.0, 0.2857]]
        # Max assignment picks (0,0) and (1,1), sum = 0.5714
        # (Alternative (0,1) and (1,0) sum = 1.0 + (-1) = 0.0)
        # Filter check:
        # Pair (0,0): IoU 0.2857 >= 0.2 -> Keep
        # Pair (1,1): IoU 0.2857 >= 0.2 -> Keep
        # Expected matches: (0,0) and (1,1)
        assert len(matches) == 2
        match_pairs = {(a, b) for a, b in matches}
        assert (0, 0) in match_pairs  # a[0] (mask1) matches b[0] (mask2)
        assert (1, 1) in match_pairs  # a[1] (mask3) matches b[1] (mask1 copy)
        assert len(unmatched_a) == 0
        assert len(unmatched_b) == 0

    def test_greedy_fallback(self, monkeypatch):
        """Test the greedy fallback when linear_sum_assignment fails."""
        dataset_a = [MockBBox([10, 10, 20, 20]), MockBBox([30, 30, 40, 40])]
        dataset_b = [MockBBox([12, 12, 22, 22]), MockBBox([32, 32, 42, 42])]

        # Mock linear_sum_assignment to raise ValueError
        def mock_linear_sum_assignment(*args, **kwargs):
            raise ValueError("Mocked error")

        # Apply monkey patch
        monkeypatch.setattr(
            "vibelab.utils.common.label_match.linear_sum_assignment", mock_linear_sum_assignment
        )

        # Should use greedy fallback and still work
        matches, unmatched_a, unmatched_b = match_instances(
            dataset_a, dataset_b, bbox_iou_fn, iou_threshold=0.1
        )

        # Should still match both boxes
        assert len(matches) == 2
        assert len(unmatched_a) == 0
        assert len(unmatched_b) == 0
