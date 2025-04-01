"""Tests for the IoU calculation utility."""

import numpy as np
import pytest

from src.utils.common.iou import calculate_iou

# --- Test Cases ---


@pytest.mark.parametrize(
    "box1, box2, expected_iou",
    [
        # Perfect overlap
        ([0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], 1.0),
        # Partial overlap (50%)
        (
            [0.0, 0.0, 1.0, 1.0],
            [0.5, 0.0, 1.5, 1.0],
            0.5 / 1.5,
        ),  # Area1=1, Area2=1, Inter=0.5, Union=1+1-0.5=1.5
        # Partial overlap (corner)
        (
            [0.0, 0.0, 0.5, 0.5],
            [0.25, 0.25, 0.75, 0.75],
            0.0625 / (0.25 + 0.25 - 0.0625),
        ),  # Area=0.25, Inter=0.25*0.25=0.0625, Union=0.5-0.0625
        # No overlap
        ([0.0, 0.0, 0.5, 0.5], [0.6, 0.6, 1.0, 1.0], 0.0),
        # One box inside another
        (
            [0.1, 0.1, 0.9, 0.9],
            [0.2, 0.2, 0.8, 0.8],
            (0.6 * 0.6) / (0.8 * 0.8),
        ),  # Area1=0.64, Area2=0.36, Inter=0.36, Union=0.64
        # Touching edges
        ([0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0], 0.0),
        # Boxes with zero area
        ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], 0.0),
        ([0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5], 0.0),
        ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], 0.0),
        # Using list inputs instead of numpy arrays
        ([0.0, 0.0, 1.0, 1.0], np.array([0.5, 0.0, 1.5, 1.0]), 0.5 / 1.5),
        (np.array([0.0, 0.0, 1.0, 1.0]), [0.5, 0.0, 1.5, 1.0], 0.5 / 1.5),
    ],
)
def test_calculate_iou(box1, box2, expected_iou):
    """Test calculate_iou with various scenarios."""
    iou = calculate_iou(np.array(box1), np.array(box2))
    assert isinstance(iou, float), "IoU should be a float"
    assert iou >= 0.0 and iou <= 1.0, "IoU must be between 0.0 and 1.0"
    assert np.isclose(iou, expected_iou, atol=1e-6), (
        f"IoU calculation error: expected {expected_iou}, got {iou}"
    )


def test_calculate_iou_identical_boxes():
    """Test IoU with identical boxes."""
    box = np.array([0.1, 0.2, 0.8, 0.9])
    assert np.isclose(calculate_iou(box, box), 1.0)


def test_calculate_iou_no_overlap():
    """Test IoU with non-overlapping boxes."""
    box1 = np.array([0.0, 0.0, 0.4, 0.4])
    box2 = np.array([0.5, 0.5, 1.0, 1.0])
    assert np.isclose(calculate_iou(box1, box2), 0.0)
