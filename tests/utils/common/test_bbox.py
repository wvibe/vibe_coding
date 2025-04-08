"""Tests for bounding box utilities: format conversion and IoU calculation."""

import numpy as np
import pytest
import torch

from src.utils.common.bbox import (
    calculate_iou,
    denormalize_boxes,
    normalize_boxes,
    xywh_to_xyxy,
    xyxy_to_xywh,
)

# --- Test Data for Format Conversion ---

# Format: (xywh_np, xyxy_np, xywh_torch, xyxy_torch)
TEST_BOXES = [
    # Simple center box
    (
        np.array([0.5, 0.5, 0.2, 0.2]),
        np.array([0.4, 0.4, 0.6, 0.6]),
        torch.tensor([0.5, 0.5, 0.2, 0.2]),
        torch.tensor([0.4, 0.4, 0.6, 0.6]),
    ),
    # Box touching edge
    (
        np.array([0.1, 0.5, 0.2, 1.0]),
        np.array([0.0, 0.0, 0.2, 1.0]),
        torch.tensor([0.1, 0.5, 0.2, 1.0]),
        torch.tensor([0.0, 0.0, 0.2, 1.0]),
    ),
    # Full image box
    (
        np.array([0.5, 0.5, 1.0, 1.0]),
        np.array([0.0, 0.0, 1.0, 1.0]),
        torch.tensor([0.5, 0.5, 1.0, 1.0]),
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
    ),
    # Small box
    (
        np.array([0.75, 0.25, 0.1, 0.05]),
        np.array([0.7, 0.225, 0.8, 0.275]),
        torch.tensor([0.75, 0.25, 0.1, 0.05]),
        torch.tensor([0.7, 0.225, 0.8, 0.275]),
    ),
]

# --- Format Conversion Tests ---


@pytest.mark.parametrize("xywh_np, xyxy_np, xywh_torch, xyxy_torch", TEST_BOXES)
def test_xywh_to_xyxy_numpy(xywh_np, xyxy_np, xywh_torch, xyxy_torch):
    """Test xywh_to_xyxy conversion with numpy arrays."""
    converted_xyxy = xywh_to_xyxy(xywh_np)
    assert isinstance(converted_xyxy, np.ndarray)
    assert np.allclose(converted_xyxy, xyxy_np, atol=1e-6)


@pytest.mark.parametrize("xywh_np, xyxy_np, xywh_torch, xyxy_torch", TEST_BOXES)
def test_xywh_to_xyxy_torch(xywh_np, xyxy_np, xywh_torch, xyxy_torch):
    """Test xywh_to_xyxy conversion with torch tensors."""
    converted_xyxy = xywh_to_xyxy(xywh_torch)
    assert isinstance(converted_xyxy, torch.Tensor)
    assert torch.allclose(converted_xyxy, xyxy_torch, atol=1e-6)


@pytest.mark.parametrize("xywh_np, xyxy_np, xywh_torch, xyxy_torch", TEST_BOXES)
def test_xyxy_to_xywh_numpy(xywh_np, xyxy_np, xywh_torch, xyxy_torch):
    """Test xyxy_to_xywh conversion with numpy arrays."""
    converted_xywh = xyxy_to_xywh(xyxy_np)
    assert isinstance(converted_xywh, np.ndarray)
    assert np.allclose(converted_xywh, xywh_np, atol=1e-6)


@pytest.mark.parametrize("xywh_np, xyxy_np, xywh_torch, xyxy_torch", TEST_BOXES)
def test_xyxy_to_xywh_torch(xywh_np, xyxy_np, xywh_torch, xyxy_torch):
    """Test xyxy_to_xywh conversion with torch tensors."""
    converted_xywh = xyxy_to_xywh(xyxy_torch)
    assert isinstance(converted_xywh, torch.Tensor)
    assert torch.allclose(converted_xywh, xywh_torch, atol=1e-6)


# Test with batch dimension
def test_batch_conversion_numpy():
    """Test conversions with a batch dimension using numpy."""
    xywh_batch = np.array([[0.5, 0.5, 0.2, 0.2], [0.1, 0.5, 0.2, 1.0]])
    xyxy_expected = np.array([[0.4, 0.4, 0.6, 0.6], [0.0, 0.0, 0.2, 1.0]])

    converted_xyxy = xywh_to_xyxy(xywh_batch)
    assert isinstance(converted_xyxy, np.ndarray)
    assert converted_xyxy.shape == xyxy_expected.shape
    assert np.allclose(converted_xyxy, xyxy_expected, atol=1e-6)

    converted_xywh = xyxy_to_xywh(xyxy_expected)
    assert isinstance(converted_xywh, np.ndarray)
    assert converted_xywh.shape == xywh_batch.shape
    assert np.allclose(converted_xywh, xywh_batch, atol=1e-6)


def test_batch_conversion_torch():
    """Test conversions with a batch dimension using torch."""
    xywh_batch = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.1, 0.5, 0.2, 1.0]])
    xyxy_expected = torch.tensor([[0.4, 0.4, 0.6, 0.6], [0.0, 0.0, 0.2, 1.0]])

    converted_xyxy = xywh_to_xyxy(xywh_batch)
    assert isinstance(converted_xyxy, torch.Tensor)
    assert converted_xyxy.shape == xyxy_expected.shape
    assert torch.allclose(converted_xyxy, xyxy_expected, atol=1e-6)

    converted_xywh = xyxy_to_xywh(xyxy_expected)
    assert isinstance(converted_xywh, torch.Tensor)
    assert converted_xywh.shape == xywh_batch.shape
    assert torch.allclose(converted_xywh, xywh_batch, atol=1e-6)


# --- IoU Calculation Tests ---


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


# --- Normalize and Denormalize Tests ---


def test_normalize_boxes_numpy():
    """Test normalize_boxes with numpy arrays."""
    img_width, img_height = 640, 480
    # Use float32 instead of integers to avoid type conversion issues
    boxes_pixels = np.array(
        [[64.0, 48.0, 320.0, 240.0], [128.0, 96.0, 256.0, 192.0]], dtype=np.float32
    )
    expected_norm = np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.4, 0.4]], dtype=np.float32)

    norm_boxes = normalize_boxes(boxes_pixels, img_width, img_height)
    assert isinstance(norm_boxes, np.ndarray)
    assert np.allclose(norm_boxes, expected_norm, atol=1e-6)


def test_normalize_boxes_torch():
    """Test normalize_boxes with torch tensors."""
    img_width, img_height = 640, 480
    # Use float tensor instead of integer tensor
    boxes_pixels = torch.tensor(
        [[64.0, 48.0, 320.0, 240.0], [128.0, 96.0, 256.0, 192.0]], dtype=torch.float32
    )
    expected_norm = torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.4, 0.4]], dtype=torch.float32)

    norm_boxes = normalize_boxes(boxes_pixels, img_width, img_height)
    assert isinstance(norm_boxes, torch.Tensor)
    assert torch.allclose(norm_boxes, expected_norm, atol=1e-6)


def test_denormalize_boxes_numpy():
    """Test denormalize_boxes with numpy arrays."""
    img_width, img_height = 640, 480
    normalized_boxes = np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.4, 0.4]], dtype=np.float32)
    expected_pixels = np.array(
        [[64.0, 48.0, 320.0, 240.0], [128.0, 96.0, 256.0, 192.0]], dtype=np.float32
    )

    pixel_boxes = denormalize_boxes(normalized_boxes, img_width, img_height)
    assert isinstance(pixel_boxes, np.ndarray)
    assert np.allclose(pixel_boxes, expected_pixels, atol=1e-6)


def test_denormalize_boxes_torch():
    """Test denormalize_boxes with torch tensors."""
    img_width, img_height = 640, 480
    normalized_boxes = torch.tensor(
        [[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.4, 0.4]], dtype=torch.float32
    )
    expected_pixels = torch.tensor(
        [[64.0, 48.0, 320.0, 240.0], [128.0, 96.0, 256.0, 192.0]], dtype=torch.float32
    )

    pixel_boxes = denormalize_boxes(normalized_boxes, img_width, img_height)
    assert isinstance(pixel_boxes, torch.Tensor)
    assert torch.allclose(pixel_boxes, expected_pixels, atol=1e-6)
