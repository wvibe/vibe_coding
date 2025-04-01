"""Tests for the bounding box format conversion utilities."""

import numpy as np
import pytest
import torch

from src.utils.common.bbox_format import xywh_to_xyxy, xyxy_to_xywh

# --- Test Data ---

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


# --- Test Functions ---


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
