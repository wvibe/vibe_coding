import logging
from unittest.mock import MagicMock, patch

import pytest
import torch

# Assume the file structure allows this import
from src.utils.metrics.compute import get_model_params, get_peak_gpu_memory_mb

# --- Tests for get_model_params ---


class MockUltralyticsModelInfo:
    def info(self, detailed=False):
        # Simulate returning (layers, params, gradients, flops)
        return (99, 1234567, 0, 0)

    # Add dummy methods/attributes to prevent fallback check failure if needed
    # Although the fallback shouldn't be reached in the success test
    parameters = None  # Make it look non-callable


class MockUltralyticsModelBadInfo:
    def info(self, detailed=False):
        return ("Layer info", "Bad Params", 0, 0)  # Params not numeric


class MockYOLO:
    def __init__(self, model_internal):
        self.model = model_internal


class MockTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 5)  # 10*5 + 5 = 55 params
        self.linear2 = torch.nn.Linear(5, 1)  # 5*1 + 1 = 6 params
        # Total trainable = 61

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


def test_get_params_ultralytics_info():
    """Test getting params from Ultralytics model.model.info()."""
    mock_yolo = MockYOLO(MockUltralyticsModelInfo())
    params = get_model_params(mock_yolo)
    assert params == 1234567


def test_get_params_ultralytics_bad_info_fallback():
    """Test fallback when Ultralytics model.model.info() is bad."""
    mock_yolo = MockYOLO(MockTorchModel())  # Use torch model for fallback count
    mock_yolo.model.info = MockUltralyticsModelBadInfo().info  # Add bad info method
    params = get_model_params(mock_yolo)
    assert params == 61  # Should get from summing torch params


def test_get_params_torch_fallback():
    """Test fallback to summing torch parameters directly."""
    mock_yolo = MockYOLO(MockTorchModel())
    params = get_model_params(mock_yolo)
    assert params == 61


def test_get_params_direct_torch_model():
    """Test passing a direct torch.nn.Module."""
    torch_model = MockTorchModel()
    params = get_model_params(torch_model)
    assert params == 61


def test_get_params_no_info_no_module():
    """Test when model has neither .info nor is a Module."""
    mock_model = MagicMock(spec=object)
    # Ensure it doesn't accidentally look like a module
    del mock_model.parameters
    params = get_model_params(mock_model)
    assert params is None


def test_get_params_info_error():
    """Test when model.model.info() raises an error."""
    mock_info = MagicMock(side_effect=Exception("Info Error"))
    mock_internal = MagicMock()
    mock_internal.info = mock_info
    mock_yolo = MockYOLO(mock_internal)
    # Make fallback fail too
    del mock_yolo.model.parameters
    params = get_model_params(mock_yolo)
    assert params is None


# --- Tests for get_peak_gpu_memory_mb ---


@patch("torch.cuda.is_available")
def test_gpu_memory_cuda_not_available(mock_is_available):
    """Test GPU memory check when CUDA is not available."""
    mock_is_available.return_value = False
    memory = get_peak_gpu_memory_mb()
    assert memory is None


# Note: Testing actual CUDA memory requires a CUDA environment and is tricky.
# These tests focus on non-CUDA paths or mocking.


@patch("torch.cuda.is_available")
def test_gpu_memory_cpu_device(mock_is_available):
    """Test GPU memory check when device is explicitly CPU."""
    mock_is_available.return_value = True  # Mock CUDA as available
    memory = get_peak_gpu_memory_mb(device="cpu")
    assert memory is None
    memory = get_peak_gpu_memory_mb(device=torch.device("cpu"))
    assert memory is None


@patch("torch.cuda.is_available")
@patch("torch.cuda.max_memory_allocated")
@patch("torch.cuda.current_device")
@patch("torch.cuda.device_count")
def test_gpu_memory_mock_success(mock_dev_count, mock_current_dev, mock_max_mem, mock_is_available):
    """Test successful GPU memory retrieval using mocks."""
    mock_is_available.return_value = True
    mock_dev_count.return_value = 1
    mock_current_dev.return_value = 0
    mock_max_mem.return_value = 512 * 1024 * 1024  # Simulate 512 MiB

    # Test with default device
    memory = get_peak_gpu_memory_mb()
    assert memory == pytest.approx(512.0)
    mock_max_mem.assert_called_with(device=0)

    # Test with string device
    mock_max_mem.reset_mock()
    memory = get_peak_gpu_memory_mb(device="cuda:0")
    assert memory == pytest.approx(512.0)
    mock_max_mem.assert_called_with(device=0)

    # Test with torch.device
    mock_max_mem.reset_mock()
    memory = get_peak_gpu_memory_mb(device=torch.device("cuda:0"))
    assert memory == pytest.approx(512.0)
    mock_max_mem.assert_called_with(device=0)


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.device_count", return_value=1)
@patch("torch.cuda.max_memory_allocated", side_effect=RuntimeError("Simulated CUDA error"))
def test_gpu_memory_device_out_of_range(mock_alloc, mock_count, mock_avail, caplog):
    """Tests handling when the requested device index is out of range."""
    with caplog.at_level(logging.ERROR):
        memory = get_peak_gpu_memory_mb(
            device=1
        )  # Request device 1 when count is 1 (so index 0 is valid)

    assert "CUDA device index 1 out of range" in caplog.text
    # The function should return None when the device is out of range
    assert memory is None  # Corrected Assertion
