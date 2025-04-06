"""
Unit tests for src/models/ext/yolov11/predict_detect.py
Focusing on logic functions and non-IO operations
"""

# Add project root to path to allow relative imports
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from src.models.ext.yolov11.predict_detect import (
    _calculate_and_log_stats,
    _extract_and_average_times,
    process_source,
)


# Mock YOLO result for testing
class MockYOLOResult:
    """Mock class to simulate YOLO result objects for testing."""

    def __init__(self, speed=None):
        self.speed = speed or {}


# Tests for _extract_and_average_times
def test_extract_times_normal_case():
    """Test with valid speed data in all results."""
    # Create test results with speed data
    results = [
        MockYOLOResult(speed={"preprocess": 10.0, "inference": 50.0, "postprocess": 15.0}),
        MockYOLOResult(speed={"preprocess": 20.0, "inference": 60.0, "postprocess": 25.0}),
        MockYOLOResult(speed={"preprocess": 30.0, "inference": 70.0, "postprocess": 35.0}),
    ]

    avg_times, valid_count = _extract_and_average_times(results)

    # Check counts
    assert valid_count == 3

    # Check averages
    assert avg_times["preprocess"] == 20.0
    assert avg_times["inference"] == 60.0
    assert avg_times["postprocess"] == 25.0
    assert avg_times["total"] == 105.0


def test_extract_times_missing_values():
    """Test with missing speed values in some results."""
    results = [
        MockYOLOResult(speed={"preprocess": 10.0, "inference": 50.0, "postprocess": 15.0}),
        MockYOLOResult(speed={"preprocess": 20.0, "inference": 60.0}),  # Missing postprocess
        MockYOLOResult(speed={"inference": 70.0, "postprocess": 35.0}),  # Missing preprocess
    ]

    avg_times, valid_count = _extract_and_average_times(results)

    # All results should be valid as they have inference time > 0
    assert valid_count == 3

    # Check averages (missing values default to 0.0)
    assert avg_times["preprocess"] == pytest.approx(10.0)  # (10 + 20 + 0) / 3
    assert avg_times["inference"] == pytest.approx(60.0)  # (50 + 60 + 70) / 3
    assert avg_times["postprocess"] == pytest.approx(16.67, abs=0.01)  # (15 + 0 + 35) / 3
    assert avg_times["total"] == pytest.approx(86.67, abs=0.01)  # Sum of averages


def test_extract_times_zero_inference_time():
    """Test with zero inference time, which should be skipped."""
    results = [
        MockYOLOResult(speed={"preprocess": 10.0, "inference": 50.0, "postprocess": 15.0}),
        MockYOLOResult(speed={"preprocess": 20.0, "inference": 0.0, "postprocess": 25.0}),
        MockYOLOResult(speed={"preprocess": 30.0, "inference": 70.0, "postprocess": 35.0}),
    ]

    avg_times, valid_count = _extract_and_average_times(results)

    # Second result should be skipped due to inference = 0
    assert valid_count == 2

    # Check averages
    assert avg_times["preprocess"] == 20.0  # (10 + 30) / 2
    assert avg_times["inference"] == 60.0  # (50 + 70) / 2
    assert avg_times["postprocess"] == 25.0  # (15 + 35) / 2
    assert avg_times["total"] == 105.0  # Sum of averages


def test_extract_times_empty_results():
    """Test with an empty results list."""
    results = []

    avg_times, valid_count = _extract_and_average_times(results)

    # No valid results
    assert valid_count == 0
    assert avg_times is None


def test_extract_times_no_speed_data():
    """Test with results that have no speed data."""
    results = [
        MockYOLOResult(speed=None),
        MockYOLOResult(speed={}),
        MockYOLOResult(),  # Default empty speed dict
    ]

    avg_times, valid_count = _extract_and_average_times(results)

    # No valid results
    assert valid_count == 0
    assert avg_times is None


# Tests for _calculate_and_log_stats
@pytest.mark.parametrize(
    "num_results, predict_duration, expected_fps, expected_avg_time",
    [
        (5, 2.5, 2.0, 500.0),  # 5 images in 2.5s = 2.0 FPS, 500ms per image
        (10, 2.0, 5.0, 200.0),  # 10 images in 2.0s = 5.0 FPS, 200ms per image
        (1, 1.5, 0.67, 1500.0),  # 1 image in 1.5s = 0.67 FPS, 1500ms per image
    ],
)
def test_calculate_stats_wall_clock(
    num_results, predict_duration, expected_fps, expected_avg_time, monkeypatch
):
    """Test wall clock statistics calculation with different configurations."""
    # Setup mocks
    mock_logger = mock.MagicMock()
    monkeypatch.setattr("src.models.ext.yolov11.predict_detect.logger", mock_logger)

    mock_extract_times = mock.MagicMock()
    mock_extract_times.return_value = (
        {"preprocess": 15.0, "inference": 55.0, "postprocess": 20.0, "total": 90.0},
        num_results,
    )
    monkeypatch.setattr(
        "src.models.ext.yolov11.predict_detect._extract_and_average_times", mock_extract_times
    )

    # Create mock results
    mock_results = [MockYOLOResult() for _ in range(num_results)]
    cli_args = mock.MagicMock()

    # Call the function
    _calculate_and_log_stats(mock_results, predict_duration, cli_args)

    # Verify logs for wall clock stats
    mock_logger.info.assert_any_call(f"Successfully processed {num_results} images.")
    mock_logger.info.assert_any_call(f"Total YOLO Prediction Wall Time: {predict_duration:.3f} s")
    mock_logger.info.assert_any_call(
        f"Average Time per Image (Wall Clock): {expected_avg_time:.2f} ms"
    )
    mock_logger.info.assert_any_call(f"Overall FPS (Wall Clock): {expected_fps:.2f}")


def test_calculate_stats_successful(monkeypatch):
    """Test normal statistics calculation with successful results."""
    # Setup mocks
    mock_logger = mock.MagicMock()
    monkeypatch.setattr("src.models.ext.yolov11.predict_detect.logger", mock_logger)

    # Setup component time stats
    mock_avg_times = {"preprocess": 15.0, "inference": 55.0, "postprocess": 20.0, "total": 90.0}
    valid_count = 5

    mock_extract_times = mock.MagicMock()
    mock_extract_times.return_value = (mock_avg_times, valid_count)
    monkeypatch.setattr(
        "src.models.ext.yolov11.predict_detect._extract_and_average_times", mock_extract_times
    )

    # Create mock results and args
    mock_results = [MockYOLOResult() for _ in range(5)]
    predict_duration = 2.5  # seconds
    cli_args = mock.MagicMock()

    # Call the function
    _calculate_and_log_stats(mock_results, predict_duration, cli_args)

    # Verify component time logs
    mock_logger.info.assert_any_call(
        f"--- Avg Times from Ultralytics 'speed' (over {valid_count} images) ---"
    )
    mock_logger.info.assert_any_call(f"  Avg Preprocess : {mock_avg_times['preprocess']:.2f} ms")
    mock_logger.info.assert_any_call(f"  Avg Inference  : {mock_avg_times['inference']:.2f} ms")
    mock_logger.info.assert_any_call(f"  Avg Postprocess: {mock_avg_times['postprocess']:.2f} ms")
    mock_logger.info.assert_any_call(f"  Avg Total      : {mock_avg_times['total']:.2f} ms")


def test_calculate_stats_no_valid_speed_data(monkeypatch):
    """Test when no valid speed data is available."""
    # Setup mocks
    mock_logger = mock.MagicMock()
    monkeypatch.setattr("src.models.ext.yolov11.predict_detect.logger", mock_logger)

    mock_extract_times = mock.MagicMock()
    mock_extract_times.return_value = (None, 0)
    monkeypatch.setattr(
        "src.models.ext.yolov11.predict_detect._extract_and_average_times", mock_extract_times
    )

    # Setup
    mock_results = [MockYOLOResult() for _ in range(3)]
    predict_duration = 1.5  # seconds
    cli_args = mock.MagicMock()

    # Call the function
    _calculate_and_log_stats(mock_results, predict_duration, cli_args)

    # Verify logs
    mock_logger.info.assert_any_call("Successfully processed 3 images.")
    mock_logger.warning.assert_any_call(
        "Could not calculate average times from Ultralytics 'speed' results."
    )


def test_calculate_stats_empty_results(monkeypatch):
    """Test with empty results list."""
    # Setup mocks
    mock_logger = mock.MagicMock()
    monkeypatch.setattr("src.models.ext.yolov11.predict_detect.logger", mock_logger)

    # Setup
    mock_results = []
    predict_duration = 1.0
    cli_args = mock.MagicMock()

    # Call the function
    _calculate_and_log_stats(mock_results, predict_duration, cli_args)

    # Verify warning log
    mock_logger.warning.assert_any_call("Prediction ran but returned zero results.")


def test_calculate_stats_none_results(monkeypatch):
    """Test with None results (prediction failed)."""
    # Setup mocks
    mock_logger = mock.MagicMock()
    monkeypatch.setattr("src.models.ext.yolov11.predict_detect.logger", mock_logger)

    # Setup
    mock_results = None
    predict_duration = 1.0
    cli_args = mock.MagicMock()

    # Call the function
    _calculate_and_log_stats(mock_results, predict_duration, cli_args)

    # Verify error log
    mock_logger.error.assert_called_with("Prediction failed or returned unexpected result type.")


# Tests for process_source
def test_process_source_empty_directory(monkeypatch):
    """Test when source directory is empty."""
    # Mock logger
    mock_logger = mock.MagicMock()
    monkeypatch.setattr("src.models.ext.yolov11.predict_detect.logger", mock_logger)

    # Mock Path
    mock_dir = mock.MagicMock(spec=Path)
    mock_dir.glob.return_value = []

    # Call function
    result = process_source(mock_dir, None)

    # Assertions
    assert result == str(mock_dir)
    mock_logger.warning.assert_called_with(f"No image files found in source directory: {mock_dir}")


def test_process_source_all_images(monkeypatch):
    """Test processing all images (no sampling)."""
    # Mock logger
    mock_logger = mock.MagicMock()
    monkeypatch.setattr("src.models.ext.yolov11.predict_detect.logger", mock_logger)

    # Mock Path with image files
    mock_dir = mock.MagicMock(spec=Path)
    mock_files = [
        Path("image1.jpg"),
        Path("image2.png"),
        Path("image3.jpeg"),
        Path("image4.JPG"),
        Path("text.txt"),  # Should be ignored
    ]
    mock_dir.glob.return_value = mock_files

    # Expected count of image files (excluding text.txt)
    expected_num_images = 4

    # Call function
    result = process_source(mock_dir, None)

    # Assertions
    assert result == str(mock_dir)
    mock_logger.info.assert_called_with(
        f"Processing all {expected_num_images} found images in directory: {mock_dir}"
    )


def test_process_source_with_sampling(monkeypatch):
    """Test sampling a subset of images."""
    # Mock logger
    mock_logger = mock.MagicMock()
    monkeypatch.setattr("src.models.ext.yolov11.predict_detect.logger", mock_logger)

    # Mock random.sample
    image_files = [
        Path("image1.jpg"),
        Path("image2.png"),
        Path("image3.jpeg"),
        Path("image4.JPG"),
    ]
    sample_count = 2
    selected_images = image_files[:sample_count]

    mock_random_sample = mock.MagicMock(return_value=selected_images)
    monkeypatch.setattr("src.models.ext.yolov11.predict_detect.random.sample", mock_random_sample)

    # Mock Path
    mock_dir = mock.MagicMock(spec=Path)
    mock_dir.glob.return_value = image_files

    # Call function
    result = process_source(mock_dir, sample_count)

    # Assertions
    assert result == [str(p) for p in selected_images]
    mock_random_sample.assert_called_with(image_files, sample_count)
    mock_logger.info.assert_called_with(f"Randomly selected {sample_count} images for processing.")


def test_process_source_sample_more_than_available(monkeypatch):
    """Test requesting more samples than available images."""
    # Mock logger
    mock_logger = mock.MagicMock()
    monkeypatch.setattr("src.models.ext.yolov11.predict_detect.logger", mock_logger)

    # Mock image files
    image_files = [
        Path("image1.jpg"),
        Path("image2.png"),
    ]

    # Mock random.sample
    sample_count = 5  # More than available
    mock_random_sample = mock.MagicMock(return_value=image_files)
    monkeypatch.setattr("src.models.ext.yolov11.predict_detect.random.sample", mock_random_sample)

    # Mock Path
    mock_dir = mock.MagicMock(spec=Path)
    mock_dir.glob.return_value = image_files

    # Call function
    result = process_source(mock_dir, sample_count)

    # Assertions
    assert result == [str(p) for p in image_files]
    mock_random_sample.assert_called_with(image_files, len(image_files))
    mock_logger.warning.assert_called_with(
        f"Requested {sample_count} images, but only found {len(image_files)}. "
        f"Using all {len(image_files)} found images."
    )


def test_process_source_glob_exception(monkeypatch):
    """Test handling of exceptions when listing files."""
    # Mock logger
    mock_logger = mock.MagicMock()
    monkeypatch.setattr("src.models.ext.yolov11.predict_detect.logger", mock_logger)

    # Mock Path with exception
    mock_dir = mock.MagicMock(spec=Path)
    mock_dir.glob.side_effect = Exception("Test exception")

    # Call function should exit
    with pytest.raises(SystemExit) as excinfo:
        process_source(mock_dir, None)

    # Assertions
    assert excinfo.value.code == 1
    mock_logger.error.assert_called_with(f"Error listing images in {mock_dir}: Test exception")
