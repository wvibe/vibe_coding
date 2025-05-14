import argparse
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Assuming predict_utils.py is importable
from vibelab.models.ext.yolov11.predict_utils import (
    _merge_config_and_args,
    _prepare_output_directory,
)

# --- Tests for _merge_config_and_args ---


def test_merge_config_no_overrides():
    """Test merging when no CLI args are provided (all None)."""
    base_config = {"device": "cpu", "save": True, "show": False, "other": "value"}
    args = argparse.Namespace(device=None, save=None, show=None, irrelevant="test")
    merged = _merge_config_and_args(base_config, args)
    assert merged == base_config  # Expect no changes


def test_merge_config_with_device_override():
    """Test merging when only device is overridden via CLI."""
    base_config = {"device": "cpu", "save": True, "show": False}
    args = argparse.Namespace(device="0", save=None, show=None)
    merged = _merge_config_and_args(base_config, args)
    expected = {"device": "0", "save": True, "show": False}
    assert merged == expected


def test_merge_config_with_save_override():
    """Test merging when save is overridden via CLI."""
    base_config = {"device": "cpu", "save": True, "show": False}
    args = argparse.Namespace(device=None, save=False, show=None)
    merged = _merge_config_and_args(base_config, args)
    expected = {"device": "cpu", "save": False, "show": False}
    assert merged == expected


def test_merge_config_with_show_override():
    """Test merging when show is overridden via CLI."""
    base_config = {"device": "cpu", "save": True, "show": False}
    args = argparse.Namespace(device=None, save=None, show=True)
    merged = _merge_config_and_args(base_config, args)
    expected = {"device": "cpu", "save": True, "show": True}
    assert merged == expected


def test_merge_config_multiple_overrides():
    """Test merging when multiple args are overridden via CLI."""
    base_config = {"device": "cpu", "save": True, "show": False, "other": "val"}
    args = argparse.Namespace(device="1", save=False, show=True)
    merged = _merge_config_and_args(base_config, args)
    expected = {"device": "1", "save": False, "show": True, "other": "val"}
    assert merged == expected


def test_merge_config_preserves_other_keys():
    """Ensure keys not in allowed_override_keys are preserved."""
    base_config = {"device": "cpu", "model": "yolo.pt", "project": "runs/test"}
    args = argparse.Namespace(device="0", save=None, show=None)
    merged = _merge_config_and_args(base_config, args)
    expected = {"device": "0", "model": "yolo.pt", "project": "runs/test"}
    assert merged == expected


# --- Tests for _prepare_output_directory ---


# Use parametrize to test different inputs
@pytest.mark.parametrize(
    "project_dir, name, expected_proj_part, expected_name_part",
    [
        ("runs/segment", "my_run", "runs/segment", "my_run"),
        ("/abs/path/runs", "test_123", "/abs/path/runs", "test_123"),
        (".", "relative_run", ".", "relative_run"),
    ],
)
@patch("vibelab.models.ext.yolov11.predict_utils.datetime")
@patch("vibelab.models.ext.yolov11.predict_utils.Path.mkdir")
def test_prepare_output_directory(
    mock_mkdir,
    mock_dt,
    project_dir,
    name,
    expected_proj_part,
    expected_name_part,
    caplog,
):
    """Test output directory preparation with mocked datetime and mkdir."""
    # Set the logging level for the test
    caplog.set_level(logging.INFO)

    # Mock datetime.now() to return a fixed timestamp
    mock_now = MagicMock()
    mock_now.strftime.return_value = "240101_120000"  # Fixed timestamp
    mock_dt.now.return_value = mock_now

    # Call the function
    proj_path_str, exp_name_ts = _prepare_output_directory(project_dir, name)

    # Assertions
    expected_timestamped_name = f"{expected_name_part}_240101_120000"
    expected_computed_dir = Path(expected_proj_part) / expected_timestamped_name

    assert proj_path_str == expected_proj_part
    assert exp_name_ts == expected_timestamped_name
    # Check the log message is now captured
    assert f"Computed target output directory: {expected_computed_dir}" in caplog.text

    # Check that Path(project_dir).mkdir was called correctly
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    # Check that the mock datetime was used
    mock_dt.now.assert_called_once()
    mock_now.strftime.assert_called_once_with("%y%m%d_%H%M%S")
