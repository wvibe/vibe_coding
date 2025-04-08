import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to sys.path to allow importing train_segment
# This assumes the test is run from the project root (vibe_coding)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

# Import the functions/classes to test AFTER modifying sys.path
from models.ext.yolov11.train_segment import (
    _determine_run_params,
    _load_model,  # Assuming we might want to test this lightly
    _setup_wandb,
    _validate_and_get_data_config_path,
    prepare_train_kwargs,
)

# Mocks for dependencies
MOCK_PROJECT_ROOT = Path("/fake/project/root")
MOCK_WANDB_DIR = "mock_wandb"


@pytest.fixture
def mock_args(tmp_path):
    """Fixture to create mock argparse.Namespace with default values."""
    return argparse.Namespace(
        config="path/to/mock_config.yaml",
        project=None,
        name="test_run",
        resume_with=None,
        wandb_dir=MOCK_WANDB_DIR,
    )


@pytest.fixture
def mock_main_config():
    """Fixture for a basic main configuration dictionary."""
    return {
        "model": "yolo11l-seg.pt",
        "data": "src/models/ext/yolov11/configs/voc_segment.yaml",
        "epochs": 10,
        "imgsz": 640,
        "batch": 16,
        "workers": 8,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.1,
        "device": "0",
        "pretrained": True,
        "project": "runs/train/default_config_project",
        "some_other_key": "should_be_ignored",
        "overlap_mask": True,
        "mask_ratio": 4,
    }


# --- Tests for _validate_and_get_data_config_path ---


def test_validate_data_config_path_success(mock_main_config, tmp_path):
    project_root = tmp_path
    relative_path = mock_main_config["data"]
    absolute_path = (project_root / relative_path).resolve()
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.touch()  # Create the dummy file

    result_path = _validate_and_get_data_config_path(mock_main_config, project_root)
    assert result_path == absolute_path


def test_validate_data_config_path_missing_key(mock_main_config):
    del mock_main_config["data"]
    with pytest.raises(ValueError, match="Missing 'data' key"):
        _validate_and_get_data_config_path(mock_main_config, MOCK_PROJECT_ROOT)


def test_validate_data_config_path_file_not_found(mock_main_config):
    project_root = MOCK_PROJECT_ROOT
    with pytest.raises(FileNotFoundError, match="Data config file specified.*not found"):
        _validate_and_get_data_config_path(mock_main_config, project_root)


# --- Tests for _determine_run_params ---


@patch("models.ext.yolov11.train_segment.datetime")
def test_determine_run_params_new_run(mock_dt, mock_args, mock_main_config):
    mock_now = datetime(2024, 1, 1, 10, 30, 0)
    mock_dt.now.return_value = mock_now
    timestamp = mock_now.strftime("%Y%m%d_%H%M%S")
    mock_args.name = "new_run_base"
    expected_name = f"new_run_base_{timestamp}"
    expected_model = mock_main_config["model"]

    model, name, resume, wandb_id = _determine_run_params(
        mock_args, mock_main_config, MOCK_PROJECT_ROOT
    )

    assert model == expected_model
    assert name == expected_name
    assert resume is False
    assert wandb_id is None


@patch("models.ext.yolov11.train_segment.find_wandb_run_id")
def test_determine_run_params_resume_success_no_wandb(
    mock_find_wandb, mock_args, mock_main_config, tmp_path
):
    mock_find_wandb.return_value = None  # Simulate not finding WandB ID
    project_root = tmp_path
    resume_run_name = "resumed_run_20240101_110000"
    resume_dir_rel = f"runs/finetune/segment/{resume_run_name}"
    resume_dir_abs = project_root / resume_dir_rel
    checkpoint_path = resume_dir_abs / "weights" / "last.pt"

    resume_dir_abs.mkdir(parents=True)
    (resume_dir_abs / "weights").mkdir()
    checkpoint_path.touch()

    mock_args.resume_with = resume_dir_rel
    mock_args.name = "resumed_run"  # Base name provided

    model, name, resume, wandb_id = _determine_run_params(mock_args, mock_main_config, project_root)

    assert model == str(checkpoint_path)
    assert name == resume_run_name
    assert resume is True
    assert wandb_id is None
    mock_find_wandb.assert_called_once_with(str(resume_dir_abs), MOCK_WANDB_DIR)


@patch("models.ext.yolov11.train_segment.find_wandb_run_id")
def test_determine_run_params_resume_success_with_wandb(
    mock_find_wandb, mock_args, mock_main_config, tmp_path
):
    found_wandb_id = "wandb123xyz"
    mock_find_wandb.return_value = found_wandb_id
    project_root = tmp_path
    resume_run_name = "resumed_run_wandb_20240101_120000"
    resume_dir_rel = f"runs/finetune/segment/{resume_run_name}"
    resume_dir_abs = project_root / resume_dir_rel
    checkpoint_path = resume_dir_abs / "weights" / "last.pt"

    resume_dir_abs.mkdir(parents=True)
    (resume_dir_abs / "weights").mkdir()
    checkpoint_path.touch()

    mock_args.resume_with = resume_dir_rel
    mock_args.name = "resumed_run_wandb"  # Base name provided

    model, name, resume, wandb_id = _determine_run_params(mock_args, mock_main_config, project_root)

    assert model == str(checkpoint_path)
    assert name == resume_run_name
    assert resume is True
    assert wandb_id == found_wandb_id
    mock_find_wandb.assert_called_once_with(str(resume_dir_abs), MOCK_WANDB_DIR)


def test_determine_run_params_resume_dir_not_found(mock_args, mock_main_config, tmp_path):
    project_root = tmp_path
    mock_args.resume_with = "non/existent/path"
    with pytest.raises(FileNotFoundError, match="Resume directory not found"):
        _determine_run_params(mock_args, mock_main_config, project_root)


def test_determine_run_params_resume_checkpoint_not_found(mock_args, mock_main_config, tmp_path):
    project_root = tmp_path
    resume_run_name = "no_checkpoint_run_20240101_130000"
    resume_dir_rel = f"runs/finetune/segment/{resume_run_name}"
    resume_dir_abs = project_root / resume_dir_rel
    resume_dir_abs.mkdir(parents=True)
    (resume_dir_abs / "weights").mkdir()  # Create weights dir, but no last.pt

    mock_args.resume_with = resume_dir_rel

    with pytest.raises(FileNotFoundError, match="Checkpoint 'last.pt' not found"):
        _determine_run_params(mock_args, mock_main_config, project_root)


def test_determine_run_params_new_run_missing_model(mock_args, mock_main_config):
    del mock_main_config["model"]
    with pytest.raises(ValueError, match="Missing 'model' key"):
        _determine_run_params(mock_args, mock_main_config, MOCK_PROJECT_ROOT)


# --- Tests for _setup_wandb ---


@patch.dict(os.environ, {}, clear=True)
def test_setup_wandb_with_id_resume():
    wandb_id = "wandb_resume_123"
    _setup_wandb(wandb_id, resume_flag=True)
    assert os.environ.get("WANDB_RESUME") == "allow"
    assert os.environ.get("WANDB_RUN_ID") == wandb_id


@patch.dict(os.environ, {}, clear=True)
def test_setup_wandb_with_id_no_resume():
    # Although less common now, test the case where ID is provided for a new run
    wandb_id = "wandb_new_456"
    _setup_wandb(wandb_id, resume_flag=False)
    assert os.environ.get("WANDB_RESUME") == "allow"  # Still set to allow
    assert os.environ.get("WANDB_RUN_ID") == wandb_id


@patch.dict(os.environ, {}, clear=True)
def test_setup_wandb_no_id():
    _setup_wandb(None, resume_flag=False)
    assert "WANDB_RESUME" not in os.environ
    assert "WANDB_RUN_ID" not in os.environ


# --- Tests for prepare_train_kwargs ---
# These tests are simpler now, just ensuring config mapping and fixed args


def test_prepare_train_kwargs_basic_mapping(mock_main_config):
    # Define the inputs that prepare_train_kwargs now expects
    run_name = "prepared_run_1"
    resume_flag = False
    effective_project = "final/project/path"
    data_config_path = Path("/absolute/path/to/data.yaml")

    expected_kwargs = {
        # Mapped from mock_main_config
        "epochs": 10,
        "imgsz": 640,
        "batch": 16,
        "workers": 8,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.1,
        "device": "0",
        "pretrained": True,
        "overlap_mask": True,
        "mask_ratio": 4,
        # Explicitly set by prepare_train_kwargs
        "project": effective_project,
        "name": run_name,
        "resume": resume_flag,
        "data": str(data_config_path),
        # Keys NOT expected: model, some_other_key
    }

    actual_kwargs = prepare_train_kwargs(
        mock_main_config, run_name, resume_flag, effective_project, data_config_path
    )

    # Check all expected keys are present and have correct values
    for key, expected_value in expected_kwargs.items():
        assert key in actual_kwargs
        assert actual_kwargs[key] == expected_value

    # Check that unexpected keys from main_config are filtered out
    assert "model" not in actual_kwargs
    assert "some_other_key" not in actual_kwargs


def test_prepare_train_kwargs_empty_device_handling(mock_main_config):
    mock_main_config["device"] = ""  # Set device to empty string in config
    run_name = "prepared_run_empty_device"
    resume_flag = False
    effective_project = "final/project/path_empty"
    data_config_path = Path("/absolute/path/to/data_empty.yaml")

    expected_kwargs = {
        # Mapped from mock_main_config
        "epochs": 10,
        "imgsz": 640,
        "batch": 16,
        "workers": 8,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.1,
        "device": None,  # Should be None
        "pretrained": True,
        "overlap_mask": True,
        "mask_ratio": 4,
        # Explicitly set by prepare_train_kwargs
        "project": effective_project,
        "name": run_name,
        "resume": resume_flag,
        "data": str(data_config_path),
    }

    actual_kwargs = prepare_train_kwargs(
        mock_main_config, run_name, resume_flag, effective_project, data_config_path
    )
    assert actual_kwargs["device"] is None


# --- Test dataset config content (remains largely the same) ---
# Removed this test as it depends directly on external file content,
# making the unit test brittle. Configuration file content validation
# is better suited for integration tests or schema validation.
# def test_dataset_config_content():
#     ...


# (Optional: Add a light test for _load_model if needed, mainly checking it calls YOLO)
@patch("models.ext.yolov11.train_segment.YOLO")
def test_load_model(mock_yolo):
    model_path = "fake/path/model.pt"
    _load_model(model_path)
    mock_yolo.assert_called_once_with(model_path)
