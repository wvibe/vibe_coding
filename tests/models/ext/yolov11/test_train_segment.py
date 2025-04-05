import sys
import argparse
from pathlib import Path
import pytest
import yaml # Added for loading dataset config

# Add project root to sys.path to allow importing train_segment
# This assumes the test is run from the project root (vibe_coding)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_PATH))

# Import the function to test AFTER modifying sys.path
from models.ext.yolov11.train_segment import prepare_train_kwargs, get_project_root

# Helper function to create mock args
def create_mock_args(
    name="test_run", project=None, resume=False, wandb_id=None
):
    return argparse.Namespace(
        name=name,
        project=project,
        resume=resume,
        wandb_id=wandb_id,
        # config is not used by prepare_train_kwargs directly
    )

# Test case 1: Basic config, no CLI overrides
def test_prepare_train_kwargs_basic():
    project_root = get_project_root()
    mock_config = {
        "model": "yolo11l-seg.pt",
        "data": str(project_root / "src/models/ext/yolov11/configs/voc_segment.yaml"),
        "epochs": 10,
        "imgsz": 640,
        "batch": 16,
        "workers": 8,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.1,
        "device": "0",
        "pretrained": True,
        "project": "runs/train/default_config_project", # Project from config
        "some_other_key": "should_be_ignored"
    }
    mock_args = create_mock_args(name="basic_run")
    effective_project_path = "runs/train/default_config_project" # From config

    expected_kwargs = {
        "data": str(project_root / "src/models/ext/yolov11/configs/voc_segment.yaml"),
        "epochs": 10,
        "imgsz": 640,
        "batch": 16,
        "workers": 8,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.1,
        "device": "0",
        "pretrained": True,
        "project": effective_project_path, # Should match config
        "name": "basic_run",
        "resume": False
    }

    actual_kwargs = prepare_train_kwargs(mock_config, mock_args, effective_project_path)
    assert actual_kwargs == expected_kwargs

# Test case 2: CLI overrides for project and resume
def test_prepare_train_kwargs_cli_overrides():
    project_root = get_project_root()
    mock_config = {
        "model": "yolo11l-seg.pt",
        "data": str(project_root / "src/models/ext/yolov11/configs/voc_segment.yaml"),
        "epochs": 10,
        "imgsz": 640,
        "batch": 16,
        "workers": 8,
        "optimizer": "AdamW",
        "device": "1,2",
        "pretrained": True,
        "project": "runs/train/config_project" # Will be overridden
    }
    # CLI overrides
    cli_project = "runs/train/cli_project"
    cli_name = "override_run"
    cli_resume = True
    mock_args = create_mock_args(name=cli_name, project=cli_project, resume=cli_resume)
    # Effective project path comes from CLI arg in this case
    effective_project_path = cli_project

    expected_kwargs = {
        "data": str(project_root / "src/models/ext/yolov11/configs/voc_segment.yaml"),
        "epochs": 10,
        "imgsz": 640,
        "batch": 16,
        "workers": 8,
        "optimizer": "AdamW",
        "device": "1,2",
        "pretrained": True,
        "project": cli_project, # Overridden by CLI
        "name": cli_name,
        "resume": cli_resume # Set by CLI
    }

    actual_kwargs = prepare_train_kwargs(mock_config, mock_args, effective_project_path)
    assert actual_kwargs == expected_kwargs

# Test case 3: Empty device string in config
def test_prepare_train_kwargs_empty_device():
    project_root = get_project_root()
    mock_config = {
        "model": "yolo11l-seg.pt",
        "data": str(project_root / "src/models/ext/yolov11/configs/voc_segment.yaml"),
        "epochs": 10,
        "device": "", # Empty string
        "project": "runs/train/empty_device_project"
    }
    mock_args = create_mock_args(name="empty_device_run")
    effective_project_path = "runs/train/empty_device_project"

    expected_kwargs = {
        "data": str(project_root / "src/models/ext/yolov11/configs/voc_segment.yaml"),
        "epochs": 10,
        "device": None, # Should be converted to None
        "project": effective_project_path,
        "name": "empty_device_run",
        "resume": False
    }

    actual_kwargs = prepare_train_kwargs(mock_config, mock_args, effective_project_path)
    assert actual_kwargs == expected_kwargs

# Test case 4: Config with segmentation-specific args
def test_prepare_train_kwargs_segmentation_args():
    project_root = get_project_root()
    mock_config = {
        "model": "yolo11l-seg.pt",
        "data": str(project_root / "src/models/ext/yolov11/configs/voc_segment.yaml"),
        "epochs": 5,
        "overlap_mask": False,
        "mask_ratio": 2,
        "project": "runs/train/seg_args_project"
    }
    mock_args = create_mock_args(name="seg_args_run")
    effective_project_path = "runs/train/seg_args_project"

    expected_kwargs = {
        "data": str(project_root / "src/models/ext/yolov11/configs/voc_segment.yaml"),
        "epochs": 5,
        "overlap_mask": False,
        "mask_ratio": 2,
        "project": effective_project_path,
        "name": "seg_args_run",
        "resume": False
    }

    actual_kwargs = prepare_train_kwargs(mock_config, mock_args, effective_project_path)
    assert actual_kwargs == expected_kwargs

# Test case 5: Verify content of the actual dataset config YAML
def test_dataset_config_content():
    project_root = get_project_root()
    dataset_config_path = project_root / "src/models/ext/yolov11/configs/voc_segment.yaml"

    assert dataset_config_path.is_file(), f"Dataset config file not found: {dataset_config_path}"

    with open(dataset_config_path, 'r') as f:
        try:
            dataset_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            pytest.fail(f"Error parsing dataset config YAML {dataset_config_path}: {e}")

    assert isinstance(dataset_config, dict), "Dataset config is not a dictionary."

    # Check required keys
    expected_keys = ['path', 'train', 'val', 'test', 'names']
    for key in expected_keys:
        assert key in dataset_config, f"Missing key '{key}' in dataset config."

    # Check specific values based on previous setup
    expected_path = "/home/ubuntu/vibe/hub/datasets/segment_VOC"
    expected_train_list = [
        'images/train2007',
        'images/train2012',
        'images/val2007',
        'images/val2012'
    ]
    expected_val = 'images/test2007'
    expected_test = 'images/test2007'
    expected_num_classes = 20

    assert dataset_config['path'] == expected_path, f"Incorrect path in dataset config. Expected {expected_path}, got {dataset_config['path']}"
    assert dataset_config['train'] == expected_train_list, f"Incorrect train list in dataset config. Expected {expected_train_list}, got {dataset_config['train']}"
    assert dataset_config['val'] == expected_val, f"Incorrect val path in dataset config. Expected {expected_val}, got {dataset_config['val']}"
    assert dataset_config['test'] == expected_test, f"Incorrect test path in dataset config. Expected {expected_test}, got {dataset_config['test']}"
    assert isinstance(dataset_config['names'], dict), "'names' key should be a dictionary."
    assert len(dataset_config['names']) == expected_num_classes, f"Incorrect number of classes. Expected {expected_num_classes}, got {len(dataset_config['names'])}"