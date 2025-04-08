import pytest
import pathlib
import yaml
import sys
import os

# Ensure the src directory is in the Python path for imports
# This might be necessary depending on how pytest is run/configured
project_root = pathlib.Path(__file__).resolve().parents[3]
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils.logging.log_finder import find_wandb_run_id

# Helper function to create mock config files
def create_mock_config(path: pathlib.Path, name: str, use_files_subdir: bool = True):
    config_content = {'name': {'desc': None, 'value': name}}
    if use_files_subdir:
        config_dir = path / 'files'
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / 'config.yaml'
    else:
        config_file = path / 'config.yaml'

    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

@pytest.fixture
def mock_dirs(tmp_path: pathlib.Path):
    """Creates mock Ultralytics and WandB directories for testing."""
    ul_run_name = "test_run_1"
    ul_dir = tmp_path / "runs" / "detect" / ul_run_name
    ul_dir.mkdir(parents=True)

    wandb_root = tmp_path / "mock_wandb"
    wandb_root.mkdir()

    # Matching run (config in files subdir)
    match_id = "abc123xyz"
    match_dir = wandb_root / f"run-20240801_100000-{match_id}"
    match_dir.mkdir()
    create_mock_config(match_dir, ul_run_name, use_files_subdir=True)

    # Non-matching run (different name)
    non_match_dir = wandb_root / "run-20240801_110000-def456"
    non_match_dir.mkdir()
    create_mock_config(non_match_dir, "different_run_name")

    # Run with config directly in root
    match_id_root_config = "ghi789"
    match_dir_root_config = wandb_root / f"run-20240801_120000-{match_id_root_config}"
    match_dir_root_config.mkdir()
    create_mock_config(match_dir_root_config, "test_run_root", use_files_subdir=False)

    # Run with no config.yaml
    no_config_dir = wandb_root / "run-20240801_130000-jkl012"
    no_config_dir.mkdir()
    (no_config_dir / 'files').mkdir(exist_ok=True) # Create files dir but no config

    # Offline run (should be ignored by default logic if it doesn't match name)
    offline_dir = wandb_root / "offline-run-20240801_140000-mno345"
    offline_dir.mkdir()
    # create_mock_config(offline_dir, ul_run_name) # Usually offline runs don't have same structure

    # Directory not starting with run-
    other_dir = wandb_root / "other_stuff"
    other_dir.mkdir()

    return ul_dir, wandb_root, match_id, match_id_root_config

def test_find_wandb_run_id_success(mock_dirs):
    """Test successful finding of the WandB run ID."""
    ul_dir, wandb_root, expected_id, _ = mock_dirs
    found_id = find_wandb_run_id(str(ul_dir), str(wandb_root))
    assert found_id == expected_id

def test_find_wandb_run_id_success_config_in_root(mock_dirs, tmp_path):
    """Test finding run ID when config.yaml is in the run dir root."""
    _, wandb_root, _, expected_id_root = mock_dirs
    # Create a specific UL run dir for this test case
    ul_dir_root_test = tmp_path / "runs" / "detect" / "test_run_root"
    ul_dir_root_test.mkdir(parents=True)

    found_id = find_wandb_run_id(str(ul_dir_root_test), str(wandb_root))
    assert found_id == expected_id_root


def test_find_wandb_run_id_not_found_name_mismatch(mock_dirs, tmp_path):
    """Test case where no WandB run has the matching name."""
    _, wandb_root, _, _ = mock_dirs
    # Create a UL run dir that won't match any mock config name
    ul_dir_no_match = tmp_path / "runs" / "detect" / "unmatched_run"
    ul_dir_no_match.mkdir(parents=True)

    found_id = find_wandb_run_id(str(ul_dir_no_match), str(wandb_root))
    assert found_id is None

def test_find_wandb_run_id_no_config_skipped(mock_dirs):
    """Test that runs without config files are skipped."""
    # This implicitly tested in test_find_wandb_run_id_success,
    # as the no_config_dir exists but shouldn't interfere.
    # We can make it more explicit by ensuring the correct ID is found
    # even with the no_config_dir present.
    ul_dir, wandb_root, expected_id, _ = mock_dirs
    found_id = find_wandb_run_id(str(ul_dir), str(wandb_root))
    assert found_id == expected_id # Correct ID still found


def test_find_wandb_run_id_invalid_ul_path(mock_dirs):
    """Test providing a non-existent Ultralytics path."""
    _, wandb_root, _, _ = mock_dirs
    non_existent_ul_dir = "/path/does/not/exist/run_x"
    found_id = find_wandb_run_id(non_existent_ul_dir, str(wandb_root))
    assert found_id is None

def test_find_wandb_run_id_invalid_wandb_path(mock_dirs):
    """Test providing a non-existent WandB root path."""
    ul_dir, _, _, _ = mock_dirs
    non_existent_wandb_root = "/path/does/not/exist/wandb"
    found_id = find_wandb_run_id(str(ul_dir), non_existent_wandb_root)
    assert found_id is None

def test_find_wandb_run_id_malformed_config(mock_dirs, tmp_path):
    """Test handling of a malformed config file."""
    ul_dir, wandb_root, _, _ = mock_dirs

    # Create a run with a malformed YAML
    malformed_dir = wandb_root / "run-20240801_150000-bad123"
    malformed_dir.mkdir()
    config_file = malformed_dir / "files" / "config.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        f.write("name: { value: 'test_run_1',") # Missing closing brace

    # We expect it to log an error and return the correct ID from the valid run
    expected_id = mock_dirs[2]
    found_id = find_wandb_run_id(str(ul_dir), str(wandb_root))
    assert found_id == expected_id # Should still find the good one

def test_find_wandb_run_id_config_missing_name_value(mock_dirs, tmp_path):
    """Test config where 'name' exists but not 'name.value'."""
    ul_dir, wandb_root, _, _ = mock_dirs
    ul_run_name = ul_dir.name

    # Create a run with config missing the 'value' subkey
    missing_value_dir = wandb_root / "run-20240801_160000-miss1"
    missing_value_dir.mkdir()
    config_content = {'name': {'desc': 'A description'}} # No 'value' key
    config_file = missing_value_dir / "files" / "config.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    # Expect the correct match from the original fixture, skipping this malformed one
    expected_id = mock_dirs[2]
    found_id = find_wandb_run_id(str(ul_dir), str(wandb_root))
    assert found_id == expected_id

def test_find_wandb_run_id_empty_wandb_dir(tmp_path):
    """Test with an empty WandB directory."""
    ul_dir = tmp_path / "runs" / "detect" / "some_run"
    ul_dir.mkdir(parents=True)
    wandb_root = tmp_path / "empty_wandb"
    wandb_root.mkdir()

    found_id = find_wandb_run_id(str(ul_dir), str(wandb_root))
    assert found_id is None