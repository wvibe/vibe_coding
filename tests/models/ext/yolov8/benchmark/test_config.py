import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError

# Assuming tests are run from the project root (vibe_coding)
from src.models.ext.yolov8.benchmark.config import BenchmarkConfig

# Define the path to the example config relative to the project root
EXAMPLE_CONFIG_PATH = Path("src/models/ext/yolov8/benchmark/detection_benchmark.yaml")

# Check if the example config file exists
if not EXAMPLE_CONFIG_PATH.is_file():
    pytest.skip(f"Example config file not found: {EXAMPLE_CONFIG_PATH}", allow_module_level=True)

def test_load_valid_config():
    """Tests if the example YAML configuration can be loaded and validated."""
    try:
        with open(EXAMPLE_CONFIG_PATH, 'r') as f:
            config_data = yaml.safe_load(f)

        assert config_data is not None, "YAML file is empty or could not be parsed."

        # Validate configuration using Pydantic
        config = BenchmarkConfig(**config_data)

        # Basic checks on parsed data
        assert isinstance(config.models_to_test, list)
        assert len(config.models_to_test) > 0
        assert isinstance(config.dataset.test_images_dir, Path)
        assert isinstance(config.dataset.num_classes, int)
        assert config.dataset.num_classes > 0
        assert isinstance(config.metrics.iou_threshold_map, float)
        assert config.metrics.object_size_definitions.large[1] == float('inf') # Check inf conversion

    except yaml.YAMLError as e:
        pytest.fail(f"Failed to parse example YAML: {e}")
    except ValidationError as e:
        pytest.fail(f"Example configuration failed validation:\n{e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during config loading: {e}")

# Optional: Add tests for invalid configurations later
# def test_invalid_config_missing_field():
#     ...

# def test_invalid_config_bad_type():
#     ...