"""
Runs YOLOv11 segmentation prediction based on a configuration YAML file.

Allows overriding config parameters via command line arguments.
Handles single images or directories, with an option for random selection
from directories. Organizes output using a project/name structure similar
to Ultralytics training, incorporating a timestamp.
"""

import argparse
import logging
from pathlib import Path

from src.models.ext.yolov11.predict_utils import (
    _load_and_validate_config,
    _merge_config_and_args,
    _prepare_output_directory,
    _process_source_path,
    _run_yolo_prediction,
    _validate_final_config,
)

# Configure basic logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run YOLOv11 segmentation prediction")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument(
        "--name-prefix", type=str, required=True, help="Prefix for output directory"
    )
    # Add other args as in predict_detect.py
    return parser.parse_args()


def predict_pipeline(config_path: Path, name_prefix: str, cli_args: argparse.Namespace):
    """Orchestrates the segmentation prediction pipeline."""
    # 1. Load base config
    base_config = _load_and_validate_config(config_path)

    # 2. Merge CLI args into config
    final_config = _merge_config_and_args(base_config, cli_args)

    # 3. Validate the final merged config (especially paths)
    _validate_final_config(final_config, config_path)

    # 4. Prepare output directory info
    computed_output_dir, project_dir_str, exp_name = _prepare_output_directory(
        final_config["project"], name_prefix
    )

    # 5. Process source path (including random selection)
    random_select_val = final_config.get("random_select")
    processed_source = _process_source_path(final_config["source"], random_select_val)

    # 6. Run YOLO segmentation prediction
    _run_yolo_prediction(
        final_config, processed_source, project_dir_str, exp_name, computed_output_dir, "segment"
    )


def main():
    args = parse_args()
    config_path = Path(args.config)
    predict_pipeline(config_path, args.name_prefix, args)


if __name__ == "__main__":
    main()
