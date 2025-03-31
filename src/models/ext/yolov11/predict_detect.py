"""
Runs YOLOv11 detection prediction based on a configuration YAML file.

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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def predict_pipeline(config_path: Path, name_prefix: str, cli_args: argparse.Namespace):
    """Orchestrates the detection prediction pipeline."""
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

    # 6. Run YOLO detection prediction
    _run_yolo_prediction(
        final_config, processed_source, project_dir_str, exp_name, computed_output_dir, "detect"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLOv11 detection prediction using a YAML configuration file, allowing CLI overrides."
    )

    # Core arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the prediction configuration YAML file.",
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        required=True,
        help="Prefix for the output directory name (e.g., 'voc_test').",
    )

    # Arguments to override config file
    parser.add_argument("--model", type=str, default=None, help="Override model path from config.")
    parser.add_argument(
        "--source", type=str, default=None, help="Override source path from config."
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Override project directory from config.",
    )
    parser.add_argument("--conf", type=float, default=None, help="Override confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size.")
    parser.add_argument("--iou", type=float, default=None, help="Override IoU threshold for NMS.")
    parser.add_argument(
        "--max_det",
        type=int,
        default=None,
        help="Override maximum detections per image.",
    )
    parser.add_argument(
        "--save_txt",
        type=lambda x: (str(x).lower() == "true"),
        default=None,
        help="Override save results as YOLO labels (True/False).",
    )
    parser.add_argument(
        "--save_conf",
        type=lambda x: (str(x).lower() == "true"),
        default=None,
        help="Override include confidence in saved labels/plots (True/False).",
    )
    parser.add_argument(
        "--save",
        type=lambda x: (str(x).lower() == "true"),
        default=None,
        help="Override save annotated images (True/False).",
    )
    parser.add_argument(
        "--random_select",
        type=int,
        default=None,
        help="Override number of random images to select from directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g., 'cpu', '0', '0,1').",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    predict_pipeline(config_path, args.name_prefix, args)


if __name__ == "__main__":
    main()
