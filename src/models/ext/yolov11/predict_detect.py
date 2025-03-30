"""
Runs YOLOv11 detection prediction based on a configuration YAML file.

Allows overriding config parameters via command line arguments.
Handles single images or directories, with an option for random selection
from directories. Organizes output using a project/name structure similar
to Ultralytics training, incorporating a timestamp.
"""

import argparse
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _load_and_validate_config(config_path: Path) -> dict:
    """Loads and performs basic validation on the YAML config file."""
    logging.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if not config:
            raise ValueError("Config file is empty or invalid.")
        return config
    except FileNotFoundError:
        logging.error(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Error in config file: {e}")
        sys.exit(1)


def _merge_config_and_args(config: dict, args: argparse.Namespace) -> dict:
    """Merges command-line arguments into the loaded config, overriding if specified."""
    merged_config = config.copy()
    cli_overrides = {
        k: v
        for k, v in vars(args).items()
        if v is not None
        and k not in ["config", "name_prefix"]  # Exclude script-control args
    }

    if cli_overrides:
        logging.info(f"Applying command-line overrides: {cli_overrides}")
        merged_config.update(cli_overrides)

    return merged_config


def _validate_final_config(final_config: dict, config_path: Path):
    """Validates required keys and paths in the final merged configuration."""
    # Validate required config keys
    required_keys = ["model", "source", "project"]
    for key in required_keys:
        if key not in final_config or not final_config[key]:
            logging.error(
                f"Error: Missing or empty required key '{key}' after merging config '{config_path}' and CLI args."
            )
            sys.exit(1)

    # Validate paths
    model_path = Path(final_config["model"])
    source_path = Path(final_config["source"])

    # Check model path (allow Ultralytics names for download)
    if not model_path.exists() and not str(model_path).endswith((".pt", ".onnx")):
        logging.warning(
            f"Model path {model_path} not found locally, assuming it's an Ultralytics model name for download."
        )
    elif model_path.exists() and not model_path.is_file():
        logging.error(
            f"Error: Specified model path '{model_path}' exists but is not a file."
        )
        sys.exit(1)

    # Check source path
    if not source_path.exists():
        logging.error(f"Error: Source path '{source_path}' does not exist.")
        sys.exit(1)

    logging.info(f"Using final model: {final_config['model']}")
    logging.info(f"Using final source: {source_path}")
    logging.info(f"Using final project directory: {Path(final_config['project'])}")


def _prepare_output_directory(
    project_dir_str: str, name_prefix: str
) -> tuple[Path, str, str]:
    """Computes the timestamped output directory name and returns paths/names.
    Note: Does not create the directory, lets Ultralytics handle it via project/name.
    """
    project_dir = Path(project_dir_str)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    exp_name = f"{name_prefix}_{timestamp}"
    # output_dir = project_dir / exp_name # We compute this, but don't create it
    # try:
    #     output_dir.mkdir(parents=True, exist_ok=True)
    #     logging.info(f"Created output directory: {output_dir}")
    # except OSError as e:
    #     logging.error(f"Error creating output directory {output_dir}: {e}")
    #     sys.exit(1)
    # Return the computed output path, project string, and experiment name
    computed_output_dir = project_dir / exp_name
    logging.info(f"Computed target output directory: {computed_output_dir}")
    return computed_output_dir, str(project_dir), exp_name


def _process_source_path(
    source_path_str: str, random_select: int | None
) -> str | list[str]:
    """Handles file/directory source and random selection."""
    source_path = Path(source_path_str)
    source_is_dir = source_path.is_dir()
    processed_source = None

    if not source_is_dir:
        logging.info("Source is a single file.")
        processed_source = str(source_path)
    elif random_select and random_select > 0:
        logging.info(
            f"Source is a directory, selecting {random_select} random image(s)."
        )
        try:
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
            all_images = [
                p for p in source_path.glob("*") if p.suffix.lower() in image_extensions
            ]
            if not all_images:
                logging.error(f"No images found in directory: {source_path}")
                sys.exit(1)

            num_found = len(all_images)
            num_to_select = min(random_select, num_found)

            if num_found < random_select:
                logging.warning(
                    f"Requested {random_select} images, but only found {num_found}. Using all found images."
                )

            processed_source = [
                str(p) for p in random.sample(all_images, num_to_select)
            ]
            logging.info(f"Selected {num_to_select} images: {processed_source}")
        except Exception as e:
            logging.error(f"Error selecting random images from {source_path}: {e}")
            sys.exit(1)
    else:
        logging.info("Source is a directory, processing all contents.")
        processed_source = str(source_path)  # Pass directory path directly to predict

    return processed_source


def _run_yolo_prediction(
    config: dict,
    processed_source: str | list[str],
    project_dir_str: str,
    exp_name: str,
    computed_output_dir: Path,
):
    """Loads the model, prepares args, and runs YOLO prediction."""
    # Prepare Prediction Arguments for model.predict()
    predict_kwargs = {
        k: v
        for k, v in config.items()
        if k not in ["model", "source", "project", "random_select"] and v is not None
    }
    logging.info(f"Prediction arguments for YOLO.predict(): {predict_kwargs}")

    try:
        logging.info("Loading YOLO model...")
        model = YOLO(config["model"])
        logging.info(f"Starting prediction on source: {processed_source}")

        results = model.predict(
            source=processed_source,
            project=project_dir_str,  # Pass project base dir
            name=exp_name,  # Pass specific experiment name
            stream=False,  # Process all at once
            **predict_kwargs,
        )
        # The actual output dir might have a suffix if exp_name already existed
        logging.info(
            f"Prediction complete. Results saved in project '{project_dir_str}' under an experiment name starting with '{exp_name}'. Check logs for exact path."
        )
        # logging.info(f"Expected output directory: {computed_output_dir}") # Keep for reference
        # results object can be large, avoid logging it directly

    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}", exc_info=True)
        sys.exit(1)


def predict_pipeline(config_path: Path, name_prefix: str, cli_args: argparse.Namespace):
    """Orchestrates the prediction pipeline."""
    # 1. Load base config
    base_config = _load_and_validate_config(config_path)

    # 2. Merge CLI args into config
    final_config = _merge_config_and_args(base_config, cli_args)

    # 3. Validate the final merged config (especially paths)
    _validate_final_config(final_config, config_path)

    # 4. Prepare output directory info (don't create)
    computed_output_dir, project_dir_str, exp_name = _prepare_output_directory(
        final_config["project"], name_prefix
    )

    # 5. Process source path (including random selection)
    random_select_val = final_config.get("random_select")
    processed_source = _process_source_path(final_config["source"], random_select_val)

    # 6. Run YOLO prediction
    _run_yolo_prediction(
        final_config, processed_source, project_dir_str, exp_name, computed_output_dir
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

    # Arguments to override config file (set default=None to detect if user provided them)
    parser.add_argument(
        "--model", type=str, default=None, help="Override model path from config."
    )
    parser.add_argument(
        "--source", type=str, default=None, help="Override source path from config."
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Override project directory from config.",
    )
    parser.add_argument(
        "--conf", type=float, default=None, help="Override confidence threshold."
    )
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size.")
    parser.add_argument(
        "--iou", type=float, default=None, help="Override IoU threshold for NMS."
    )
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
    # Add other Ultralytics predict() args here if needed, e.g., device, augment, etc.
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
