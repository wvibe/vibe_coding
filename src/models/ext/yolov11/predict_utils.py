"""
Common utility functions for YOLOv11 prediction scripts.
"""

import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO

# Configure logger
logger = logging.getLogger(__name__)


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


def _merge_config_and_args(config: dict, args) -> dict:
    """Merges command-line arguments into the loaded config, overriding if specified."""
    merged_config = config.copy()
    cli_overrides = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k not in ["config", "name_prefix"]  # Exclude script-control args
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
        logging.error(f"Error: Specified model path '{model_path}' exists but is not a file.")
        sys.exit(1)

    # Check source path
    if not source_path.exists():
        logging.error(f"Error: Source path '{source_path}' does not exist.")
        sys.exit(1)

    logging.info(f"Using final model: {final_config['model']}")
    logging.info(f"Using final source: {source_path}")
    logging.info(f"Using final project directory: {Path(final_config['project'])}")


def _prepare_output_directory(project_dir_str: str, name_prefix: str) -> tuple[Path, str, str]:
    """Computes the timestamped output directory name and returns paths/names."""
    project_dir = Path(project_dir_str)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    exp_name = f"{name_prefix}_{timestamp}"
    computed_output_dir = project_dir / exp_name
    logging.info(f"Computed target output directory: {computed_output_dir}")
    return computed_output_dir, str(project_dir), exp_name


def _process_source_path(source_path_str: str, random_select: int | None) -> str | list[str]:
    """Handles file/directory source and random selection."""
    source_path = Path(source_path_str)
    source_is_dir = source_path.is_dir()
    processed_source = None

    if not source_is_dir:
        logging.info("Source is a single file.")
        processed_source = str(source_path)
    elif random_select and random_select > 0:
        logging.info(f"Source is a directory, selecting {random_select} random image(s).")
        try:
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
            all_images = [p for p in source_path.glob("*") if p.suffix.lower() in image_extensions]
            if not all_images:
                logging.error(f"No images found in directory: {source_path}")
                sys.exit(1)

            num_found = len(all_images)
            num_to_select = min(random_select, num_found)

            if num_found < random_select:
                logging.warning(
                    f"Requested {random_select} images, but only found {num_found}. Using all found images."
                )

            processed_source = [str(p) for p in random.sample(all_images, num_to_select)]
            logging.info(f"Selected {num_to_select} images: {processed_source}")
        except Exception as e:
            logging.error(f"Error selecting random images from {source_path}: {e}")
            sys.exit(1)
    else:
        logging.info("Source is a directory, processing all contents.")
        processed_source = str(source_path)

    return processed_source


def _run_yolo_prediction(
    config: dict,
    processed_source: str | list[str],
    project_dir_str: str,
    exp_name: str,
    computed_output_dir: Path,
    task_type: str = "detect",  # "detect" or "segment"
):
    """Loads the model, prepares args, and runs YOLO prediction."""
    # Filter to only include valid YOLO arguments
    # Base valid parameters for all prediction types
    valid_keys = {
        "conf",
        "iou",
        "imgsz",
        "device",
        "max_det",
        "classes",
        "save_txt",
        "save_conf",
        "save",
        "save_crop",
        "half",
        "visualize",
        "show",
        "stream",
        "verbose",
        "view_img",
    }

    # Add task-specific parameters
    if task_type == "segment":
        # Valid segmentation-specific parameters from YOLOv11
        valid_keys.update({"retina_masks", "overlap_mask"})

    # Filter config to only include valid keys
    predict_kwargs = {k: v for k, v in config.items() if k in valid_keys and v is not None}

    # Add default values for segmentation if needed
    if task_type == "segment":
        if "retina_masks" not in predict_kwargs:
            predict_kwargs["retina_masks"] = True
        if "overlap_mask" not in predict_kwargs:
            predict_kwargs["overlap_mask"] = True

    logger.info(f"Prediction arguments for YOLO.predict(): {predict_kwargs}")

    try:
        logger.info(f"Loading YOLO {task_type} model...")
        model = YOLO(config["model"])
        logger.info(f"Starting {task_type} prediction on source: {processed_source}")

        results = model.predict(
            source=processed_source,
            project=project_dir_str,
            name=exp_name,
            stream=False,
            **predict_kwargs,
        )
        logger.info(
            f"{task_type.capitalize()} prediction complete. Results saved in project '{project_dir_str}' under an experiment name starting with '{exp_name}'. Check logs for exact path."
        )

    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}", exc_info=True)
        sys.exit(1)
