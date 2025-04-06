"""
Common utility functions for YOLOv11 prediction scripts.
"""

import logging
import os  # Added for expandvars
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv  # Added for env loading
from ultralytics import YOLO

# Configure logger
logger = logging.getLogger(__name__)


def find_project_root(marker=".git") -> Path:
    """Find the project root directory by searching upwards for a marker."""
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / marker).exists():
            return parent
    logger.warning(
        f"Could not find project root marker '{marker}'. Using current file's directory."
    )
    return current_path.parent


def _load_expand_validate_config(config_path: Path) -> dict:
    """Loads YAML config, expands env vars, and performs basic validation."""
    logger.info(f"Loading configuration from: {config_path}")
    if not config_path.exists():
        logger.error(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)

    # Load .env file from project root
    project_root = find_project_root()
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        logger.info(f"Loading environment variables from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)
    else:
        logger.warning(
            f".env file not found at {dotenv_path}, skipping environment variable loading."
        )

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if not config:
            raise ValueError("Config file is empty or invalid.")

        # Expand environment variables in string values (recursive for nested dicts/lists if needed)
        def expand_vars(item):
            if isinstance(item, dict):
                return {k: expand_vars(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [expand_vars(elem) for elem in item]
            elif isinstance(item, str):
                return os.path.expandvars(item)
            else:
                return item

        expanded_config = expand_vars(config)
        # logger.info(f"Loaded and expanded config: {expanded_config}") # Can be verbose
        return expanded_config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Error in config file {config_path}: {e}")
        sys.exit(1)


def _merge_config_and_args(config: dict, args: object) -> dict:
    """Merges limited command-line arguments into the loaded config."""
    merged_config = config.copy()
    # Define the limited set of keys allowed for CLI override
    allowed_override_keys = {"device", "save", "show"}

    cli_overrides = {
        k: v for k, v in vars(args).items() if k in allowed_override_keys and v is not None
    }

    if cli_overrides:
        logging.info(f"Applying command-line overrides: {cli_overrides}")
        merged_config.update(cli_overrides)

    return merged_config


def _validate_final_config(final_config: dict, config_path: Path):
    """Validates required keys and paths in the final merged configuration."""
    # Validate required config keys (model and project now)
    required_keys = ["model", "project"]
    for key in required_keys:
        if key not in final_config or not final_config[key]:
            logging.error(
                f"Error: Missing or empty required key '{key}' in config '{config_path}'."
            )
            sys.exit(1)

    # Validate model path
    model_path = Path(final_config["model"])

    # Check model path (allow Ultralytics names for download)
    if not model_path.exists() and not str(model_path).endswith((".pt", ".onnx")):
        logging.warning(
            f"Model path '{model_path}' not found locally. "
            f"Assuming it's an Ultralytics model name for download."
        )
    elif model_path.exists() and not model_path.is_file():
        logging.error(f"Error: Specified model path '{model_path}' exists but is not a file.")
        sys.exit(1)

    # Validate project path is a directory (or can be created)
    project_path = Path(final_config["project"])
    if project_path.exists() and not project_path.is_dir():
        logging.error(
            f"Error: Specified project path '{project_path}' exists but is not a directory."
        )
        sys.exit(1)
    # We don't check if source exists here, as it's constructed later

    logging.info(f"Using final model: {final_config['model']}")
    logging.info(f"Using final project directory: {project_path}")


def _prepare_output_directory(project_dir_str: str, name: str) -> tuple[str, str]:
    """Computes final output dir path with timestamp, returns project path and exp name."""
    project_dir = Path(project_dir_str)
    # Append timestamp to the provided name
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    exp_name = f"{name}_{timestamp}"
    computed_output_dir = project_dir / exp_name
    logging.info(f"Computed target output directory: {computed_output_dir}")
    # Ensure parent project directory exists
    project_dir.mkdir(parents=True, exist_ok=True)
    # Return project_dir string and the timestamped experiment name
    return str(project_dir), exp_name


# _process_source_path function removed as source is constructed in main script


def _run_yolo_prediction(
    config: dict,
    source: str | list[str],  # Renamed from processed_source
    project_dir_str: str,
    exp_name: str,
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
        "exist_ok",  # Pass exist_ok if set
    }

    # Add task-specific parameters
    if task_type == "segment":
        # Valid segmentation-specific parameters from Ultralytics docs/defaults
        valid_keys.update({"retina_masks", "show_boxes"})  # Changed boxes to show_boxes

    # Filter config to only include valid keys
    predict_kwargs = {k: v for k, v in config.items() if k in valid_keys and v is not None}

    # Add default values for segmentation if needed and not overridden
    # These are often True by default in YOLO, but being explicit can help
    # if task_type == "segment":
    #     if "retina_masks" not in predict_kwargs:
    #         predict_kwargs["retina_masks"] = True
    #     if "boxes" not in predict_kwargs:
    #         predict_kwargs["boxes"] = True

    logger.info(f"Prediction arguments for YOLO.predict(): {predict_kwargs}")

    try:
        logger.info(f"Loading YOLO {task_type} model: {config['model']}...")
        model = YOLO(config["model"])

        logger.info(
            f"Starting {task_type} prediction on source..."
        )  # Don't log source list if long
        # computed_output_dir is no longer passed here, YOLO handles project/name
        results = model.predict(
            source=source,
            project=project_dir_str,
            name=exp_name,
            stream=False,  # Process all for statistics
            **predict_kwargs,
        )
        # The exact output path is determined by YOLO based on project/name/exist_ok
        output_path_base = Path(project_dir_str) / exp_name
        logger.info(
            f"{task_type.capitalize()} prediction complete. "
            f"Results likely saved under: {output_path_base}"
        )
        # Return results for statistics calculation
        return results

    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}", exc_info=True)
        # Re-raise or return None/empty list based on desired handling in calling script
        # sys.exit(1) # Avoid exiting from utility function
        return None  # Indicate failure
