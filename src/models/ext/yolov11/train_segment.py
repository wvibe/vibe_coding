"""
Main script for initiating YOLOv11 SEGMENTATION model training (finetuning or from scratch)
based on a YAML configuration file.

Mirrors the structure and capabilities of the YOLOv11 detection training script.
"""

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO
from ultralytics.utils import RANK, SETTINGS

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _setup_project_environment() -> Path:
    """Setup the project environment and return the project root.

    Returns:
        Path: Project root directory path.
    """
    # Inline get_project_root()
    project_root = Path(__file__).resolve().parents[4]
    logging.info(f"Project Root: {project_root}")

    # Inline setup_environment()
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f".env loaded from: {dotenv_path}")
    else:
        logging.info(".env file not found, proceeding without it.")

    return project_root


def _parse_config_yaml(config_path: Path) -> dict:
    """Load configuration from a YAML file.

    Args:
        config_path (Path): Path to the YAML configuration file.
    Returns:
        dict: Configuration dictionary loaded from the YAML file.
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file is empty or invalid.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    logging.info(f"Loading configuration from: {config_path}")
    if not config_path.is_file():
        logging.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if not config:
            raise ValueError("Config file is empty or invalid.")
        logging.info(f"Configuration loaded successfully from: {config_path}")
        return config
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred loading config: {e}")
        raise


def _load_training_config(args: argparse.Namespace, project_root: Path) -> tuple[dict, Path]:
    """Load training configurations.

    Args:
        args (argparse.Namespace): Command-line arguments.
        project_root (Path): Project root directory path.
    Returns:
        tuple[dict, Path]: Training configuration dictionary and absolute path to data config file.
    Raises:
        FileNotFoundError: If configuration files are not found.
        ValueError: If configurations are invalid.
        yaml.YAMLError: If there is an error parsing YAML files.
    """
    config_path_abs = (project_root / args.config).resolve()
    train_config = _parse_config_yaml(config_path_abs)

    # Verify project path is specified in config
    if "project" not in train_config:
        raise ValueError("Missing 'project' key in the main training configuration.")

    # determine the data config path
    relative_data_config_path = train_config.get("data")
    if not relative_data_config_path:
        raise ValueError("Missing 'data' key in the main training configuration.")
    data_config_path = (project_root / relative_data_config_path).resolve()
    if not data_config_path.is_file():
        raise FileNotFoundError(
            f"Data config file specified in main config not found: {data_config_path}"
        )
    logging.info(f"Using data config file: {data_config_path}")

    return train_config, data_config_path


def _parse_tracker_log(tracker_file: Path) -> dict | None:
    """Parse the training run tracker log file.

    Args:
        tracker_file (Path): Path to the tracker log file.

    Returns:
        dict | None: Dictionary with parsed tracking data if successful, None otherwise.
    """
    try:
        if not tracker_file.exists():
            return None

        with open(tracker_file, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)  # Read header row
            data = next(reader)  # Read data row

            # Convert to dictionary
            log_data = dict(zip(headers, data))
            return log_data
    except (FileNotFoundError, csv.Error, IndexError, StopIteration) as e:
        logging.warning(f"Error parsing tracker log {tracker_file}: {e}")
        return None


def log_run_tracker(tracker_file_path: Path, base_name: str, train_kwargs: dict) -> None:
    """Write run information to the tracker file for potential future resuming.

    Args:
        tracker_file_path (Path): Path to the tracker log file.
        base_name (str): Base name of the run (from args.name).
        train_kwargs (dict): Training arguments dictionary for model.train().
    """
    try:
        # Extract required information from train_kwargs
        project_path = train_kwargs["project"]
        actual_run_name = train_kwargs["name"]
        config_epochs = train_kwargs["epochs"]
        timestamp = datetime.now().isoformat()

        # Define columns and data
        columns = [
            "project_path",
            "base_name",
            "actual_run_name",
            "config_epochs",
            "log_timestamp",
        ]
        data = [
            project_path,
            base_name,
            actual_run_name,
            config_epochs,
            timestamp,
        ]

        # Write to tracker file (overwrite)
        with open(tracker_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerow(data)

        logging.info(f"Run tracker information logged to {tracker_file_path}")
    except Exception as e:
        logging.error(f"Failed to write run tracker log to {tracker_file_path}: {e}")


def _load_model(model_path: str) -> YOLO:
    """Load the YOLO model from the specified path.

    Args:
        model_path (str): Path to the model file or identifier.
    Returns:
        YOLO: Loaded YOLO model instance.
    Raises:
        Exception: If there is an error loading the model.
    """
    try:
        # YOLO class should automatically handle segmentation models based on name/architecture
        model = YOLO(model_path)
        logging.info(f"Model '{model_path}' loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model '{model_path}': {e}", exc_info=True)
        raise  # Re-raise after logging


def prepare_train_kwargs(
    train_config: dict,
    run_name: str,
    resume_flag: bool,
    project_path: str,
    data_config_path: Path,
) -> dict:
    """Prepare the keyword arguments for the model.train() call.

    Args:
        train_config (dict): Training configuration dictionary.
        run_name (str): Name of the training run.
        resume_flag (bool): Flag indicating if training is being resumed.
        project_path (str): Project directory path for saving runs.
        data_config_path (Path): Absolute path to the data configuration file.
    Returns:
        dict: Dictionary of training arguments for model.train().
    """
    # Filter main_config to only include valid YOLO train() arguments
    # Define the set of valid training arguments recognized by the Ultralytics trainer
    # Sorted alphabetically for readability.
    valid_train_args = {
        "augment",
        "batch",
        "cache",
        "close_mosaic",
        "copy_paste",
        "cos_lr",
        "csv_root",
        "degrees",
        "deterministic",
        "device",
        "dropout",
        "epochs",
        "exist_ok",
        "fliplr",
        "flipud",
        "fraction",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "imgsz",
        "lr0",
        "lrf",
        "mask_ratio",
        "mixup",
        "momentum",
        "optimizer",
        "overlap_mask",
        "patience",
        "perspective",
        "pretrained",
        "save_period",
        "scale",
        "seed",
        "shear",
        "translate",
        "warmup_epochs",
        "wdir_root",
        "weight_decay",
        "workers",
    }

    # The filtering logic remains the same
    train_kwargs = {k: v for k, v in train_config.items() if k in valid_train_args}

    # Add/override arguments from orchestration logic
    train_kwargs["project"] = project_path
    train_kwargs["name"] = run_name
    train_kwargs["resume"] = resume_flag
    train_kwargs["data"] = str(data_config_path)

    # Handle potential None/empty string for device (YOLO expects None or str)
    if "device" in train_kwargs and not train_kwargs["device"]:
        train_kwargs["device"] = None

    logging.info("--- Training Arguments --- ")
    # Sort for consistent printing
    for key, val in sorted(train_kwargs.items()):
        logging.info(f"  {key}: {val}")
    logging.info("------------------------")

    return train_kwargs


def execute_training(
    model: YOLO,
    train_kwargs: dict,
) -> tuple[Path | None, bool]:
    """Execute the training process and capture results.

    Args:
        model (YOLO): Loaded YOLO model instance.
        train_kwargs (dict): Training arguments for model.train().
    Returns:
        tuple[Path | None, bool]: Final output directory (if available) and training success flag.
    """
    logging.info("Starting segmentation training...")
    final_output_dir = None
    training_successful = False

    try:
        # The train method should work for segmentation task type based on the loaded model
        model.train(**train_kwargs)
        logging.info("Segmentation training finished successfully.")
        training_successful = True

        # Capture the actual save directory after training
        if hasattr(model, "trainer") and hasattr(model.trainer, "save_dir"):
            final_output_dir = Path(model.trainer.save_dir)
            logging.info(f"Results saved to: {final_output_dir}")
        else:
            logging.warning("Could not determine final save directory from trainer.")
    except Exception as e:
        logging.error(f"Error during segmentation training: {e}", exc_info=True)

    if not training_successful:
        logging.error("Segmentation training did not complete successfully.")

    return final_output_dir, training_successful


def _determine_run_parameters(
    args: argparse.Namespace, train_config: dict, project_dir: Path, tracker_file: Path
) -> tuple[bool, str | None, str]:
    """Determine run parameters based on args, config, and tracker log.

    Args:
        args (argparse.Namespace): Command-line arguments.
        train_config (dict): Loaded training configuration.
        project_dir (Path): Project directory path.
        tracker_file (Path): Path to the tracker log file.

    Returns:
        tuple[bool, str | None, str]: (resume_flag, model_to_load, run_name)
    """
    # Initialize default values for a new run
    resume_flag = False
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.name}_{timestamp}"

    # Default model path is always from the config unless overridden by --model
    if args.model:
        logging.info(
            f"Model specified via --model={args.model}, starting new training (auto-resume disabled)"
        )
        model_to_load = args.model
        return resume_flag, model_to_load, run_name

    model_to_load = train_config.get("model")
    if not model_to_load:
        raise ValueError(
            "Missing 'model' key in configuration for new run or not provided via --model."
        )

    # Only attempt to resume if auto-resume is enabled and no model override
    if args.auto_resume:
        logging.info(f"Auto-resume enabled, checking for previous run with name '{args.name}'")
        log_data = _parse_tracker_log(tracker_file)

        if log_data:
            logged_project = log_data.get("project_path")
            logged_base_name = log_data.get("base_name")

            if logged_project == str(project_dir) and logged_base_name == args.name:
                logging.info("Found matching run in log, will attempt to resume training")
                resume_flag = True
                run_name = log_data.get("actual_run_name")

                # Verify the run directory exists
                run_dir = project_dir / run_name
                if not run_dir.exists():
                    logging.warning(
                        f"Run directory {run_dir} from tracker log does not exist. Starting new run."
                    )
                    resume_flag = False
                    run_name = f"{args.name}_{timestamp}"
                else:
                    logging.info(f"Verified run directory exists: {run_dir}")
                    # Still use model from config, trainer will handle loading last.pt with resume=True
                    return resume_flag, model_to_load, run_name
            else:
                logging.info(
                    "Last run log exists but doesn't match current project/name. Starting new run."
                )
        else:
            logging.info("No last run log found or couldn't parse it. Starting new run.")
    else:
        logging.info("Auto-resume disabled. Starting new run.")

    # At this point, we're doing a new run with the model from config
    return resume_flag, model_to_load, run_name


# --- Training Pipeline Steps --- #


def setup_environment(project_root: Path) -> None:
    """Load environment variables from .env file if available.

    Args:
        project_root (Path): Project root directory path.
    """
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f".env loaded from: {dotenv_path}")
    else:
        logging.info(".env file not found, proceeding without it.")


def run_training_pipeline(args: argparse.Namespace):
    """Orchestrate the steps for loading config and running training.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Step 1: Validate arguments - name is required
    if not args.name:
        raise ValueError("--name is required for all training runs")

    # Step 2: Setup project environment
    project_root = _setup_project_environment()

    # Step 3: Load configurations
    try:
        train_config, data_config_path = _load_training_config(args, project_root)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        logging.error(f"Configuration error: {e}")
        sys.exit(1)

    # Step 4: Determine project path from config (required)
    project_path = train_config.get("project")
    if not project_path:
        raise ValueError("Project path must be specified in the config file ('project' key).")
    project_dir = Path(project_path)
    logging.info(f"Using project directory: {project_path}")

    # Create project directory if it doesn't exist
    if RANK in {-1, 0} and not project_dir.exists():
        project_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created project directory: {project_path}")

    # Define tracker file path (used both for reading and writing)
    tracker_file_path = project_dir / "last_run.log"

    # Step 5: Determine run parameters using the helper function
    try:
        resume_flag, model_to_load, run_name = _determine_run_parameters(
            args, train_config, project_dir, tracker_file_path
        )
    except ValueError as e:
        logging.error(f"Failed to determine run parameters: {e}")
        sys.exit(1)

    # Step 6: Setup WandB to True for YOLO settings if wandb_dir is provided
    if args.wandb_dir:
        SETTINGS["wandb"] = True
        logging.info(f"WandB enabled. Using directory: {args.wandb_dir}")

    # Step 7: Setup model
    try:
        model = _load_model(model_to_load)
    except Exception:
        # Error already logged in _load_model
        sys.exit(1)

    # Step 8: Prepare training arguments
    train_kwargs = prepare_train_kwargs(
        train_config,
        run_name,
        resume_flag,
        project_path,
        data_config_path,
    )

    # Step 9: Log run information to tracker file before training starts
    log_run_tracker(tracker_file_path, args.name, train_kwargs)

    # Step 10: Execute training
    execute_training(model, train_kwargs)

    logging.info("Script finished.")


def main():
    """Parse command-line arguments and initiate the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 segmentation models using configuration files."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Path to the main segmentation training configuration YAML file "
            "(relative to project root). It must contain 'project' and 'data' keys."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Base name for the training run. A timestamp will be appended for new runs.",
    )
    parser.add_argument(
        "--wandb-dir",
        type=str,
        default="wandb",
        help=(
            "Path to the root WandB directory (e.g., 'wandb'). Used to automatically find "
            "the run ID when resuming."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Path to a model file (e.g., last.pt) to use for a new training job. "
            "Overrides the 'model' key in the configuration file and disables auto-resume."
        ),
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        default=False,
        help=(
            "Automatically resume training from the last run with the same base name "
            "if a checkpoint exists. Disabled if --model is specified."
        ),
    )

    args = parser.parse_args()

    run_training_pipeline(args)


if __name__ == "__main__":
    # Example Usage (New training):
    # python src/models/ext/yolov11/train_segment.py \
    #     --config configs/yolov11/finetune_segment_voc.yaml \
    #     --name voc11_seg_finetune_run1
    #
    # Example Usage (With custom model weights):
    # python src/models/ext/yolov11/train_segment.py \
    #     --config configs/yolov11/finetune_segment_voc.yaml \
    #     --name voc11_seg_finetune_run1 \
    #     --model path/to/custom/weights.pt
    #
    # Note: Auto-resume is enabled by default and can be configured in the YAML file.
    # If a previous run with the same base name exists, it will attempt to resume
    # from the last checkpoint.

    main()
