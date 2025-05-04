"""
Main script for initiating YOLOv11 SEGMENTATION model training (finetuning or from scratch)
based on a YAML configuration file.

Mirrors the structure and capabilities of the YOLOv11 detection training script.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

from utils.logging.log_finder import find_wandb_run_id

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


def _determine_run_params(
    args: argparse.Namespace, train_config: dict, project_root: Path
) -> tuple:
    """Determine model path, run name, resume flag, and attempt to find WandB ID on resume.

    Args:
        args (argparse.Namespace): Command-line arguments.
        train_config (dict): Training configuration dictionary.
        project_root (Path): Project root directory path.
    Returns:
        tuple: (model_to_load, name_to_use, resume_flag, wandb_id_to_use)
    Raises:
        FileNotFoundError: If resume directory or checkpoint file is not found.
        ValueError: If 'model' key is missing in configuration for a new run.
    """
    model_to_load = None
    name_to_use = args.name  # Base name, might be modified
    resume_flag = False
    wandb_id_to_use = None  # Default to None

    if args.resume_with:
        logging.info(f"Attempting to resume training from: {args.resume_with}")
        resume_dir = (project_root / args.resume_with).resolve()

        if not resume_dir.is_dir():
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")

        checkpoint_path_for_resume = resume_dir / "weights" / "last.pt"
        if not checkpoint_path_for_resume.is_file():
            raise FileNotFoundError(
                f"Checkpoint 'last.pt' not found in resume directory: {checkpoint_path_for_resume}"
            )

        model_to_load = str(checkpoint_path_for_resume)
        name_to_use = resume_dir.name  # Use the exact name of the folder being resumed
        resume_flag = True
        logging.info(f"Resuming with checkpoint: {model_to_load}")
        logging.info(f"Run name set to resumed directory: {name_to_use}")

        # Warning if CLI --name differs significantly from resumed name
        if args.name:
            # Check if provided name is a prefix of the resumed name (ignoring timestamp)
            base_resumed_name = (
                "_".join(name_to_use.split("_")[:-1]) if "_" in name_to_use else name_to_use
            )
            if args.name != base_resumed_name:
                logging.warning(
                    f"Provided --name '{args.name}' differs from resumed run base "
                    f"'{base_resumed_name}'. Using full resumed name '{name_to_use}'."
                )
            # If it matches the base name, no warning needed as timestamp is the difference

        # --- Attempt to find WandB ID automatically --- #
        logging.info(f"Attempting to find corresponding WandB run ID in: {args.wandb_dir}")
        wandb_id_to_use = find_wandb_run_id(str(resume_dir), args.wandb_dir)
        if wandb_id_to_use:
            logging.info(
                f"Automatically found WandB run ID: {wandb_id_to_use}. Will attempt to resume."
            )
            os.environ["WANDB_RESUME"] = "allow"
            os.environ["WANDB_RUN_ID"] = wandb_id_to_use
        else:
            logging.warning(
                f"Could not automatically find a matching WandB run ID for {name_to_use} "
                f"in {args.wandb_dir}. WandB (if enabled) will start as a new run."
            )
        # --- End WandB ID Lookup --- #
    else:
        # New run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_to_use = f"{args.name}_{timestamp}"
        model_to_load = args.model if args.model else train_config.get("model")
        resume_flag = False  # Explicitly false
        logging.info(f"Starting new run with name: {name_to_use}")
        if not model_to_load:
            raise ValueError(
                "Missing 'model' key in configuration for new run or not provided via --model."
            )

    return model_to_load, name_to_use, resume_flag


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
    name: str,
    resume: bool,
    effective_project_path: str,
    absolute_data_config_path: Path,
) -> dict:
    """Prepare the keyword arguments for the model.train() call.

    Args:
        train_config (dict): Training configuration dictionary.
        name (str): Name of the training run.
        resume (bool): Flag indicating if training is being resumed.
        effective_project_path (str): Project directory path for saving runs.
        absolute_data_config_path (Path): Absolute path to the data configuration file.
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
    train_kwargs["project"] = effective_project_path
    train_kwargs["name"] = name
    train_kwargs["resume"] = resume
    train_kwargs["data"] = str(absolute_data_config_path)

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
    effective_project_path: str,
    name_to_use: str,
    project_root: Path,
) -> tuple[Path | None, bool]:
    """Execute the training process and capture results.

    Args:
        model (YOLO): Loaded YOLO model instance.
        train_kwargs (dict): Training arguments for model.train().
        effective_project_path (str): Project directory path for saving runs.
        name_to_use (str): Name of the training run.
        project_root (Path): Project root directory path.
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
            fallback_dir = project_root / effective_project_path / name_to_use
            logging.info(f"Expected results directory: {fallback_dir}")
    except Exception as e:
        logging.error(f"Error during segmentation training: {e}", exc_info=True)

    if not training_successful:
        logging.error("Segmentation training did not complete successfully.")

    return final_output_dir, training_successful


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
    # Step 1: Validate arguments
    if not args.resume_with and not args.name:
        raise ValueError("--name is required when not using --resume-with")

    # Step 2: Setup project environment
    project_root = _setup_project_environment()

    # Step 3: Load configurations
    try:
        train_config, data_config_path = _load_training_config(args, project_root)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        logging.error(f"Configuration error: {e}")
        sys.exit(1)

    # Step 4: Determine run parameters (includes auto WandB ID lookup)
    try:
        model_to_load, name_to_use, resume_flag = _determine_run_params(
            args, train_config, project_root
        )
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Failed to determine run parameters: {e}")
        sys.exit(1)

    # Step 5: Determine project path (Fail if not specified)
    effective_project_path = (
        args.project if args.project is not None else train_config.get("project")
    )
    if not effective_project_path:
        raise ValueError(
            "Project path must be specified either via --project argument or in the config file."
        )
    logging.info(f"Using project directory: {effective_project_path}")

    # Step 6: Setup WandB to True for YOLO settings if wandb_dir is provided
    if args.wandb_dir:
        SETTINGS["wandb"] = True
        logging.info(f"WandB enabled. Using directory: {args.wandb_dir}")

    # Step 7: Setup model (Directly call _load_model)
    try:
        model = _load_model(model_to_load)
    except Exception:
        # Error already logged in _load_model
        sys.exit(1)

    # Step 8: Prepare training arguments
    train_kwargs = prepare_train_kwargs(
        train_config, name_to_use, resume_flag, effective_project_path, data_config_path
    )

    # Step 9: Execute training
    execute_training(model, train_kwargs, effective_project_path, name_to_use, project_root)

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
            "(relative to project root). It must contain a 'data' key "
            "pointing to the dataset-specific config."
        ),
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help=(
            "Override the base directory to save runs. If None, uses 'project' from config "
            "or default ('runs/train/segment')."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default=None,
        help="Base name for the training run. A timestamp will be appended for new runs. "
        "Required if not using --resume-with.",
    )
    parser.add_argument(
        "--resume-with",
        type=str,
        default=None,
        help=(
            "Path to the exact training run directory "
            "(e.g., runs/train/segment/run_YYMMDD_HHMMSS) to resume from. "
            "Overrides --name logic for naming and loads last.pt."
        ),
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
            "Overrides the 'model' key in the configuration file."
        ),
    )

    args = parser.parse_args()

    run_training_pipeline(args)


if __name__ == "__main__":
    # Example Usage (Fine-tuning):
    # python src/models/ext/yolov11/train_segment.py \
    #     --config configs/yolov11/finetune_segment_voc.yaml \
    #     --name voc11_seg_finetune_run1
    #
    # Example Usage (Resuming):
    # python src/models/ext/yolov11/train_segment.py \
    #     --config configs/yolov11/finetune_segment_voc.yaml \
    #     --resume-with runs/train/segment/voc11_seg_finetune_run1_20240101_000000 \
    #     --name voc11_seg_finetune_run1
    #
    # Example Usage (With Custom Weights Directory):
    # python src/models/ext/yolov11/train_segment.py \
    #     --config configs/yolov11/finetune_segment_cov_segm.yaml \
    #     --name cov_segm_ft_yolo11l_260k \
    #     --weights-dir /path/to/nfs/mount

    main()
