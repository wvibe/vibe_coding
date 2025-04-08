"""
Main script for initiating YOLOv11 detection model training (finetuning or from scratch)
based on a YAML configuration file.

Mirrors the structure and capabilities of the YOLOv8 training script.
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

# --- Vibe Imports --- #
# Assuming src is in PYTHONPATH or handled by execution environment
try:
    from utils.common.wandb_utils import find_wandb_run_id
except ImportError:
    # Fallback if run directly and utils path needs adjustment
    project_root_for_import = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(project_root_for_import / "src"))
    try:
        from utils.common.wandb_utils import find_wandb_run_id
    except ImportError as e:
        logging.error("Could not import find_wandb_run_id. Ensure src is in PYTHONPATH.")
        raise e
# --- End Vibe Imports --- #

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_project_root() -> Path:
    """Find the project root directory relative to this script."""
    # Assumes script is in project_root/src/models/ext/yolov11
    return Path(__file__).resolve().parents[4]


def load_config(config_path: Path) -> dict:
    """Load training configuration from a YAML file."""
    logging.info(f"Loading configuration from: {config_path}")
    if not config_path.is_file():
        logging.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if not config:
            raise ValueError("Config file is empty or invalid.")
        logging.info("Configuration loaded successfully.")
        return config
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred loading config: {e}")
        raise


def _validate_and_get_data_config_path(main_config: dict, project_root: Path) -> Path:
    """Validates and resolves the data config path from the main config."""
    relative_data_config_path = main_config.get("data")
    if not relative_data_config_path:
        raise ValueError("Missing 'data' key in the main training configuration.")
    absolute_data_config_path = (project_root / relative_data_config_path).resolve()
    if not absolute_data_config_path.is_file():
        raise FileNotFoundError(
            f"Data config file specified in main config not found: {absolute_data_config_path}"
        )
    logging.info(f"Using data config file: {absolute_data_config_path}")
    return absolute_data_config_path


def _determine_run_params(args: argparse.Namespace, main_config: dict, project_root: Path) -> tuple:
    """Determines model path, run name, resume flag, and attempts to find wandb ID on resume."""
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
        model_to_load = main_config.get("model")
        resume_flag = False  # Explicitly false
        wandb_id_to_use = None  # New runs don't automatically reuse IDs
        logging.info(f"Starting new run with name: {name_to_use}")
        if not model_to_load:
            raise ValueError("Missing 'model' key in configuration for new run.")

    return model_to_load, name_to_use, resume_flag, wandb_id_to_use


def _setup_wandb(wandb_id: str | None, resume_flag: bool):
    """Sets environment variables for WandB based on provided ID."""
    if wandb_id:
        if resume_flag:
            logging.info(f"Setting up WandB to resume run ID: {wandb_id}")
        else:
            # This case is less likely now as we don't automatically assign IDs to new runs
            logging.info(f"Setting up WandB with provided run ID: {wandb_id}")
        os.environ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_RUN_ID"] = wandb_id
    # else: Let Ultralytics handle default WandB initialization/behavior


def _load_model(model_path: str) -> YOLO:
    """Loads the YOLO model."""
    try:
        model = YOLO(model_path)
        logging.info(f"Model '{model_path}' loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model '{model_path}': {e}", exc_info=True)
        raise  # Re-raise after logging


def prepare_train_kwargs(
    main_config: dict,
    name: str,
    resume: bool,
    effective_project_path: str,
    absolute_data_config_path: Path,
) -> dict:
    """Prepares the keyword arguments for the model.train() call."""
    # Filter main_config to only include valid YOLO train() arguments
    valid_train_args = {
        "epochs",
        "imgsz",
        "batch",
        "workers",
        "optimizer",
        "lr0",
        "lrf",
        "close_mosaic",
        "device",
        "pretrained",
        "save_period",
        "weight_decay",
        "momentum",
        "warmup_epochs",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "cos_lr",
        "patience",
        "seed",
        "deterministic",
        "exist_ok",
        # Add more as needed based on configs
    }
    train_kwargs = {k: v for k, v in main_config.items() if k in valid_train_args}

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


def run_training_pipeline(args: argparse.Namespace):
    """Orchestrates the steps for loading config and running training."""
    project_root = get_project_root()
    logging.info(f"Project Root: {project_root}")

    # --- Load .env --- #
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f".env loaded from: {dotenv_path}")
    else:
        logging.info(".env file not found, proceeding without it.")

    # --- Load and Validate Configurations --- #
    try:
        config_path_abs = (project_root / args.config).resolve()
        main_config = load_config(config_path_abs)
        absolute_data_config_path = _validate_and_get_data_config_path(main_config, project_root)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        logging.error(f"Configuration error: {e}")
        sys.exit(1)

    # --- Determine Run Parameters (includes auto WandB ID lookup) --- #
    try:
        model_to_load, name_to_use, resume_flag, wandb_id_to_use = _determine_run_params(
            args, main_config, project_root
        )
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Failed to determine run parameters: {e}")
        sys.exit(1)

    # --- Determine Effective Project Path --- #
    default_project = "runs/train/detect"
    effective_project_path = (
        args.project if args.project is not None else main_config.get("project", default_project)
    )
    logging.info(f"Using project directory: {effective_project_path}")

    # --- Setup WandB --- #
    _setup_wandb(wandb_id_to_use, resume_flag)

    # --- Load Model --- #
    try:
        model = _load_model(model_to_load)
    except Exception:
        # Error already logged in _load_model
        sys.exit(1)

    # --- Prepare Training Arguments --- #
    train_kwargs = prepare_train_kwargs(
        main_config, name_to_use, resume_flag, effective_project_path, absolute_data_config_path
    )

    # --- Run Training --- #
    logging.info("Starting training...")
    final_output_dir = None
    training_successful = False
    try:
        model.train(**train_kwargs)
        logging.info("Training finished successfully.")
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
        logging.error(f"Error during training: {e}", exc_info=True)

    if not training_successful:
        logging.error("Training did not complete successfully.")

    logging.info("Script finished.")


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 detection models using configuration files."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Path to the main training configuration YAML file (relative to project root). "
            "It must contain a 'data' key pointing to the dataset-specific config."
        ),
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help=(
            "Override the base directory to save runs. If None, uses 'project' from config "
            "or default."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Base name for the training run. A timestamp will be appended for new runs.",
    )
    parser.add_argument(
        "--resume_with",
        type=str,
        default=None,
        help=(
            "Path to the exact training run directory "
            "(e.g., runs/train/detect/run_YYMMDD_HHMMSS) to resume from. "
            "Overrides --name."
        ),
    )
    parser.add_argument(
        "--wandb-dir",  # Changed from --wandb-id
        type=str,
        default="wandb",  # Default WandB directory
        help=(
            "Path to the root WandB directory (e.g., 'wandb'). Used to automatically find "
            "the run ID when resuming."
        ),
    )

    args = parser.parse_args()
    run_training_pipeline(args)


if __name__ == "__main__":
    main()
