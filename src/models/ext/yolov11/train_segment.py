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

# --- Vibe Imports --- #
# Assuming src is in PYTHONPATH or handled by execution environment
try:
    from utils.logging.log_finder import find_wandb_run_id
except ImportError:
    # Fallback if run directly and utils path needs adjustment
    project_root_for_import = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(project_root_for_import / "src"))
    try:
        from utils.logging.log_finder import find_wandb_run_id
    except ImportError as e:
        logging.error("Could not import find_wandb_run_id. Ensure src is in PYTHONPATH.")
        raise e
# --- End Vibe Imports --- #


# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_project_root() -> Path:
    """Find the project root directory relative to this script.

    Assumes the script is located in project_root/src/models/ext/yolov11.
    Returns:
        Path: The absolute path to the project root directory.
    """
    return Path(__file__).resolve().parents[4]


def load_config(config_path: Path) -> dict:
    """Load training configuration from a YAML file.

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
        logging.info("Configuration loaded successfully.")
        return config
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred loading config: {e}")
        raise


def _validate_and_get_data_config_path(main_config: dict, project_root: Path) -> Path:
    """Validate and resolve the data config path from the main config.

    Args:
        main_config (dict): Main training configuration dictionary.
        project_root (Path): Project root directory path.
    Returns:
        Path: Absolute path to the data configuration YAML file.
    Raises:
        ValueError: If the 'data' key is missing in the main configuration.
        FileNotFoundError: If the data config file does not exist.
    """
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
    """Determine model path, run name, resume flag, and attempt to find WandB ID on resume.

    Args:
        args (argparse.Namespace): Command-line arguments.
        main_config (dict): Main training configuration dictionary.
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
    """Set environment variables for WandB based on provided ID.

    Args:
        wandb_id (str | None): WandB run ID to resume, if available.
        resume_flag (bool): Flag indicating if training is being resumed.
    """
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
    main_config: dict,
    name: str,
    resume: bool,
    effective_project_path: str,
    absolute_data_config_path: Path,
) -> dict:
    """Prepare the keyword arguments for the model.train() call.

    Args:
        main_config (dict): Main training configuration dictionary.
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
        "batch",
        "close_mosaic",
        "copy_paste",
        "cos_lr",
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
        "weight_decay",
        "workers",
    }

    # The filtering logic remains the same
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


def load_configurations(args: argparse.Namespace, project_root: Path) -> tuple[dict, Path]:
    """Load main and data configurations for training.

    Args:
        args (argparse.Namespace): Command-line arguments.
        project_root (Path): Project root directory path.
    Returns:
        tuple[dict, Path]: Main configuration dictionary and absolute path to data config file.
    Raises:
        FileNotFoundError: If configuration files are not found.
        ValueError: If configurations are invalid.
        yaml.YAMLError: If there is an error parsing YAML files.
    """
    config_path_abs = (project_root / args.config).resolve()
    main_config = load_config(config_path_abs)
    absolute_data_config_path = _validate_and_get_data_config_path(main_config, project_root)
    return main_config, absolute_data_config_path


def determine_project_and_weights_paths(
    args: argparse.Namespace, main_config: dict, name_to_use: str
) -> tuple[str, Path | None]:
    """Determine the effective project path and custom weights directory.

    Args:
        args (argparse.Namespace): Command-line arguments.
        main_config (dict): Main training configuration dictionary.
        name_to_use (str): Name of the training run to use.
    Returns:
        tuple[str, Path | None]: Effective project path and custom weights directory if specified.
    """
    default_project = "runs/train/segment"  # Default base for segmentation runs
    # If resuming, the project path should ideally match the original run's project path.
    # However, Ultralytics might handle this automatically with 'resume=True'.
    # For simplicity here, we use the same logic as detection: CLI override > config > default.
    # If the config being used for resume *differs* from the original run's config in 'project',
    # this could potentially save results to a different base folder than expected,
    # though the resumed run folder itself (`name_to_use`) will be correct.
    effective_project_path = (
        args.project if args.project is not None else main_config.get("project", default_project)
    )
    # Handle custom weights directory if provided
    custom_weights_dir = None
    if args.weights_dir:
        weights_dir = Path(args.weights_dir).resolve()
        if not weights_dir.exists():
            weights_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created weights directory: {weights_dir}")
        else:
            logging.info(f"Using weights directory: {weights_dir}")
        # Construct full weights directory as {weights_dir}/{project}/{name_to_use}/weights
        custom_weights_dir = weights_dir / effective_project_path / name_to_use / "weights"
        if not custom_weights_dir.exists():
            custom_weights_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created full weights directory: {custom_weights_dir}")
        else:
            logging.info(f"Using existing full weights directory: {custom_weights_dir}")
    logging.info(f"Using project directory: {effective_project_path}")
    return effective_project_path, custom_weights_dir


def setup_model_and_callbacks(
    model_to_load: str, name_to_use: str, custom_weights_dir: Path | None
) -> YOLO:
    """Load the YOLO model and register callbacks for custom weights saving.

    Args:
        model_to_load (str): Path or identifier of the model to load.
        name_to_use (str): Name of the training run to use in filenames.
        custom_weights_dir (Path | None): Custom directory for saving weights, if specified.
    Returns:
        YOLO: Loaded YOLO model instance with callbacks registered if applicable.
    Raises:
        Exception: If there is an error loading the model.
    """
    model = _load_model(model_to_load)
    # Register callback for custom weights directory if provided
    if custom_weights_dir:

        def on_model_save_callback(trainer):
            if custom_weights_dir:
                # Get current epoch for filename
                epoch = trainer.epoch if hasattr(trainer, "epoch") else 0
                # Determine if it's a periodic save or standard save (last/best)
                if hasattr(trainer, "last") and trainer.last:
                    last_path = custom_weights_dir / "last.pt"
                    trainer.model.save(last_path)
                    logging.info(f"Saved last model to: {last_path}")
                if (
                    hasattr(trainer, "best")
                    and trainer.best
                    and hasattr(trainer, "best_fitness")
                    and trainer.best_fitness == trainer.fitness
                ):
                    best_path = custom_weights_dir / "best.pt"
                    trainer.model.save(best_path)
                    logging.info(f"Saved best model to: {best_path}")
                # Check if it's a periodic save
                if (
                    hasattr(trainer, "save_period")
                    and trainer.save_period > 0
                    and (epoch % trainer.save_period == 0)
                ):
                    periodic_path = custom_weights_dir / f"epoch_{epoch}.pt"
                    trainer.model.save(periodic_path)
                    logging.info(f"Saved periodic model for epoch {epoch} to: {periodic_path}")

        model.add_callback("on_model_save", on_model_save_callback)
        logging.info(
            f"Registered on_model_save callback for custom weights directory: {custom_weights_dir}"
        )
    return model


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
            # If custom weights dir was used, log it explicitly
            custom_weights_dir_value = getattr(args, "weights_dir", None)
            if custom_weights_dir_value and hasattr(model.trainer, "wdir"):
                logging.info(f"Weights saved to custom directory: {model.trainer.wdir}")
        else:
            logging.warning("Could not determine final save directory from trainer.")
            fallback_dir = project_root / effective_project_path / name_to_use
            logging.info(f"Expected results directory: {fallback_dir}")
    except Exception as e:
        logging.error(f"Error during segmentation training: {e}", exc_info=True)

    if not training_successful:
        logging.error("Segmentation training did not complete successfully.")

    return final_output_dir, training_successful


def run_training_pipeline(args: argparse.Namespace):
    """Orchestrate the steps for loading config and running training.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    project_root = get_project_root()
    logging.info(f"Project Root: {project_root}")

    # Step 1: Setup environment
    setup_environment(project_root)

    # Step 2: Load configurations
    try:
        main_config, absolute_data_config_path = load_configurations(args, project_root)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        logging.error(f"Configuration error: {e}")
        sys.exit(1)

    # Step 3: Determine run parameters (includes auto WandB ID lookup)
    try:
        model_to_load, name_to_use, resume_flag, wandb_id_to_use = _determine_run_params(
            args, main_config, project_root
        )
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Failed to determine run parameters: {e}")
        sys.exit(1)

    # Step 4: Determine project and weights paths
    effective_project_path, custom_weights_dir = determine_project_and_weights_paths(
        args, main_config, name_to_use
    )

    # Step 5: Setup WandB
    _setup_wandb(wandb_id_to_use, resume_flag)

    # Step 6: Setup model and callbacks
    try:
        model = setup_model_and_callbacks(model_to_load, name_to_use, custom_weights_dir)
    except Exception:
        # Error already logged in _load_model
        sys.exit(1)

    # Step 7: Prepare training arguments
    train_kwargs = prepare_train_kwargs(
        main_config, name_to_use, resume_flag, effective_project_path, absolute_data_config_path
    )

    # Step 8: Execute training
    final_output_dir, training_successful = execute_training(
        model, train_kwargs, effective_project_path, name_to_use, project_root
    )

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
        "Required if not using --resume_with.",
    )
    parser.add_argument(
        "--resume_with",  # Changed from --resume
        type=str,
        default=None,
        help=(
            "Path to the exact training run directory "
            "(e.g., runs/train/segment/run_YYMMDD_HHMMSS) to resume from. "
            "Overrides --name logic for naming and loads last.pt."
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
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help=(
            "Directory for saving model weights, useful for saving to NFS mounts on spot "
            "instances. If set, weights are saved to {weights_dir}/{project}/{name}/weights."
        ),
    )

    args = parser.parse_args()

    # --- Argument Validation --- # Added validation block
    if not args.resume_with and not args.name:
        parser.error("--name is required when not using --resume_with")
    # --- End Argument Validation ---

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
    #     --resume_with runs/train/segment/voc11_seg_finetune_run1_20240101_000000 \
    #     --name voc11_seg_finetune_run1
    #
    # Example Usage (With Custom Weights Directory):
    # python src/models/ext/yolov11/train_segment.py \
    #     --config configs/yolov11/finetune_segment_cov_segm.yaml \
    #     --name cov_segm_ft_yolo11l_260k \
    #     --weights-dir /path/to/nfs/mount

    main()
