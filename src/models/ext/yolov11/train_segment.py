"""
Main script for initiating YOLOv11 SEGMENTATION model training (finetuning or from scratch)
based on a YAML configuration file.

Mirrors the structure and capabilities of the YOLOv8/YOLOv11 detection training script.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


def resolve_data_path(config: dict, project_root: Path) -> dict:
    """Resolve the data path in config to an absolute path if relative."""
    if "data" in config and config["data"] and not Path(config["data"]).is_absolute():
        abs_data_path = (project_root / config["data"]).resolve()
        logging.info(
            f"Resolving relative data path '{config['data']}' to '{abs_data_path}'"
        )
        if not abs_data_path.exists():
            logging.warning(
                f"Resolved data config path does not exist: {abs_data_path}"
            )
        config["data"] = str(abs_data_path)
    elif "data" in config and config["data"]:
        logging.info(f"Data path is absolute: {config['data']}")
    else:
        logging.error("Missing 'data' key in configuration.")
        # Or handle default? For now, error.
        raise ValueError("Missing 'data' key in training configuration.")
    return config


def determine_model_to_load(
    config: dict,
    args: argparse.Namespace,
    project_root: Path,
    effective_project_path: str,
) -> str:
    """Determines the model path to load (base model or checkpoint for resume)."""
    base_model_name = config.get("model")
    if not base_model_name:
        raise ValueError("Missing 'model' key in training configuration.")

    model_to_load = base_model_name
    if args.resume:
        checkpoint_path = (
            project_root / effective_project_path / args.name / "weights" / "last.pt"
        )
        if checkpoint_path.is_file():
            logging.info(
                f"Resume flag set and checkpoint found. Loading: {checkpoint_path}"
            )
            model_to_load = str(checkpoint_path)
        else:
            logging.warning(
                f"Resume flag set, but checkpoint not found at: {checkpoint_path}"
            )
            logging.warning(
                "Attempting to resume using Ultralytics internal mechanism..."
            )
            # Keep resume=True in train_kwargs, Ultralytics might still find it based on name/project
            model_to_load = (
                base_model_name  # Need to load base model if checkpoint missing
            )

    logging.info(f"Resolved model to load: {model_to_load}")
    return model_to_load


def prepare_train_kwargs(
    config: dict, args: argparse.Namespace, effective_project_path: str
) -> dict:
    """Prepares the keyword arguments for the model.train() call."""
    # Filter config to only include valid YOLO train() arguments
    # List common/expected args. Ultralytics train() handles extras, but filtering is cleaner.
    valid_train_args = {
        "data",
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
        # Segmentation specific args (though YOLO often auto-detects task type)
        "overlap_mask",
        "mask_ratio",
    }
    train_kwargs = {k: v for k, v in config.items() if k in valid_train_args}

    # Add/override arguments from CLI
    train_kwargs["project"] = effective_project_path
    train_kwargs["name"] = args.name
    train_kwargs["resume"] = args.resume
    # train_kwargs['exist_ok'] = args.exist_ok # If adding --exist-ok flag

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

    # --- Load Environment Variables (Optional) ---
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f".env loaded from: {dotenv_path}")
    else:
        logging.info(".env file not found, proceeding without it.")

    # --- Load Training Configuration ---
    config_path_rel = args.config
    logging.info(f"Loading training configuration from: {config_path_rel}")
    config_path_abs = (project_root / config_path_rel).resolve()
    logging.info(f"Resolved training configuration path: {config_path_abs}")
    try:
        config = load_config(config_path_abs)
        # Resolve the path to the dataset YAML file, making it absolute if needed
        config = resolve_data_path(config, project_root)
        # The config['data'] now holds the path to the dataset YAML file.
        logging.info(f"Resolved dataset path: {config['data']}")

    except FileNotFoundError as e:
        logging.error(f"Configuration file error: {e}")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to load or process configurations: {e}", exc_info=True)
        sys.exit(1)

    # --- Determine Effective Project Path ----
    # Use CLI --project if provided, else config 'project', else default
    default_project = "runs/train/segment"  # Default for segmentation
    effective_project_path = (
        args.project
        if args.project is not None
        else config.get("project", default_project)
    )
    logging.info(f"Using project directory: {effective_project_path}")

    # --- Handle WandB Resume ID ---
    if args.wandb_id:
        logging.info(f"Setting up WandB to resume run ID: {args.wandb_id}")
        os.environ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_RUN_ID"] = args.wandb_id

    # --- Determine & Load Model ---
    try:
        model_to_load = determine_model_to_load(
            config, args, project_root, effective_project_path
        )
        # The YOLO class should automatically handle segmentation models
        model = YOLO(model_to_load)
        logging.info(f"Model '{model_to_load}' loaded successfully.")
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading model '{model_to_load}': {e}", exc_info=True)
        sys.exit(1)

    # --- Prepare Training Arguments ---
    train_kwargs = prepare_train_kwargs(config, args, effective_project_path)
    logging.info(f"Training arguments: {train_kwargs}")

    # --- Run Training ---
    logging.info("Starting segmentation training...") # Updated log message
    try:
        # The train method should work for segmentation task type
        model.train(**train_kwargs)
        logging.info("Segmentation training finished successfully.") # Updated log message
        # Use the effective project/name values for the final message
        final_output_dir = project_root / effective_project_path / args.name
        logging.info(f"Results saved to: {final_output_dir}")

    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 segmentation models using a configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the segmentation training configuration YAML file (relative to project root).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Output project directory (overrides config file if provided, defaults to runs/train/segment).",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for the specific training run directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint in the specified run.",
    )
    parser.add_argument(
        "--wandb-id", type=str, default=None, help="WandB run ID to resume logging to."
    )
    # parser.add_argument('--exist-ok', action='store_true', help='Allow overwriting existing run.')

    args = parser.parse_args()
    run_training_pipeline(args)


if __name__ == "__main__":
    # Ensure you are in the project root or environment is set up correctly.
    # Example usage from project root:
    # Finetune:
    # python src/models/ext/yolov11/train_segment.py --config src/models/ext/yolov11/configs/voc_segment_finetune.yaml --name voc11_segment_finetune_run1
    # Resume run (assuming WandB ID 'def456uvw' and name 'voc11_segment_finetune_run1'):
    # python src/models/ext/yolov11/train_segment.py --config src/models/ext/yolov11/configs/voc_segment_finetune.yaml --resume --wandb-id def456uvw --name voc11_segment_finetune_run1
    main()