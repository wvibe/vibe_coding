import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO


def get_project_root() -> Path:
    """Find the project root directory relative to this script."""
    # Assumes script is in project_root/src/models/ext/yolov8
    return Path(__file__).resolve().parents[4]


def load_config(config_path: Path) -> dict:
    """Load training configuration from a YAML file."""
    print(f"Loading configuration from: {config_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("Configuration loaded successfully.")
    return config


def resolve_paths(config: dict, project_root: Path) -> dict:
    """Resolve relative paths in config to absolute paths."""
    if "data" in config and not Path(config["data"]).is_absolute():
        config["data"] = str((project_root / config["data"]).resolve())
        print(f"Resolved dataset config path: {config['data']}")
    # Project path resolution is handled later based on CLI args/config value
    return config


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 using a configuration file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration YAML file (relative to project root).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Output project directory (overrides config file if provided).",
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
        "--wandb-id",
        type=str,
        default=None,
        help="WandB run ID to resume logging to.",
    )
    # Add --exist-ok if needed, or manage through run names
    # parser.add_argument('--exist-ok', action='store_true', help='Allow overwriting existing run.')

    args = parser.parse_args()

    project_root = get_project_root()
    print(f"Project Root: {project_root}")

    # --- Load Environment Variables (Optional but good practice) ---
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        print(f".env loaded from: {dotenv_path}")
    else:
        print(".env file not found, proceeding without it.")

    # --- Load Training Configuration ---
    config_path_rel = args.config
    config_path_abs = (project_root / config_path_rel).resolve()
    try:
        config = load_config(config_path_abs)
        config = resolve_paths(config, project_root)
    except Exception as e:
        print(f"Error loading or processing config: {e}")
        return

    # --- Determine Effective Project Path ---
    project_value = (
        args.project if args.project is not None else config.get("project", "runs/detect")
    )
    print(f"Using project directory: {project_value}")

    # --- Handle WandB Resume ID ---
    # If wandb_id is provided via CLI, set environment variables for wandb
    if args.wandb_id:
        print(f"Setting up WandB to resume run ID: {args.wandb_id}")
        os.environ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_RUN_ID"] = args.wandb_id

    # --- Load Base Model ---
    base_model_name = config.get("model", "yolov8n.pt")  # Default if missing
    try:
        model_to_load = base_model_name
        # Check if resuming and if the last checkpoint exists
        if args.resume:
            checkpoint_path = project_root / project_value / args.name / "weights" / "last.pt"
            if checkpoint_path.exists():
                print(f"Resume flag set and checkpoint found. Loading: {checkpoint_path}")
                model_to_load = str(checkpoint_path)
            else:
                print(f"Resume flag set, but checkpoint not found at: {checkpoint_path}")
                print("Attempting to resume without explicit checkpoint (might start new run)...")
                # Keep resume=True in train_kwargs, ultralytics might still find it

        print(f"Loading model: {model_to_load}")
        # Device specified during train() can override initial load device if specified there
        model = YOLO(model_to_load)

        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model '{model_to_load}': {e}")
        return

    # --- Prepare Training Arguments from Config & CLI ---
    # Filter config to only include valid YOLO train() arguments
    valid_train_args = [
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
        "patience",  # Add more as needed from config
        # Note: project, name, resume are added separately from CLI args below
    ]
    train_kwargs = {k: v for k, v in config.items() if k in valid_train_args}

    # Add arguments from CLI
    train_kwargs["project"] = project_value
    train_kwargs["name"] = args.name
    train_kwargs["resume"] = args.resume
    # train_kwargs['exist_ok'] = args.exist_ok # Add if using --exist-ok

    # Handle potential None for device (YOLO expects None or str)
    if "device" in train_kwargs and not train_kwargs["device"]:
        train_kwargs["device"] = None

    print("--- Training Arguments ---")
    # Sort for consistent printing
    for key, val in sorted(train_kwargs.items()):
        print(f"  {key}: {val}")
    print("------------------------")

    # --- Run Training ---
    print("Starting training...")
    try:
        model.train(**train_kwargs)
        print("Training finished successfully.")
        # Use the effective project/name values for the final message
        print(f"Results saved to: {project_root / project_value / args.name}")

    except Exception as e:
        print(f"Error during training: {e}")


if __name__ == "__main__":
    # Activate conda env first: conda activate vbl
    # Ensure you are in the project root: cd /path/to/vibe/vibe_coding
    # Example usage from project root:
    # Start new run:
    # python src/models/ext/yolov8/finetune_yolov8.py --config src/models/ext/yolov8/configs/voc_finetune_config.yaml --name my_first_run
    # Resume run (assuming run ID 'abc123xyz'):
    # python src/models/ext/yolov8/finetune_yolov8.py --config src/models/ext/yolov8/configs/voc_finetune_config.yaml --resume --wandb-id abc123xyz --name my_first_run
    main()
