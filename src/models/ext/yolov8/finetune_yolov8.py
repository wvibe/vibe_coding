import argparse
import yaml
import os
from pathlib import Path
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
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("Configuration loaded successfully.")
    return config

def resolve_paths(config: dict, project_root: Path) -> dict:
    """Resolve relative paths in config to absolute paths."""
    if 'data' in config and not Path(config['data']).is_absolute():
        config['data'] = str((project_root / config['data']).resolve())
        print(f"Resolved dataset config path: {config['data']}")
    # Add other paths to resolve if needed (e.g., project, name)
    # For project/name, YOLOv8 typically handles them relative to CWD,
    # so resolving might not be strictly necessary unless you want explicit control.
    # config['project'] = str((project_root / config['project']).resolve())
    return config

def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 using a configuration file.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the training configuration YAML file (relative to project root).')
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

    # --- Load Base Model ---
    base_model_name = config.get('model', 'yolov8n.pt') # Default if missing
    try:
        print(f"Loading base model: {base_model_name}")
        # Device specified during train() overrides initial load
        model = YOLO(base_model_name)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading base model '{base_model_name}': {e}")
        return

    # --- Prepare Training Arguments from Config ---
    # Filter config to only include valid YOLO train() arguments
    # This prevents passing unknown keys like 'enable_wandb' to train()
    # Get valid args from YOLO model if possible (more robust) or use a known list
    # Simple approach: manually list expected args
    valid_train_args = [
        'data', 'epochs', 'imgsz', 'batch', 'workers', 'optimizer',
        'lr0', 'lrf', 'close_mosaic', 'device', 'pretrained', 'project',
        'name', 'exist_ok', 'save_period', 'weight_decay', 'momentum',
        'warmup_epochs', 'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate',
        'scale', 'shear', 'perspective', 'flipud', 'fliplr' # Add more as needed
        # Add other args you might have in your config
    ]
    train_kwargs = {k: v for k, v in config.items() if k in valid_train_args}

    # Handle potential None for device (YOLO expects None or str)
    if 'device' in train_kwargs and not train_kwargs['device']:
        train_kwargs['device'] = None

    print("--- Training Arguments ---")
    for key, val in train_kwargs.items():
        print(f"  {key}: {val}")
    print("------------------------")

    # --- Run Training ---
    print("Starting training...")
    try:
        model.train(**train_kwargs)
        print("Training finished successfully.")
        print(f"Results saved to: {project_root / config.get('project', 'runs/detect') / config.get('name', 'train')}")

    except Exception as e:
        print(f"Error during training: {e}")


if __name__ == "__main__":
    # Activate conda env first: conda activate vbl
    # Ensure you are in the project root: cd /path/to/vibe/vibe_coding
    # Example usage from project root:
    # python src/models/ext/yolov8/finetune_yolov8.py --config src/models/ext/yolov8/configs/voc_finetune_config.yaml
    main()