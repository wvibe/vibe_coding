"""
Runs YOLOv11 detection evaluation based on a configuration YAML file.

This script:
1. Loads a trained YOLOv11 model
2. Runs inference on a specified dataset
3. Calculates detection metrics (mAP, mAP by size, confusion matrix)
4. Reports computational metrics (parameters, inference time, memory usage)
5. Generates various plots and output files

Implementation plan:
- [x] 5.3.1: Setup script structure with single `--config` argument and basic imports
- [x] 5.3.2: Create `evaluate_default.yaml` configuration file with comprehensive options
- [x] 5.3.3: Implement configuration loading and validation
- [x] 5.3.4: Add model loading with parameter counting via `get_model_params`
- [ ] 5.3.5: Implement inference with warmup and measurement of time/memory
- [ ] 5.3.6: Implement ground truth loading and format conversion
- [ ] 5.3.7: Integrate metric calculation (`match_predictions`, `calculate_map`, etc.)
- [ ] 5.3.8: Implement visualization and result saving
- [x] 5.3.9: Create evaluation documentation in `docs/yolov11/evaluate.md`
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO

# Import custom metrics utilities
from src.utils.metrics.detection import (
    calculate_iou,
    match_predictions,
    calculate_pr_data,
    calculate_ap,
    calculate_map,
    calculate_map_by_size,
    generate_confusion_matrix
)
from src.utils.metrics.compute import (
    get_model_params,
    get_peak_gpu_memory_mb
)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: str) -> Dict:
    """Loads and performs basic validation on the YAML config file.

    Args:
        config_path: Path to the configuration YAML file

    Returns:
        Dict containing the loaded configuration

    Raises:
        SystemExit: If the config file is not found, invalid, or missing required keys
    """
    config_path = Path(config_path)
    logging.info(f"Loading configuration from: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError("Config file is empty or invalid.")

        # Validate required top-level sections
        required_sections = ["model", "dataset", "evaluation_params", "metrics"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in config.")

        # Validate required dataset fields
        required_dataset_fields = ["image_dir", "label_dir", "class_names"]
        for field in required_dataset_fields:
            if field not in config["dataset"]:
                raise ValueError(f"Missing required field '{field}' in dataset section.")

        # Ensure class_names is a list
        if not isinstance(config["dataset"]["class_names"], list) or not config["dataset"]["class_names"]:
            raise ValueError("class_names must be a non-empty list in dataset section.")

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


def setup_output_directory(config: Dict) -> Path:
    """Creates the output directory based on the configuration.

    Args:
        config: The loaded configuration dictionary

    Returns:
        Path object for the output directory
    """
    # Get output configuration (with defaults if not provided)
    output_config = config.get("output", {})
    project_dir = Path(output_config.get("project", "runs/evaluate/detect"))

    # Generate name based on model name and timestamp if not specified
    name = output_config.get("name")
    if not name:
        model_name = Path(config["model"]).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{model_name}_{timestamp}"

    output_dir = project_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")

    return output_dir


def load_model(model_path: str, device: Union[str, int]) -> Tuple[Any, int]:
    """Loads the YOLO model and returns information about it.

    Args:
        model_path: Path to the model file or a model name (e.g., "yolo11n.pt")
        device: Device to load the model on (e.g., 0, "cuda:0", "cpu")

    Returns:
        Tuple of (model, num_parameters)
    """
    logging.info(f"Loading model from {model_path} on device {device}")

    try:
        # Reset GPU memory stats before loading model
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model = YOLO(model_path)

        # Get model parameters
        num_params = get_model_params(model)
        if num_params is not None:
            logging.info(f"Model has {num_params:,} parameters")
        else:
            logging.warning("Could not determine model parameter count")
            num_params = 0

        return model, num_params

    except Exception as e:
        logging.error(f"Error loading model: {e}")
        sys.exit(1)


def main():
    """Main entry point for the evaluation script."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11 detection model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the evaluation configuration YAML file"
    )
    args = parser.parse_args()

    # Load configuration (DONE - 5.3.3)
    config = load_config(args.config)

    # Setup output directory (DONE - 5.3.3)
    output_dir = setup_output_directory(config)

    # Save a copy of the configuration
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # TODO - 5.3.4: Load model and get parameter count
    # model, num_params = load_model(config["model"], config["evaluation_params"]["device"])

    # TODO - 5.3.5: Run inference with timing and memory measurement
    # Implementation steps:
    # 1. Reset GPU memory stats
    # 2. Perform warmup iterations
    # 3. Time inference on all images
    # 4. Record peak GPU memory
    # 5. Store predictions for metric calculation

    # TODO - 5.3.6: Load and process ground truth data
    # Implementation steps:
    # 1. Read YOLO format labels from label_dir
    # 2. Convert to format compatible with metric functions
    # 3. Organize by image for efficient matching

    # TODO - 5.3.7: Calculate metrics
    # Implementation steps:
    # 1. Match predictions to ground truth
    # 2. Calculate precision-recall data
    # 3. Calculate mAP across IoU thresholds
    # 4. Calculate mAP by size
    # 5. Generate confusion matrix

    # TODO - 5.3.8: Generate visualizations and save results
    # Implementation steps:
    # 1. Create precision-recall curve plots
    # 2. Create confusion matrix visualization
    # 3. Save metrics to JSON/CSV
    # 4. Save inference stats
    # 5. Print summary to console

    logging.info(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()