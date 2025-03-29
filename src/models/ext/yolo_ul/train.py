"""
Script for training Ultralytics YOLO models
"""

import argparse
import os

from ultralytics import YOLO

from utils import load_yaml_config


def train_model(
    model_name="yolo11n.pt",  # Model name or path
    data_yaml=None,  # Path to dataset YAML
    train_yaml=None,  # Path to training configuration YAML
    project=None,  # Project name for saving results
    **kwargs,  # Additional arguments for model.train()
):
    """
    Train a YOLO model

    Args:
        model_name: Model name or path
        data_yaml: Path to dataset YAML
        train_yaml: Path to training configuration YAML
        project: Project name for saving results
        **kwargs: Additional arguments for model.train()

    Returns:
        Training results and validation metrics
    """
    # Validate required parameters
    if not data_yaml:
        raise ValueError("Dataset YAML must be provided")

    # Set up training configuration
    train_config = {}

    # Load training configuration from YAML if provided
    if train_yaml:
        train_config = load_yaml_config(train_yaml)

    # Add dataset YAML to config
    train_config["data"] = data_yaml

    # Add project if provided
    if project:
        train_config["project"] = project

    # Add any additional kwargs to config
    train_config.update(kwargs)

    # Load model
    model = YOLO(model_name)

    # Train the model
    print(f"Training model {model_name} with {data_yaml}")
    print(f"Training configuration: {train_config}")
    results = model.train(**train_config)

    # Validate the model
    print("Validating model...")
    metrics = model.val()

    # Print summary
    print("\nTraining Complete!")
    print(f"Model saved to: {os.path.join(results.save_dir, 'weights/best.pt')}")
    if hasattr(metrics, "box"):
        print(f"mAP50-95: {metrics.box.map:.4f}, mAP50: {metrics.box.map50:.4f}")

    return results, metrics


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Model path or name")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML file")

    # Optional parameters
    parser.add_argument("--train", type=str, help="Path to training configuration YAML file")
    parser.add_argument("--project", type=str, help="Project name for saving results")

    # Additional training parameters
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--device", type=str, help="Device (cpu, 0, 0,1)")

    args = parser.parse_args()

    # Extract additional kwargs from args to pass to train_model
    kwargs = {}
    if args.epochs is not None:
        kwargs["epochs"] = args.epochs
    if args.batch is not None:
        kwargs["batch"] = args.batch
    if args.imgsz is not None:
        kwargs["imgsz"] = args.imgsz
    if args.device is not None:
        kwargs["device"] = args.device

    # Call train function with provided arguments
    results, metrics = train_model(
        model_name=args.model,
        data_yaml=args.data,
        train_yaml=args.train,
        project=args.project,
        **kwargs,
    )
