"""
Environment setup and argument parsing utilities for YOLOv3 training
"""

import argparse
import datetime
import os

import numpy as np
import torch

import wandb


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train YOLOv3 model")

    # Fixed paths
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for model checkpoints (default: auto-generated based on run date)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to weights file to load",
    )

    # Data directory argument
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.getenv("DATA_ROOT", "/Users/wmu/vibe/hub/data"),
        help="Directory containing the dataset",
    )

    # Dataset options
    parser.add_argument(
        "--dataset",
        type=str,
        default="voc",
        choices=["voc"],
        help="Dataset to use (default: voc)",
    )
    parser.add_argument(
        "--year",
        type=str,
        default="2007,2012",
        help="VOC dataset years to use as comma-separated string (default: 2007,2012)",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        choices=["train", "trainval"],
        help="Dataset split to use for training (default: train)",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to use for validation (default: val)",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default=None,
        choices=[None, "test"],
        help="Dataset split to use for final testing (default: None)",
    )

    # Debug/Development parameters
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to use per dataset (for debugging)",
    )
    parser.add_argument(
        "--subset-percent",
        type=float,
        default=None,
        help="Percentage of dataset to use (0.01 to 1.0, for debugging)",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default=None,
        help="Train only on images containing a specific class (e.g., 'car', 'person')",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Enable additional debug logging",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Fast dev run mode - use minimal batches for train/val/test",
    )

    # Model parameters
    parser.add_argument(
        "--input-size", type=int, default=416, help="Input image size (default: 416)"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=20,
        help="Number of classes (default: 20 for Pascal VOC)",
    )
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained backbone weights")

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train (default: 100)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight decay (default: 5e-4)"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer (default: 0.9)",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone during initial training",
    )
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=10,
        help="Number of epochs to train with frozen backbone (default: 10)",
    )

    # Output parameters
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Name for this training run (default: timestamp)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Interval for saving checkpoints (default: 10 epochs)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Interval for evaluation during training (default: 1 epoch)",
    )

    # W&B parameters
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="yolov3",
        help="Weights & Biases project name (default: yolov3)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (default: None)",
    )

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help=("Device to use (default: auto - selects CUDA if available, then MPS, then CPU)"),
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for evaluation (default: 0.5)",
    )

    # Anchor optimization parameter
    parser.add_argument(
        "--no-anchor-optimization",
        action="store_true",
        help="Disable anchor optimization for VOC dataset",
    )

    # Argument for using dummy data
    parser.add_argument(
        "--use-dummy-data",
        action="store_true",
        help="Use dummy data instead of real VOC dataset (for debugging)",
    )

    return parser.parse_args()


def setup_environment(args):
    """
    Set up training environment

    Args:
        args: Command line arguments

    Returns:
        tuple: Updated args, device, wandb_run
    """
    # Debug mode setup
    if args.fast_dev_run:
        print("DEBUG MODE: Running with minimal dataset and epochs")
        if not args.max_images:
            args.max_images = 10
        args.epochs = 1

    # Set up run name
    if args.run_name is None:
        args.run_name = f"yolov3_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Set default output directory if None
    if args.output_dir is None:
        args.output_dir = "output"

    # Create output directory
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize W&B
    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
        print(f"W&B initialized: {wandb.run.name}")

    # Set device based on availability and user preference
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    return args, device, wandb_run
