#!/usr/bin/env python3
"""
Debug script for YOLOv3 model.

This script runs a quick training cycle on a small subset of the dataset (10%)
to verify that all parts of the pipeline (train, validate, evaluate) are working properly.
"""

import os
import random

import torch
from torch.utils.data import DataLoader, Subset

from src.models.py.yolov3.config import YOLOv3Config
from src.models.py.yolov3.loss import YOLOv3Loss
from src.models.py.yolov3.train import (
    parse_args,
    setup_environment,
    train_epoch,
    validate,
)
from src.models.py.yolov3.yolov3 import YOLOv3
from src.utils.data_loaders.cv.voc import PascalVOCDataset


def create_subset_dataloader(
    dataset, subset_fraction=0.1, batch_size=8, num_workers=2, shuffle=True
):
    """
    Create a dataloader with only a subset of the dataset

    Args:
        dataset: The full dataset
        subset_fraction: Fraction of the dataset to use (default: 0.1 or 10%)
        batch_size: Batch size for the dataloader
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader: A dataloader containing the subset of data
    """
    # Calculate subset size
    full_size = len(dataset)
    subset_size = max(int(full_size * subset_fraction), 1)

    # Create random indices for the subset
    indices = random.sample(range(full_size), subset_size)
    subset = Subset(dataset, indices)

    print(f"Created subset with {subset_size}/{full_size} samples ({subset_fraction * 100:.1f}%)")

    # Create dataloader
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )


def debug_train():
    """
    Run a quick debug training cycle on a small subset of the dataset
    """
    # Parse arguments
    args = parse_args()

    # Override some settings for debugging
    args.epochs = 1  # Just do one epoch
    args.batch_size = 2  # Small batch size
    args.workers = 2  # Fewer workers
    args.checkpoint_interval = 1  # Save after each epoch
    args.output_dir = os.path.join("model_outputs", "yolov3", "debug")
    args.run_name = "debug_run"
    args.no_wandb = True  # Disable wandb logging for debugging

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up environment
    args, device, _ = setup_environment(args)

    print("=" * 50)
    print("RUNNING DEBUG TRAINING")
    print(f"Device: {device}")
    print("=" * 50)

    # Create datasets
    print("\nLoading datasets:")
    subset_fraction = 0.1  # Use 10% of the dataset

    # Create train dataset
    train_dataset = PascalVOCDataset(
        years=[args.year] if isinstance(args.year, str) else args.year.split(","),
        split_file=f"{args.train_split}.txt",
        sample_pct=subset_fraction,
    )
    train_loader = create_subset_dataloader(
        train_dataset,
        subset_fraction=subset_fraction,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # Create validation dataset
    val_dataset = PascalVOCDataset(
        years=[args.year] if isinstance(args.year, str) else args.year.split(","),
        split_file=f"{args.val_split}.txt",
        sample_pct=subset_fraction,
    )
    val_loader = create_subset_dataloader(
        val_dataset,
        subset_fraction=subset_fraction,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
    )

    # Create model
    print("\nCreating model...")
    config = YOLOv3Config(
        input_size=args.input_size,
        num_classes=train_dataset.num_classes,
        darknet_pretrained=args.pretrained,
    )
    model = YOLOv3(config)
    model.to(device)

    # Create loss function
    loss_fn = YOLOv3Loss(config)

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Training loop
    print("\n=== TRAINING CYCLE ===")
    print("Running training for 1 epoch...")
    train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device, 1, args)

    print("\n=== VALIDATION CYCLE ===")
    print("Running validation...")
    val_metrics = validate(model, val_loader, loss_fn, device, 1, None, args)

    # Save checkpoint
    print("\n=== SAVING MODEL ===")
    checkpoint_path = os.path.join(args.output_dir, "debug_checkpoint.pt")
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": model.config,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

    print("\n=== DEBUG SUMMARY ===")
    print(f"Training loss: {train_metrics.get('loss', 'N/A')}")
    print(f"Validation loss: {val_metrics.get('val_loss', 'N/A')}")
    print(f"Validation mAP: {val_metrics.get('val_mAP', 'N/A')}")
    print("\nDebug training completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    debug_train()
