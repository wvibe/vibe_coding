"""
Training script for YOLOv3 model
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from datasets.object_detection.voc import PascalVOCDataset
from models.vanilla.yolov3.config import YOLOv3Config
from models.vanilla.yolov3.loss import YOLOv3Loss
from models.vanilla.yolov3.yolov3 import YOLOv3


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train YOLOv3 model")

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="voc",
        choices=["voc", "bdd"],
        help="Dataset to use (default: voc)",
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
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to weights file for loading a pretrained model",
    )

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
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for saving models and logs (default: output)",
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
        default=5,
        help="Interval for evaluation during training (default: 5 epochs)",
    )

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda)")

    return parser.parse_args()


def train_epoch(model, dataloader, loss_fn, optimizer, device, epoch, args):
    """
    Train model for one epoch

    Args:
        model: YOLOv3 model
        dataloader: DataLoader for training data
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch
        args: Command line arguments

    Returns:
        dict: Training metrics
    """
    model.train()

    # Initialize metrics
    total_loss = 0
    total_loc_loss = 0
    total_obj_loss = 0
    total_cls_loss = 0

    start_time = time.time()

    # Loop over batches
    for batch_idx, batch in enumerate(dataloader):
        # Get batch data
        images = batch["images"].to(device)
        targets = {
            "boxes": [boxes.to(device) for boxes in batch["boxes"]],
            "labels": [labels.to(device) for labels in batch["labels"]],
        }

        # Forward pass
        predictions = model(images)

        # Compute loss
        loss_dict = loss_fn(predictions, targets)
        loss = loss_dict["loss"]

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        total_loc_loss += loss_dict["loc_loss"].item()
        total_obj_loss += loss_dict["obj_loss"].item()
        total_cls_loss += loss_dict["cls_loss"].item()

        # Print progress
        if batch_idx % 10 == 0:
            print(
                f"Epoch: {epoch} | Batch: {batch_idx}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Loc Loss: {loss_dict['loc_loss'].item():.4f} | "
                f"Obj Loss: {loss_dict['obj_loss'].item():.4f} | "
                f"Cls Loss: {loss_dict['cls_loss'].item():.4f}"
            )

    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_loc_loss = total_loc_loss / len(dataloader)
    avg_obj_loss = total_obj_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)

    # Calculate elapsed time
    time_elapsed = time.time() - start_time

    # Print epoch summary
    print(
        f"Epoch: {epoch} completed in {time_elapsed:.2f}s | "
        f"Avg Loss: {avg_loss:.4f} | "
        f"Avg Loc Loss: {avg_loc_loss:.4f} | "
        f"Avg Obj Loss: {avg_obj_loss:.4f} | "
        f"Avg Cls Loss: {avg_cls_loss:.4f}"
    )

    return {
        "loss": avg_loss,
        "loc_loss": avg_loc_loss,
        "obj_loss": avg_obj_loss,
        "cls_loss": avg_cls_loss,
        "time_elapsed": time_elapsed,
    }


def validate(model, dataloader, loss_fn, device):
    """
    Validate model on validation set

    Args:
        model: YOLOv3 model
        dataloader: DataLoader for validation data
        loss_fn: Loss function
        device: Device to use

    Returns:
        dict: Validation metrics
    """
    model.eval()

    # Initialize metrics
    total_loss = 0
    total_loc_loss = 0
    total_obj_loss = 0
    total_cls_loss = 0

    with torch.no_grad():
        for _batch_idx, batch in enumerate(dataloader):
            # Get batch data
            images = batch["images"].to(device)
            targets = {
                "boxes": [boxes.to(device) for boxes in batch["boxes"]],
                "labels": [labels.to(device) for labels in batch["labels"]],
            }

            # Forward pass
            predictions = model(images)

            # Compute loss
            loss_dict = loss_fn(predictions, targets)
            loss = loss_dict["loss"]

            # Update metrics
            total_loss += loss.item()
            total_loc_loss += loss_dict["loc_loss"].item()
            total_obj_loss += loss_dict["obj_loss"].item()
            total_cls_loss += loss_dict["cls_loss"].item()

    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_loc_loss = total_loc_loss / len(dataloader)
    avg_obj_loss = total_obj_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)

    # Print validation summary
    print(
        f"Validation | "
        f"Avg Loss: {avg_loss:.4f} | "
        f"Avg Loc Loss: {avg_loc_loss:.4f} | "
        f"Avg Obj Loss: {avg_obj_loss:.4f} | "
        f"Avg Cls Loss: {avg_cls_loss:.4f}"
    )

    return {
        "loss": avg_loss,
        "loc_loss": avg_loc_loss,
        "obj_loss": avg_obj_loss,
        "cls_loss": avg_cls_loss,
    }


def save_checkpoint(model, optimizer, epoch, metrics, args, is_best=False):
    """
    Save model checkpoint

    Args:
        model: YOLOv3 model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Training metrics
        args: Command line arguments
        is_best: Whether this is the best model so far
    """
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": model.config,
    }

    checkpoint_path = os.path.join(args.output_dir, f"yolov3_checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    if is_best:
        best_path = os.path.join(args.output_dir, "yolov3_best.pt")
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")


def main():
    """Main training function"""
    args = parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    if args.dataset == "voc":
        train_dataset = PascalVOCDataset(split="train")
        val_dataset = PascalVOCDataset(split="val")
    else:
        # In future, add BDD100K dataset
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented yet")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=train_dataset.collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=val_dataset.collate_fn,
    )

    # Create model config
    config = YOLOv3Config(
        input_size=args.input_size,
        num_classes=args.num_classes,
        darknet_pretrained=args.pretrained,
        darknet_weights_path=None,  # We'll load full model weights separately if provided
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )

    # Create model
    model = YOLOv3(config)
    model.to(device)

    # Load weights if provided
    if args.weights:
        print(f"Loading weights from {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    # Create loss function
    loss_fn = YOLOv3Loss(config)

    # Training with frozen backbone if specified
    if args.freeze_backbone and args.freeze_epochs > 0:
        # Freeze backbone parameters
        for param in model.backbone.parameters():
            param.requires_grad = False

        # Create optimizer for unfrozen parameters
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # Train with frozen backbone
        print(f"Training with frozen backbone for {args.freeze_epochs} epochs")
        for epoch in range(1, args.freeze_epochs + 1):
            metrics = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch, args)

            # Validate model
            if epoch % args.eval_interval == 0:
                val_metrics = validate(model, val_loader, loss_fn, device)
                metrics.update({"val_" + k: v for k, v in val_metrics.items()})

            # Save checkpoint
            if epoch % args.checkpoint_interval == 0 or epoch == args.freeze_epochs:
                save_checkpoint(model, optimizer, epoch, metrics, args)

        # Unfreeze backbone for full training
        for param in model.backbone.parameters():
            param.requires_grad = True
        print("Unfrozen backbone for full training")

    # Create optimizer for full training
    if args.freeze_backbone:
        # Reset learning rate after unfreezing
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate / 10,  # Use lower learning rate after unfreezing
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.freeze_epochs if args.freeze_backbone else args.epochs,
        eta_min=args.learning_rate / 100,
    )

    # Train model
    start_epoch = args.freeze_epochs + 1 if args.freeze_backbone else 1
    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        # Train one epoch
        metrics = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch, args)

        # Validate model
        if epoch % args.eval_interval == 0:
            val_metrics = validate(model, val_loader, loss_fn, device)
            metrics.update({"val_" + k: v for k, v in val_metrics.items()})

            # Check if this is the best model
            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
        else:
            is_best = False

        # Save checkpoint
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            save_checkpoint(model, optimizer, epoch, metrics, args, is_best)

        # Update learning rate
        lr_scheduler.step()

    # Save final model
    final_path = os.path.join(args.output_dir, "yolov3_final.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": model.config}, final_path)
    print(f"Training completed. Saved final model to {final_path}")


if __name__ == "__main__":
    main()
