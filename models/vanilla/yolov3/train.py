"""
Training script for YOLOv3 model

This script should be run from the project root directory:
python -m models.vanilla.yolov3.train [args]

For quick training with default parameters, use:
./models/vanilla/yolov3/scripts/run_train_and_eval.sh

For debugging the training pipeline, use:
./models/vanilla/yolov3/scripts/run_debug.sh
"""

import argparse
import datetime
import os
import sys
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

# Add project root to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from data_loaders.object_detection.voc import PascalVOCDataset
from models.vanilla.yolov3.config import YOLOv3Config
from models.vanilla.yolov3.loss import YOLOv3Loss
from models.vanilla.yolov3.yolov3 import YOLOv3

# Load environment variables
load_dotenv()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train YOLOv3 model")

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="voc",
        choices=["voc"],
        help="Dataset to use (default: voc)",
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
    parser.add_argument(
        "--year",
        type=str,
        default="2007",
        choices=["2007"],
        help="VOC dataset year (default: 2007)",
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
    parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained backbone weights"
    )
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
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: 16)"
    )
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
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
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
        help="Device to use (default: auto - selects CUDA if available, then MPS, then CPU)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for evaluation (default: 0.5)",
    )

    return parser.parse_args()


def train_epoch(
    model, dataloader, loss_fn, optimizer, device, epoch, args, wandb_run=None
):
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
        wandb_run: Weights & Biases run object for logging

    Returns:
        dict: Training metrics
    """
    model.train()

    # Initialize metrics
    total_loss = 0
    total_loc_loss = 0
    total_obj_loss = 0
    total_cls_loss = 0
    batch_times = []

    start_time = time.time()

    # Loop over batches
    for batch_idx, batch in enumerate(dataloader):
        batch_start = time.time()

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

        # Calculate batch time
        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_times.append(batch_time)

        # Print progress
        if batch_idx % 10 == 0:
            print(
                f"Epoch: {epoch} | Batch: {batch_idx}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Loc Loss: {loss_dict['loc_loss'].item():.4f} | "
                f"Obj Loss: {loss_dict['obj_loss'].item():.4f} | "
                f"Cls Loss: {loss_dict['cls_loss'].item():.4f} | "
                f"Batch Time: {batch_time:.2f}s"
            )

            # Log batch metrics to W&B
            if wandb_run:
                wandb_run.log(
                    {
                        "batch": batch_idx + epoch * len(dataloader),
                        "batch_loss": loss.item(),
                        "batch_loc_loss": loss_dict["loc_loss"].item(),
                        "batch_obj_loss": loss_dict["obj_loss"].item(),
                        "batch_cls_loss": loss_dict["cls_loss"].item(),
                        "batch_time": batch_time,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_loc_loss = total_loc_loss / len(dataloader)
    avg_obj_loss = total_obj_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_batch_time = sum(batch_times) / len(batch_times)

    # Calculate elapsed time
    time_elapsed = time.time() - start_time

    # Print epoch summary
    print(
        f"Epoch: {epoch} completed in {time_elapsed:.2f}s | "
        f"Avg Loss: {avg_loss:.4f} | "
        f"Avg Loc Loss: {avg_loc_loss:.4f} | "
        f"Avg Obj Loss: {avg_obj_loss:.4f} | "
        f"Avg Cls Loss: {avg_cls_loss:.4f} | "
        f"Avg Batch Time: {avg_batch_time:.2f}s"
    )

    # Log epoch metrics to W&B
    metrics = {
        "epoch": epoch,
        "train_loss": avg_loss,
        "train_loc_loss": avg_loc_loss,
        "train_obj_loss": avg_obj_loss,
        "train_cls_loss": avg_cls_loss,
        "epoch_time": time_elapsed,
        "avg_batch_time": avg_batch_time,
    }

    if wandb_run:
        wandb_run.log(metrics)

    return metrics


def validate(model, dataloader, loss_fn, device, epoch=None, wandb_run=None, args=None):
    """
    Validate model on validation set

    Args:
        model: YOLOv3 model
        dataloader: DataLoader for validation data
        loss_fn: Loss function
        device: Device to use
        epoch: Current epoch (for logging)
        wandb_run: Weights & Biases run object for logging
        args: Command line arguments

    Returns:
        dict: Validation metrics
    """
    model.eval()
    start_time = time.time()

    # Initialize metrics
    total_loss = 0
    total_loc_loss = 0
    total_obj_loss = 0
    total_cls_loss = 0

    # For visualization
    if wandb_run and epoch is not None and epoch % 5 == 0:  # Log every 5 epochs
        validation_images = []

    # Evaluate model using the evaluate module
    print("Running validation" + (f" for epoch {epoch}" if epoch is not None else ""))

    with torch.no_grad():
        # Compute loss on validation set
        progress_bar = tqdm(dataloader, desc="Computing validation loss")
        for batch_idx, batch in enumerate(progress_bar):
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

            # Log predictions for visualization
            if wandb_run and epoch is not None and epoch % 5 == 0 and batch_idx < 4:
                # Get predictions
                detections = model.predict(images)

                # Process each image in batch
                for img_idx in range(min(2, len(images))):  # Log max 2 images per batch
                    img = images[img_idx].cpu()
                    # Denormalize image
                    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    img = (img * 255).byte().permute(1, 2, 0).numpy()

                    # Create figure for visualization
                    plt.figure(figsize=(10, 10))
                    plt.imshow(img)

                    # Draw ground truth boxes in green
                    for box, label in zip(
                        targets["boxes"][img_idx], targets["labels"][img_idx]
                    ):
                        x1, y1, x2, y2 = box.cpu().numpy()
                        rect = patches.Rectangle(
                            (x1, y1),
                            x2 - x1,
                            y2 - y1,
                            linewidth=2,
                            edgecolor="g",
                            facecolor="none",
                        )
                        plt.gca().add_patch(rect)
                        plt.text(
                            x1,
                            y1 - 5,
                            f"GT: {dataloader.dataset.class_names[label]}",
                            color="g",
                        )

                    # Draw predicted boxes in red
                    for det in detections[img_idx]:
                        if det.size(0) > 0:  # If there are detections
                            x1, y1, x2, y2, conf, cls_id = det
                            rect = patches.Rectangle(
                                (x1, y1),
                                x2 - x1,
                                y2 - y1,
                                linewidth=2,
                                edgecolor="r",
                                facecolor="none",
                            )
                            plt.gca().add_patch(rect)
                            plt.text(
                                x1,
                                y2 + 15,
                                f"Pred: {dataloader.dataset.class_names[int(cls_id)]} {conf:.2f}",
                                color="r",
                            )

                    plt.axis("off")
                    validation_images.append(wandb.Image(plt))
                    plt.close()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": loss.item(),
                    "loc_loss": loss_dict["loc_loss"].item(),
                    "obj_loss": loss_dict["obj_loss"].item(),
                    "cls_loss": loss_dict["cls_loss"].item(),
                }
            )

    # Calculate average losses
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_loc_loss = total_loc_loss / num_batches
    avg_obj_loss = total_obj_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches

    # Create metrics dictionary
    metrics = {
        "val_loss": avg_loss,
        "val_loc_loss": avg_loc_loss,
        "val_obj_loss": avg_obj_loss,
        "val_cls_loss": avg_cls_loss,
        "val_time": time.time() - start_time,
    }

    # Log validation images to W&B
    if wandb_run and epoch is not None and epoch % 5 == 0:
        wandb_run.log(
            {"validation_predictions": validation_images, **metrics, "epoch": epoch}
        )
    elif wandb_run:
        wandb_run.log({**metrics, "epoch": epoch})

    return metrics


def save_checkpoint(
    model, optimizer, epoch, metrics, args, is_best=False, wandb_run=None
):
    """
    Save model checkpoint

    Args:
        model: YOLOv3 model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Training metrics
        args: Command line arguments
        is_best: Whether this is the best model so far
        wandb_run: Weights & Biases run object for logging
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": model.config,
    }

    checkpoint_path = os.path.join(
        args.output_dir, f"yolov3_checkpoint_epoch_{epoch}.pt"
    )
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    if is_best:
        best_path = os.path.join(args.output_dir, "yolov3_best.pt")
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")

        # Log best model to W&B
        if wandb_run:
            wandb.save(best_path)


def setup_environment(args):
    """
    Set up training environment

    Args:
        args: Command line arguments

    Returns:
        tuple: Updated args, device, wandb_run
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create unique run name with timestamp if not provided
    if not args.run_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"yolov3_{timestamp}"

    # Create output directory for this run
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

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


def create_datasets(args):
    """
    Create datasets and dataloaders

    Args:
        args: Command line arguments

    Returns:
        tuple: train_loader, val_loader, test_loader (None if not requested)
    """
    if args.dataset == "voc":
        # Create train dataset
        train_dataset = PascalVOCDataset(split=args.train_split, year=args.year)
        train_collate_fn = train_dataset.collate_fn

        # Create validation dataset
        val_dataset = PascalVOCDataset(split=args.val_split, year=args.year)
        val_collate_fn = val_dataset.collate_fn

        # Create test dataset if specified
        test_dataset = None
        test_collate_fn = None
        if args.test_split:
            test_dataset = PascalVOCDataset(split=args.test_split, year=args.year)
            test_collate_fn = test_dataset.collate_fn
    else:
        # In future, add BDD100K dataset
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented yet")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=train_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=val_collate_fn,
    )

    # Create test loader if test dataset exists
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=test_collate_fn,
        )

    print(f"Created datasets: Train ({len(train_dataset)}), Val ({len(val_dataset)})")
    if test_loader:
        print(f"Test ({len(test_dataset)})")

    return train_loader, val_loader, test_loader


def create_model(args, device, wandb_run=None):
    """
    Create model, loss function and optimizer

    Args:
        args: Command line arguments
        device: Device to use
        wandb_run: Weights & Biases run object for logging

    Returns:
        tuple: model, loss_fn, optimizer, lr_scheduler
    """
    # Create model config
    config = YOLOv3Config(
        input_size=args.input_size,
        num_classes=args.num_classes,
        darknet_pretrained=args.pretrained,
        darknet_weights_path=os.getenv("DARKNET53_WEIGHTS"),  # Use environment variable
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )

    # Create model
    model = YOLOv3(config)
    model.to(device)

    # Log model summary to W&B
    if wandb_run:
        wandb.watch(model, log="all", log_freq=100)

    # Load weights if provided
    if args.weights:
        print(f"Loading weights from {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    elif args.pretrained:
        print("Loading pretrained backbone weights")
        model.backbone.load_pretrained()

    # Create loss function
    loss_fn = YOLOv3Loss(config)

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate / 100,
    )

    return model, loss_fn, optimizer, lr_scheduler


def train_with_frozen_backbone(
    model, train_loader, val_loader, loss_fn, args, device, wandb_run=None
):
    """
    Train model with frozen backbone

    Args:
        model: YOLOv3 model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        loss_fn: Loss function
        args: Command line arguments
        device: Device to use
        wandb_run: Weights & Biases run object for logging

    Returns:
        model: Updated model with unfrozen backbone
    """
    if not (args.freeze_backbone and args.freeze_epochs > 0):
        return model

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
        metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch, args, wandb_run
        )

        # Validate model
        if epoch % args.eval_interval == 0:
            val_metrics = validate(
                model, val_loader, loss_fn, device, epoch, wandb_run, args
            )
            metrics.update(val_metrics)

        # Save checkpoint
        if epoch % args.checkpoint_interval == 0 or epoch == args.freeze_epochs:
            save_checkpoint(model, optimizer, epoch, metrics, args, wandb_run=wandb_run)

    # Unfreeze backbone for full training
    for param in model.backbone.parameters():
        param.requires_grad = True
    print("Unfrozen backbone for full training")

    return model


def train_model(
    model, train_loader, val_loader, test_loader, loss_fn, args, device, wandb_run=None
):
    """
    Train model

    Args:
        model: YOLOv3 model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        loss_fn: Loss function
        args: Command line arguments
        device: Device to use
        wandb_run: Weights & Biases run object for logging
    """
    # First, train with frozen backbone if specified
    model = train_with_frozen_backbone(
        model, train_loader, val_loader, loss_fn, args, device, wandb_run
    )

    # Create optimizer for full training
    if args.freeze_backbone and args.freeze_epochs > 0:
        # Reset learning rate after unfreezing
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate / 10,  # Use lower learning rate after unfreezing
            weight_decay=args.weight_decay,
        )
        # Create learning rate scheduler for remaining epochs
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - args.freeze_epochs,
            eta_min=args.learning_rate / 100,
        )
        start_epoch = args.freeze_epochs + 1
    else:
        # No frozen training, train everything from scratch
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        # Create learning rate scheduler for all epochs
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.learning_rate / 100,
        )
        start_epoch = 1

    # Initialize best validation metrics
    best_val_loss = float("inf")
    best_val_mAP = 0.0

    # Main training loop
    for epoch in range(start_epoch, args.epochs + 1):
        # Train one epoch
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch, args, wandb_run
        )

        # Validate model
        val_metrics = validate(
            model, val_loader, loss_fn, device, epoch, wandb_run, args
        )

        # Check if this is the best model
        is_best_loss = val_metrics["val_loss"] < best_val_loss
        if is_best_loss:
            best_val_loss = val_metrics["val_loss"]

        is_best_mAP = val_metrics.get("val_mAP", 0) > best_val_mAP
        if is_best_mAP:
            best_val_mAP = val_metrics.get("val_mAP", 0)

        is_best = is_best_mAP or is_best_loss

        # Save checkpoint
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {**train_metrics, **val_metrics},
                args,
                is_best,
                wandb_run,
            )

        # Update learning rate
        lr_scheduler.step()

    # Save final model
    final_path = os.path.join(args.output_dir, "yolov3_final.pt")
    torch.save(
        {"model_state_dict": model.state_dict(), "config": model.config}, final_path
    )
    print(f"Training completed. Saved final model to {final_path}")

    # Run final evaluation on test set if available
    if test_loader:
        print("Running final evaluation on test set...")
        test_metrics = validate(
            model, test_loader, loss_fn, device, args.epochs, wandb_run, args
        )
        print(f"Test set results: mAP: {test_metrics.get('val_mAP', 0):.4f}")

        # Save test results
        test_results_path = os.path.join(args.output_dir, "test_results.pt")
        torch.save(test_metrics, test_results_path)
        print(f"Saved test results to {test_results_path}")


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()

    # Set up environment
    args, device, wandb_run = setup_environment(args)

    # Create datasets and dataloaders
    train_loader, val_loader, test_loader = create_datasets(args)

    # Create model, loss function, and optimizer
    model, loss_fn, optimizer, lr_scheduler = create_model(args, device, wandb_run)

    # Train model
    train_model(
        model, train_loader, val_loader, test_loader, loss_fn, args, device, wandb_run
    )

    # Finish W&B run
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
