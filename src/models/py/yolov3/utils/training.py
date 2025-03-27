"""
Training utilities and functions for YOLOv3
"""

from pathlib import Path

import torch
from tqdm import tqdm

import wandb

from .visualization import generate_validation_visualizations


def train_one_epoch(model, train_loader, optimizer, device, epoch, args, ema=None):
    """
    Train model for one epoch

    Args:
        model: YOLOv3 model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        args: Training arguments
        ema: Exponential moving average of model weights

    Returns:
        dict: Training metrics for the epoch
    """
    model.train()
    total_loss = 0
    total_obj_loss = 0
    total_box_loss = 0
    total_cls_loss = 0

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        images = batch["images"].to(device)
        targets = {
            "boxes": [boxes.to(device) for boxes in batch["boxes"]],
            "labels": [labels.to(device) for labels in batch["labels"]],
        }

        # Forward pass
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update EMA parameters
        if ema is not None:
            ema.update_params()

        # Update metrics
        total_loss += loss.item()
        total_obj_loss += loss_dict["obj_loss"].item()
        total_box_loss += loss_dict["box_loss"].item()
        total_cls_loss += loss_dict["cls_loss"].item()

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "obj_loss": f"{loss_dict['obj_loss'].item():.4f}",
                "box_loss": f"{loss_dict['box_loss'].item():.4f}",
                "cls_loss": f"{loss_dict['cls_loss'].item():.4f}",
            }
        )

        # Log to wandb
        if args.use_wandb:
            wandb.log(
                {
                    "batch/loss": loss.item(),
                    "batch/obj_loss": loss_dict["obj_loss"].item(),
                    "batch/box_loss": loss_dict["box_loss"].item(),
                    "batch/cls_loss": loss_dict["cls_loss"].item(),
                }
            )

    # Calculate epoch metrics
    num_batches = len(train_loader)
    metrics = {
        "loss": total_loss / num_batches,
        "obj_loss": total_obj_loss / num_batches,
        "box_loss": total_box_loss / num_batches,
        "cls_loss": total_cls_loss / num_batches,
    }

    return metrics


@torch.no_grad()
def evaluate(model, val_loader, device, epoch, args):
    """
    Evaluate model on validation set

    Args:
        model: YOLOv3 model
        val_loader: Validation data loader
        device: Device to use
        epoch: Current epoch number
        args: Training arguments

    Returns:
        dict: Validation metrics
    """
    model.eval()
    total_loss = 0
    total_obj_loss = 0
    total_box_loss = 0
    total_cls_loss = 0

    pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
    for batch in pbar:
        # Move batch to device
        images = batch["images"].to(device)
        targets = {
            "boxes": [boxes.to(device) for boxes in batch["boxes"]],
            "labels": [labels.to(device) for labels in batch["labels"]],
        }

        # Forward pass
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        # Update metrics
        total_loss += loss.item()
        total_obj_loss += loss_dict["obj_loss"].item()
        total_box_loss += loss_dict["box_loss"].item()
        total_cls_loss += loss_dict["cls_loss"].item()

    # Calculate validation metrics
    num_batches = len(val_loader)
    metrics = {
        "val_loss": total_loss / num_batches,
        "val_obj_loss": total_obj_loss / num_batches,
        "val_box_loss": total_box_loss / num_batches,
        "val_cls_loss": total_cls_loss / num_batches,
    }

    # Generate and log validation visualizations
    if args.use_wandb:
        validation_images = generate_validation_visualizations(model, val_loader, device)
        wandb.log({"validation_predictions": validation_images}, commit=False)

    return metrics


def save_checkpoint(model, optimizer, ema, epoch, metrics, args):
    """
    Save model checkpoint

    Args:
        model: YOLOv3 model
        optimizer: Optimizer
        ema: Exponential moving average of model weights
        epoch: Current epoch number
        metrics: Current metrics
        args: Training arguments
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    if ema is not None:
        checkpoint["ema_state_dict"] = ema.state_dict()

    # Save latest checkpoint
    latest_path = Path(args.output_dir) / "latest.pth"
    torch.save(checkpoint, latest_path)

    # Save checkpoint if it's time
    if (epoch + 1) % args.checkpoint_interval == 0:
        checkpoint_path = Path(args.output_dir) / f"checkpoint_{epoch + 1}.pth"
        torch.save(checkpoint, checkpoint_path)

    # Log checkpoint to wandb
    if args.use_wandb:
        wandb.save(str(latest_path))
