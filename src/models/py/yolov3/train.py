"""
YOLOv3 training script
"""

import wandb
from utils.data import create_datasets
from utils.model import EMA, create_model
from utils.setup import parse_args, setup_environment
from utils.training import evaluate, save_checkpoint, train_one_epoch


def main():
    # Parse arguments and setup environment
    args = parse_args()
    device = setup_environment(args)

    # Create datasets and dataloaders
    train_loader, val_loader, test_loader = create_datasets(args)

    # Create model, loss function, optimizer and EMA
    model, loss_fn, optimizer, scheduler = create_model(args, device)
    ema = EMA(model) if args.use_ema else None

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        # Train for one epoch
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch, args, ema)

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Evaluate on validation set
        if ema is not None:
            ema.apply_shadow()
        val_metrics = evaluate(model, val_loader, device, epoch, args)
        if ema is not None:
            ema.restore()

        # Save checkpoint
        metrics = {**train_metrics, **val_metrics}
        save_checkpoint(model, optimizer, ema, epoch, metrics, args)

        # Log metrics
        if args.use_wandb:
            wandb.log(metrics)

        # Update best validation loss
        val_loss = val_metrics["val_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = f"{args.output_dir}/best.pth"
            save_checkpoint(model, optimizer, ema, epoch, metrics, args)

    # Final evaluation on test set
    if test_loader is not None:
        if ema is not None:
            ema.apply_shadow()
        test_metrics = evaluate(model, test_loader, device, epoch, args)
        if ema is not None:
            ema.restore()

        if args.use_wandb:
            wandb.log({"test/" + k: v for k, v in test_metrics.items()})


if __name__ == "__main__":
    main()
