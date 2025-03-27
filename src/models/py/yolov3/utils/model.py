"""
Model creation and EMA utilities for YOLOv3 training
"""

import os

import torch
import torch.optim as optim

import wandb
from src.models.py.yolov3.config import YOLOv3Config
from src.models.py.yolov3.loss import YOLOv3Loss
from src.models.py.yolov3.yolov3 import YOLOv3


class EMA:
    """
    Exponential Moving Average for model weights.
    Maintains shadow parameters to produce a more stable model.
    """

    def __init__(self, model, decay=0.9999):
        """
        Initialize EMA

        Args:
            model: PyTorch model
            decay: EMA decay rate (higher = slower)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """Register model parameters for EMA"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA parameters after each training step"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA parameters to model for evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters for training continuation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def create_model(args, device, wandb_run=None, optimized_anchors=None):
    """
    Create model, loss function and optimizer

    Args:
        args: Command line arguments
        device: Device to use
        wandb_run: Weights & Biases run object for logging
        optimized_anchors: Optimized anchors for VOC dataset (optional)

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

    # Use optimized anchors if available
    if optimized_anchors is not None:
        print("Using optimized anchors for VOC dataset")
        config.anchors = optimized_anchors

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
