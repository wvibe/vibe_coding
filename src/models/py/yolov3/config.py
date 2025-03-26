"""
YOLOv3 model configuration parameters
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class YOLOv3Config:
    """Configuration settings for YOLOv3 model"""

    # Input parameters
    input_size: int = 416
    num_classes: int = 20  # Default for Pascal VOC

    # Model architecture parameters
    darknet_pretrained: bool = True
    darknet_weights_path: str = None

    # Custom anchor boxes for Pascal VOC (width, height) - computed with k-means
    # These anchors are better suited for VOC dataset compared to default COCO anchors
    anchors: List[Tuple[float, float]] = (
        # Scale 1 (13x13) - for detecting large objects
        (119.2, 207.5),
        (216.4, 208.8),
        (252.6, 305.8),
        # Scale 2 (26x26) - for detecting medium objects
        (45.2, 90.4),
        (81.9, 151.5),
        (134.7, 127.0),
        # Scale 3 (52x52) - for detecting small objects
        (11.9, 19.9),
        (24.8, 52.5),
        (45.1, 39.3),
    )

    # Anchors grouped by detection scale
    anchors_per_scale: int = 3

    # Detection parameters
    conf_threshold: float = 0.5  # Confidence threshold for inference
    nms_threshold: float = 0.4  # Non-maximum suppression IoU threshold
    eval_conf_threshold: float = 0.1  # Confidence threshold for evaluation (lower to ensure recall)

    # Training parameters - adjusted for better stability
    learning_rate: float = (
        5e-4  # Reduced from 1e-3 for more stable training with pretrained weights
    )
    weight_decay: float = 1e-4  # Reduced from 5e-4
    momentum: float = 0.9
    lr_scheduler: str = "cosine"  # Options: "cosine", "step", "multistep"

    # Learning rate scheduler parameters
    warmup_epochs: int = 3  # Number of epochs for warmup
    lr_decay_epochs: List[int] = (30, 60, 90)  # For step and multistep schedulers
    lr_decay_gamma: float = 0.1  # For step and multistep schedulers
    min_lr_factor: float = 0.01  # Minimum LR as a factor of initial LR for cosine scheduler

    # Optimizer settings
    optimizer: str = "adam"  # Options: "adam", "sgd"
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    sgd_momentum: float = 0.9

    # Gradient clipping value (0 to disable)
    grad_clip_norm: float = 10.0

    # Multiscale detection
    scales: List[int] = (13, 26, 52)  # Output sizes for 416x416 input

    # Loss function weights - adjusted for balanced training
    lambda_coord: float = 5.0  # Adjusted from 10.0
    lambda_noobj: float = 0.5  # Adjusted from 1.0

    # Debugging and monitoring
    debug_gradients: bool = False  # Monitor gradient norms during training


# Default configuration
DEFAULT_CONFIG = YOLOv3Config()
