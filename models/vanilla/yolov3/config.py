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

    # Anchor boxes (height, width) - default anchors for COCO dataset
    # These can be recomputed for specific datasets using k-means clustering
    anchors: List[Tuple[float, float]] = (
        (10, 13),
        (16, 30),
        (33, 23),  # Small objects
        (30, 61),
        (62, 45),
        (59, 119),  # Medium objects
        (116, 90),
        (156, 198),
        (373, 326),  # Large objects
    )

    # Anchors grouped by detection scale
    anchors_per_scale: int = 3

    # Detection parameters
    conf_threshold: float = 0.5  # Confidence threshold for inference
    nms_threshold: float = 0.4  # Non-maximum suppression IoU threshold
    eval_conf_threshold: float = 0.1  # Confidence threshold for evaluation (lower to ensure recall)

    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    momentum: float = 0.9

    # Multiscale detection
    scales: List[int] = (13, 26, 52)  # Output sizes for 416x416 input

    # Loss function weights - adjusted for better training
    lambda_coord: float = 10.0  # Increased from 5.0 to emphasize localization accuracy
    lambda_noobj: float = 1.0  # Increased from 0.5 to reduce false positives


# Default configuration
DEFAULT_CONFIG = YOLOv3Config()
