"""
YOLOv3 configuration for person detection
Single-class person detection task
"""

from dataclasses import dataclass
from typing import List, Tuple

from models.vanilla.yolov3.config import YOLOv3Config


@dataclass
class PersonDetectionConfig(YOLOv3Config):
    """Configuration for person detection YOLOv3 model"""

    # Dataset parameters
    num_classes: int = 1  # Only person class
    class_name: str = "person"  # Class to detect

    # Training parameters optimized for single class detection
    learning_rate: float = 1e-4  # Lower learning rate
    weight_decay: float = 1e-5  # Less regularization
    batch_size: int = 16
    num_epochs: int = 60
    warmup_epochs: int = 3

    # Custom anchor boxes for person detection
    # These are optimized for the distribution of person bounding boxes
    anchors: List[Tuple[float, float]] = (
        # Scale 1 (13x13) - for detecting large persons
        (90.0, 205.0),
        (120.0, 275.0),
        (200.0, 320.0),
        # Scale 2 (26x26) - for detecting medium persons
        (45.0, 100.0),
        (70.0, 175.0),
        (100.0, 220.0),
        # Scale 3 (52x52) - for detecting small persons
        (20.0, 35.0),
        (25.0, 75.0),
        (40.0, 120.0),
    )

    # Evaluation parameters
    conf_threshold: float = 0.5
    nms_threshold: float = 0.4
    eval_conf_threshold: float = 0.1  # Lower threshold for evaluation


# Default configuration for person detection
DEFAULT_PERSON_CONFIG = PersonDetectionConfig()
