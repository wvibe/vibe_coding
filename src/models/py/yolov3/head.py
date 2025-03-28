"""
Detection head implementations for YOLOv3
These heads convert feature maps to bounding box predictions
"""

import torch.nn as nn

from src.models.py.yolov3.darknet import ConvBlock


class DetectionHead(nn.Module):
    """
    YOLOv3 detection head for a single scale

    Takes feature maps from the Feature Pyramid Network and outputs
    bounding box predictions at that scale.
    """

    def __init__(self, in_channels, num_classes):
        """
        Initialize detection head

        Args:
            in_channels: Number of input channels from the FPN
            num_classes: Number of classes to predict
        """
        super().__init__()

        # Each bounding box prediction contains:
        # - 4 box coordinates (tx, ty, tw, th)
        # - 1 objectness score
        # - num_classes class probabilities
        self.num_classes = num_classes
        self.prediction_size = 4 + 1 + num_classes

        # Each grid cell predicts 3 bounding boxes
        self.num_anchors = 3

        # Final output channels = num_anchors * prediction_size
        self.out_channels = self.num_anchors * self.prediction_size

        # Convolutional layers
        self.conv1 = ConvBlock(in_channels, 2 * in_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(2 * in_channels, self.out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of detection head

        Args:
            x: Feature map from the FPN at a specific scale
               e.g., (batch_size, 512, 13, 13) for large scale

        Returns:
            Tensor: Predictions with shape
                   (batch_size, num_anchors, grid_size, grid_size, prediction_size)
                   where prediction_size = 4 + 1 + num_classes
        """
        batch_size = x.size(0)

        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)

        # Reshape to (batch_size, num_anchors, prediction_size, grid_size, grid_size)
        grid_size = x.size(2)
        x = x.view(batch_size, self.num_anchors, self.prediction_size, grid_size, grid_size)  # noqa: E501

        # Permute to (batch_size, num_anchors, grid_size, grid_size, prediction_size)
        x = x.permute(0, 1, 3, 4, 2)

        return x


class YOLOv3Head(nn.Module):
    """
    Complete YOLOv3 detection head for all three scales

    Combines three DetectionHead modules, one for each scale
    """

    def __init__(self, num_classes):
        """
        Initialize YOLOv3 head with detection heads for three scales

        Args:
            num_classes: Number of classes to predict
        """
        super().__init__()

        # Detection heads for each scale
        # Input channels match the output channels from the FPN
        self.large_scale_head = DetectionHead(512, num_classes)  # For 13x13 grid
        self.medium_scale_head = DetectionHead(256, num_classes)  # For 26x26 grid
        self.small_scale_head = DetectionHead(128, num_classes)  # For 52x52 grid

    def forward(self, features):
        """
        Forward pass through all three detection heads

        Args:
            features: Tuple of feature maps from the FPN
                      (large_scale, medium_scale, small_scale)

        Returns:
            tuple: Predictions from all three scales
                   large_scale_pred: For detecting large objects (13x13 grid)
                   medium_scale_pred: For detecting medium objects (26x26 grid)
                   small_scale_pred: For detecting small objects (52x52 grid)
        """
        large_scale, medium_scale, small_scale = features

        # Forward through each detection head
        large_scale_pred = self.large_scale_head(large_scale)
        medium_scale_pred = self.medium_scale_head(medium_scale)
        small_scale_pred = self.small_scale_head(small_scale)

        return large_scale_pred, medium_scale_pred, small_scale_pred
