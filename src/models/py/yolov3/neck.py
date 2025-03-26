"""
Feature Pyramid Network (FPN) implementation for YOLOv3
This module connects the Darknet-53 backbone to the detection heads
"""

import torch
import torch.nn as nn

from src.models.py.yolov3.darknet import ConvBlock


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for YOLOv3

    Takes features from Darknet-53 backbone at three scales and
    creates a feature pyramid by upsampling and concatenating features.
    """

    def __init__(self):
        super().__init__()

        # Convolutions for the largest scale (13x13)
        self.conv1_1 = ConvBlock(1024, 512, kernel_size=1)
        self.conv1_2 = ConvBlock(512, 1024, kernel_size=3)
        self.conv1_3 = ConvBlock(1024, 512, kernel_size=1)
        self.conv1_4 = ConvBlock(512, 1024, kernel_size=3)
        self.conv1_5 = ConvBlock(1024, 512, kernel_size=1)

        # Convolutions for the medium scale (26x26)
        # First, we need to reduce channels in the large scale for upsampling
        self.conv2_reduce = ConvBlock(512, 256, kernel_size=1)
        # Upsample from 13x13 to 26x26
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")

        # Process concatenated feature map (26x26)
        self.conv2_1 = ConvBlock(256 + 512, 256, kernel_size=1)  # 512 from route_2 of Darknet
        self.conv2_2 = ConvBlock(256, 512, kernel_size=3)
        self.conv2_3 = ConvBlock(512, 256, kernel_size=1)
        self.conv2_4 = ConvBlock(256, 512, kernel_size=3)
        self.conv2_5 = ConvBlock(512, 256, kernel_size=1)

        # Convolutions for the small scale (52x52)
        # First, we need to reduce channels in the medium scale for upsampling
        self.conv3_reduce = ConvBlock(256, 128, kernel_size=1)
        # Upsample from 26x26 to 52x52
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")

        # Process concatenated feature map (52x52)
        self.conv3_1 = ConvBlock(128 + 256, 128, kernel_size=1)  # 256 from route_1 of Darknet
        self.conv3_2 = ConvBlock(128, 256, kernel_size=3)
        self.conv3_3 = ConvBlock(256, 128, kernel_size=1)
        self.conv3_4 = ConvBlock(128, 256, kernel_size=3)
        self.conv3_5 = ConvBlock(256, 128, kernel_size=1)

    def forward(self, features):
        """
        Forward pass of Feature Pyramid Network

        Args:
            features: Tuple of feature maps from Darknet-53
                      (large_scale, medium_scale, small_scale)
                      large_scale: (batch_size, 1024, 13, 13)
                      medium_scale: (batch_size, 512, 26, 26)
                      small_scale: (batch_size, 256, 52, 52)

        Returns:
            tuple: Three processed feature maps for detection at different scales
                   large_scale_out: (batch_size, 512, 13, 13)
                   medium_scale_out: (batch_size, 256, 26, 26)
                   small_scale_out: (batch_size, 128, 52, 52)
        """
        large_scale, medium_scale, small_scale = features

        # Process large scale (13x13)
        x_large = self.conv1_1(large_scale)
        x_large = self.conv1_2(x_large)
        x_large = self.conv1_3(x_large)
        x_large = self.conv1_4(x_large)
        x_large = self.conv1_5(x_large)
        large_scale_out = x_large  # Output for largest scale detection head

        # Prepare for medium scale
        x_medium = self.conv2_reduce(x_large)
        x_medium = self.upsample1(x_medium)
        x_medium = torch.cat([x_medium, medium_scale], dim=1)

        # Process medium scale (26x26)
        x_medium = self.conv2_1(x_medium)
        x_medium = self.conv2_2(x_medium)
        x_medium = self.conv2_3(x_medium)
        x_medium = self.conv2_4(x_medium)
        x_medium = self.conv2_5(x_medium)
        medium_scale_out = x_medium  # Output for medium scale detection head

        # Prepare for small scale
        x_small = self.conv3_reduce(x_medium)
        x_small = self.upsample2(x_small)
        x_small = torch.cat([x_small, small_scale], dim=1)

        # Process small scale (52x52)
        x_small = self.conv3_1(x_small)
        x_small = self.conv3_2(x_small)
        x_small = self.conv3_3(x_small)
        x_small = self.conv3_4(x_small)
        x_small = self.conv3_5(x_small)
        small_scale_out = x_small  # Output for small scale detection head

        return large_scale_out, medium_scale_out, small_scale_out
