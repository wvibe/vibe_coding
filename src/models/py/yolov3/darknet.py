"""
Darknet-53 backbone implementation for YOLOv3
"""

import os

import torch
import torch.nn as nn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and LeakyReLU"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block used in Darknet-53"""

    def __init__(self, channels):
        super().__init__()
        reduced_channels = channels // 2

        self.conv1 = ConvBlock(channels, reduced_channels, kernel_size=1)
        self.conv2 = ConvBlock(reduced_channels, channels, kernel_size=3)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return residual + out


class Darknet53(nn.Module):
    """
    Darknet-53 backbone architecture for YOLOv3

    Returns intermediate feature maps for feature pyramid network
    """

    def __init__(self):
        super().__init__()

        # Initial convolution
        self.conv1 = ConvBlock(3, 32, kernel_size=3)

        # Downsample 1: 416 -> 208
        self.conv2 = ConvBlock(32, 64, kernel_size=3, stride=2)
        self.res_block1 = self._make_layer(64, num_blocks=1)

        # Downsample 2: 208 -> 104
        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=2)
        self.res_block2 = self._make_layer(128, num_blocks=2)

        # Downsample 3: 104 -> 52 (Route 1)
        self.conv4 = ConvBlock(128, 256, kernel_size=3, stride=2)
        self.res_block3 = self._make_layer(256, num_blocks=8)

        # Downsample 4: 52 -> 26 (Route 2)
        self.conv5 = ConvBlock(256, 512, kernel_size=3, stride=2)
        self.res_block4 = self._make_layer(512, num_blocks=8)

        # Downsample 5: 26 -> 13 (Route 3)
        self.conv6 = ConvBlock(512, 1024, kernel_size=3, stride=2)
        self.res_block5 = self._make_layer(1024, num_blocks=4)

    def _make_layer(self, channels, num_blocks):
        """Create a layer with the specified number of residual blocks"""
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of Darknet-53

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            tuple: Three feature maps at different scales for the FPN
                  (batch_size, 1024, 13, 13)
                  (batch_size, 512, 26, 26)
                  (batch_size, 256, 52, 52)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res_block1(x)

        x = self.conv3(x)
        x = self.res_block2(x)

        x = self.conv4(x)
        x = self.res_block3(x)
        route_1 = x  # 52x52

        x = self.conv5(x)
        x = self.res_block4(x)
        route_2 = x  # 26x26

        x = self.conv6(x)
        x = self.res_block5(x)
        route_3 = x  # 13x13

        return route_3, route_2, route_1

    def load_pretrained(self, weights_path=None):
        """
        Load pretrained weights for Darknet-53

        Args:
            weights_path: Path to pretrained weights file
                         If None, uses default path from environment variable DARKNET53_WEIGHTS
        """
        if weights_path is None:
            weights_path = os.getenv("DARKNET53_WEIGHTS")
            if weights_path is None:
                print("Warning: DARKNET53_WEIGHTS environment variable not set")
                return

        if not os.path.exists(weights_path):
            print(f"Warning: Pretrained weights not found at {weights_path}")
            print(
                "Please run scripts/download_darknet_weights.sh to download and convert the weights"
            )
            return

        try:
            # Load weights
            state_dict = torch.load(weights_path, map_location=torch.device("cpu"))

            # Load weights that match the model
            model_state_dict = self.state_dict()
            for name, param in state_dict.items():
                if name in model_state_dict:
                    if param.shape == model_state_dict[name].shape:
                        model_state_dict[name].copy_(param)
                    else:
                        print(f"Warning: Size mismatch for layer {name}")
                else:
                    print(f"Warning: Layer {name} not found in model")

            print(f"Successfully loaded pretrained weights from {weights_path}")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
