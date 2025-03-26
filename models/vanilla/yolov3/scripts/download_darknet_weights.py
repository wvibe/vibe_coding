#!/usr/bin/env python
"""
Script to download and convert Darknet53 weights
The weights are originally trained on ImageNet classification

Note: If you're behind a firewall or in regions with restricted internet access,
you may need to set up proxy in your environment before running this script.
"""

import os
import sys
from pathlib import Path

import numpy as np
import requests
import torch
from dotenv import load_dotenv

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[3]
sys.path.append(str(project_root))


def download_file(url, local_path):
    """Download a file from URL to local path with progress indicator"""
    print(f"Downloading from {url} to {local_path}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Get file size for progress reporting
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        # Write the file
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Print progress
                    if total_size > 0:
                        percent = int(100 * downloaded / total_size)
                        sys.stdout.write(
                            f"\rProgress: {percent}% ({downloaded / 1024 / 1024:.1f} MB)"
                        )
                        sys.stdout.flush()

        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def convert_weights_to_pytorch(weights_path, output_path):
    """Convert Darknet weights to PyTorch format"""
    print("Converting weights to PyTorch format...")

    try:
        # Import here to ensure we have access to the model
        from models.vanilla.yolov3.darknet import Darknet53

        # Create model
        model = Darknet53()

        # Load weights
        with open(weights_path, "rb") as f:
            # Skip header
            header = np.fromfile(f, dtype=np.int32, count=3)
            weights = np.fromfile(f, dtype=np.float32)

        ptr = 0
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                # Conv layer has bias if not followed by BN
                has_bias = not isinstance(getattr(m, "bias", None), type(None))
                conv_w = torch.from_numpy(weights[ptr : ptr + m.weight.numel()]).view_as(m.weight)
                ptr += m.weight.numel()
                m.weight.data.copy_(conv_w)

                if has_bias:
                    conv_b = torch.from_numpy(weights[ptr : ptr + m.bias.numel()]).view_as(m.bias)
                    ptr += m.bias.numel()
                    m.bias.data.copy_(conv_b)

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_w = torch.from_numpy(weights[ptr : ptr + m.weight.numel()]).view_as(m.weight)
                ptr += m.weight.numel()
                bn_b = torch.from_numpy(weights[ptr : ptr + m.bias.numel()]).view_as(m.bias)
                ptr += m.bias.numel()
                bn_rm = torch.from_numpy(weights[ptr : ptr + m.running_mean.numel()]).view_as(
                    m.running_mean
                )
                ptr += m.running_mean.numel()
                bn_rv = torch.from_numpy(weights[ptr : ptr + m.running_var.numel()]).view_as(
                    m.running_var
                )
                ptr += m.running_var.numel()

                m.weight.data.copy_(bn_w)
                m.bias.data.copy_(bn_b)
                m.running_mean.data.copy_(bn_rm)
                m.running_var.data.copy_(bn_rv)

        print(f"Converted weights: {ptr}/{len(weights)} values used")

        # Save converted weights
        torch.save(model.state_dict(), output_path)
        print(f"Saved converted weights to {output_path}")

        # Remove the original weights file to save space
        os.remove(weights_path)
        return True

    except Exception as e:
        print(f"Error during conversion: {e}")
        return False


def main():
    """Main function to download and convert weights"""
    # Load environment variables
    load_dotenv()

    # Get paths from environment
    darknet_weights_path = os.getenv("DARKNET53_WEIGHTS")
    if not darknet_weights_path:
        print("Error: DARKNET53_WEIGHTS environment variable not set")
        return False

    # Create weights directory if it doesn't exist
    weights_dir = os.path.dirname(darknet_weights_path)
    os.makedirs(weights_dir, exist_ok=True)

    # Check if weights already exist
    if os.path.exists(darknet_weights_path):
        print(f"Darknet53 weights already exist at {darknet_weights_path}")
        return True

    # Download Darknet53 weights
    temp_weights_path = os.path.join(weights_dir, "darknet53.conv.74")
    download_url = "https://pjreddie.com/media/files/darknet53.conv.74"

    if not download_file(download_url, temp_weights_path):
        return False

    # Convert weights to PyTorch format
    if not convert_weights_to_pytorch(temp_weights_path, darknet_weights_path):
        return False

    print("Darknet53 weights successfully downloaded and converted!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
