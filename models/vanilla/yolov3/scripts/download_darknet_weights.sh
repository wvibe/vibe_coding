#!/bin/bash

# Script to download and convert Darknet53 weights
# The weights are originally trained on ImageNet classification

# Change to project root directory to ensure .env file is accessible
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

# Load environment variables
source .env

# Create weights directory if it doesn't exist
WEIGHTS_DIR="$(dirname "${DARKNET53_WEIGHTS}")"
mkdir -p "${WEIGHTS_DIR}"

# Download Darknet53 weights if they don't exist
if [ ! -f "${DARKNET53_WEIGHTS}" ]; then
    echo "Downloading Darknet53 weights..."

    # Use Python to download the weights
    python3 -c '
import os
import requests
import torch
import numpy as np
from models.vanilla.yolov3.darknet import Darknet53

def download_file(url, filename):
    print(f"Downloading {url} to {filename}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192
    downloaded = 0

    with open(filename, "wb") as f:
        for data in response.iter_content(block_size):
            downloaded += len(data)
            f.write(data)
            done = int(50 * downloaded / total_size)
            print(f"\rProgress: [{"=" * done}{" " * (50-done)}] {downloaded}/{total_size} bytes", end="")
    print("\nDownload completed")

def convert_darknet53_weights(weights_path, output_path):
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
            conv_w = torch.from_numpy(
                weights[ptr:ptr + m.weight.numel()]
            ).view_as(m.weight)
            ptr += m.weight.numel()
            m.weight.data.copy_(conv_w)

            if has_bias:
                conv_b = torch.from_numpy(
                    weights[ptr:ptr + m.bias.numel()]
                ).view_as(m.bias)
                ptr += m.bias.numel()
                m.bias.data.copy_(conv_b)

        elif isinstance(m, torch.nn.BatchNorm2d):
            bn_w = torch.from_numpy(
                weights[ptr:ptr + m.weight.numel()]
            ).view_as(m.weight)
            ptr += m.weight.numel()
            bn_b = torch.from_numpy(
                weights[ptr:ptr + m.bias.numel()]
            ).view_as(m.bias)
            ptr += m.bias.numel()
            bn_rm = torch.from_numpy(
                weights[ptr:ptr + m.running_mean.numel()]
            ).view_as(m.running_mean)
            ptr += m.running_mean.numel()
            bn_rv = torch.from_numpy(
                weights[ptr:ptr + m.running_var.numel()]
            ).view_as(m.running_var)
            ptr += m.running_var.numel()

            m.weight.data.copy_(bn_w)
            m.bias.data.copy_(bn_b)
            m.running_mean.data.copy_(bn_rm)
            m.running_var.data.copy_(bn_rv)

    print(f"Converted weights: {ptr}/{len(weights)} values used")

    # Save converted weights
    torch.save(model.state_dict(), output_path)
    print(f"Saved converted weights to {output_path}")

# Download and convert weights
weights_path = "'${WEIGHTS_DIR}/darknet53.weights'"
try:
    download_file("https://pjreddie.com/media/files/darknet53.conv.74", weights_path)
    convert_darknet53_weights(weights_path, "'${DARKNET53_WEIGHTS}'")
    os.remove(weights_path)  # Remove the original weights file
    print("Conversion completed successfully")
except Exception as e:
    print(f"Error during download or conversion: {e}")
'
else
    echo "Darknet53 weights already exist at ${DARKNET53_WEIGHTS}"
fi

# Make the script executable
chmod +x "${SCRIPT_DIR}/download_darknet_weights.sh"