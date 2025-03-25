#!/bin/bash

# Script to download and convert Darknet53 weights
# The weights are originally trained on ImageNet classification
#
# Note: If you're behind a firewall or in regions with restricted internet access (like China),
# you may need to set up a proxy before running this script:
# export http_proxy=http://your.proxy.address:port
# export https_proxy=http://your.proxy.address:port
# export no_proxy=localhost,127.0.0.1
#
# Example:
# export http_proxy=http://192.168.0.250:10809
# export https_proxy=http://192.168.0.250:10809
# export no_proxy=192.168.0.0/16,localhost,127.0.0.1

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

    # Use wget directly for simple download
    echo "Downloading weights to ${WEIGHTS_DIR}/darknet53.conv.74..."

    # Try to download using wget with proxy settings
    wget -O "${WEIGHTS_DIR}/darknet53.conv.74" "https://pjreddie.com/media/files/darknet53.conv.74" || {
        echo "Failed to download using wget, trying with curl..."
        curl -L "https://pjreddie.com/media/files/darknet53.conv.74" -o "${WEIGHTS_DIR}/darknet53.conv.74"
    }

    if [ $? -ne 0 ]; then
        echo "Error: Failed to download weights. Please check your internet connection or proxy settings."
        exit 1
    fi

    echo "Converting weights to PyTorch format..."
    # Create a simple standalone Python script for conversion
    python - << END
import torch
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, "${PROJECT_ROOT}")

try:
    from models.vanilla.yolov3.darknet import Darknet53

    # Create model
    model = Darknet53()

    # Load weights
    weights_path = "${WEIGHTS_DIR}/darknet53.conv.74"
    output_path = "${DARKNET53_WEIGHTS}"

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

    # Remove the original weights file to save space
    os.remove(weights_path)
    print("Conversion completed successfully")

except Exception as e:
    print(f"Error during conversion: {e}")
    sys.exit(1)
END

    if [ $? -ne 0 ]; then
        echo "Error: Failed to convert weights to PyTorch format."
        exit 1
    fi

    echo "Darknet53 weights successfully downloaded and converted to ${DARKNET53_WEIGHTS}"
else
    echo "Darknet53 weights already exist at ${DARKNET53_WEIGHTS}"
fi