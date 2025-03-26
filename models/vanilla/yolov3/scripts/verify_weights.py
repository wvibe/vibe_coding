#!/usr/bin/env python
"""
Verify the pretrained Darknet53 weights
"""

import os
import sys
from pathlib import Path

import torch

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../.."))
sys.path.append(project_root)

from dotenv import load_dotenv

from models.vanilla.yolov3.darknet import Darknet53


def main():
    """Verify the pretrained Darknet53 weights"""
    # Load environment variables
    load_dotenv()

    # Get weights path from environment
    weights_path = os.environ.get("DARKNET53_WEIGHTS")
    if not weights_path:
        print("Error: DARKNET53_WEIGHTS environment variable not set")
        return False

    # Check if weights file exists
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        return False

    print(f"Weights path: {weights_path}")
    print(f"Weights file size: {Path(weights_path).stat().st_size / (1024 * 1024):.2f} MB")

    try:
        # Load the model
        print("Loading Darknet53 model...")
        model = Darknet53()
        initial_params = sum(p.numel() for p in model.parameters())
        print(f"Initial parameter count: {initial_params}")

        # Load the weights
        print("Loading pretrained weights...")
        state_dict = torch.load(weights_path, map_location="cpu")

        # Check keys
        model_keys = set(model.state_dict().keys())
        weights_keys = set(state_dict.keys())
        print(f"Expected keys: {len(model_keys)}")
        print(f"Model keys: {len(weights_keys)}")
        print(f"Intersection: {len(model_keys.intersection(weights_keys))}")

        # Check for missing keys
        missing_keys = model_keys - weights_keys
        if missing_keys:
            print(f"Warning: {len(missing_keys)} keys missing from weights")
            print("First few missing keys:", list(missing_keys)[:5])

        # Load weights
        model.load_state_dict(state_dict)

        # Verify forward pass works
        print("Testing forward pass...")
        dummy_input = torch.randn(1, 3, 416, 416)
        with torch.no_grad():
            features = model(dummy_input)

        # Print feature shapes
        print("Forward pass successful!")
        for i, feature in enumerate(features):
            print(f"  Feature {i} shape: {feature.shape}")

        print("\nVerification successful! Darknet53 weights are correctly loaded.")
        return True

    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
