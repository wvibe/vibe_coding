#!/usr/bin/env python
"""
Run inference with YOLOv3 model on a few VOC images to verify the implementation
Uses the existing inference.py module for consistency
"""

import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../.."))
yolov3_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

from data_loaders.object_detection.voc import PascalVOCDataset
from models.vanilla.yolov3.config import YOLOv3Config
from models.vanilla.yolov3.inference import draw_detections, get_class_names, postprocess_detections
from models.vanilla.yolov3.yolov3 import YOLOv3

# Configuration
OUTPUT_DIR = os.path.join(yolov3_dir, "inference_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def custom_preprocess_image(image_path, input_size=416):
    """
    Preprocess image for inference - custom implementation to avoid normalization issues

    Args:
        image_path: Path to input image
        input_size: Input size for the model

    Returns:
        tuple: (original_image, processed_tensor, scale)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert from BGR to RGB
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get original image dimensions
    height, width = original_image.shape[:2]

    # Calculate scale factor
    scale = min(input_size / width, input_size / height)

    # Resize image
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(original_image, (new_width, new_height))

    # Create canvas with input_size x input_size
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)

    # Paste resized image onto canvas
    canvas[:new_height, :new_width, :] = resized_image

    # Convert to tensor
    tensor = torch.from_numpy(canvas.transpose(2, 0, 1)).float() / 255.0

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return original_image, tensor, (scale, width, height)


def run_inference():
    """Run inference on a few VOC images"""
    # Load environment variables
    load_dotenv(os.path.join(project_root, ".env"))

    # Get configuration
    darknet_weights = os.environ.get("DARKNET53_WEIGHTS")
    config = YOLOv3Config(
        darknet_pretrained=True, darknet_weights_path=darknet_weights if darknet_weights else None
    )

    # Setup device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load model
    print("Loading YOLOv3 model with pretrained weights...")
    model = YOLOv3(config)
    model.to(device)
    model.eval()

    # Get class names for VOC dataset
    class_names = get_class_names("voc")

    # Load dataset to sample images
    try:
        dataset = PascalVOCDataset(years=["2007"], split="val", debug_mode=True)
        print(f"Loaded dataset with {len(dataset)} images")

        # Randomly select a few images for inference
        num_samples = min(5, len(dataset))
        sample_indices = random.sample(range(len(dataset)), num_samples)

        print(f"\n=== Running inference on {num_samples} random VOC images ===\n")

        for i, idx in enumerate(sample_indices):
            sample = dataset[idx]
            img_id = sample["image_id"]
            # Get image path directly from the dataset's image_info
            img_path = dataset.image_info[idx]["image_path"]

            print(f"Sample {i + 1}/{num_samples}: {img_id}")
            print(f"  Processing image: {img_path}")

            # Preprocess image using our custom function
            original_image, tensor, scale_info = custom_preprocess_image(
                img_path, config.input_size
            )
            tensor = tensor.to(device)

            # Run inference
            with torch.no_grad():
                detections = model.predict(
                    tensor,
                    conf_threshold=0.1,
                    nms_threshold=0.4,
                )[0]  # Get detections for the first image in batch

            # Postprocess detections to original image coordinates
            processed_detections = postprocess_detections(detections, scale_info, config.input_size)

            # Draw and save detections
            if processed_detections.shape[0] > 0:
                print(f"  Found {processed_detections.shape[0]} detections")
                fig = draw_detections(
                    original_image,
                    processed_detections,
                    class_names,
                    threshold=0.1,
                )

                # Save output image
                output_path = os.path.join(OUTPUT_DIR, f"detection_{i}_{Path(img_path).name}")
                fig.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
                print(f"  Saved detection result to {output_path}")
                plt.close(fig)
            else:
                print("  No detections found")

        print(f"\n=== Inference completed. Results saved to {OUTPUT_DIR} ===")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_inference()
