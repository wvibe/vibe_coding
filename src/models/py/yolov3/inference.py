"""
Inference script for YOLOv3 model
"""

import argparse
import os

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.py.yolov3.config import YOLOv3Config
from src.models.py.yolov3.yolov3 import YOLOv3


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="YOLOv3 Inference")

    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image or directory of images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Directory to save output images (default: output)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--nms-threshold", type=float, default=0.4, help="NMS threshold (default: 0.4)"
    )
    parser.add_argument(
        "--input-size", type=int, default=416, help="Input image size (default: 416)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda if available, else cpu)",
    )
    parser.add_argument("--show", action="store_true", help="Show detection results")
    parser.add_argument(
        "--dataset",
        type=str,
        default="voc",
        choices=["voc", "bdd"],
        help="Dataset class names (default: voc)",
    )

    return parser.parse_args()


def preprocess_image(image_path, input_size=416):
    """
    Preprocess image for inference

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

    # Normalize with ImageNet mean and std
    tensor = torch.nn.functional.normalize(
        tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    return original_image, tensor, (scale, width, height)


def draw_detections(image, detections, class_names, threshold=0.5):
    """
    Draw bounding boxes and labels on image

    Args:
        image: Original image
        detections: Detections from model
        class_names: List of class names
        threshold: Confidence threshold for drawing

    Returns:
        image: Image with detections drawn
    """
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # Colors for different classes
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))

    # Draw each detection
    for detection in detections:
        # Skip if confidence is below threshold
        confidence = detection[5].item()
        if confidence < threshold:
            continue

        # Get coordinates (convert to original image size)
        x1, y1, x2, y2 = detection[1:5].cpu().numpy()
        class_id = int(detection[6].item())

        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=colors[class_id],
            facecolor="none",
        )
        ax.add_patch(rect)

        # Add label
        class_name = class_names[class_id]
        label = f"{class_name}: {confidence:.2f}"
        ax.text(
            x1,
            y1 - 5,
            label,
            color=colors[class_id],
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    # Remove axis
    plt.axis("off")
    return fig


def postprocess_detections(detections, scale_info, input_size=416):
    """
    Convert detections from model output to original image coordinates

    Args:
        detections: Detections from model
        scale_info: Scale information (scale, width, height)
        input_size: Input size of the model

    Returns:
        detections: Detections in original image coordinates
    """
    scale, orig_width, orig_height = scale_info

    # If no detections, return empty tensor
    if len(detections) == 0 or detections.shape[0] == 0:
        return torch.zeros((0, 7), device=detections.device)

    # Convert to original image coordinates
    processed_detections = detections.clone()

    # Adjust coordinates back to original image size
    processed_detections[:, 1] /= scale  # x1
    processed_detections[:, 2] /= scale  # y1
    processed_detections[:, 3] /= scale  # x2
    processed_detections[:, 4] /= scale  # y2

    # Clip to image boundaries
    processed_detections[:, 1].clamp_(0, orig_width)  # x1
    processed_detections[:, 2].clamp_(0, orig_height)  # y1
    processed_detections[:, 3].clamp_(0, orig_width)  # x2
    processed_detections[:, 4].clamp_(0, orig_height)  # y2

    return processed_detections


def get_class_names(dataset_name):
    """
    Get class names for a dataset

    Args:
        dataset_name: Name of the dataset

    Returns:
        list: Class names
    """
    if dataset_name == "voc":
        return [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
    elif dataset_name == "bdd":
        return [
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
            "traffic light",
            "traffic sign",
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    """Main inference function"""
    args = parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get class names
    class_names = get_class_names(args.dataset)
    num_classes = len(class_names)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load model
    try:
        print(f"Loading model from {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device)

        if "config" in checkpoint:
            config = checkpoint["config"]
        else:
            config = YOLOv3Config(input_size=args.input_size, num_classes=num_classes)

        model = YOLOv3(config)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Process input (file or directory)
    if os.path.isfile(args.input):
        input_paths = [args.input]
    elif os.path.isdir(args.input):
        input_paths = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    else:
        print(f"Input path does not exist: {args.input}")
        return

    # Process each image
    for image_path in input_paths:
        try:
            print(f"Processing: {image_path}")

            # Preprocess image
            original_image, tensor, scale_info = preprocess_image(image_path, args.input_size)
            tensor = tensor.to(device)

            # Run inference
            with torch.no_grad():
                detections = model.predict(
                    tensor,
                    conf_threshold=args.conf_threshold,
                    nms_threshold=args.nms_threshold,
                )[0]  # Get detections for the first (and only) image in batch

            # Postprocess detections to original image size
            processed_detections = postprocess_detections(detections, scale_info, args.input_size)

            # Draw detections
            if processed_detections.shape[0] > 0:
                print(f"Found {processed_detections.shape[0]} detections")
                fig = draw_detections(
                    original_image,
                    processed_detections,
                    class_names,
                    args.conf_threshold,
                )

                # Save output image
                output_path = os.path.join(
                    args.output,
                    f"{os.path.splitext(os.path.basename(image_path))[0]}_detection.png",
                )
                fig.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
                print(f"Saved detection result to {output_path}")

                # Show if requested
                if args.show:
                    plt.show()
                else:
                    plt.close(fig)
            else:
                print("No detections found")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print("Inference completed")


if __name__ == "__main__":
    main()
