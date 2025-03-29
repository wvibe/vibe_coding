"""
Script for running inference with Ultralytics YOLO models
"""

import argparse

from ultralytics import YOLO

from utils import resolve_source_path


def run_inference(
    model_name="yolo11n.pt",  # Model name or path
    source=None,  # Path to image/video or directory
    conf=0.25,  # Confidence threshold
    save=False,  # Save results (default: False)
    show=True,  # Display results
    data_yaml=None,  # Optional YAML file with class names
    project=None,  # Project directory for saving results
    **kwargs,  # Additional arguments for model.predict()
):
    """
    Run inference using a YOLO model

    Args:
        model_name: Name or path of model
        source: Path to image/video or directory
        conf: Confidence threshold
        save: Save results to disk (default: False)
        show: Display results
        data_yaml: Path to YAML file with class names (optional)
        project: Project directory for saving results (only used if save=True)
        **kwargs: Additional arguments for model.predict()

    Returns:
        Results from prediction
    """
    # Validate required parameters
    if not source:
        raise ValueError("Source must be provided")

    # Set up prediction arguments from function parameters
    predict_args = {"conf": conf, "save": save, "show": show, **kwargs}

    # Add project directory only if we're saving results
    if save and project:
        predict_args["project"] = project

    # If data YAML is provided
    if data_yaml:
        predict_args["data"] = data_yaml

    # Resolve source path using environment variables
    source_path = resolve_source_path(source)
    predict_args["source"] = source_path
    print(f"Using source: {source_path}")

    # Load model
    model = YOLO(model_name)

    # Run inference
    print(f"Running inference with parameters: {predict_args}")
    results = model.predict(**predict_args)

    return results


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Model path or name")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source path (image/video/dir). Can use dataset prefixes like 'VOC2007/JPEGImages/000001.jpg'",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--save", action="store_true", help="Save results to disk")
    parser.add_argument("--show", action="store_true", help="Show results")
    parser.add_argument("--data", type=str, help="Path to dataset YAML file with class names")
    parser.add_argument(
        "--project", type=str, help="Project directory for saving results (only if --save is used)"
    )

    args = parser.parse_args()

    results = run_inference(
        model_name=args.model,
        source=args.source,
        conf=args.conf,
        save=args.save,
        show=args.show,
        data_yaml=args.data,
        iou=args.iou,
        project=args.project,
    )

    # Print detection counts
    for i, r in enumerate(results):
        print(f"Image {i + 1}: {len(r.boxes)} detections")
