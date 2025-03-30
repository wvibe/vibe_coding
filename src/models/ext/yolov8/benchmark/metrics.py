"""Functions for calculating benchmark metrics."""

import logging
from pathlib import Path
from typing import Dict, List

import torch
from ultralytics.engine.results import Results  # For type hinting

# Ultralytics Metrics
from ultralytics.utils import metrics

from .config import BenchmarkConfig


class GroundTruthBox(object):
    """Simple container for ground truth bounding box data."""

    def __init__(
        self, class_id: int, x_center: float, y_center: float, width: float, height: float
    ):
        self.class_id = class_id
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        # Store box in xyxy format (normalized) for potential future use
        self.xyxy_norm = [
            x_center - width / 2,
            y_center - height / 2,
            x_center + width / 2,
            y_center + height / 2,
        ]

    def __repr__(self):
        return f"GroundTruthBox(class_id={self.class_id}, xywh=[{self.x_center:.3f}, {self.y_center:.3f}, {self.width:.3f}, {self.height:.3f}])"


def parse_yolo_labels(label_path: Path) -> List[GroundTruthBox]:
    """
    Parses a YOLO format label file (.txt).

    Each line: class_id x_center y_center width height (normalized)

    Args:
        label_path: Path to the YOLO label file.

    Returns:
        A list of GroundTruthBox objects.
    """
    boxes = []
    if not label_path or not label_path.is_file():
        return boxes  # Return empty list if no label file

    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        # Basic validation for normalized coordinates
                        if not (
                            0 <= x_center <= 1
                            and 0 <= y_center <= 1
                            and 0 <= width <= 1
                            and 0 <= height <= 1
                        ):
                            logging.warning(
                                f"Non-normalized value found in {label_path}: {line.strip()}, skipping box."
                            )
                            continue
                        boxes.append(GroundTruthBox(class_id, x_center, y_center, width, height))
                    except ValueError:
                        logging.warning(
                            f"Skipping invalid numeric value in {label_path}: {line.strip()}"
                        )
                elif line.strip():  # Ignore empty lines but warn about malformed lines
                    logging.warning(f"Skipping malformed line in {label_path}: {line.strip()}")
    except Exception as e:
        logging.error(f"Error reading label file {label_path}: {e}")

    return boxes


def calculate_detection_metrics(
    predictions: List[Results],
    ground_truths: List[List[GroundTruthBox]],
    num_classes: int,
    config: BenchmarkConfig,
) -> Dict[str, float]:
    """
    Calculates all detection metrics (mAP, etc.) using Ultralytics DetMetrics.

    Args:
        predictions: A list of prediction results (Ultralytics Results objects).
        ground_truths: A list of ground truth boxes for each image.
        num_classes: The total number of classes in the dataset.
        config: The benchmark configuration.

    Returns:
        A dictionary containing calculated metrics.
    """
    logging.info("Calculating detection metrics using Ultralytics DetMetrics...")
    if not predictions:
        logging.warning("No predictions available to calculate metrics.")
        return {
            "mAP_50": 0.0,
            "mAP_50_95": 0.0,
            "mAP_small": 0.0,
            "mAP_medium": 0.0,
            "mAP_large": 0.0,
        }

    # --- Initialize DetMetrics ---
    iou_low = config.metrics.iou_range_coco[0]
    iou_high = config.metrics.iou_range_coco[1]
    iou_step = config.metrics.iou_range_coco[2]
    # Create the IoU vector for mAP 50:95 calculation
    iou_vector = torch.linspace(iou_low, iou_high, int(round((iou_high - iou_low) / iou_step)) + 1)

    try:
        det_metrics = metrics.DetMetrics(nc=num_classes, iou_vector=iou_vector)
    except Exception as e:
        logging.error(f"Failed to initialize DetMetrics: {e}", exc_info=True)
        # Return error state if DetMetrics fails to initialize
        return {
            "mAP_50": -1.0,
            "mAP_50_95": -1.0,
            "mAP_small": -1.0,
            "mAP_medium": -1.0,
            "mAP_large": -1.0,
        }

    # --- Process predictions and ground truths ---
    device = predictions[0].boxes.xyxyn.device  # Use device from first prediction

    for i, result in enumerate(predictions):
        gt_boxes_list = ground_truths[i]

        # Format predictions [N, 6] tensor [x1, y1, x2, y2, conf, cls]
        if result.boxes and len(result.boxes.xyxyn) > 0:
            preds_for_metric = torch.cat(
                [
                    result.boxes.xyxyn,
                    result.boxes.conf.unsqueeze(1),
                    result.boxes.cls.int().unsqueeze(1),
                ],
                dim=1,
            )
        else:
            preds_for_metric = torch.empty((0, 6), device=device)

        # Format ground truths [M, 5] tensor [cls, x1, y1, x2, y2]
        if gt_boxes_list:
            gt_cls_list = [box.class_id for box in gt_boxes_list]
            gt_boxes_xyxy_list = [box.xyxy_norm for box in gt_boxes_list]
            gt_for_metric = torch.zeros((len(gt_boxes_list), 5), device=device)
            gt_for_metric[:, 0] = torch.tensor(gt_cls_list, device=device)
            gt_for_metric[:, 1:] = torch.tensor(gt_boxes_xyxy_list, device=device)
        else:
            gt_for_metric = torch.empty((0, 5), device=device)

        # Ensure tensors are valid before processing
        # (Basic check for NaNs or Infs which can break metric calculation)
        if torch.isnan(preds_for_metric).any() or torch.isinf(preds_for_metric).any():
            logging.warning(
                f"Invalid values found in predictions for image index {i}. Skipping metrics update."
            )
            continue
        if torch.isnan(gt_for_metric).any() or torch.isinf(gt_for_metric).any():
            logging.warning(
                f"Invalid values found in ground truth for image index {i}. Skipping metrics update."
            )
            continue

        try:
            det_metrics.process(preds=preds_for_metric, gt=gt_for_metric)
        except Exception as e:
            logging.error(f"Error processing metrics for image index {i}: {e}", exc_info=True)
            # Decide whether to continue or halt; continuing might skew results
            # For now, log error and continue to process other images

    # --- Calculate final metrics ---
    try:
        results = det_metrics.results_dict
        # Map Ultralytics keys to our desired output keys
        metrics_output = {
            "mAP_50": results.get("metrics/mAP50(B)", 0.0),
            "mAP_50_95": results.get("metrics/mAP50-95(B)", 0.0),
            "mAP_small": results.get("metrics/mAP50-95(S)", 0.0),
            "mAP_medium": results.get("metrics/mAP50-95(M)", 0.0),
            "mAP_large": results.get("metrics/mAP50-95(L)", 0.0),
        }
        logging.info(
            f"Calculated Metrics: mAP50={metrics_output['mAP_50']:.4f}, mAP50-95={metrics_output['mAP_50_95']:.4f}"
        )
    except Exception as e:
        logging.error(f"Failed to calculate final metrics results: {e}", exc_info=True)
        metrics_output = {
            "mAP_50": -1.0,
            "mAP_50_95": -1.0,
            "mAP_small": -1.0,
            "mAP_medium": -1.0,
            "mAP_large": -1.0,
        }

    return metrics_output


# TODO: Add function to parse voc_xml labels
# TODO: Add function to calculate confusion matrix plot data
# Test comment added via terminal
