"""
Evaluation utilities for YOLOv3 model

This module contains functions for evaluating object detection models,
including IoU calculation, AP calculation, and mAP calculation.
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm


def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.
    Optimized for efficiency.

    Args:
        box1 (torch.Tensor): First box in format (x1, y1, x2, y2)
        box2 (torch.Tensor): Second box in format (x1, y1, x2, y2)

    Returns:
        float: IoU value
    """
    # Directly get coordinates for more efficient calculation
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Fast calculation of intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate area of each box once
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Fast calculation of union area
    union_area = box1_area + box2_area - intersection_area

    # Safeguard against division by zero
    if union_area < 1e-6:  # Virtually zero
        return 0.0

    # Calculate and return IoU
    return intersection_area / union_area


def calculate_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Calculate Average Precision (AP) using the 11-point interpolation

    Args:
        recalls (np.ndarray): Array of recall values
        precisions (np.ndarray): Array of precision values

    Returns:
        float: AP value
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11

    return ap


def calculate_mean_ap(
    all_detections: List[List[List[Dict]]],
    all_ground_truths: List[List[List[Dict]]],
    iou_threshold: float = 0.5,
) -> Tuple[float, Dict[int, float]]:
    """
    Calculate mean Average Precision (mAP) across all classes

    Args:
        all_detections (list): List of detections for each image and class
        all_ground_truths (list): List of ground truths for each image and class
        iou_threshold (float): IoU threshold for a true positive

    Returns:
        Tuple[float, Dict[int, float]]: mAP value and AP values for each class
    """
    average_precisions = {}
    # Determine number of classes from the data structure
    if len(all_detections) > 0 and len(all_detections[0]) > 0:
        num_classes = len(all_detections[0])
    else:
        return 0.0, {}

    for c in range(num_classes):
        # Extract detections and ground truths for current class
        detections = []
        ground_truths = []

        for i in range(len(all_detections)):
            detections.extend(all_detections[i][c])
            ground_truths.extend(all_ground_truths[i][c])

        # Create arrays for confidence, TP, and FP
        confidence = np.array([d["confidence"] for d in detections])
        TP = np.zeros(len(detections))
        FP = np.zeros(len(detections))

        # Count number of ground truth objects
        num_ground_truths = len(ground_truths)

        # No ground truths means AP for this class is 0
        if num_ground_truths == 0:
            average_precisions[c] = 0.0
            continue

        # Sort detections by confidence
        indices = np.argsort(-confidence)

        # Create array to keep track of detected ground truths
        detected_gt = np.zeros(num_ground_truths)

        # Assign detections to ground truth objects
        for d_idx in indices:
            # Get ground truth with highest IoU
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truths):
                if detected_gt[gt_idx]:
                    continue

                iou = calculate_iou(detections[d_idx]["bbox"], gt["bbox"])

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # If IoU exceeds threshold, it's a true positive
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                TP[d_idx] = 1
                detected_gt[best_gt_idx] = 1
            else:
                FP[d_idx] = 1

        # Compute cumulative sum of TP and FP
        TP_cumsum = np.cumsum(TP)
        FP_cumsum = np.cumsum(FP)

        # Calculate precision and recall
        precision = TP_cumsum / (TP_cumsum + FP_cumsum)
        recall = TP_cumsum / num_ground_truths

        # Add sentinel values to make sure the precision curve starts at 0 recall
        precision = np.concatenate(([0], precision, [0]))
        recall = np.concatenate(([0], recall, [1]))

        # Ensure precision is monotonically decreasing
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])

        # Calculate AP as the area under the precision-recall curve
        ap = calculate_ap(recall, precision)
        average_precisions[c] = ap

    # Calculate mAP
    mAP = np.mean(list(average_precisions.values()))

    return mAP, average_precisions


def collect_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[List[List[List[Dict]]], List[List[List[Dict]]]]:
    """
    Collect predictions and ground truths from model for mAP calculation

    Args:
        model (torch.nn.Module): The model to evaluate
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data
        device (torch.device): Device to use

    Returns:
        Tuple[List, List]: Detected objects and ground truth objects
    """
    model.eval()
    all_detections = []
    all_ground_truths = []

    # Get number of classes from model config
    num_classes = model.config.num_classes

    # Set up progress bar
    progress_bar = tqdm(dataloader, desc="Collecting predictions")

    with torch.no_grad():
        for batch in progress_bar:
            # Get batch data
            images = batch["images"].to(device)
            targets = {
                "boxes": [boxes.to(device) for boxes in batch["boxes"]],
                "labels": [labels.to(device) for labels in batch["labels"]],
            }

            # Process batch predictions and targets
            batch_detections = []
            batch_ground_truths = []

            # Post-process predictions to get bounding boxes
            # Using the model's predict method with the evaluation confidence threshold
            processed_predictions_list = model.predict(
                images,
                conf_threshold=model.config.eval_conf_threshold,
                nms_threshold=model.config.nms_threshold,
            )

            for i in range(len(images)):
                # Process predictions for image i
                img_detections = [[] for _ in range(num_classes)]

                # Get detections for this image - already filtered for this image
                image_preds = processed_predictions_list[i]

                if image_preds.shape[0] > 0:
                    # Extract predictions for this image
                    # Format is [batch_idx, x1, y1, x2, y2, confidence, class_id]
                    pred_boxes = image_preds[:, 1:5]  # x1, y1, x2, y2
                    pred_scores = image_preds[:, 5]  # confidence
                    pred_classes = image_preds[:, 6]  # class_id

                    for box, score, cls_id in zip(pred_boxes, pred_scores, pred_classes):
                        cls_idx = int(cls_id.item())
                        if cls_idx < num_classes:  # Ensure class index is valid
                            img_detections[cls_idx].append(
                                {"bbox": box.cpu(), "confidence": score.item()}
                            )

                # Process ground truths for image i
                img_ground_truths = [[] for _ in range(num_classes)]
                gt_boxes = targets["boxes"][i]
                gt_classes = targets["labels"][i]

                for box, cls in zip(gt_boxes, gt_classes):
                    class_idx = cls.item()
                    if class_idx < num_classes:  # Ensure class index is valid
                        img_ground_truths[class_idx].append({"bbox": box.cpu()})

                batch_detections.append(img_detections)
                batch_ground_truths.append(img_ground_truths)

            all_detections.extend(batch_detections)
            all_ground_truths.extend(batch_ground_truths)

            # Update progress bar with counts
            progress_bar.set_postfix(
                {
                    "images": len(all_detections),
                    "detections": sum(
                        len(cls_dets) for img_dets in all_detections for cls_dets in img_dets
                    ),
                }
            )

    return all_detections, all_ground_truths


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    iou_threshold: float = 0.5,
) -> Dict[str, Union[float, Dict[int, float]]]:
    """
    Evaluate model performance using mAP

    Args:
        model (torch.nn.Module): The model to evaluate
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data
        device (torch.device): Device to use
        iou_threshold (float): IoU threshold for evaluation

    Returns:
        Dict: Evaluation metrics
    """

    print("Starting model evaluation...")

    # Collect predictions and ground truths
    all_detections, all_ground_truths = collect_predictions(model, dataloader, device)

    print(f"Calculating mAP at IoU threshold {iou_threshold}...")

    # Calculate mAP
    mAP, class_APs = calculate_mean_ap(all_detections, all_ground_truths, iou_threshold)

    # Create metrics dictionary
    metrics = {
        "mAP": mAP,
        "class_APs": class_APs,
        "iou_threshold": iou_threshold,
    }

    return metrics
