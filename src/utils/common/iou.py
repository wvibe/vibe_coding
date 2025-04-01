"""Intersection over Union (IoU) calculation utility."""

import numpy as np


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculates the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (np.ndarray): The first bounding box in format [x_min, y_min, x_max, y_max]. Shape (4,).
        box2 (np.ndarray): The second bounding box in format [x_min, y_min, x_max, y_max]. Shape (4,).

    Returns:
        float: The IoU value, ranging from 0.0 to 1.0. Returns 0.0 if there is no
               overlap or if the union area is zero.
    """
    # Ensure inputs are numpy arrays
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)

    # Determine the coordinates of the intersection rectangle
    x_left = np.maximum(box1[0], box2[0])
    y_top = np.maximum(box1[1], box2[1])
    x_right = np.minimum(box1[2], box2[2])
    y_bottom = np.minimum(box1[3], box2[3])

    # Calculate the area of intersection rectangle
    # np.maximum(0.0, ...) handles cases where there is no overlap
    intersection_width = np.maximum(0.0, x_right - x_left)
    intersection_height = np.maximum(0.0, y_bottom - y_top)
    intersection_area = intersection_width * intersection_height

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the area of union
    union_area = box1_area + box2_area - intersection_area

    # Compute the IoU
    # Add a small epsilon to avoid division by zero
    iou = intersection_area / (union_area + 1e-16)

    # Ensure IoU is between 0 and 1
    return float(np.clip(iou, 0.0, 1.0))
