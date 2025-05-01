"""Bounding box utilities for format conversion and IoU calculations."""

import numpy as np
import torch


def xywh_to_xyxy(box):
    """
    Convert bounding box from [x_center, y_center, width, height] to
    [x_min, y_min, x_max, y_max] format

    Args:
        box: Bounding box in [x_center, y_center, width, height] format (normalized 0-1)
             Can be numpy array or tensor

    Returns:
        box in [x_min, y_min, x_max, y_max] format (normalized 0-1)
    """
    if isinstance(box, torch.Tensor):
        x, y, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)
    else:
        x, y, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return np.stack([x1, y1, x2, y2], axis=-1)


def xyxy_to_xywh(box):
    """
    Convert bounding box from [x_min, y_min, x_max, y_max] to
    [x_center, y_center, width, height] format

    Args:
        box: Bounding box in [x_min, y_min, x_max, y_max] format (normalized 0-1)
             Can be numpy array or tensor

    Returns:
        box in [x_center, y_center, width, height] format (normalized 0-1)
    """
    if isinstance(box, torch.Tensor):
        x1, y1, x2, y2 = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        return torch.stack([x, y, w, h], dim=-1)
    else:
        x1, y1, x2, y2 = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        return np.stack([x, y, w, h], axis=-1)


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


def calculate_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Calculates the IoU matrix between two sets of boxes.

    Args:
        boxes1 (np.ndarray): First set of boxes in format [x_min, y_min, x_max, y_max].
                            Shape (N, 4).
        boxes2 (np.ndarray): Second set of boxes in format [x_min, y_min, x_max, y_max].
                            Shape (M, 4).

    Returns:
        np.ndarray: IoU matrix of shape (N, M) where matrix[i, j] contains
                   the IoU between the i-th box in boxes1 and the j-th box in boxes2.
    """
    # Ensure inputs are numpy arrays
    boxes1 = np.asarray(boxes1)
    boxes2 = np.asarray(boxes2)

    # Get the number of boxes in each set
    n_boxes1 = boxes1.shape[0]
    n_boxes2 = boxes2.shape[0]

    # Reshape to enable broadcasting (N, 1, 4) and (1, M, 4)
    boxes1_expanded = np.expand_dims(boxes1, axis=1)  # (N, 1, 4)
    boxes2_expanded = np.expand_dims(boxes2, axis=0)  # (1, M, 4)

    # Calculate intersection coordinates
    # Both will be (N, M, 2) after broadcasting for left_top and right_bottom
    left_top = np.maximum(boxes1_expanded[..., :2], boxes2_expanded[..., :2])  # (N, M, 2)
    right_bottom = np.minimum(boxes1_expanded[..., 2:], boxes2_expanded[..., 2:])  # (N, M, 2)

    # Calculate intersection areas
    wh = np.maximum(0.0, right_bottom - left_top)  # (N, M, 2)
    intersection_area = wh[..., 0] * wh[..., 1]  # (N, M)

    # Calculate areas of all boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)

    # Reshape to enable broadcasting
    area1_expanded = np.expand_dims(area1, axis=1)  # (N, 1)
    area2_expanded = np.expand_dims(area2, axis=0)  # (1, M)

    # Calculate union areas
    union_area = area1_expanded + area2_expanded - intersection_area  # (N, M)

    # Calculate IoU matrix and handle division by zero
    iou_matrix = intersection_area / (union_area + 1e-16)  # (N, M)

    # Clip values to range [0, 1]
    return np.clip(iou_matrix, 0.0, 1.0)


def normalize_boxes(boxes, image_width, image_height):
    """
    Normalize box coordinates from pixel values to [0, 1] range.

    Args:
        boxes: Box coordinates in [x_min, y_min, x_max, y_max] format, in pixel coordinates
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels

    Returns:
        Normalized boxes with coordinates in [0, 1] range
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.clone()  # Create a copy to avoid modifying the original
        boxes[..., [0, 2]] /= image_width
        boxes[..., [1, 3]] /= image_height
    else:
        boxes = boxes.copy()  # Create a copy to avoid modifying the original
        boxes[..., [0, 2]] /= image_width
        boxes[..., [1, 3]] /= image_height

    return boxes


def denormalize_boxes(boxes, image_width, image_height):
    """
    Convert normalized box coordinates [0, 1] to pixel values.

    Args:
        boxes: Box coordinates in [x_min, y_min, x_max, y_max] format, normalized to [0, 1]
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels

    Returns:
        Boxes with coordinates in pixel values
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.clone()  # Create a copy to avoid modifying the original
        boxes[..., [0, 2]] *= image_width
        boxes[..., [1, 3]] *= image_height
    else:
        boxes = boxes.copy()  # Create a copy to avoid modifying the original
        boxes[..., [0, 2]] *= image_width
        boxes[..., [1, 3]] *= image_height

    return boxes