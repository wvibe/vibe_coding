"""
Bounding box utilities for object detection datasets
"""

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


def box_iou(box1, box2):
    """
    Calculate IoU between two boxes

    Args:
        box1: Box in [x_min, y_min, x_max, y_max] format
              Shape: (..., 4)
        box2: Box in [x_min, y_min, x_max, y_max] format
              Shape: (..., 4)

    Returns:
        IoU between box1 and box2
    """
    # Get coordinates
    if isinstance(box1, torch.Tensor):
        # Make sure both are tensors
        if not isinstance(box2, torch.Tensor):
            box2 = torch.tensor(box2, device=box1.device, dtype=box1.dtype)

        # Get coordinates
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

        # Calculate intersection area
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)

        # Clip to ensure width and height are positive
        inter_width = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0)
        inter_height = torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

        inter_area = inter_width * inter_height

        # Calculate union area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        union_area = b1_area + b2_area - inter_area

        # Calculate IoU
        iou = inter_area / (union_area + 1e-16)

        return iou
    else:
        # NumPy implementation
        # Get coordinates
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

        # Calculate intersection area
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)

        # Clip to ensure width and height are positive
        inter_width = np.maximum(inter_rect_x2 - inter_rect_x1, 0)
        inter_height = np.maximum(inter_rect_y2 - inter_rect_y1, 0)

        inter_area = inter_width * inter_height

        # Calculate union area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        union_area = b1_area + b2_area - inter_area

        # Calculate IoU
        iou = inter_area / (union_area + 1e-16)

        return iou


def generate_anchors(dataset, num_clusters=9, method="kmeans"):
    """
    Generate anchor boxes for a dataset using k-means clustering

    Args:
        dataset: Dataset with bounding boxes
        num_clusters: Number of anchor boxes to generate
        method: Method to use for clustering ('kmeans' or 'random')

    Returns:
        anchors: Generated anchor boxes
    """
    # Extract all bounding boxes from dataset
    boxes = []
    for i in range(len(dataset)):
        sample = dataset[i]
        boxes.append(sample["boxes"])

    if isinstance(boxes[0], torch.Tensor):
        boxes = torch.cat(boxes, dim=0)
    else:
        boxes = np.concatenate(boxes, axis=0)

    # Convert to width and height
    widths = boxes[..., 2]
    heights = boxes[..., 3]

    if method == "kmeans":
        # Use k-means clustering to find anchor boxes
        from sklearn.cluster import KMeans

        # Convert to numpy for sklearn
        if isinstance(widths, torch.Tensor):
            widths = widths.cpu().numpy()
            heights = heights.cpu().numpy()

        # Stack widths and heights for clustering
        X = np.stack([widths, heights], axis=1)

        # Apply k-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
        anchors = kmeans.cluster_centers_

        # Sort anchors by area
        anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]

        if isinstance(boxes, torch.Tensor):
            # Convert back to tensor
            anchors = torch.tensor(anchors, device=boxes.device, dtype=boxes.dtype)
    else:
        # Random selection
        if isinstance(boxes, torch.Tensor):
            # Randomly select indices
            indices = torch.randperm(len(widths))[:num_clusters]
            anchors = torch.stack([widths[indices], heights[indices]], dim=1)
        else:
            indices = np.random.permutation(len(widths))[:num_clusters]
            anchors = np.stack([widths[indices], heights[indices]], axis=1)

    return anchors


def non_max_suppression(boxes, scores, threshold=0.5):
    """
    Apply non-maximum suppression to eliminate overlapping bounding boxes

    Args:
        boxes: Bounding boxes in [x_min, y_min, x_max, y_max] format
               Shape: (num_boxes, 4)
        scores: Confidence scores for each box
                Shape: (num_boxes,)
        threshold: IoU threshold for suppression

    Returns:
        indices: Indices of boxes to keep
    """
    if isinstance(boxes, torch.Tensor):
        # No boxes
        if boxes.shape[0] == 0:
            return torch.tensor([], dtype=torch.int64)

        # Get coordinates
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # Calculate areas
        areas = (x2 - x1) * (y2 - y1)

        # Sort boxes by score
        order = torch.argsort(scores, descending=True)

        keep = []
        while order.numel() > 0:
            # Pick the box with highest score
            i = order[0].item()
            keep.append(i)

            # If only one box left, break
            if order.numel() == 1:
                break

            # Get IoU with remaining boxes
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])

            # Calculate intersection area
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            intersection = w * h

            # Calculate IoU
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-16)

            # Keep boxes with IoU less than threshold
            mask = iou <= threshold
            if not mask.any():
                break
            order = order[1:][mask]

        return torch.tensor(keep, dtype=torch.int64)
    else:
        # NumPy implementation
        # No boxes
        if boxes.shape[0] == 0:
            return np.array([], dtype=np.int64)

        # Get coordinates
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # Calculate areas
        areas = (x2 - x1) * (y2 - y1)

        # Sort boxes by score
        order = np.argsort(scores)[::-1]

        keep = []
        while order.size > 0:
            # Pick the box with highest score
            i = order[0]
            keep.append(i)

            # If only one box left, break
            if order.size == 1:
                break

            # Get IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # Calculate intersection area
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            # Calculate IoU
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-16)

            # Keep boxes with IoU less than threshold
            mask = iou <= threshold
            if not np.any(mask):
                break
            order = order[1:][mask]

        return np.array(keep, dtype=np.int64)
