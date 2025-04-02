"""Bounding box format conversion utilities."""

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

