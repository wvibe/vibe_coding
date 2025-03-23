"""
Loss function implementation for YOLOv3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOv3Loss(nn.Module):
    """
    YOLOv3 loss function

    Computes the loss for YOLOv3 predictions, consisting of:
    1. Localization loss: MSE for bounding box coordinates
    2. Objectness loss: BCE for objectness score
    3. Classification loss: BCE for class probabilities
    """

    def __init__(self, config):
        """
        Initialize YOLOv3 loss

        Args:
            config: YOLOv3 configuration object
        """
        super().__init__()
        self.lambda_coord = config.lambda_coord
        self.lambda_noobj = config.lambda_noobj
        self.num_classes = config.num_classes
        self.anchors = config.anchors
        self.num_anchors = config.anchors_per_scale
        self.ignore_threshold = 0.5  # IoU threshold for ignoring objectness loss

        # Reshape anchors for each scale
        self.anchors_large = self.anchors[0:3]  # For 13x13 grid
        self.anchors_medium = self.anchors[3:6]  # For 26x26 grid
        self.anchors_small = self.anchors[6:9]  # For 52x52 grid

    def forward(self, predictions, targets):
        """
        Compute the YOLOv3 loss

        Args:
            predictions: Tuple of predictions from all three scales
                         (large_scale_pred, medium_scale_pred, small_scale_pred)
            targets: Dictionary containing ground truth:
                    {
                        'boxes': List of tensors with bounding boxes [x, y, w, h],
                        'labels': List of tensors with class labels,
                        'scale_mask': List of tensors indicating which scale to use
                    }

        Returns:
            dict: Dictionary containing the different loss components
                  {
                      'loss': Total loss,
                      'loc_loss': Localization loss,
                      'obj_loss': Objectness loss,
                      'cls_loss': Classification loss
                  }
        """
        device = predictions[0].device

        # Unpack predictions
        large_pred, medium_pred, small_pred = predictions
        batch_size = large_pred.size(0)

        # Initialize losses
        total_loss = torch.tensor(0.0, device=device)
        loc_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)

        # Compute loss for each scale
        scale_preds = [large_pred, medium_pred, small_pred]
        scale_anchors = [self.anchors_large, self.anchors_medium, self.anchors_small]
        grid_sizes = [13, 26, 52]  # Grid sizes for 416x416 input

        for scale_idx, (pred, anchors, grid_size) in enumerate(
            zip(scale_preds, scale_anchors, grid_sizes, strict=False)
        ):
            # Create targets tensor for this scale
            target_tensor = self._build_target(
                targets, anchors, grid_size, scale_idx, batch_size
            )

            # Get masks
            obj_mask = target_tensor[..., 4].bool()  # Objectness mask
            noobj_mask = ~obj_mask

            # Compute losses
            # Localization loss (only for cells with objects)
            if obj_mask.sum() > 0:
                # Box coordinates loss
                xy_loss = F.mse_loss(
                    pred[..., 0:2][obj_mask],
                    target_tensor[..., 0:2][obj_mask],
                    reduction="sum",
                )
                wh_loss = F.mse_loss(
                    pred[..., 2:4][obj_mask],
                    target_tensor[..., 2:4][obj_mask],
                    reduction="sum",
                )
                loc_loss += xy_loss + wh_loss

            # Objectness loss
            obj_loss += F.binary_cross_entropy_with_logits(
                pred[..., 4][obj_mask], target_tensor[..., 4][obj_mask], reduction="sum"
            )

            # No object loss (with lower weight)
            obj_loss += self.lambda_noobj * F.binary_cross_entropy_with_logits(
                pred[..., 4][noobj_mask],
                target_tensor[..., 4][noobj_mask],
                reduction="sum",
            )

            # Classification loss (only for cells with objects)
            if obj_mask.sum() > 0:
                cls_loss += F.binary_cross_entropy_with_logits(
                    pred[..., 5:][obj_mask],
                    target_tensor[..., 5:][obj_mask],
                    reduction="sum",
                )

        # Combine losses with weighting
        total_loss = self.lambda_coord * loc_loss + obj_loss + cls_loss

        return {
            "loss": total_loss,
            "loc_loss": loc_loss,
            "obj_loss": obj_loss,
            "cls_loss": cls_loss,
        }

    def _build_target(self, targets, anchors, grid_size, scale_idx, batch_size):
        """
        Build target tensor for a specific scale

        Args:
            targets: Dictionary with ground truth
            anchors: Anchors for this scale
            grid_size: Grid size for this scale
            scale_idx: Index of the scale (0, 1, 2)
            batch_size: Batch size

        Returns:
            torch.Tensor: Target tensor for this scale with shape
                        (batch_size, num_anchors, grid_size, grid_size, 5 + num_classes)
        """
        device = targets["boxes"][0].device
        dtype = targets["boxes"][0].dtype

        # Initialize target tensor
        target = torch.zeros(
            (batch_size, self.num_anchors, grid_size, grid_size, 5 + self.num_classes),
            dtype=dtype,
            device=device,
        )

        # For each image in the batch
        for b in range(batch_size):
            # Get ground truth boxes and labels for this image
            gt_boxes = targets["boxes"][b]
            gt_labels = targets["labels"][b]

            # Skip if no ground truth boxes
            if len(gt_boxes) == 0:
                continue

            # For each ground truth box
            for box_idx, (box, label) in enumerate(
                zip(gt_boxes, gt_labels, strict=False)
            ):
                # Check if this box should be assigned to this scale
                if (
                    "scale_mask" in targets
                    and not targets["scale_mask"][b][box_idx][scale_idx]
                ):
                    continue

                # Convert box coordinates to grid cell coordinates
                x, y, w, h = box

                # Grid cell coordinates
                grid_x = int(x * grid_size)
                grid_y = int(y * grid_size)

                # Clamp to grid boundaries
                grid_x = min(grid_x, grid_size - 1)
                grid_y = min(grid_y, grid_size - 1)

                # Box width and height relative to grid cell
                box_w = w * grid_size
                box_h = h * grid_size

                # Find best anchor based on IoU
                best_anchor_idx = self._find_best_anchor(box_w, box_h, anchors)

                # Set target values
                # Box coordinates relative to grid cell (0 to 1)
                x_cell = x * grid_size - grid_x
                y_cell = y * grid_size - grid_y

                # Width and height relative to anchor
                w_cell = torch.log(box_w / anchors[best_anchor_idx][0] + 1e-16)
                h_cell = torch.log(box_h / anchors[best_anchor_idx][1] + 1e-16)

                # Set target box
                target[b, best_anchor_idx, grid_y, grid_x, 0] = x_cell
                target[b, best_anchor_idx, grid_y, grid_x, 1] = y_cell
                target[b, best_anchor_idx, grid_y, grid_x, 2] = w_cell
                target[b, best_anchor_idx, grid_y, grid_x, 3] = h_cell

                # Set objectness
                target[b, best_anchor_idx, grid_y, grid_x, 4] = 1.0

                # Set class (one-hot encoding)
                target[b, best_anchor_idx, grid_y, grid_x, 5 + label] = 1.0

        return target

    def _find_best_anchor(self, box_w, box_h, anchors):
        """
        Find best anchor box based on IoU

        Args:
            box_w: Box width relative to grid cell
            box_h: Box height relative to grid cell
            anchors: Anchor boxes for this scale

        Returns:
            int: Index of the best anchor
        """
        # Convert anchors to tensor if not already
        if not isinstance(anchors, torch.Tensor):
            anchors = torch.tensor(anchors, device=box_w.device, dtype=box_w.dtype)

        # Calculate IoU between the box and all anchors
        inter_w = torch.min(box_w, anchors[:, 0])
        inter_h = torch.min(box_h, anchors[:, 1])
        inter_area = inter_w * inter_h

        box_area = box_w * box_h
        anchor_area = anchors[:, 0] * anchors[:, 1]

        iou = inter_area / (box_area + anchor_area - inter_area + 1e-16)

        # Return the index of the anchor with the highest IoU
        return torch.argmax(iou).item()
