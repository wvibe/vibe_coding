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

    With improved anchor assignment and ignore mechanism.
    """

    def __init__(self, config):
        """
        Initialize YOLOv3 loss

        Args:
            config: YOLOv3 configuration object
        """
        super().__init__()
        self.lambda_coord = 5.0  # Weight for coordinate loss
        self.lambda_noobj = 0.5  # Weight for no-object loss
        self.num_classes = config.num_classes
        self.num_anchors = 3  # Number of anchors per grid cell
        self.ignore_threshold = 0.5  # IoU threshold for ignoring predictions

        # Save input size for box conversion
        self.input_size = config.input_size

        # Scale-specific anchors
        anchor_list = config.anchors
        anchor_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        # Anchors for each scale
        self.anchors_large = [anchor_list[i] for i in anchor_indices[0]]  # 13x13 grid anchors
        self.anchors_medium = [anchor_list[i] for i in anchor_indices[1]]  # 26x26 grid anchors
        self.anchors_small = [anchor_list[i] for i in anchor_indices[2]]  # 52x52 grid anchors

        # All anchors as tensor
        self.anchors = torch.tensor(anchor_list, dtype=torch.float32)

        # Scaled anchors for each grid level
        self.scaled_anchors = [self.anchors_large, self.anchors_medium, self.anchors_small]

    def forward(self, predictions, targets):
        """
        Compute the YOLOv3 loss with improved mechanism

        Args:
            predictions: Tuple of predictions from all three scales
                         (large_scale_pred, medium_scale_pred, small_scale_pred)
            targets: Dictionary containing ground truth:
                    {
                        'boxes': List of tensors with bounding boxes [x, y, w, h],
                        'labels': List of tensors with class labels,
                        'scale_mask': List of tensors indicating which scale to use (optional)
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

        # Add loss balance weights (FIXED: Improved balance between components)
        box_gain = 0.05  # Lower weight for box regression (xy, wh)
        obj_gain = 1.0  # Standard weight for objectness
        cls_gain = 0.5  # Medium weight for classification

        # Keep track of total objects for normalization
        total_objects = 0

        # Compute loss for each scale
        scale_preds = [large_pred, medium_pred, small_pred]
        scale_anchors = [self.anchors_large, self.anchors_medium, self.anchors_small]
        grid_sizes = [
            large_pred.shape[2],
            medium_pred.shape[2],
            small_pred.shape[2],
        ]  # Fixed to use dynamic grid sizes

        for scale_idx, (pred, anchors, grid_size) in enumerate(
            zip(scale_preds, scale_anchors, grid_sizes, strict=False)
        ):
            # Transform predictions to the same format as targets for loss computation
            transformed_pred = self._transform_predictions(pred, anchors, grid_size)

            # Create target tensor with improved assignment, including ignore mask
            target_tensor, obj_mask, noobj_mask, ignore_mask = self._build_targets_improved(
                targets, anchors, grid_size, scale_idx, batch_size, pred
            )

            # Update noobj_mask to exclude ignored predictions
            noobj_mask = noobj_mask & ~ignore_mask

            # Count objects for normalization
            total_objects += obj_mask.sum().item()

            # Compute losses with improved mechanism
            # Localization loss (only for cells with objects)
            if obj_mask.sum() > 0:
                # Box coordinates loss
                xy_loss = F.mse_loss(
                    transformed_pred[..., 0:2][obj_mask],
                    target_tensor[..., 0:2][obj_mask],
                    reduction="sum",
                )
                wh_loss = F.mse_loss(
                    transformed_pred[..., 2:4][obj_mask],
                    target_tensor[..., 2:4][obj_mask],
                    reduction="sum",
                )
                loc_loss += xy_loss + wh_loss

            # Objectness loss for positive samples (with objects)
            if obj_mask.sum() > 0:
                obj_loss += F.binary_cross_entropy_with_logits(
                    pred[..., 4][obj_mask],
                    torch.ones_like(pred[..., 4][obj_mask]),
                    reduction="sum",
                )

            # No object loss (with lower weight and ignore mask)
            if noobj_mask.sum() > 0:
                obj_loss += self.lambda_noobj * F.binary_cross_entropy_with_logits(
                    pred[..., 4][noobj_mask],
                    torch.zeros_like(pred[..., 4][noobj_mask]),
                    reduction="sum",
                )

            # Classification loss (only for cells with objects)
            if obj_mask.sum() > 0:
                cls_loss += F.binary_cross_entropy_with_logits(
                    pred[..., 5:][obj_mask],
                    target_tensor[..., 5:][obj_mask],
                    reduction="sum",
                )

        # FIXED: Improved loss normalization
        # Ensure we have at least 1 object for normalization to avoid division by zero
        total_objects = max(1, total_objects)

        # Normalize localization and classification losses by number of objects
        # This makes the loss more stable and less affected by batch size variations
        loc_loss = loc_loss / total_objects
        cls_loss = cls_loss / total_objects

        # Normalize objectness loss by batch size (as it applies to all grid cells)
        obj_loss = obj_loss / batch_size

        # Apply loss component weights and combine
        weighted_loc_loss = box_gain * loc_loss
        weighted_obj_loss = obj_gain * obj_loss
        weighted_cls_loss = cls_gain * cls_loss

        # Combine losses with improved weighting
        total_loss = weighted_loc_loss + weighted_obj_loss + weighted_cls_loss

        return {
            "loss": total_loss,
            "loc_loss": loc_loss,
            "obj_loss": obj_loss,
            "cls_loss": cls_loss,
        }

    def _transform_predictions(self, pred, anchors, grid_size):
        """
        Transform raw model predictions to the target format for loss computation

        Args:
            pred: Raw predictions from model [batch_size, num_anchors, grid_size, grid_size, 5+num_classes]
            anchors: Anchor boxes for this scale
            grid_size: Grid size for this scale

        Returns:
            torch.Tensor: Transformed predictions in target format
        """
        batch_size = pred.size(0)
        num_anchors = len(anchors)
        device = pred.device

        # Create grid
        grid_x, grid_y = torch.meshgrid(
            torch.arange(grid_size, device=device),
            torch.arange(grid_size, device=device),
            indexing="ij",
        )

        # Reshape for broadcasting
        grid_x = grid_x.reshape(1, 1, grid_size, grid_size).expand(
            batch_size, num_anchors, grid_size, grid_size
        )
        grid_y = grid_y.reshape(1, 1, grid_size, grid_size).expand(
            batch_size, num_anchors, grid_size, grid_size
        )

        # Reshape anchors for broadcasting
        anchors_tensor = torch.tensor(anchors, device=device, dtype=torch.float32)
        anchors_tensor = anchors_tensor.reshape(1, num_anchors, 1, 1, 2).expand(
            batch_size, num_anchors, grid_size, grid_size, 2
        )

        # Apply sigmoid to x, y predictions (cell-relative coordinates)
        pred_xy = torch.sigmoid(pred[..., 0:2])

        # Cell-relative coordinates to grid-relative
        pred_xy = (pred_xy + torch.stack((grid_x, grid_y), dim=-1)) / grid_size

        # Apply exp to width, height predictions and multiply by anchors
        pred_wh = torch.exp(pred[..., 2:4]) * anchors_tensor / grid_size

        # Combine transformed predictions
        transformed_pred = torch.cat([pred_xy, pred_wh, torch.sigmoid(pred[..., 4:])], dim=-1)

        return transformed_pred

    def _build_targets_improved(
        self, targets, anchors, grid_size, scale_idx, batch_size, predictions
    ):
        """
        Build target tensor for a specific scale with improved anchor assignment and ignore mask

        Args:
            targets: Dictionary with ground truth
            anchors: Anchors for this scale
            grid_size: Grid size for this scale
            scale_idx: Index of the scale (0, 1, 2)
            batch_size: Batch size
            predictions: Raw predictions for this scale

        Returns:
            tuple: (target_tensor, obj_mask, noobj_mask, ignore_mask)
                  target_tensor: Target tensor for this scale
                  obj_mask: Mask for cells with objects
                  noobj_mask: Mask for cells without objects
                  ignore_mask: Mask for cells to ignore in noobj loss
        """
        device = targets["boxes"][0].device if len(targets["boxes"]) > 0 else predictions.device
        dtype = predictions.dtype

        # Initialize tensors
        target = torch.zeros(
            (batch_size, self.num_anchors, grid_size, grid_size, 5 + self.num_classes),
            dtype=dtype,
            device=device,
        )
        obj_mask = torch.zeros(
            (batch_size, self.num_anchors, grid_size, grid_size), dtype=torch.bool, device=device
        )
        noobj_mask = torch.ones(
            (batch_size, self.num_anchors, grid_size, grid_size), dtype=torch.bool, device=device
        )
        ignore_mask = torch.zeros(
            (batch_size, self.num_anchors, grid_size, grid_size), dtype=torch.bool, device=device
        )

        # Convert anchors to tensor
        anchors_tensor = torch.tensor(anchors, device=device, dtype=dtype)

        # For each image in the batch
        for b in range(batch_size):
            # Skip if no ground truth boxes for this image
            if len(targets["boxes"]) <= b or len(targets["boxes"][b]) == 0:
                continue

            # Get ground truth boxes and labels for this image
            gt_boxes = targets["boxes"][b]
            gt_labels = targets["labels"][b]

            # Process each ground truth box
            for box_idx, (box, label) in enumerate(zip(gt_boxes, gt_labels, strict=False)):
                # Skip dummy boxes (with width or height near 0.1)
                if torch.allclose(box[2:4], torch.tensor([0.1, 0.1], device=device), atol=1e-2):
                    continue

                # Get all anchors from all scales to find best anchor across all scales
                all_anchor_list = []
                scale_indices = []

                for s in range(3):  # Three scales: large (13x13), medium (26x26), small (52x52)
                    scale_anchor_list = self.scaled_anchors[s]
                    all_anchor_list.extend(scale_anchor_list)
                    scale_indices.extend([s] * len(scale_anchor_list))

                # Convert lists to tensors
                all_anchors = torch.tensor(all_anchor_list, device=device, dtype=dtype)
                scale_indices = torch.tensor(scale_indices, device=device)

                # Calculate box width and height in absolute terms
                _, _, box_w, box_h = box
                box_w_abs = box_w * self.input_size  # Convert to absolute pixels
                box_h_abs = box_h * self.input_size

                # Calculate IoU with all anchors
                ious = []
                for anchor_idx, anchor in enumerate(all_anchors):
                    anchor_w, anchor_h = anchor
                    # Calculate intersection
                    inter_w = torch.min(box_w_abs, anchor_w)
                    inter_h = torch.min(box_h_abs, anchor_h)
                    inter_area = inter_w * inter_h
                    # Calculate union
                    box_area = box_w_abs * box_h_abs
                    anchor_area = anchor_w * anchor_h
                    union_area = box_area + anchor_area - inter_area + 1e-16
                    # Calculate IoU
                    iou = inter_area / union_area
                    ious.append(iou)

                ious = torch.tensor(ious, device=device)
                best_anchor_idx = torch.argmax(ious)
                best_scale = scale_indices[best_anchor_idx]

                # Check if this box should be assigned to the current scale
                if best_scale != scale_idx:
                    # Skip if this box is assigned to a different scale
                    continue

                # Get anchor index within this scale (0, 1, or 2)
                within_scale_idx = best_anchor_idx % 3

                # Continue with original code for target assignment
                # Box coordinates (normalized)
                x, y, w, h = box

                # Convert to grid cell coordinates
                grid_x = int(x * grid_size)
                grid_y = int(y * grid_size)

                # Clamp to grid boundaries
                grid_x = max(0, min(grid_x, grid_size - 1))
                grid_y = max(0, min(grid_y, grid_size - 1))

                # Box width and height relative to grid cell
                box_w = w * grid_size
                box_h = h * grid_size

                # Use the best anchor determined by IoU
                best_anchor_idx = within_scale_idx

                # Set target values
                # Coordinates within cell (0 to 1)
                x_cell = x * grid_size - grid_x
                y_cell = y * grid_size - grid_y

                # Width and height relative to anchor
                w_cell = torch.log(w * grid_size / anchors_tensor[best_anchor_idx][0] + 1e-16)
                h_cell = torch.log(h * grid_size / anchors_tensor[best_anchor_idx][1] + 1e-16)

                # Set target box coordinates
                target[b, best_anchor_idx, grid_y, grid_x, 0] = x_cell
                target[b, best_anchor_idx, grid_y, grid_x, 1] = y_cell
                target[b, best_anchor_idx, grid_y, grid_x, 2] = w_cell
                target[b, best_anchor_idx, grid_y, grid_x, 3] = h_cell

                # Set objectness
                target[b, best_anchor_idx, grid_y, grid_x, 4] = 1.0

                # Set class (one-hot encoding)
                target[b, best_anchor_idx, grid_y, grid_x, 5 + label] = 1.0

                # Update masks
                obj_mask[b, best_anchor_idx, grid_y, grid_x] = True
                noobj_mask[b, best_anchor_idx, grid_y, grid_x] = False

                # Mark anchors with high IoU as ignored for noobj loss
                for anchor_idx in range(len(anchors)):
                    if anchor_idx != best_anchor_idx:
                        # Calculate IoU with other anchors
                        iou = self._calculate_anchor_box_iou(
                            box_w,
                            box_h,
                            anchors_tensor[anchor_idx][0],
                            anchors_tensor[anchor_idx][1],
                        )

                        # If IoU exceeds threshold, ignore this anchor for noobj loss
                        if iou > self.ignore_threshold:
                            ignore_mask[b, anchor_idx, grid_y, grid_x] = True

        return target, obj_mask, noobj_mask, ignore_mask

    def _find_best_anchor_iou(self, box_w, box_h, anchors):
        """
        Find best anchor based on IoU

        Args:
            box_w: Box width (in grid units)
            box_h: Box height (in grid units)
            anchors: Anchors tensor

        Returns:
            tuple: (best_anchor_idx, best_iou)
        """
        # Calculate IoU for all anchors
        anchor_ious = torch.zeros(len(anchors), device=box_w.device)

        for i, anchor in enumerate(anchors):
            anchor_w, anchor_h = anchor

            # Calculate intersection area
            intersect_w = torch.min(box_w, anchor_w)
            intersect_h = torch.min(box_h, anchor_h)
            intersect_area = intersect_w * intersect_h

            # Calculate union area
            box_area = box_w * box_h
            anchor_area = anchor_w * anchor_h
            union_area = box_area + anchor_area - intersect_area + 1e-16

            # Calculate IoU
            anchor_ious[i] = intersect_area / union_area

        # Find best anchor
        best_anchor_idx = torch.argmax(anchor_ious).item()
        best_iou = anchor_ious[best_anchor_idx].item()

        return best_anchor_idx, best_iou

    def _calculate_anchor_box_iou(self, box_w, box_h, anchor_w, anchor_h):
        """
        Calculate IoU between a box and an anchor

        Args:
            box_w: Box width
            box_h: Box height
            anchor_w: Anchor width
            anchor_h: Anchor height

        Returns:
            float: IoU value
        """
        # Calculate intersection area
        intersect_w = torch.min(box_w, anchor_w)
        intersect_h = torch.min(box_h, anchor_h)
        intersect_area = intersect_w * intersect_h

        # Calculate union area
        box_area = box_w * box_h
        anchor_area = anchor_w * anchor_h
        union_area = box_area + anchor_area - intersect_area + 1e-16

        # Calculate IoU
        iou = intersect_area / union_area

        return iou

    def _find_best_anchor(self, box_w, box_h, anchors):
        """
        Find best anchor box based on IoU (legacy method, kept for compatibility)

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


def calculate_batch_loss(loss_dict):
    """
    Extract individual loss components from loss dictionary.

    Args:
        loss_dict: Dictionary containing loss components
                  {'loss': total_loss, 'loc_loss': loc_loss,
                   'obj_loss': obj_loss, 'cls_loss': cls_loss}

    Returns:
        tuple: (total_loss, loc_loss, obj_loss, cls_loss) as scalar values
    """
    return (
        loss_dict["loss"].item(),
        loss_dict["loc_loss"].item(),
        loss_dict["obj_loss"].item(),
        loss_dict["cls_loss"].item(),
    )


def accumulate_losses(loss_tuple, accumulated_losses):
    """
    Accumulate loss values into running totals.

    Args:
        loss_tuple: Tuple of (total_loss, loc_loss, obj_loss, cls_loss)
        accumulated_losses: Tuple of accumulated (total_loss, loc_loss, obj_loss, cls_loss)

    Returns:
        tuple: Updated accumulated losses
    """
    return (
        accumulated_losses[0] + loss_tuple[0],  # total
        accumulated_losses[1] + loss_tuple[1],  # loc
        accumulated_losses[2] + loss_tuple[2],  # obj
        accumulated_losses[3] + loss_tuple[3],  # cls
    )


def average_losses(accumulated_losses, num_batches):
    """
    Calculate average losses over multiple batches.

    Args:
        accumulated_losses: Tuple of accumulated (total_loss, loc_loss, obj_loss, total_cls_loss)
        num_batches: Number of batches the losses were accumulated over

    Returns:
        dict: Dictionary of average losses
              {'loss': avg_loss, 'loc_loss': avg_loc_loss,
               'obj_loss': avg_obj_loss, 'cls_loss': avg_cls_loss}
    """
    return {
        "loss": accumulated_losses[0] / num_batches,
        "loc_loss": accumulated_losses[1] / num_batches,
        "obj_loss": accumulated_losses[2] / num_batches,
        "cls_loss": accumulated_losses[3] / num_batches,
    }
