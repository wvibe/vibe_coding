"""
YOLOv3 model implementation
"""

import torch
import torch.nn as nn

from src.models.py.yolov3.config import DEFAULT_CONFIG
from src.models.py.yolov3.darknet import Darknet53
from src.models.py.yolov3.head import YOLOv3Head
from src.models.py.yolov3.neck import FeaturePyramidNetwork


class YOLOv3(nn.Module):
    """
    YOLOv3 object detection model

    Combines Darknet-53 backbone, Feature Pyramid Network, and detection heads
    """

    def __init__(self, config=None):
        """
        Initialize YOLOv3 model

        Args:
            config: YOLOv3Config object with model parameters
                   (uses default config if None)
        """
        super().__init__()
        self.config = config if config is not None else DEFAULT_CONFIG

        # Backbone: Darknet-53
        self.backbone = Darknet53()

        # Neck: Feature Pyramid Network
        self.neck = FeaturePyramidNetwork()

        # Head: Detection heads for different scales
        self.head = YOLOv3Head(self.config.num_classes)

        # Initialize weights
        self._initialize_weights()

        # Load pretrained backbone weights if specified
        if self.config.darknet_pretrained:
            self.backbone.load_pretrained(self.config.darknet_weights_path)

    def forward(self, x):
        """
        Forward pass of YOLOv3

        Args:
            x: Input image tensor of shape (batch_size, 3, height, width)
               where height and width should ideally be equal to config.input_size

        Returns:
            tuple: Predictions from all three scales
                   (large_scale_pred, medium_scale_pred, small_scale_pred)
                   Each with shape (batch_size, num_anchors, grid_size, grid_size, 5 + num_classes)
        """
        # Get backbone features
        backbone_features = self.backbone(x)

        # Apply Feature Pyramid Network
        fpn_features = self.neck(backbone_features)

        # Get predictions from detection heads
        predictions = self.head(fpn_features)

        return predictions

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def predict(self, x, conf_threshold=None, nms_threshold=None):
        """
        Make predictions with post-processing (confidence thresholding and NMS)

        Args:
            x: Input image tensor of shape (batch_size, 3, height, width)
            conf_threshold: Confidence threshold (uses config value if None)
            nms_threshold: NMS IoU threshold (uses config value if None)

        Returns:
            list: List of detections for each image in the batch
                  Each detection is [x1, y1, x2, y2, confidence, class_id]
        """
        # Set thresholds
        conf_threshold = (
            conf_threshold if conf_threshold is not None else self.config.conf_threshold
        )
        nms_threshold = nms_threshold if nms_threshold is not None else self.config.nms_threshold

        # Get raw predictions
        with torch.no_grad():
            predictions = self(x)

        # Process predictions
        batch_detections = []
        batch_size = x.size(0)

        # Process each image in the batch
        for batch_idx in range(batch_size):
            image_detections = []

            # Process predictions for each scale
            scale_predictions = [pred[batch_idx] for pred in predictions]

            # FIXED: Dynamically determine grid sizes from prediction tensor shapes
            # Instead of hardcoded grid sizes for 416x416 input
            grid_sizes = [
                pred.shape[1] for pred in scale_predictions
            ]  # Get grid size from prediction shape
            anchor_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

            # Process each scale
            for scale_idx, (pred, grid_size) in enumerate(
                zip(scale_predictions, grid_sizes, strict=False)
            ):
                # Get the anchors for this scale
                scale_anchors = [self.config.anchors[i] for i in anchor_indices[scale_idx]]

                # Process this scale
                detections = self._process_scale_batch(
                    pred,
                    scale_anchors,
                    grid_size,
                    self.config.input_size,
                    conf_threshold,
                    batch_idx,
                )

                if detections.size(0) > 0:
                    image_detections.append(detections)

            # Combine detections from all scales
            if len(image_detections) > 0:
                image_detections = torch.cat(image_detections, dim=0)

                # Apply Non-Maximum Suppression
                image_detections = self._non_max_suppression(
                    image_detections, self.config.num_classes, nms_threshold
                )
            else:
                image_detections = torch.empty((0, 7), device=x.device)

            batch_detections.append(image_detections)

        return batch_detections

    def _process_scale_batch(
        self, predictions, anchors, grid_size, input_size, conf_threshold, batch_idx
    ):
        """
        Process predictions from a single scale for a single image in batch

        Args:
            predictions: Predictions from a single scale (for one image)
                         Shape: (num_anchors, grid_size, grid_size, 5 + num_classes)
            anchors: Anchor boxes for this scale
            grid_size: Grid size for this scale
            input_size: Input image size
            conf_threshold: Confidence threshold for filtering
            batch_idx: Index of image in the batch

        Returns:
            torch.Tensor: Filtered and transformed detections
                          Shape: (n, 7) where n is the number of detections
                          Each detection is [batch_idx, x1, y1, x2, y2, confidence, class_id]
        """
        device = predictions.device
        num_anchors = len(anchors)
        num_classes = predictions.size(-1) - 5

        # Ensure anchors is a tensor
        anchors = torch.tensor(anchors, device=device)

        # Get grid coordinates for this scale
        stride = input_size // grid_size

        # Create grid coordinates
        grid_x, grid_y = torch.meshgrid(
            torch.arange(grid_size, device=device),
            torch.arange(grid_size, device=device),
            indexing="ij",
        )

        # Reshape grid for broadcasting
        grid_x = grid_x.reshape(grid_size, grid_size, 1).repeat(1, 1, num_anchors)
        grid_y = grid_y.reshape(grid_size, grid_size, 1).repeat(1, 1, num_anchors)

        # Reshape predictions to [grid_size, grid_size, num_anchors, 5+num_classes]
        predictions = predictions.permute(1, 2, 0, 3)

        # Extract box coordinates, confidence and class scores
        pred_boxes = predictions[..., :4].clone()
        pred_conf = predictions[..., 4].clone()
        pred_classes = predictions[..., 5:].clone()

        # Apply sigmoid to convert tx, ty to normalized coordinates (0-1)
        pred_boxes[..., 0:2] = torch.sigmoid(pred_boxes[..., 0:2])

        # Add grid cell offsets
        pred_boxes[..., 0] += grid_x
        pred_boxes[..., 1] += grid_y

        # Scale to real coordinates
        pred_boxes[..., 0:2] *= stride

        # Apply exp to width and height and multiply by anchors
        anchors = anchors.reshape(1, 1, num_anchors, 2)
        pred_boxes[..., 2:4] = torch.exp(pred_boxes[..., 2:4]) * anchors

        # Apply sigmoid to confidence and class predictions
        pred_conf = torch.sigmoid(pred_conf)
        pred_classes = torch.sigmoid(pred_classes)

        # Get class with highest confidence
        class_scores, class_ids = torch.max(pred_classes, dim=-1)

        # Calculate final detection confidence
        det_confidence = pred_conf * class_scores

        # Filter by confidence threshold
        mask = det_confidence > conf_threshold

        # Flatten for filtering
        pred_boxes = pred_boxes.reshape(-1, 4)
        det_confidence = det_confidence.reshape(-1)
        class_ids = class_ids.reshape(-1)
        mask = mask.reshape(-1)

        # Apply filtering
        filtered_boxes = pred_boxes[mask]
        filtered_confidence = det_confidence[mask]
        filtered_class_ids = class_ids[mask]

        # If no boxes, return empty tensor
        if filtered_boxes.shape[0] == 0:
            return torch.empty((0, 7), device=device)

        # Convert from center-width-height to top-left, bottom-right
        x1y1 = filtered_boxes[..., 0:2] - filtered_boxes[..., 2:4] / 2
        x2y2 = filtered_boxes[..., 0:2] + filtered_boxes[..., 2:4] / 2
        boxes = torch.cat([x1y1, x2y2], dim=-1)

        # Create detections tensor: [batch_idx, x1, y1, x2, y2, confidence, class_id]
        batch_indices = torch.full((filtered_boxes.shape[0], 1), batch_idx, device=device)
        detections = torch.cat(
            [
                batch_indices,
                boxes,
                filtered_confidence.unsqueeze(-1),
                filtered_class_ids.float().unsqueeze(-1),
            ],
            dim=-1,
        )

        return detections

    def _non_max_suppression(self, detections, num_classes, nms_threshold):
        """
        Apply class-aware Non-Maximum Suppression to detections using a vectorized approach

        Args:
            detections: Tensor of detections
                        Shape: (n, 7) where n is the number of detections
                        Each detection is [batch_idx, x1, y1, x2, y2, confidence, class_id]
            num_classes: Number of classes
            nms_threshold: IoU threshold for NMS

        Returns:
            torch.Tensor: Filtered detections after NMS
        """
        # If no detections, return empty tensor
        if detections.shape[0] == 0:
            return detections

        # Initialize storage for kept detections
        kept_detections = []

        # Process each class separately to perform class-aware NMS
        # This avoids comparing boxes across different classes
        for class_id in range(num_classes):
            # Get detections for this class
            class_mask = detections[:, 6] == class_id
            class_detections = detections[class_mask]

            if class_detections.shape[0] == 0:
                continue

            # Get coordinates and scores for this class
            boxes = class_detections[:, 1:5]  # [x1, y1, x2, y2]
            scores = class_detections[:, 5]  # confidence scores

            # Calculate areas once (vectorized)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            # Sort by confidence (descending)
            _, order = scores.sort(descending=True)

            # Perform NMS
            keep = []
            while order.numel() > 0:
                # Pick the one with highest confidence
                i = order[0].item()
                keep.append(i)

                # If only one detection left, break
                if order.numel() == 1:
                    break

                # Calculate IoUs with remaining boxes (vectorized)
                # Get the current box
                curr_box = boxes[i]

                # Get remaining boxes
                rest_boxes = boxes[order[1:]]

                # Calculate intersection dimensions (vectorized)
                inter_x1 = torch.max(curr_box[0], rest_boxes[:, 0])
                inter_y1 = torch.max(curr_box[1], rest_boxes[:, 1])
                inter_x2 = torch.min(curr_box[2], rest_boxes[:, 2])
                inter_y2 = torch.min(curr_box[3], rest_boxes[:, 3])

                # Calculate intersection area (vectorized)
                w = torch.clamp(inter_x2 - inter_x1, min=0)
                h = torch.clamp(inter_y2 - inter_y1, min=0)
                inter_area = w * h

                # Calculate union area (vectorized)
                curr_area = areas[i]
                rest_areas = areas[order[1:]]
                union_area = curr_area + rest_areas - inter_area

                # Calculate IoU (vectorized)
                iou = inter_area / (union_area + 1e-16)

                # Keep detections with IoU less than threshold (vectorized)
                mask = iou <= nms_threshold
                if not mask.any():
                    break

                order = order[1:][mask]

            # Add kept detections for this class
            kept_detections.append(class_detections[keep])

        # Combine all kept detections
        if kept_detections:
            return torch.cat(kept_detections, dim=0)
        else:
            # Return empty tensor with correct shape
            return torch.zeros((0, 7), device=detections.device)

    def load_state_dict(self, state_dict, strict=True):
        """
        Load model state dictionary

        If strict is False, it will ignore missing keys
        """
        super().load_state_dict(state_dict, strict=strict)

    def save(self, path):
        """
        Save model to path

        Args:
            path: Path to save model
        """
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config,
            },
            path,
        )

    @classmethod
    def load(cls, path, device=None):
        """
        Load model from path

        Args:
            path: Path to load model from
            device: Device to load model on

        Returns:
            YOLOv3: Loaded model
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint.get("config", DEFAULT_CONFIG)
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
