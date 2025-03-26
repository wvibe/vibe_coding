"""
Tests for YOLOv3 model
"""

import torch

from src.models.py.yolov3.config import YOLOv3Config
from src.models.py.yolov3.darknet import Darknet53
from src.models.py.yolov3.head import DetectionHead, YOLOv3Head
from src.models.py.yolov3.neck import FeaturePyramidNetwork
from src.models.py.yolov3.yolov3 import YOLOv3


class TestYOLOv3:
    """Test YOLOv3 model components"""

    def test_darknet53(self):
        """Test Darknet-53 backbone"""
        # Create model
        model = Darknet53()

        # Test with different input sizes
        for size in [320, 416, 608]:
            # Create dummy input
            x = torch.randn(2, 3, size, size)

            # Forward pass
            features = model(x)

            # Check output shapes
            assert len(features) == 3, "Darknet-53 should return 3 feature maps"

            # Expected output sizes for different scales
            expected_sizes = [
                (size // 32, size // 32),  # 13×13 for 416 input
                (size // 16, size // 16),  # 26×26 for 416 input
                (size // 8, size // 8),  # 52×52 for 416 input
            ]

            # Check each feature map
            for i, feature in enumerate(features):
                assert feature.shape[0] == 2, f"Batch size of feature {i} should be 2"
                expected_channels = [1024, 512, 256][i]
                assert feature.shape[1] == expected_channels, (
                    f"Feature {i} should have {expected_channels} channels"
                )
                assert feature.shape[2] == expected_sizes[i][0], (
                    f"Feature {i} should have height {expected_sizes[i][0]}"
                )
                assert feature.shape[3] == expected_sizes[i][1], (
                    f"Feature {i} should have width {expected_sizes[i][1]}"
                )

    def test_fpn(self):
        """Test Feature Pyramid Network"""
        # Create backbone model to get features
        backbone = Darknet53()
        fpn = FeaturePyramidNetwork()

        # Create dummy input
        x = torch.randn(2, 3, 416, 416)

        # Get backbone features
        backbone_features = backbone(x)

        # Forward pass through FPN
        fpn_features = fpn(backbone_features)

        # Check output shapes
        assert len(fpn_features) == 3, "FPN should return 3 feature maps"

        # Expected output channels
        expected_channels = [512, 256, 128]

        # Expected output sizes
        expected_sizes = [(13, 13), (26, 26), (52, 52)]

        # Check each feature map
        for i, feature in enumerate(fpn_features):
            assert feature.shape[0] == 2, f"Batch size of feature {i} should be 2"
            assert feature.shape[1] == expected_channels[i], (
                f"Feature {i} should have {expected_channels[i]} channels"
            )
            assert feature.shape[2] == expected_sizes[i][0], (
                f"Feature {i} should have height {expected_sizes[i][0]}"
            )
            assert feature.shape[3] == expected_sizes[i][1], (
                f"Feature {i} should have width {expected_sizes[i][1]}"
            )

    def test_detection_head(self):
        """Test Detection Head"""
        # Test parameters
        num_classes = 20
        batch_size = 2

        # Create detection head for each scale
        large_head = DetectionHead(512, num_classes)
        medium_head = DetectionHead(256, num_classes)
        small_head = DetectionHead(128, num_classes)

        # Create dummy input for each scale
        large_input = torch.randn(batch_size, 512, 13, 13)
        medium_input = torch.randn(batch_size, 256, 26, 26)
        small_input = torch.randn(batch_size, 128, 52, 52)

        # Forward pass
        large_output = large_head(large_input)
        medium_output = medium_head(medium_input)
        small_output = small_head(small_input)

        # Check output shapes
        # Each head should output (batch_size, num_anchors, grid_size, grid_size, 5 + num_classes)
        # where 5 = 4 box coordinates + 1 objectness score
        assert large_output.shape == (batch_size, 3, 13, 13, 5 + num_classes)
        assert medium_output.shape == (batch_size, 3, 26, 26, 5 + num_classes)
        assert small_output.shape == (batch_size, 3, 52, 52, 5 + num_classes)

    def test_yolov3_head(self):
        """Test YOLOv3 Head (combining all detection heads)"""
        # Test parameters
        num_classes = 20
        batch_size = 2

        # Create YOLOv3 head
        head = YOLOv3Head(num_classes)

        # Create dummy input for each scale
        large_input = torch.randn(batch_size, 512, 13, 13)
        medium_input = torch.randn(batch_size, 256, 26, 26)
        small_input = torch.randn(batch_size, 128, 52, 52)

        # Forward pass
        outputs = head((large_input, medium_input, small_input))

        # Check output
        assert len(outputs) == 3, "YOLOv3 head should return 3 outputs"

        # Check each output shape
        assert outputs[0].shape == (batch_size, 3, 13, 13, 5 + num_classes)
        assert outputs[1].shape == (batch_size, 3, 26, 26, 5 + num_classes)
        assert outputs[2].shape == (batch_size, 3, 52, 52, 5 + num_classes)

    def test_full_yolov3(self):
        """Test full YOLOv3 model"""
        # Test parameters
        batch_size = 2
        input_size = 416
        num_classes = 20

        # Create config
        config = YOLOv3Config(
            input_size=input_size,
            num_classes=num_classes,
            darknet_pretrained=False,  # Don't load pretrained weights for testing
        )

        # Create model
        model = YOLOv3(config)

        # Create dummy input
        x = torch.randn(batch_size, 3, input_size, input_size)

        # Forward pass
        outputs = model(x)

        # Check output
        assert len(outputs) == 3, "YOLOv3 should return 3 outputs"

        # Check each output shape
        # Each output should be (batch_size, num_anchors, grid_size, grid_size, 5 + num_classes)
        assert outputs[0].shape == (batch_size, 3, 13, 13, 5 + num_classes)
        assert outputs[1].shape == (batch_size, 3, 26, 26, 5 + num_classes)
        assert outputs[2].shape == (batch_size, 3, 52, 52, 5 + num_classes)

    def test_predict_function(self):
        """Test predict function with post-processing"""
        # Test parameters
        batch_size = 2
        input_size = 416
        num_classes = 20

        # Create config
        config = YOLOv3Config(
            input_size=input_size,
            num_classes=num_classes,
            darknet_pretrained=False,  # Don't load pretrained weights for testing
        )

        # Create model
        model = YOLOv3(config)

        # Create dummy input
        x = torch.randn(batch_size, 3, input_size, input_size)

        # Test predict function
        with torch.no_grad():
            detections = model.predict(x)

        # Check output
        assert len(detections) == batch_size, "Should return detections for each image in batch"

        # Each detection should be in format [batch_idx, x1, y1, x2, y2, conf, class_id]
        for detection in detections:
            # With random input, we might not have any detections
            # but if we do, check the format
            if detection.shape[0] > 0:
                assert detection.shape[1] == 7, "Each detection should have 7 values"
