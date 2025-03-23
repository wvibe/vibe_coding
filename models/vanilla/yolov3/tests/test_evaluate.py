"""
Unit tests for evaluate.py
"""

import unittest
from typing import Dict, List

import numpy as np
import torch

from models.vanilla.yolov3.evaluate import (
    calculate_ap,
    calculate_iou,
    calculate_mean_ap,
)


class TestIoU(unittest.TestCase):
    """Test IoU calculation"""

    def test_perfect_overlap(self):
        """Test IoU calculation with perfect overlap (IoU = 1.0)"""
        box1 = torch.tensor([0.0, 0.0, 10.0, 10.0])
        box2 = torch.tensor([0.0, 0.0, 10.0, 10.0])
        iou = calculate_iou(box1, box2)
        self.assertAlmostEqual(iou, 1.0)

    def test_no_overlap(self):
        """Test IoU calculation with no overlap (IoU = 0.0)"""
        box1 = torch.tensor([0.0, 0.0, 10.0, 10.0])
        box2 = torch.tensor([20.0, 20.0, 30.0, 30.0])
        iou = calculate_iou(box1, box2)
        self.assertAlmostEqual(iou, 0.0)

    def test_partial_overlap(self):
        """Test IoU calculation with partial overlap"""
        box1 = torch.tensor([0.0, 0.0, 10.0, 10.0])  # Area = 100
        box2 = torch.tensor([5.0, 5.0, 15.0, 15.0])  # Area = 100
        # Intersection area = 5*5 = 25
        # Union area = 100 + 100 - 25 = 175
        # IoU = 25/175 = 0.1428...
        iou = calculate_iou(box1, box2)
        self.assertAlmostEqual(iou, 25.0 / 175.0)


class TestAP(unittest.TestCase):
    """Test AP calculation"""

    def test_perfect_ap(self):
        """Test AP calculation with perfect precision (AP = 1.0)"""
        recalls = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        precisions = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ap = calculate_ap(recalls, precisions)
        self.assertAlmostEqual(ap, 1.0)

    def test_zero_ap(self):
        """Test AP calculation with zero precision (AP = 0.0)"""
        recalls = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        precisions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ap = calculate_ap(recalls, precisions)
        self.assertAlmostEqual(ap, 0.0)

    def test_partial_ap(self):
        """Test AP calculation with partial precision"""
        recalls = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        precisions = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        ap = calculate_ap(recalls, precisions)
        # For 11-point interpolation, this should be around 0.5
        self.assertGreater(ap, 0.4)
        self.assertLess(ap, 0.6)


class TestMAP(unittest.TestCase):
    """Test mAP calculation"""

    def create_mock_detections(
        self,
        scores: List[float],
        boxes: List[List[float]],
        cls_idx: int,
        num_classes: int,
    ) -> List[List[List[Dict]]]:
        """Create mock detections for testing"""
        detections = [[[] for _ in range(num_classes)] for _ in range(1)]
        for score, box in zip(scores, boxes):
            detections[0][cls_idx].append(
                {"bbox": torch.tensor(box), "confidence": score}
            )
        return detections

    def create_mock_ground_truths(
        self, boxes: List[List[float]], cls_idx: int, num_classes: int
    ) -> List[List[List[Dict]]]:
        """Create mock ground truths for testing"""
        ground_truths = [[[] for _ in range(num_classes)] for _ in range(1)]
        for box in boxes:
            ground_truths[0][cls_idx].append({"bbox": torch.tensor(box)})
        return ground_truths

    def test_perfect_detection(self):
        """Test mAP calculation with perfect detection (mAP = 1.0)"""
        num_classes = 2
        # Create perfect detections for class 0
        detections = self.create_mock_detections(
            [1.0, 1.0],
            [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
            0,
            num_classes,
        )
        ground_truths = self.create_mock_ground_truths(
            [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], 0, num_classes
        )
        mAP, class_APs = calculate_mean_ap(detections, ground_truths)
        self.assertAlmostEqual(mAP, 0.5)  # One class has perfect AP, one has 0 AP
        self.assertAlmostEqual(class_APs[0], 1.0)
        self.assertAlmostEqual(class_APs[1], 0.0)

    def test_no_detection(self):
        """Test mAP calculation with no detection (mAP = 0.0)"""
        num_classes = 1
        # Create empty detections
        detections = [[[] for _ in range(num_classes)] for _ in range(1)]
        # Create ground truths
        ground_truths = self.create_mock_ground_truths(
            [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], 0, num_classes
        )
        mAP, class_APs = calculate_mean_ap(detections, ground_truths)
        self.assertAlmostEqual(mAP, 0.0)
        self.assertAlmostEqual(class_APs[0], 0.0)


if __name__ == "__main__":
    unittest.main()
