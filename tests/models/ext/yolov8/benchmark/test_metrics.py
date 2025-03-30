import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path

from src.models.ext.yolov8.benchmark.config import (
    BenchmarkConfig,
    ComputeConfig,
    DatasetConfig,
    MetricsConfig,
    ObjectSizeDefinition,
    OutputConfig,
)

# Import the function to test (adjust path as necessary)
from src.models.ext.yolov8.benchmark.metrics import calculate_detection_metrics


# Define a basic mock config for testing
class MockArgs:
    config = "dummy_config_path.yaml"


# Sample config for context (values don't heavily influence this specific test)
mock_object_size_defs = ObjectSizeDefinition(
    small=[0, 1024], medium=[1024, 9216], large=[9216, float("inf")]
)
mock_metrics_config = MetricsConfig(
    iou_threshold_map=0.5,
    iou_range_coco=[0.5, 0.95, 0.05],
    object_size_definitions=mock_object_size_defs,
    confusion_matrix_classes=None,
)
mock_compute_config = ComputeConfig(device="cpu", batch_size=1)
mock_output_config = OutputConfig(
    output_dir="dummy_out",
    results_csv="res.csv",
    results_html="rep.html",
    save_plots=False,
    save_qualitative_results=False,
    num_qualitative_images=0,
)


class TestCalculateDetectionMetrics(unittest.TestCase):
    def setUp(self):
        "Create temporary directories for dataset paths."
        self.test_dir = Path(tempfile.mkdtemp())
        self.img_dir = self.test_dir / "images"
        self.ann_dir = self.test_dir / "annotations"
        self.img_dir.mkdir()
        self.ann_dir.mkdir()

        # Define configs that require temp paths here
        self.mock_dataset_config = DatasetConfig(
            test_images_dir=self.img_dir,
            annotations_dir=self.ann_dir,
            annotation_format='yolo_txt',
            subset_method='all',
            subset_size=1,
            image_list_file=None,
            num_classes=20 # Added missing field
        )
        self.mock_config = BenchmarkConfig(
            models_to_test=['yolov8n'],
            dataset=self.mock_dataset_config,
            metrics=mock_metrics_config,
            compute=mock_compute_config,
            output=mock_output_config
        )

    def tearDown(self):
        "Remove temporary directory."
        shutil.rmtree(self.test_dir)

    @patch("src.models.ext.yolov8.benchmark.metrics.metrics.DetMetrics")
    def test_calculate_detection_metrics_extraction(self, MockDetMetrics):
        """Test that detailed metrics are correctly extracted."""
        # --- Setup Mock DetMetrics ---
        mock_instance = MockDetMetrics.return_value

        # Mock the results_dict attribute
        mock_instance.results_dict = {
            "metrics/mAP50(B)": 0.55,
            "metrics/mAP50-95(B)": 0.45,
            "metrics/mAP50-95(S)": 0.15,
            "metrics/mAP50-95(M)": 0.40,
            "metrics/mAP50-95(L)": 0.60,
        }

        # Mock the attributes needed for detailed data extraction
        # Ensure tensors are created on CPU for numpy conversion
        mock_instance.iouv = torch.linspace(0.5, 0.95, 10)  # Example IoU vector
        mock_instance.ap = torch.rand(20, 10)  # Example AP array (num_classes, num_iou_thresholds)
        mock_instance.confusion_matrix = MagicMock()
        mock_instance.confusion_matrix.matrix = torch.randint(
            0, 10, (20, 20)
        )  # Example confusion matrix

        # Expected extracted data (converted to numpy)
        expected_ious = mock_instance.iouv.cpu().numpy()
        expected_mean_ap = mock_instance.ap.mean(0).cpu().numpy()
        expected_cm = mock_instance.confusion_matrix.matrix.cpu().numpy()

        # --- Setup Inputs (Minimal needed) ---
        # Mock predictions (need at least one to avoid early return)
        # Create a mock Results object with necessary attributes
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.xyxyn = torch.empty((1, 4), device="cpu")  # Minimal tensor on cpu
        mock_result.boxes.conf = torch.empty((1,), device="cpu")
        mock_result.boxes.cls = torch.empty((1,), device="cpu")
        predictions = [mock_result]
        ground_truths = [[]]  # Empty GT list is handled
        num_classes = 20  # Match mock AP/CM dimensions

        # --- Call the function ---
        results = calculate_detection_metrics(
            predictions=predictions,
            ground_truths=ground_truths,
            num_classes=num_classes,
            config=self.mock_config,
        )

        # --- Assertions ---
        # Check standard metrics
        self.assertAlmostEqual(results["mAP_50"], 0.55)
        self.assertAlmostEqual(results["mAP_50_95"], 0.45)
        self.assertAlmostEqual(results["mAP_small"], 0.15)
        self.assertAlmostEqual(results["mAP_medium"], 0.40)
        self.assertAlmostEqual(results["mAP_large"], 0.60)

        # Check detailed metrics extraction
        self.assertIsNotNone(results["iou_thresholds"])
        np.testing.assert_allclose(results["iou_thresholds"], expected_ious)

        self.assertIsNotNone(results["mean_ap_per_iou"])
        np.testing.assert_allclose(results["mean_ap_per_iou"], expected_mean_ap)

        self.assertIsNotNone(results["confusion_matrix"])
        np.testing.assert_array_equal(results["confusion_matrix"], expected_cm)

        # Verify DetMetrics was initialized and processed correctly
        MockDetMetrics.assert_called_once()
        mock_instance.process.assert_called_once()

    @patch("src.models.ext.yolov8.benchmark.metrics.metrics.DetMetrics")
    def test_calculate_detection_metrics_no_predictions(self, MockDetMetrics):
        """Test the behavior when no predictions are provided."""
        results = calculate_detection_metrics(
            predictions=[],  # Empty list
            ground_truths=[],
            num_classes=20,
            config=self.mock_config,
        )

        # Assert default values are returned, including None for new keys
        self.assertEqual(results["mAP_50"], 0.0)
        self.assertEqual(results["mAP_50_95"], 0.0)
        self.assertIsNone(results["iou_thresholds"])
        self.assertIsNone(results["mean_ap_per_iou"])
        self.assertIsNone(results["confusion_matrix"])
        MockDetMetrics.assert_not_called()  # DetMetrics should not be initialized

    @patch(
        "src.models.ext.yolov8.benchmark.metrics.metrics.DetMetrics",
        side_effect=Exception("Init failed"),
    )
    def test_calculate_detection_metrics_init_fails(self, MockDetMetrics):
        """Test behavior when DetMetrics initialization fails."""
        # Mock predictions
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.xyxyn = torch.empty((1, 4), device="cpu")  # Minimal tensor on cpu
        mock_result.boxes.conf = torch.empty((1,), device="cpu")
        mock_result.boxes.cls = torch.empty((1,), device="cpu")
        predictions = [mock_result]
        ground_truths = [[]]
        num_classes = 20

        results = calculate_detection_metrics(
            predictions=predictions,
            ground_truths=ground_truths,
            num_classes=num_classes,
            config=self.mock_config,
        )

        # Assert error values are returned, including None for new keys
        self.assertEqual(results["mAP_50"], -1.0)
        self.assertEqual(results["mAP_50_95"], -1.0)
        self.assertIsNone(results["iou_thresholds"])
        self.assertIsNone(results["mean_ap_per_iou"])
        self.assertIsNone(results["confusion_matrix"])
        MockDetMetrics.assert_called_once()  # Attempted to initialize


if __name__ == "__main__":
    unittest.main()
