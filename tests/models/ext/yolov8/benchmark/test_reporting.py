import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import numpy as np
import pandas as pd
import yaml  # For loading config data in test
import jinja2 # Import for patching

# Import functions to test
from src.models.ext.yolov8.benchmark.reporting import (
    generate_html_report,
    save_confusion_matrix_plot,
    save_map_iou_plot,
)

# Sample data for testing
MOCK_CLASS_NAMES = ["class_a", "class_b", "class_c"]
MOCK_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
MOCK_RESULTS_DF = pd.DataFrame(
    {
        "model_name": ["model1", "model2"],
        "mAP_50": [0.7, 0.8],
        "mAP_50_95": [0.5, 0.6],
        "inference_time_ms_mean": [10.5, 9.8],
        "peak_gpu_memory_mb": [1024, 1536],
    }
)
MOCK_MEAN_AP_PER_IOU = {
    "model1": np.random.rand(10) * 0.3 + 0.5,  # shape (10,)
    "model2": np.random.rand(10) * 0.2 + 0.6,  # shape (10,)
}
MOCK_CM_DATA = {
    "model1": np.random.randint(0, 50, size=(len(MOCK_CLASS_NAMES), len(MOCK_CLASS_NAMES))),
    "model2": np.random.randint(0, 40, size=(len(MOCK_CLASS_NAMES), len(MOCK_CLASS_NAMES))),
}
MOCK_CONFIG_DATA = {
    'models_to_test': ['model1', 'model2'],
    'dataset': {
        'test_images_dir': '/path/to/images',
        'annotations_dir': '/path/to/labels',
        'annotation_format': 'yolo_txt',
        'num_classes': len(MOCK_CLASS_NAMES),
        'subset_method': 'first_n',
        'subset_size': 10,
        'image_list_file': None,
    },
    'metrics': {
        'iou_threshold_map': 0.5,
        'iou_range_coco': [0.5, 0.95, 0.05],
        'object_size_definitions': {
            'small': [0, 1024],
            'medium': [1024, 9216],
            'large': [9216, float('inf')]
        },
        'confusion_matrix_classes': MOCK_CLASS_NAMES
    },
    'compute': {'device': 'cpu', 'batch_size': 1},
    'output': {
        'output_dir': 'benchmark_results/test_run',
        'results_csv': 'metrics.csv',
        'results_html': 'report.html',
        'save_plots': True,
        'save_qualitative_results': False,
        'num_qualitative_images': 0
    }
}


class TestReporting(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test outputs
        self.test_dir = Path(tempfile.mkdtemp())  # Create a real temp dir

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    @patch("src.models.ext.yolov8.benchmark.reporting.plt")
    def test_save_map_iou_plot_success(self, mock_plt):
        """Test save_map_iou_plot successfully generates and saves plot."""
        relative_path = save_map_iou_plot(
            results_df=MOCK_RESULTS_DF,
            iou_thresholds=MOCK_IOU_THRESHOLDS,
            mean_ap_per_iou_dict=MOCK_MEAN_AP_PER_IOU,
            output_dir=self.test_dir,
        )

        expected_filename = self.test_dir / "plots" / "map_vs_iou.png"
        mock_plt.figure.assert_called_once()
        self.assertEqual(mock_plt.plot.call_count, 2)  # Called once per model
        mock_plt.title.assert_called_with("Mean Average Precision (mAP) vs. IoU Threshold")
        mock_plt.xlabel.assert_called_with("IoU Threshold")
        mock_plt.ylabel.assert_called_with("mAP")
        mock_plt.grid.assert_called_with(True)
        mock_plt.legend.assert_called_once()
        mock_plt.ylim.assert_called_with(0, 1)
        mock_plt.tight_layout.assert_called_once()
        mock_plt.savefig.assert_called_once_with(expected_filename)
        mock_plt.close.assert_called_once()
        self.assertEqual(relative_path, "plots/map_vs_iou.png")

    @patch("src.models.ext.yolov8.benchmark.reporting.plt")
    def test_save_map_iou_plot_no_data(self, mock_plt):
        """Test save_map_iou_plot handles missing data."""
        relative_path = save_map_iou_plot(
            results_df=MOCK_RESULTS_DF,
            iou_thresholds=None,  # Missing IoUs
            mean_ap_per_iou_dict=MOCK_MEAN_AP_PER_IOU,
            output_dir=self.test_dir,
        )
        self.assertIsNone(relative_path)
        mock_plt.savefig.assert_not_called()

        relative_path = save_map_iou_plot(
            results_df=MOCK_RESULTS_DF,
            iou_thresholds=MOCK_IOU_THRESHOLDS,
            mean_ap_per_iou_dict={},
            output_dir=self.test_dir,
        )
        self.assertIsNone(relative_path)
        mock_plt.savefig.assert_not_called()

    @patch("src.models.ext.yolov8.benchmark.reporting.sns")
    @patch("src.models.ext.yolov8.benchmark.reporting.plt")
    def test_save_confusion_matrix_plot_success(self, mock_plt, mock_sns):
        """Test save_confusion_matrix_plot successfully generates and saves plot."""
        model_name = "model1"
        relative_path = save_confusion_matrix_plot(
            model_name=model_name,
            cm_data=MOCK_CM_DATA[model_name],
            class_names=MOCK_CLASS_NAMES,
            output_dir=self.test_dir,
        )

        expected_filename = self.test_dir / "plots" / f"confusion_matrix_{model_name}.png"
        mock_plt.figure.assert_called_once()
        mock_sns.heatmap.assert_called_once()
        mock_plt.title.assert_called_with(f"Confusion Matrix: {model_name}")
        mock_plt.xlabel.assert_called_with("Predicted Label")
        mock_plt.ylabel.assert_called_with("True Label")
        mock_plt.tight_layout.assert_called_once()
        mock_plt.savefig.assert_called_once_with(expected_filename)
        mock_plt.close.assert_called_once()
        self.assertEqual(relative_path, f"plots/confusion_matrix_{model_name}.png")

    @patch("src.models.ext.yolov8.benchmark.reporting.plt")
    def test_save_confusion_matrix_plot_no_data(self, mock_plt):
        """Test save_confusion_matrix_plot handles missing data."""
        relative_path = save_confusion_matrix_plot(
            model_name="model1",
            cm_data=None,
            class_names=MOCK_CLASS_NAMES,
            output_dir=self.test_dir,
        )
        self.assertIsNone(relative_path)
        mock_plt.savefig.assert_not_called()

    @patch("src.models.ext.yolov8.benchmark.reporting.plt")
    def test_save_confusion_matrix_plot_mismatched_dims(self, mock_plt):
        """Test save_confusion_matrix_plot handles mismatched dimensions."""
        wrong_cm = np.zeros((2, 2))  # Only 2x2
        relative_path = save_confusion_matrix_plot(
            model_name="model1",
            cm_data=wrong_cm,
            class_names=MOCK_CLASS_NAMES,  # 3 classes
            output_dir=self.test_dir,
        )
        self.assertIsNone(relative_path)
        mock_plt.savefig.assert_not_called()

    @patch('src.models.ext.yolov8.benchmark.reporting.save_map_iou_plot')
    @patch('src.models.ext.yolov8.benchmark.reporting.save_confusion_matrix_plot')
    @patch('builtins.open', new_callable=unittest.mock.mock_open) # Mock file opening
    @patch('src.models.ext.yolov8.benchmark.reporting.save_comparison_barplot') # Add missing patch
    def test_generate_html_report(
        self, mock_save_barplot, mock_open, mock_save_cm, mock_save_map_iou
    ):
        """Test generate_html_report calls helpers and renders template."""
        # --- Setup Mocks ---
        # Mock return values for plot saving functions (relative paths)
        # Make mocks *always* return the expected relative path
        def barplot_side_effect(df, metric, title, fname, **kwargs):
            if fname.name.startswith("map50_comparison"):
                return "plots/map50_comparison.png"
            elif fname.name.startswith("map5095_comparison"):
                return "plots/map5095_comparison.png"
            elif fname.name.startswith("time_comparison"):
                return "plots/time_comparison.png"
            elif fname.name.startswith("gpu_comparison"):
                return "plots/gpu_comparison.png"
            else:
                return None # Or raise error for unexpected calls
        mock_save_barplot.side_effect = barplot_side_effect
        mock_save_map_iou.return_value = "plots/map_vs_iou.png"
        # Correct lambda signature to match the actual function call
        mock_save_cm.side_effect = lambda model_name, cm_data, class_names, output_dir: f"plots/confusion_matrix_{model_name}.png"

        # Path.exists patch removed

        # --- Call Function ---
        report_filename = "final_report.html"
        generate_html_report(
            results_df=MOCK_RESULTS_DF,
            config_data=MOCK_CONFIG_DATA,
            iou_thresholds=MOCK_IOU_THRESHOLDS,
            mean_ap_per_iou_dict=MOCK_MEAN_AP_PER_IOU,
            confusion_matrices_dict=MOCK_CM_DATA,
            class_names=MOCK_CLASS_NAMES,
            output_dir=self.test_dir,
            report_filename=report_filename,
        )

        # --- Assertions ---
        # Check barplot calls
        self.assertGreaterEqual(mock_save_barplot.call_count, 2)  # At least map50, map5095 called

        # Check new plot calls
        mock_save_map_iou.assert_called_once_with(
            MOCK_RESULTS_DF, MOCK_IOU_THRESHOLDS, MOCK_MEAN_AP_PER_IOU, self.test_dir
        )
        self.assertEqual(mock_save_cm.call_count, 2)  # Called for model1 and model2
        mock_save_cm.assert_has_calls(
            [
                call(
                    model_name="model1",
                    cm_data=MOCK_CM_DATA["model1"],
                    class_names=MOCK_CLASS_NAMES,
                    output_dir=self.test_dir,
                ),
                call(
                    model_name="model2",
                    cm_data=MOCK_CM_DATA["model2"],
                    class_names=MOCK_CLASS_NAMES,
                    output_dir=self.test_dir,
                ),
            ],
            any_order=True,
        )

        # Check file writing
        expected_report_path = self.test_dir / report_filename
        mock_open.assert_called_once_with(expected_report_path, "w", encoding="utf-8")
        mock_open().write.assert_called_once()


if __name__ == "__main__":
    unittest.main()
