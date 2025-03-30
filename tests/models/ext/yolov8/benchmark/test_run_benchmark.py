import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import yaml

# Assume run_benchmark.py is accessible (might need path adjustment)
from src.models.ext.yolov8.benchmark import run_benchmark

# Sample data consistent with test_reporting.py
MOCK_CLASS_NAMES = ["class_a", "class_b", "class_c"]
MOCK_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
MOCK_CONFIG_DATA_WITH_CLASSES = {
    "models_to_test": ["mock_model1.pt", "mock_model2.pt"],
    "dataset": {
        "test_images_dir": "/fake/images",
        "annotations_dir": "/fake/labels",
        "annotation_format": "yolo_txt",
        "subset_method": "first_n",
        "subset_size": 2,
    },
    "metrics": {
        "iou_threshold_map": 0.5,
        "iou_range_coco": [0.5, 0.95, 0.05],
        "object_size_definitions": {
            "small": [0, 1024],
            "medium": [1024, 9216],
            "large": [9216, float("inf")],
        },
        "confusion_matrix_classes": MOCK_CLASS_NAMES,  # Provide class names
    },
    "compute": {"device": "cpu", "batch_size": 1},
    "output": {
        "output_dir": "benchmark_results/run_{timestamp:%Y%m%d_%H%M%S}",  # Use placeholder
        "results_csv": "metrics.csv",
        "results_html": "report.html",
        "save_plots": True,
        "save_qualitative_results": False,
        "num_qualitative_images": 0,
    },
}

MOCK_STD_METRICS_1 = {"model_name": "mock_model1", "mAP_50": 0.7, "mAP_50_95": 0.5}
MOCK_DTL_METRICS_1 = {
    "iou_thresholds": MOCK_IOU_THRESHOLDS,
    "mean_ap_per_iou": np.random.rand(10),
    "confusion_matrix": np.random.randint(10, size=(3, 3)),
}

MOCK_STD_METRICS_2 = {"model_name": "mock_model2", "mAP_50": 0.8, "mAP_50_95": 0.6}
MOCK_DTL_METRICS_2 = {
    "iou_thresholds": MOCK_IOU_THRESHOLDS,
    "mean_ap_per_iou": np.random.rand(10),
    "confusion_matrix": np.random.randint(10, size=(3, 3)),
}


class TestRunBenchmarkMain(unittest.TestCase):
    # Define mock_argv as a class attribute
    mock_argv = ['run_benchmark.py', '--config', '/tmp/dummy_config_for_class.yaml'] # Placeholder path

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())  # Temp dir for config file
        self.config_path = self.test_dir / "test_config.yaml"

        # Update the MOCK_CONFIG_DATA dictionary *before* writing it to YAML
        MOCK_CONFIG_DATA_WITH_CLASSES['dataset']['test_images_dir'] = str(self.test_dir / "fake_images")
        MOCK_CONFIG_DATA_WITH_CLASSES['dataset']['annotations_dir'] = str(self.test_dir / "fake_labels")
        MOCK_CONFIG_DATA_WITH_CLASSES['dataset']['num_classes'] = len(MOCK_CLASS_NAMES) # Add required field

        # Create dummy dirs for validation
        (self.test_dir / "fake_images").mkdir(exist_ok=True)
        (self.test_dir / "fake_labels").mkdir(exist_ok=True)

        # Write the *updated* config data to the temp file
        with open(self.config_path, 'w') as f:
            yaml.dump(MOCK_CONFIG_DATA_WITH_CLASSES, f)

        # Update class attribute with the actual temp config path
        TestRunBenchmarkMain.mock_argv = ['run_benchmark.py', '--config', str(self.config_path)]

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("src.models.ext.yolov8.benchmark.run_benchmark.YOLO")  # Mock model loading
    @patch(
        "src.models.ext.yolov8.benchmark.run_benchmark.find_dataset_files"
    )  # Mock dataset finding
    @patch("src.models.ext.yolov8.benchmark.run_benchmark.select_subset")  # Mock subset selection
    @patch(
        "src.models.ext.yolov8.benchmark.run_benchmark.benchmark_single_model"
    )  # Mock core benchmark fn
    @patch("src.models.ext.yolov8.benchmark.run_benchmark.pd.DataFrame.to_csv")  # Mock CSV saving
    @patch(
        "src.models.ext.yolov8.benchmark.run_benchmark.generate_html_report"
    )  # Mock report generation
    @patch("src.models.ext.yolov8.benchmark.run_benchmark.Path.mkdir")  # Mock dir creation
    @patch("builtins.open", new_callable=unittest.mock.mock_open)  # Mock open for saving config
    @patch.object(sys, "argv", new_callable=lambda: TestRunBenchmarkMain.mock_argv)  # Mock sys.argv
    def test_main_flow_with_detailed_metrics(
        self,
        mock_sys_argv,
        mock_builtin_open,  # Patched open
        mock_mkdir,
        mock_generate_report,
        mock_to_csv,
        mock_benchmark_single,
        mock_select_subset,
        mock_find_files,
        mock_yolo_init,
    ):
        """Test the main function correctly processes and passes detailed metrics."""

        # --- Setup Mocks ---
        mock_find_files.return_value = [
            (Path("/fake/images/img1.jpg"), Path("/fake/labels/img1.txt")),
            (Path("/fake/images/img2.jpg"), Path("/fake/labels/img2.txt")),
        ]
        mock_select_subset.return_value = mock_find_files.return_value  # Assume subset is all
        mock_yolo_init.return_value = MagicMock()  # Mock YOLO instance

        # Define side effects for benchmark_single_model to return different results per model
        mock_benchmark_single.side_effect = [
            (MOCK_STD_METRICS_1, MOCK_DTL_METRICS_1),  # Results for model1
            (MOCK_STD_METRICS_2, MOCK_DTL_METRICS_2),  # Results for model2
        ]

        # Mock file opening for reading the config
        # Need to handle the actual read call separately from the write calls
        config_content = yaml.dump(MOCK_CONFIG_DATA_WITH_CLASSES)
        mock_builtin_open.side_effect = [
            unittest.mock.mock_open(read_data=config_content).return_value,  # For reading config
            unittest.mock.mock_open().return_value,  # For writing used config
            # Add more if other files are opened
        ]

        # --- Run main function ---
        run_benchmark.main()

        # --- Assertions ---
        # Check config loading
        mock_builtin_open.assert_any_call(self.config_path, "r")

        # Check benchmark calls
        self.assertEqual(mock_benchmark_single.call_count, 2)  # Called for 2 models
        mock_benchmark_single.assert_has_calls(
            [
                call(
                    model=mock_yolo_init(),
                    model_name="mock_model1.pt",
                    benchmark_files=mock_select_subset.return_value,
                    config=unittest.mock.ANY,
                    output_dir=unittest.mock.ANY,
                    num_classes=len(MOCK_CLASS_NAMES),
                ),
                call(
                    model=mock_yolo_init(),
                    model_name="mock_model2.pt",
                    benchmark_files=mock_select_subset.return_value,
                    config=unittest.mock.ANY,
                    output_dir=unittest.mock.ANY,
                    num_classes=len(MOCK_CLASS_NAMES),
                ),
            ],
            any_order=False,
        )  # Check calls in order

        # Check CSV saving (optional, check if called)
        mock_to_csv.assert_called_once()

        # Check report generation call
        mock_generate_report.assert_called_once()
        report_args, report_kwargs = mock_generate_report.call_args

        # Check arguments passed to generate_html_report
        self.assertIsInstance(report_kwargs["results_df"], pd.DataFrame)
        self.assertEqual(len(report_kwargs["results_df"]), 2)
        self.assertEqual(report_kwargs["config_data"], MOCK_CONFIG_DATA_WITH_CLASSES)
        np.testing.assert_array_equal(report_kwargs["iou_thresholds"], MOCK_IOU_THRESHOLDS)
        self.assertEqual(
            list(report_kwargs["mean_ap_per_iou_dict"].keys()), ["mock_model1.pt", "mock_model2.pt"]
        )
        np.testing.assert_array_equal(
            report_kwargs["mean_ap_per_iou_dict"]["mock_model1.pt"],
            MOCK_DTL_METRICS_1["mean_ap_per_iou"],
        )
        self.assertEqual(
            list(report_kwargs["confusion_matrices_dict"].keys()),
            ["mock_model1.pt", "mock_model2.pt"],
        )
        np.testing.assert_array_equal(
            report_kwargs["confusion_matrices_dict"]["mock_model1.pt"],
            MOCK_DTL_METRICS_1["confusion_matrix"],
        )
        self.assertEqual(report_kwargs["class_names"], MOCK_CLASS_NAMES)
        self.assertIsInstance(report_kwargs["output_dir"], Path)
        self.assertEqual(
            report_kwargs["report_filename"],
            MOCK_CONFIG_DATA_WITH_CLASSES["output"]["results_html"],
        )

        # Check directory creation (optional)
        # self.assertGreaterEqual(mock_mkdir.call_count, 1)

        # Check config saving
        # mock_builtin_open.assert_any_call(unittest.mock.ANY, "w", encoding="utf-8") # Check config save


if __name__ == "__main__":
    unittest.main()
