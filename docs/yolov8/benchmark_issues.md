# YOLOv8 Benchmark Enhancement: Current Status & Issues (As of 2025-03-30)

## Goal

Enhance the YOLOv8 benchmark report (`src/models/ext/yolov8/benchmark/`) by adding two key visualizations:

1.  A line plot showing mAP vs. IoU Threshold for all tested models.
2.  Confusion matrix heatmaps for each tested model.

## Implementation Status

Significant progress has been made, with changes implemented across several files:

*   **`src/models/ext/yolov8/benchmark/metrics.py`**:
    *   Modified `calculate_detection_metrics` to attempt extracting detailed data (`iou_thresholds`, `mean_ap_per_iou`, `confusion_matrix`) from the `ultralytics.utils.metrics.DetMetrics` object.
    *   Removed `nc` and `iou_vector` arguments from the `DetMetrics()` constructor call due to `TypeError` exceptions encountered during runtime, suggesting API differences in the installed `ultralytics` version.
*   **`src/models/ext/yolov8/benchmark/reporting.py`**:
    *   Implemented `save_map_iou_plot` function using matplotlib.
    *   Implemented `save_confusion_matrix_plot` function using seaborn/matplotlib.
    *   Updated `generate_html_report` to accept the new detailed metrics data, call the plotting functions, and pass relative plot paths to the Jinja2 template context.
    *   Updated the embedded `HTML_TEMPLATE` string to include sections for the new plots.
    *   Added `encoding="utf-8"` to the `open()` call when writing the final HTML report file.
*   **`src/models/ext/yolov8/benchmark/run_benchmark.py`**:
    *   Modified `benchmark_single_model` to return standard and detailed metrics separately.
    *   Updated the `main()` function to collect detailed metrics across all models.
    *   Added logic to determine `class_names` (required for confusion matrices) primarily from the configuration file.
    *   Updated the call to `generate_html_report` to pass the collected detailed metrics and class names.
*   **`src/models/ext/yolov8/benchmark/detection_benchmark.yaml`**:
    *   Added the required `metrics.confusion_matrix_classes` key and populated it with the standard 20 PASCAL VOC class names.
*   **`tests/models/ext/yolov8/benchmark/`**:
    *   Added and significantly updated unit tests (`test_metrics.py`, `test_reporting.py`, `test_run_benchmark.py`).
    *   Addressed various test setup issues related to mocking (`unittest.mock`), Pydantic validation, and decorator argument order.
    *   All unit tests are currently passing after multiple rounds of fixes.

## Known Issues & Uncertainties

1.  **Metric Calculation Failure:**
    *   **Symptom:** Running the benchmark results in `-1.0` for all mAP values in the final `report.html` and `metrics.csv`.
    *   **Cause:** This indicates an error occurs within the final `try...except` block in the `calculate_detection_metrics` function (`metrics.py`) when attempting to extract results from the `det_metrics` object (e.g., accessing `det_metrics.results_dict`, `det_metrics.ap`, `det_metrics.iouv`, etc.). The function logs an error but returns the default error dictionary containing `-1.0`.
    *   **Uncertainty:** The exact point of failure (which attribute access fails or why) is unknown without further debugging.

2.  **Missing Plots in Report:**
    *   **Symptom:** The mAP vs. IoU line plot and the confusion matrix heatmaps are not present in the generated `report.html`.
    *   **Cause:** This is a direct consequence of Issue #1. Since the detailed metric data extraction fails, `None` values are passed to `generate_html_report`, and the corresponding plotting functions (`save_map_iou_plot`, `save_confusion_matrix_plot`) likely skip execution or return `None`, preventing the plots from being added to the report.

3.  **`DetMetrics` API Uncertainty:**
    *   **Symptom:** `TypeError` exceptions were encountered when initializing `ultralytics.utils.metrics.DetMetrics` with `nc` and `iou_vector` arguments, requiring their removal from the constructor call.
    *   **Cause:** The API of the `DetMetrics` class in the specific installed version of the `ultralytics` library differs from the assumed or previously documented API.
    *   **Uncertainty:** The correct way to configure the IoU range for calculation and the exact structure/attributes of the `DetMetrics` object *after* processing data are still uncertain and require investigation.

4.  **GPU Memory Reporting:**
    *   **Symptom:** `peak_gpu_memory_mb` shows `-1.0` in the report.
    *   **Cause:** The benchmark was run using the CPU (as indicated by logs: `Using auto-detected device: cpu`). GPU memory measurement is skipped when not running on CUDA.
    *   **Note:** This is expected behavior when running on CPU, not necessarily an issue unless GPU execution was intended.

## Next Debugging Steps

1.  Add detailed logging inside the final `try...except` block of `calculate_detection_metrics` (`metrics.py`) to inspect the `det_metrics` object (e.g., `dir(det_metrics)`, `hasattr(...)` checks, print values if they exist) before extraction is attempted.
2.  Re-run the benchmark script (`python -m src.models.ext.yolov8.benchmark.run_benchmark ...`).
3.  Analyze the logs to understand the available attributes/methods on `det_metrics` and why the current extraction fails.
4.  Adjust the data extraction logic in `calculate_detection_metrics` based on the findings.
5.  (Optional) Consult the specific `ultralytics` version's documentation or source code for the `DetMetrics` class API.