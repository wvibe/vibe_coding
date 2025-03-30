# Object Detection Benchmark Design

This document outlines the design for a configurable benchmarking tool for object detection models within the `vibe_coding` project.

## 1. Goal

Develop a flexible and reusable benchmark tool to evaluate and compare object detection models, initially focusing on Ultralytics YOLO families (v5, v8, etc.) and custom-trained models. The benchmark will assess performance on standard datasets like PASCAL VOC using various accuracy and computational resource metrics.

## 2. Location

-   **Code:** `src/models/ext/yolov8/benchmark/`
    -   Main script: `run_benchmark.py`
    -   Helpers (potential): `metrics.py`, `reporting.py`, `config.py`, `utils.py`
-   **Configuration:** `src/models/ext/yolov8/benchmark/detection_benchmark.yaml` (Example)
-   **Documentation:** `docs/yolov8/benchmark_design.md` (This file)
-   **Tests:** `tests/models/ext/yolov8/benchmark/`

## 3. Configuration (`detection_benchmark.yaml`)

The benchmark run will be configured using a single YAML file.

```yaml
# Example: configs/benchmarking/detection_benchmark.yaml

# List of models to test.
# Can be Ultralytics model names (e.g., 'yolov8n') which implies downloading yolov8n.pt,
# or paths to local checkpoint files (.pt).
# Future: Could add a 'type' field if different loading logic is needed.
models_to_test:
  - yolov8n
  - yolov5s
  # - /home/wmu/vibe/hub/models/custom_yolov8_finetuned.pt
  # - /home/wmu/vibe/hub/models/another_model.pt

dataset:
  # Path to the directory containing test images
  test_images_dir: "/home/wmu/vibe/hub/datasets/VOC2007/test/JPEGImages" # Example
  # Path to the directory containing annotations
  annotations_dir: "/home/wmu/vibe/hub/datasets/VOC2007/test/Annotations" # Example
  # Annotation format ('voc_xml', 'yolo_txt', 'coco_json')
  annotation_format: "voc_xml"
  # Method to select a subset of images: 'random', 'first_n', 'all'
  subset_method: random
  # Number of images for the subset (ignored if subset_method is 'all')
  subset_size: 100
  # Optional: Path to a file listing specific image basenames (without extension) to use
  image_list_file: null # e.g., "configs/benchmarking/voc_test_subset.txt"

metrics:
  # Primary IoU threshold for main mAP reporting
  iou_threshold_map: 0.5
  # IoU range for COCO-style mAP [start, end_inclusive, step]
  iou_range_coco: [0.5, 0.95, 0.05]
  # Object size definitions for mAP_small, mAP_medium, mAP_large (area in pixels^2)
  object_size_definitions:
    small: [0, 1024]      # area < 32*32
    medium: [1024, 9216]  # 32*32 <= area < 96*96
    large: [9216, .inf]   # area >= 96*96 # Use '.inf' for infinity
  # Optional: List of class names for confusion matrix. If null, use all classes found in the dataset.
  confusion_matrix_classes: null # e.g., ["person", "car", "dog"]

compute:
  # Device to run inference on ('cpu', 'cuda:0', etc.)
  # 'auto' will let Ultralytics decide (usually CUDA if available)
  device: 'auto'
  # Batch size for inference (if model/dataloader supports batching)
  batch_size: 1 # Default to 1 for consistent timing per image initially

output:
  # Directory to save all results (plots, csv, html)
  # Can use placeholders like {timestamp}
  output_dir: "benchmark_results/run_{timestamp:%Y%m%d_%H%M%S}"
  # Filename for the CSV containing aggregated metrics
  results_csv: "metrics.csv"
  # Filename for the main HTML report
  results_html: "report.html"
  # Save generated plots (e.g., confusion matrices)?
  save_plots: true
  # Save qualitative results (images with boxes)?
  save_qualitative_results: true
  # Number of qualitative result images to save
  num_qualitative_images: 5

```

## 4. Core Features & Logic (`src/models/ext/yolov8/benchmark/`)

-   **`run_benchmark.py`:**
    -   Parses command-line arguments (e.g., `--config path/to/config.yaml`).
    *   Loads and validates the configuration using Pydantic models (defined in `config.py`) for robustness.
    -   Sets up logging.
    -   Creates the output directory (handling timestamp placeholders).
    -   Loads the dataset index (image paths, annotation paths/data) based on config.
    -   Initializes a list to store results per model.
    -   **Sequential Model Loop:** Iterates through `models_to_test`.
        -   Prints progress (e.g., "Benchmarking model: yolov8n...").
        *   Calls a `benchmark_single_model` function.
        *   Stores the returned metrics dictionary.
    -   Aggregates results from all models into a pandas DataFrame.
    -   Saves the DataFrame to CSV (using `output.results_csv` path).
    -   Calls a `generate_html_report` function.
    -   Prints a summary to the console.
-   **`benchmark_single_model` function:**
    -   Input: Model identifier (name or path), dataset information, config object.
    -   Output: Dictionary of metrics for this model.
    -   Loads the model (using `ultralytics.YOLO` or custom logic if needed). Handle potential download for UL names.
    *   Measures model disk size.
    -   Determines device based on config and availability. Moves model to device.
    *   Initializes lists for per-image inference times and predictions/ground truths.
    *   Resets GPU stats if applicable (`torch.cuda.reset_peak_memory_stats`).
    *   **Image Loop:** Iterates through the image subset.
        *   Loads image and ground truth annotation.
        *   Starts timer.
        *   Performs inference (`model(image, device=device, verbose=False)`).
        *   Stops timer, records inference time.
        *   Stores predictions and ground truths in a standardized format suitable for evaluation.
    *   Calculates inference time statistics (mean, 75th, 90th, 95th percentiles) using `numpy.percentile`.
    *   Gets peak GPU memory usage if applicable (`torch.cuda.max_memory_allocated`).
    *   **Metric Calculation (using `metrics.py` helpers):**
        *   Calculate mAP for the primary IoU threshold (`metrics.iou_threshold_map`).
        *   Calculate mAP for the COCO IoU range (`metrics.iou_range_coco`).
        *   Calculate mAP_small, mAP_medium, mAP_large based on `metrics.object_size_definitions`. This requires access to ground truth box areas during evaluation.
        *   Generate confusion matrix data.
    *   Collects all metrics into a dictionary.
    -   Returns the metrics dictionary.
-   **`metrics.py`:**
    -   Contains functions for calculating specific metrics.
    -   Leverages `ultralytics.metrics` where possible, especially for standard mAP calculations.
    *   Implements logic for area-based mAP (mAP_s/m/l) if not directly available. This involves filtering detections/ground truths based on GT bounding box area before calling the mAP calculation for each category.
    *   Handles confusion matrix calculation (potentially using `ultralytics` or `sklearn`).
-   **`reporting.py`:**
    *   Contains `generate_html_report` function.
    *   Uses `jinja2` to render an HTML template.
    *   Generates plots (using `matplotlib`/`seaborn`) if `output.save_plots` is true:
        *   Confusion Matrix heatmap (per model).
        *   Bar charts comparing key mAP values across models.
        *   Box plots comparing inference time distributions across models.
    *   Generates qualitative result images (using `opencv-python` or `PIL` to draw boxes) if `output.save_qualitative_results` is true. Selects a few images and draws GT vs. predictions for each model.
    *   Embeds plots and qualitative images into the HTML report.
    *   Includes summary tables.
-   **`config.py`:**
    *   Defines Pydantic models mirroring the YAML structure for validation and type hinting.
-   **`utils.py`:**
    *   Helper functions (e.g., dataset loading, annotation parsing, path handling).

## 5. Dependencies

Ensure the following are installed in the `vbl` conda environment (add to `requirements.txt`):

-   `ultralytics>=8.0.0`
-   `torch>=2.0.0`
-   `torchvision>=0.15.0`
-   `pyyaml>=6.0`
-   `pandas>=1.0`
-   `numpy>=1.24.0`
-   `matplotlib>=3.7.0`
-   `seaborn>=0.12.0`
-   `jinja2>=3.0`
-   `opencv-python>=4.9.0`
-   `scikit-learn>=1.0.0` (for potential metric calculation/confusion matrix)
-   `tqdm` (for progress bars)
-   `pydantic` (for config validation)

*(Check `requirements.txt` for existing versions)*

## 6. Execution

1.  Activate the `conda` environment: `conda activate vbl`
2.  Run the benchmark:
    ```bash
    # Ensure config path is correct
    python src/models/ext/yolov8/benchmark/run_benchmark.py --config src/models/ext/yolov8/benchmark/detection_benchmark.yaml
    ```
3.  Results will be saved in the directory specified by `output.output_dir` in the config file.

## 7. Development Strategy

-   **Step 1: Setup & Config.** Create directory structure, `requirements.txt` updates (done), YAML config structure, Pydantic models (`config.py`), basic `run_benchmark.py` argument parsing and config loading. Add basic unit tests for config loading.
-   **Step 2: Data Loading.** Implement dataset loading (`utils.py`) based on config (image paths, annotations, subsetting). Add unit tests for data loading logic.
-   **Step 3: Core Inference Loop.** Implement `benchmark_single_model` skeleton focusing on model loading and the inference loop over images. Add basic timing (mean inference time). Add tests for model loading.
-   **Step 4: Basic Metrics.** Integrate standard mAP (@0.5) calculation, likely using `ultralytics.metrics`. Calculate percentile timings and model size. Add tests for metric calculation wrappers.
-   **Step 5: Advanced Metrics.** Implement mAP@0.5:0.95, area-based mAPs (s/m/l), and confusion matrix generation (`metrics.py`). Add tests.
-   **Step 6: Resource Metrics.** Add peak GPU memory measurement.
-   **Step 7: Reporting.** Implement CSV saving and the basic HTML report structure (`reporting.py` with `jinja2`).
-   **Step 8: Visualization.** Implement plot generation (matplotlib/seaborn) and qualitative result image generation (opencv). Integrate into HTML report. Add tests for reporting functions (checking file creation, perhaps basic content).
-   **Step 9: Refinement.** Add logging, error handling, documentation strings, and refine usability.

## 8. Future Enhancements

-   Support for other dataset formats (e.g., COCO JSON).
-   Support for other task types (segmentation, classification).
-   Parallel execution of model benchmarks (needs careful resource management).
-   Integration with experiment tracking tools (e.g., WandB).
-   More sophisticated analysis in the report.