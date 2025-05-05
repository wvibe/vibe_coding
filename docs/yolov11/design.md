# YOLOv11 Integration Design

This document outlines the design choices for integrating YOLOv11 models into the `vibelab` project, focusing on training, prediction, and evaluation workflows.

## Core Principles

- **Leverage Ultralytics:** Utilize the `ultralytics` library for core model loading, training (`model.train()`), and prediction (`model.predict()`).
- **Configuration Driven:** Use YAML files for managing parameters for training, prediction, and evaluation to ensure reproducibility and ease of modification.
- **Clear Separation:** Maintain separate scripts for distinct tasks (training, prediction, evaluation), although training is now unified.
- **Standardized Output:** Define consistent output directory structures for runs.
- **Namespace:** All code resides under the `vibelab.models.ext.yolov11` namespace.

## Key Components

1.  **Unified Training Script (`train_yolo.py`)**
    -   **Purpose:** Handles both detection and segmentation model training/fine-tuning.
    -   **Inputs:** `--config` (path to main training YAML), `--name` (base run name), optional `--resume-with` (path to previous run dir), optional `--model` (override model in config).
    -   **Configuration (`<task>_config.yaml`):** Contains hyperparameters (`epochs`, `batch`, `lr0`, etc.), model path (`model`), data configuration path (`data`), output project directory (`project`).
    -   **Data Configuration (`<dataset>_<task>.yaml`):** Standard Ultralytics data YAML specifying `path`, `train`, `val`, `test` splits, and `names`.
    -   **Logic:**
        -   Parses args and loads main config.
        -   Validates data config path.
        -   Determines run parameters (model path, run name, resume status, WandB ID) based on args and potential resume path.
        -   Sets up WandB environment variables if resuming with a known ID.
        -   Loads the YOLO model (using `YOLO(model_path)`).
        -   Prepares keyword arguments for `model.train()` by filtering the main config.
        -   Calls `model.train(**kwargs)`.
        -   Logs run information for potential future auto-resuming.
    -   **Output:** Standard Ultralytics run directory structure under `<config.project>/<run_name>_<timestamp>`.

2.  **Unified Prediction Script (`predict_yolo.py`)**
    -   **Purpose:** Run inference using a trained YOLOv11 model for detection or segmentation tasks.
    -   **Inputs:** `--config` (path to prediction YAML), `--dataset` (source path or env var name), `--task` (`detect` or `segment`), `--tag`, `--name`, optional `--device`, `--save`, `--show`, `--sample_count`.
    -   **Configuration (`predict_<task>.yaml`):** Specifies `model`, `project`, and YOLO prediction parameters (e.g., `conf`, `imgsz`, `save`, `save_txt`).
    -   **Logic:**
        -   Load config and merge CLI overrides (`device`, `save`, `show`).
        -   Resolve the source path (either directly from a provided path or via an environment variable).
        -   Process source directory (list images, optionally sample).
        -   Prepare output directory (`<config.project>/<name>_<timestamp>`).
        -   Call `_run_yolo_prediction` with appropriate parameters, passing the `task_type` from the CLI argument.
        -   Calculate and log performance stats (FPS, component times).
    -   **Output:** Predictions saved (if `save=True`) in the output directory.

3.  **Evaluation Script (`evaluate_detect.py`)**
    -   **Purpose:** Evaluate a trained detection model against a ground truth dataset.
    -   **Inputs:** `--config` (path to evaluation YAML).
    -   **Configuration (`evaluate_detect.yaml`):** Specifies `model`, dataset details (`image_dir`, `label_dir`, `class_names`), evaluation parameters (`conf_thres`, `iou_thres`, `device`), metric parameters (`map_iou_threshold`, `conf_threshold_cm`, etc.), output settings (`project`, `name`).
    -   **Logic:**
        -   Load config.
        -   Setup output directory.
        -   Load model and count parameters.
        -   Run inference on the dataset images, measure time/memory.
        -   Load ground truth labels.
        -   Calculate detection metrics (mAP, AP per class, confusion matrix) using `vibelab.utils.metrics.detection`.
        -   Save results (JSON, summary text) and generate plots (PR curves, confusion matrix).
    -   **Output:** Evaluation results, plots, and summary saved in the output directory.

4.  **Utilities (`predict_utils.py`, `evaluate_utils.py`, `vibelab.utils.metrics.*`)**
    -   Contain helper functions for common tasks like path construction, config merging, statistics calculation, metric computations, output directory setup, ground truth loading, result saving, etc., promoting DRY principles.

## Initial Scope: Detection

- **Configuration:**
  - Dataset configurations (e.g., `voc_detect.yaml`) follow the Ultralytics format for training/finetuning.
  - Prediction uses separate YAML configs (e.g., `predict_detect.yaml`) specifying model, output project dir, and Ultralytics prediction args (`conf`, `imgsz`, `save_txt`, `save`, etc.).
- **Prediction Script:** The unified `predict_yolo.py` script provides a command-line interface to run pre-trained YOLOv11 models for either detection or segmentation tasks.
  - Takes `--config <yaml_path>`, `--dataset <env_var_or_path>`, `--task <detect|segment>`, `--tag <split_tag>`, and `--name <run_name>` as input.
  - Reads parameters from the specified YAML config.
  - Resolves the source path either directly from the provided path or via an environment variable, then appends `/images/{tag}`.
  - Supports random selection of N images if `--sample_count N` is specified.
  - Saves results to `<config.project>/<name>_<timestamp>/`.
  - Calculates and reports prediction time statistics (wall clock time, FPS, component times).
  - Leverages `ultralytics.YOLO.predict` for core logic and output generation (annotated images if `save: True`, YOLO labels if `save_txt: True`).
- **Training/Finetuning:** A script (`train_detect.py`) allows finetuning pre-trained models or training from scratch on custom datasets (like VOC).
  - Takes `--config <yaml_path>` and `--name <run_name>` as input.
  - Uses dataset configurations (`voc_detect.yaml`) and training configurations (`voc_finetune.yaml`, `voc_retrain.yaml`).
  - The training config determines whether it's finetuning (`pretrained: True`, `model: *.pt`) or training from scratch (`pretrained: False`, `model: *.yaml`).
  - Leverages `ultralytics.YOLO.train` for the core training logic.
- **Metrics Calculation:** Functionality will be added to compute detailed performance metrics.
  - Will leverage/extend `ultralytics` validation metrics where possible (e.g., standard mAP@0.5, mAP@0.5:0.95).
  - Custom calculations for mAP based on object size (Small/Medium/Large) will be implemented, configurable via YAML.
  - Confusion matrix generation for a specified subset of classes (defined in config), grouping others into an 'others' category.
  - Capture computational metrics: inference time per image, total parameters, peak GPU memory usage.
  - **Evaluation Script (`evaluate_detect.py`):** A dedicated evaluation script implements comprehensive metrics evaluation:
    - Uses a separate YAML configuration file (`evaluate_default.yaml`) with nested sections for model, dataset, evaluation parameters, metrics, computation, and output settings.
    - Leverages custom metric utilities (`src/utils/metrics/detection.py`) for IoU calculation, predictions matching, precision-recall calculations, AP/mAP, mAP by size, and confusion matrix generation.
    - Uses computational utilities (`src/utils/metrics/compute.py`) to measure model parameters and GPU memory usage.
    - Generates visualizations including precision-recall curves and confusion matrices.
    - Outputs metrics in various formats (JSON, CSV, text summary).
  - Unit tests verify the correctness of custom metric calculations.
- **Benchmarking & Reporting:** Tools for comparing model performance and generating reports.
  - Compare metrics from predictions against ground truth dataset labels.
  - Compare metrics across different models or training runs.
  - Generate reports summarizing metrics in plain text format.
  - Generate enhanced HTML reports including visualizations (e.g., PR curves, confusion matrices).
  - A dedicated script or CLI interface will orchestrate benchmarking runs and report generation.

## Future Scope: Segmentation & Expansion

- **Segmentation:** Similar scripts (`predict_segment.py`, `train_segment.py`) and configurations (`*_seg.yaml`, `voc_segment.yaml`, `finetune_segment_voc.yaml`) exist. The training script (`train_segment.py`) mirrors the detection script's functionality, with added features like `--auto-resume` for resuming interrupted training runs. It includes proper tracking of training runs to facilitate resumption and integrates with WandB for experiment logging. Evaluation script (`evaluate_segment.py`) and metrics are future work.
- **Training:** Scripts for training from scratch (`train_*.py`) might be added if needed.
- **Datasets:** Support for COCO will be added via `coco_*.yaml` configurations.
- **Tests:** Unit tests cover basic script execution and argument parsing for prediction and training scripts.