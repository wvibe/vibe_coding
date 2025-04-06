# YOLOv11 Design Notes

## Core Library

- This module relies heavily on the `ultralytics` Python package.
- We will leverage its `YOLO` class for loading models, running inference (`predict`), and training/finetuning (`train`).

## Initial Scope: Detection

- **Configuration:**
  - Dataset configurations (e.g., `voc_detect.yaml`) follow the Ultralytics format for training/finetuning.
  - Prediction uses separate YAML configs (e.g., `predict_detect.yaml`) specifying model, output project dir, and Ultralytics prediction args (`conf`, `imgsz`, `save_txt`, `save`, etc.).
- **Prediction Scripts:** The scripts (`predict_detect.py`, `predict_segment.py`) provide a command-line interface to run pre-trained YOLOv11 models.
  - Takes `--config <yaml_path>`, `--dataset <dataset_id>`, `--tag <split_tag>`, and `--name <run_name>` as input.
  - Reads parameters from the specified YAML config.
  - Constructs the source path from dataset and tag (`${DATASET_BASE_PATH}/images/{tag}`).
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

- **Segmentation:** Similar scripts (`predict_segment.py`, `train_segment.py`) and configurations (`*_seg.yaml`, `voc_segment.yaml`, `voc_segment_finetune.yaml`) are being added. Evaluation script (`evaluate_segment.py`) and metrics are future work.
- **Training:** Scripts for training from scratch (`train_*.py`) might be added if needed.
- **Datasets:** Support for COCO will be added via `coco_*.yaml` configurations.
- **Tests:** Unit tests will cover basic script execution and argument parsing.