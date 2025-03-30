# YOLOv11 Design Notes

## Core Library

- This module relies heavily on the `ultralytics` Python package.
- We will leverage its `YOLO` class for loading models, running inference (`predict`), and training/finetuning (`train`).

## Initial Scope: Detection

- **Configuration:**
  - Dataset configurations (e.g., `voc_detect.yaml`) follow the Ultralytics format for training/finetuning.
  - Prediction uses separate YAML configs (e.g., `predict_demo.yaml`) specifying model, source, output project dir, Ultralytics prediction args (`conf`, `imgsz`, `save_txt`, `save`, etc.), and custom args (`random_select`).
- **Prediction (New Script):** A script (`predict_detect.py`) provides a command-line interface to run pre-trained YOLOv11 detection models.
  - Takes `--config <yaml_path>` and `--name-prefix <prefix>` as input.
  - Reads parameters from the specified YAML config.
  - Handles single image files or directories as `source`.
  - Supports random selection of N images if `source` is a directory and `random_select: N` is set in config.
  - Saves results to `<config.project>/<name-prefix>_<timestamp>/`.
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

- **Segmentation:** Similar scripts (`inference_seg.py`, `train_seg.py`) and configurations (`*_seg.yaml`) will be added once segmentation datasets are available.
- **Training:** Scripts for training from scratch (`train_*.py`) might be added if needed.
- **Datasets:** Support for COCO will be added via `coco_*.yaml` configurations.
- **Tests:** Unit tests will cover basic script execution and argument parsing.