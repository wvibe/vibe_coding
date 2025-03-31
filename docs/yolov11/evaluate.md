# YOLOv11 Evaluation

This document describes the evaluation functionality for YOLOv11 detection models implemented in `evaluate_detect.py`.

## Overview

The evaluation script provides a comprehensive framework for assessing YOLOv11 model performance on object detection tasks. It leverages the metric utilities developed in `src/utils/metrics/detection.py` and `src/utils/metrics/compute.py` to calculate standard detection metrics like mAP and provide computational insights.

## Configuration

The evaluation script is driven by a YAML configuration file with the following structure:

```yaml
# --- Model ---
model: "yolo11n.pt" # REQUIRED: Model path or name

# --- Dataset ---
dataset:
  image_dir: "/path/to/images"  # REQUIRED: Path to evaluation images
  label_dir: "/path/to/labels"  # REQUIRED: Path to corresponding YOLO format labels
  class_names:                  # REQUIRED: Class names list
    - class1
    - class2
    # ...

# --- Evaluation Parameters ---
evaluation_params:
  imgsz: 640             # Image size for inference
  batch_size: 16         # Batch size for model.predict
  device: 0              # Compute device
  iou_thres_nms: 0.65    # IoU threshold for NMS
  conf_thres: 0.001      # Confidence threshold
  max_det: 300           # Maximum detections per image
  warmup_iterations: 5   # Warmup iterations before timing

# --- Metrics Configuration ---
metrics:
  iou_thresholds: [0.5, 0.55, ..., 0.95]  # IoU thresholds for mAP

  confidence_threshold_cm: 0.3 # Confidence threshold for confusion matrix
  iou_threshold_cm: 0.5        # IoU threshold for confusion matrix
  target_classes_cm:           # Classes for confusion matrix
    - class1
    - class2
    # ...

  size_ranges:                 # Area ranges for mAP@Size
    small: [0, 1024]           # area < 32*32
    medium: [1024, 9216]       # 32*32 <= area < 96*96
    large: [9216, null]        # area >= 96*96

# --- Computation Measurement ---
computation:
  measure_inference_time: True  # Measure inference time
  measure_memory: True          # Measure peak GPU memory usage

# --- Output Control ---
output:
  project: "runs/evaluate/detect"  # Base directory for output
  name: null                       # Run name (defaults to model+timestamp)
  save_results: False              # Save annotated images and YOLO format txt for each image (in `individual_results` subdir)
  # Ultralytics predict() save flags (NOT directly used by this script's main logic, but may affect internal predict behavior if passed):
  # save_json: True                  # (UL flag) Save results to JSON - Handled by our script's metrics saving
  # save_txt: True                   # (UL flag) Save results as text files - Handled by `save_results` if needed
  # save_conf: True                  # (UL flag) Include confidence in text files - Handled by `save_results` if needed
  # Metrics/Plot Saving (Controlled by our script):
  save_metrics_json: True          # Save final computed metrics to metrics.json
  plot_confusion_matrix: True      # Generate confusion matrix plot
  plot_precision_recall: True      # Generate P-R curve plots
```