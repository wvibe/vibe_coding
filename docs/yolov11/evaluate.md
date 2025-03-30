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
  save_json: True                  # Save results to JSON
  save_txt: True                   # Save results as text files
  save_conf: True                  # Include confidence in text files
  save_metrics: True               # Save metrics as CSV/JSON
  plot_confusion_matrix: True      # Generate confusion matrix plot
  plot_precision_recall: True      # Generate P-R curve plots
```

A default configuration is provided at `src/models/ext/yolov11/configs/evaluate_default.yaml`.

## Usage

To run the evaluation script:

```bash
python -m src.models.ext.yolov11.evaluate_detect --config path/to/config.yaml
```

## Implementation Details

The evaluation script follows this process:

1. **Configuration Loading**: Loads and validates the configuration YAML file
2. **Output Directory Setup**: Creates a timestamped output directory
3. **Model Loading**: Loads the model and counts parameters with `get_model_params`
4. **Inference**: Runs predictions on all evaluation images and measures performance:
   - Performs warmup iterations
   - Measures inference time
   - Tracks peak GPU memory with `get_peak_gpu_memory_mb`
5. **Ground Truth Processing**: Loads and processes ground truth annotations
6. **Metric Calculation**:
   - Matches predictions to ground truth using `match_predictions`
   - Calculates precision-recall data with `calculate_pr_data`
   - Calculates mAP across IoU thresholds with `calculate_map`
   - Calculates mAP by object size with `calculate_map_by_size`
   - Generates confusion matrix with `generate_confusion_matrix`
7. **Result Reporting**: Outputs metrics to console and saves to files
8. **Visualization**: Generates precision-recall curves and confusion matrix plots

## Metrics

The script calculates the following metrics:

- **mAP (mean Average Precision)**: Averaged across classes and IoU thresholds
- **mAP@0.5**: AP at IoU threshold of 0.5
- **mAP@0.5:0.95**: AP averaged over IoU thresholds from 0.5 to 0.95
- **mAP by Size**: AP for small, medium, and large objects
- **Confusion Matrix**: True positives, false positives, and false negatives by class
- **Computational Metrics**:
  - Model parameter count
  - Inference time per image
  - Peak GPU memory usage

## Output

The script creates the following output files in the specified output directory:

- **config.yaml**: Copy of the evaluation configuration
- **metrics.json**: Complete metrics results
- **predictions.json**: Detection results in JSON format
- **confusion_matrix.png**: Confusion matrix visualization
- **pr_curves.png**: Precision-recall curves
- **inference_stats.txt**: Inference time and memory statistics

## Future Extensions

- Integration with benchmarking to compare multiple models
- Support for custom metrics and visualizations
- Integration with external evaluation frameworks (COCO, etc.)