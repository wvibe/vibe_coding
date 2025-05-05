# YOLOv11 Detection Evaluation

This document explains how to use the `evaluate_detect.py` script to evaluate the performance of YOLOv11 detection models.

## Overview

The `vibelab.models.ext.yolov11.evaluate_detect` script takes a configuration file specifying a trained model and a dataset, runs inference on the dataset, compares the predictions against ground truth labels, and calculates standard detection metrics (mAP, AP per class, confusion matrix).

## Usage

Execute the script from the project root using `python -m`:

```bash
python -m vibelab.models.ext.yolov11.evaluate_detect --config <path_to_evaluation_config.yaml>
```

## Configuration File (`evaluate_detect.yaml`)

A YAML configuration file controls the evaluation process. See `configs/yolov11/evaluate_detect.yaml` for a detailed example.

## Implementation Details

1.  **Setup:** Loads configuration, sets up the output directory using helpers from `evaluate_utils.py`.
2.  **Model Loading:** Loads the specified YOLO model using `ultralytics.YOLO`.
3.  **Parameter Counting:** Uses `vibelab.utils.metrics.compute.get_model_params` to count trainable parameters.
4.  **Inference:** Runs `model.predict()` on all images specified in the dataset configuration. Includes warmup runs. Measures wall time and attempts to get peak GPU memory using `vibelab.utils.metrics.compute.get_peak_gpu_memory_mb`.
5.  **Ground Truth Loading:** Uses `evaluate_utils.load_ground_truth` to read label files (`.txt` in YOLO format) and convert them to the required format (list of dictionaries with `box` and `class_id`).
6.  **Metric Calculation:**
    -   Calls `vibelab.utils.metrics.detection.calculate_all_metrics`, which orchestrates:
        -   Matching predictions to ground truths for various IoU thresholds (`match_predictions`).
        -   Calculating precision-recall data (`calculate_pr_data`).
        -   Calculating Average Precision (AP) per class and mean AP (mAP) at the specified IoU threshold (`calculate_ap`, `calculate_map`).
        -   Calculating mAP@0.5:0.95 (COCO standard) using `_calculate_map_coco`.
        -   Generating a confusion matrix (`generate_confusion_matrix`).
7.  **Result Saving:** Uses `evaluate_utils.save_evaluation_results` to:
    -   Generate a text summary (`_generate_text_summary`).
    -   Save computed metrics and configuration to `evaluation_results.json`.
    -   Save the text summary to `summary.txt`.
    -   Generate and save plots for PR curves (`_plot_pr_curve`) and the confusion matrix (`_plot_confusion_matrix`) using matplotlib.