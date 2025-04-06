# YOLOv11 Segmentation Experiment Plan

This document outlines the plan for experimenting with YOLOv11 segmentation capabilities within the `vibe_coding` project.

## Goals

1.  Perform simple inference using pre-trained YOLOv11 segmentation models.
2.  Fine-tune a pre-trained YOLOv11 segmentation model on a target dataset.
3.  Train a YOLOv11 segmentation model from scratch.
4.  Benchmark and compare the performance of the original, fine-tuned, and retrained models.

## Prerequisites

*   Workspace: `/home/ubuntu/vibe/vibe_coding`
*   Environment: `conda activate vbl`
*   Dependency: `ultralytics` Python package installed (`pip install ultralytics`)
*   Review: Familiarity with `docs/yolov11/`, `src/models/ext/yolov11/` (as read-only reference), and Ultralytics documentation.

## Milestones

1.  **Milestone 1: Simple Segmentation Inference (Notebook)**
    *   **Goal:** Demonstrate basic inference with a pre-trained YOLOv11 segmentation model.
    *   **Artifact:** `notebooks/model/yolov11_segmentation_inference.ipynb`.
    *   **Implementation:** Use `ultralytics` library (`YOLO('yolo11n-seg.pt').predict(...)`).
    *   **Status:** Completed. Notebook includes loading VOC data, inference, and basic quantitative/visual comparison.
    *   **Documentation:** In-notebook explanations and results.

2.  **Milestone 2: Segmentation Fine-tuning (Script)**
    *   **Goal:** Fine-tune a pre-trained YOLOv11 segmentation model on a target dataset.
    *   **Artifact:** `scripts/yolov11_finetune_segment.py`.
    *   **Implementation:** Use `ultralytics` library (`model.train(...)` mode), loading a `.pt` model. Configure via args/YAML. Dataset likely from `hub/datasets/`.
    *   **Documentation:** Script docstrings/comments. Update project docs (`docs/yolov11/`) with usage and output locations (e.g., `hub/checkpoints/` or default `runs/`).

3.  **Milestone 3: Segmentation Retraining from Scratch (Script)**
    *   **Goal:** Train a YOLOv11 segmentation model from scratch.
    *   **Artifact:** `scripts/yolov11_retrain_segment.py`.
    *   **Implementation:** Use `ultralytics` library (`model.train(...)` mode), loading a `.yaml` configuration. Use the same dataset as Milestone 2.
    *   **Documentation:** Script docstrings/comments. Update project docs similar to Milestone 2.

4.  **Milestone 4: Benchmarking Models (Script)**
    *   **Goal:** Evaluate and compare the performance of the different segmentation models.
    *   **Artifact:** `scripts/yolov11_benchmark_segment.py`.
    *   **Implementation:** Use `ultralytics` library (`model.val()` mode). Evaluate pre-trained (M1), fine-tuned (M2), and retrained (M3) models on a consistent validation set. Output structured results.
    *   **Documentation:** Script docstrings/comments. Update `docs/yolov11/evaluate.md` or main README with usage and results interpretation.

5.  **Final Documentation Consolidation:**
    *   **Goal:** Ensure all experiments, scripts, and findings are well-documented.
    *   **Action:** Update `docs/yolov11/README.md` for a cohesive overview. Link artifacts. Summarize benchmark results and conclusions.

## Next Steps: Prediction Script

Following Milestone 1, the immediate next step is to develop a robust command-line script for segmentation prediction (`src/models/ext/yolov11/predict_segment.py`).

**Revised Plan:**
1.  **Configuration (`predict_segment.yaml`):** Define model/inference params (`model`, `project` root, `imgsz`, `conf`, etc.). Exclude `source` and `name`.
2.  **Command-Line Arguments (`predict_segment.py`):**
    *   Required: `--config`, `--dataset` (e.g., `voc`), `--tag` (e.g., `val2007`), `--name` (output run name).
    *   Optional: `--device`, `--save`, `--show` (limited overrides that take precedence over config if provided).
3.  **Script Logic (`predict_segment.py`):** Load config, construct `source_path` from `--dataset`/`--tag` and env vars, merge limited CLI args (checking for `None`), call prediction utility with params, calculate and log summary stats (total wall time, avg wall time/image, avg component times from Ultralytics speed dict).
4.  **Utility Functions (`predict_utils.py`):** Adapt existing functions (`_load_config`, `_merge_args`, `_validate_config`, `_prepare_output_dir`, `_run_yolo_prediction`) to align with the new argument structure and data flow (e.g., `name` instead of `name_prefix`, removing `_process_source_path`).
5.  **Documentation:** Update `docs/yolov11/README.md` with the new CLI usage and config structure for `predict_segment.py`. Update this doc (`segment.md`) to reflect the final implementation details.
6.  **Unit Tests:** Add unit tests for helper functions in `predict_utils.py` and `predict_segment.py` (excluding file operations and external library calls).