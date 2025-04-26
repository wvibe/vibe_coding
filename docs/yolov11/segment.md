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

# YOLOv11 Segmentation Training

This document outlines the process and considerations for training YOLOv11 models specifically for segmentation tasks within the project.

## Overview

YOLOv11 segmentation models are trained using the `train_segment.py` script located in `src/models/ext/yolov11/`. This script leverages the Ultralytics library to handle the training process, supporting both finetuning from pretrained models and resuming interrupted training runs.

## Training Process

[Existing content about training process...]

## Limitations and Workarounds

### Resuming Training Limitations with Ultralytics YOLO

A known limitation with the Ultralytics YOLO library (as of version 8.3.117) is that resuming training from a checkpoint (`last.pt`) may fail if the library determines that the total number of epochs specified in the checkpoint's metadata has been reached. When this occurs, attempting to resume with `resume=True` results in an `AssertionError` indicating that training is already finished, even if additional epochs are desired.

- **Error Message**: `AssertionError: <path/to/last.pt> training to <N> epochs is finished, nothing to resume.`
- **Cause**: The Ultralytics trainer checks the `epochs` value embedded in the checkpoint file and compares it against the current epoch. If the current epoch is equal to or greater than the total epochs, it prevents resuming.
- **Impact**: This prevents extending training beyond the originally specified number of epochs when using the `--resume-with` argument in `train_segment.py`.

### Workaround for Additional Training

To perform additional training based on the last model output when resuming fails due to the epoch limit:

1. **Start a New Training Job with the Checkpoint**:
   Instead of using `--resume-with`, use the `--model` argument to specify the path to the `last.pt` checkpoint file. This starts a new training job from epoch 0, using the weights from the checkpoint as the starting point, without attempting to resume the optimizer state or other metadata.

   ```bash
   python -m src.models.ext.yolov11.train_segment --config configs/yolov11/finetune_segment_cov_segm.yaml --name new_finetune_run --model runs/finetune/segment_cov_segm/finetune_yolo11l-seg_260k_20250426_142406/weights/last.pt
   ```

   - **Effect**: This creates a new run with a fresh WandB trace (if enabled) and allows setting a new number of epochs or other training parameters via the configuration file.
   - **Trade-off**: Unlike a true resume, this approach does not preserve the optimizer state, learning rate scheduler, or current epoch from the checkpoint, which may affect training convergence or require additional epochs to stabilize.

2. **Adjust Configuration**: Ensure the configuration YAML file specifies the desired number of epochs and other parameters for the new training job.

This workaround is implemented in the updated `train_segment.py` script, which supports the `--model` argument to override the model specified in the configuration file and start a new training session.

## Usage Examples

[Existing content about usage examples...]