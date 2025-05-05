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

## Resuming Training Runs

The `train_segment.py` script provides two methods for resuming training:

1. **Automatic Resume (using `--auto-resume`):**
   - When the `--auto-resume` flag is enabled, the script will check for previous runs with the same base name (specified by `--name`) in the project directory.
   - If a matching run is found, the script will resume training from the last checkpoint, preserving the original run directory and WandB logging (if enabled).
   - This is useful for continuing training that was interrupted.
   - Usage example:
     ```bash
     python -m src.models.ext.yolov11.train_segment \
         --config configs/yolov11/finetune_segment_voc.yaml \
         --name voc11_seg_finetune_run1 \
         --auto-resume
     ```

2. **Manual Resume (using `--model`):**
   - Alternatively, you can explicitly specify a checkpoint to resume from using the `--model` parameter.
   - This approach starts a new training run but initializes weights from the specified checkpoint.
   - This is useful when you want more control over the resumption process or when extending training beyond the original epoch count.
   - Usage example:
     ```bash
     python -m src.models.ext.yolov11.train_segment \
         --config configs/yolov11/finetune_segment_voc.yaml \
         --name voc11_seg_finetune_run1_extended \
         --model runs/segment/cov_segm/voc11_seg_finetune_run1_YYYYMMDD_HHMMSS/weights/best.pt
     ```

Note that the `--auto-resume` flag is disabled by default, and it will be ignored if `--model` is specified.

## Limitations and Workarounds

#### Extending Training Runs

A potential limitation with the Ultralytics YOLO library is managing the total number of epochs when resuming. If a run was initially configured for `N` epochs and completed, simply resuming it might not allow training for *additional* epochs beyond `N` using the standard resume mechanism (which restores the exact state, including the epoch count).

**Workaround for Additional Training (using `--model`):**

To perform additional training based on the output of a previous run (whether completed or interrupted), the recommended approach is to use the `--model` argument with the new `train_segment.py` script:

1.  **Identify the Checkpoint**: Locate the desired checkpoint file from the previous run (e.g., `runs/train/segment/<previous_run_name>/weights/best.pt` or `last.pt`).
2.  **Start a New Training Job**: Initiate a new training job using the `--model` argument pointing to that checkpoint. Assign a *new base name* using `--name` to avoid conflicts with the auto-resume logic for the *original* name.

    ```bash
    # Example: Continue training based on the best weights of a previous run
    python -m src.models.ext.yolov11.train_segment \
        --config configs/yolov11/finetune_segment_voc.yaml \
        --name voc11_seg_finetune_run1_extended \
        --model runs/train/segment/voc11_seg_finetune_run1_YYYYMMDD_HHMMSS/weights/best.pt
    ```

    - **Effect**: This starts a new training run from epoch 0, initializing weights from the specified checkpoint. It creates a new run directory (e.g., `voc11_seg_finetune_run1_extended_TIMESTAMP`) and a distinct WandB trace (if enabled).
    - **Configuration**: You can adjust the total `epochs` and other parameters in your YAML configuration file for this new phase of training.
    - **Benefit**: This cleanly separates the extended training phase and avoids potential issues with the internal epoch tracking of the Ultralytics resume state.

This approach leverages the flexibility of starting new runs from specific weights, providing a clear way to continue or extend training.

## Usage Examples

[Existing content about usage examples...]

# YOLOv11 Segmentation Prediction

This document describes how to use the `predict_segment.py` script for running inference with YOLOv11 segmentation models.

## Overview

The `vibelab.models.ext.yolov11.predict_segment` script provides a command-line interface to load a YOLOv11 segmentation model and run predictions on various sources (single image, directory, or standard datasets like VOC/COCO).

## Usage

Run the script from the project root using `python -m`:

```bash
python -m vibelab.models.ext.yolov11.predict_segment --config <path_to_config.yaml> [options]
```

## Configuration File (`predict_segment.yaml`)

The script relies on a YAML configuration file to specify most parameters. See `configs/yolov11/predict_segment_demo.yaml` for an example.

## Implementation Details

- The script utilizes `vibelab.models.ext.yolov11.predict_utils` for shared logic like config loading, output directory setup, and source processing.
- It loads the specified model using `ultralytics.YOLO(config['model'])`.
- Inference is performed via `model.predict(...)`, passing relevant parameters from the config (e.g., `conf`, `iou`, `device`, `save`).
- Performance statistics (FPS, timing) are calculated based on the `results` object returned by `predict` and logged.