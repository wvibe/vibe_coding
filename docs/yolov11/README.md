# YOLOv11 Experiments

This directory contains code and documentation related to experiments with YOLOv11 models using the Ultralytics library.

## Scope

- Initial focus: Object Detection (Inference, Finetuning) using VOC dataset.
- Future scope: Segmentation, COCO dataset integration, potential full training.

## Core Dependency

- [Ultralytics](https://github.com/ultralytics/ultralytics)
- Official YOLOv11 Docs: [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)

## Source Code

- Main scripts: [`src/vibelab/models/ext/yolov11/`](../../src/vibelab/models/ext/yolov11/)
- Configurations: [`configs/yolov11/`](../../configs/yolov11/)
- Tests:
  - Prediction: `tests/models/ext/yolov11/test_predict.py`

## Usage

### Prediction (YOLOv11)

The `predict_yolo.py` script runs inference using a YOLOv11 model (detection or segmentation) based on parameters defined in a YAML configuration file. It supports command-line arguments for dataset selection, task type, and limited overrides.

**Configuration (`configs/yolov11/predict_detect.yaml` or `configs/yolov11/predict_segment.yaml`):**

- `model`: Path to the model (`.pt`) or Ultralytics model name (e.g., `yolo11n.pt` for detection, `yolo11n-seg.pt` for segmentation).
- `project`: Base directory to save output runs (e.g., `runs/predict/detect` or `runs/segment`).
- Other optional keys correspond to `ultralytics.YOLO.predict` arguments (e.g., `conf`, `imgsz`, `save`, `save_txt`, `save_conf`, `device`).

**Command-Line:**

```bash
python -m vibelab.models.ext.yolov11.predict_yolo \
    --config configs/yolov11/predict_<task>.yaml \
    --dataset <dataset_src> \
    --task <detect|segment> \
    --tag <split_tag> \
    --name <your_run_name> \
    [--device <device_id>] \
    [--save <True|False>] \
    [--show <True|False>] \
    [--sample_count <N>]
```

**Arguments:**

- `--config`: Path to the YAML configuration file. *Required.*
- `--dataset`: Specifies the dataset source. Can be either:
  - A direct path to the dataset base directory (e.g., `/path/to/voc/detect`)
  - The name of an environment variable containing the base path (e.g., `VOC_DETECT`, `VOC_SEGMENT`). *Required.*
- `--task`: Specifies the prediction task type (`detect` or `segment`). *Required.*
- `--tag`: The specific split/tag for the dataset (e.g., `val2007`, `test2007`). The script looks for images in `${DATASET_BASE_PATH}/images/{tag}`. *Required.*
- `--name`: The name for this specific prediction run. This will be the name of the output directory created under the `project` specified in the config file. *Required.*
- `--device` (optional): Override the compute device specified in the config file. If omitted, the config value is used.
- `--save` (optional): Override whether to save annotated images (True/False). If omitted, the config value is used.
- `--show` (optional): Override whether to display results in a window (True/False). If omitted, the config value is used.
- `--sample_count` (optional): Randomly sample `N` images from the source directory. Processes all if omitted.

**Example (Detection on VOC test2007):**

```bash
python -m vibelab.models.ext.yolov11.predict_yolo \
    --config configs/yolov11/predict_detect.yaml \
    --dataset VOC_DETECT \
    --task detect \
    --tag test2007 \
    --name voc_test2007_detect_run1
```

**Example (Segmentation on VOC val2007):**

```bash
python -m vibelab.models.ext.yolov11.predict_yolo \
    --config configs/yolov11/predict_segment.yaml \
    --dataset VOC_SEGMENT \
    --task segment \
    --tag val2007 \
    --name voc_val2007_segment_run1
```

**Example (Using specific device and random sampling):**

```bash
python -m vibelab.models.ext.yolov11.predict_yolo \
    --config configs/yolov11/predict_detect.yaml \
    --dataset VOC_DETECT \
    --task detect \
    --tag val2007 \
    --name voc_val2007_detect_run2 \
    --device cpu \
    --save True \
    --sample_count 10
```

**Example (Using a direct path to dataset):**

```bash
python -m vibelab.models.ext.yolov11.predict_yolo \
    --config configs/yolov11/predict_segment.yaml \
    --dataset /path/to/voc/segment \
    --task segment \
    --tag val2007 \
    --name voc_val2007_segment_run2
```

Results are saved to `<config.project>/<name>_<timestamp>/`.

**Output:**

- The console output includes logs of the configuration, process, and prediction time statistics (total time, average time per image (wall clock), FPS (wall clock), and average preprocess/inference/postprocess times reported by Ultralytics).
- Annotated images are saved if `save: True` (either in config or via `--save True`).

### Training (Detection)

The `train_detect.py` script initiates training (finetuning or from scratch) using a YOLOv11 model. It requires a main training configuration YAML. It automatically appends a timestamp (`_YYYYMMDD_HHMMSS`) to the provided run name for new runs and supports resuming specific previous runs.

**Configuration (`src/models/configs/training/<config_name>.yaml`):**

- `model`: Path to the base model weights (`.pt`) or architecture YAML (`.yaml`). *Required.*
- `data`: Path (relative to project root) to the *dataset-specific* configuration YAML (e.g., `src/models/configs/datasets/voc_detect.yaml`). This file should contain paths for `train`, `val`, `test` (optional), along with class `names`. *Required.*
- `project`: Base directory to save training runs (e.g., `runs/train/detect`). *Optional, defaults defined in script.*
- `pretrained`: Boolean (`True` for finetuning, `False` for scratch). Relevant only for *new* runs.
- Other keys correspond to `ultralytics.YOLO.train` arguments (e.g., `epochs`, `batch`, `imgsz`, `optimizer`, `lr0`, `device`, etc.).

**Command-Line:**

```bash
# Start a new training run (timestamp will be appended to name)
python src/models/ext/yolov11/train_detect.py \\
    --config <path_to_main_training_config.yaml> \\
    --name <your_base_run_name> \\
    [--project <output_project_dir>] \\
    [--wandb-dir <path_to_wandb_root>] # Optional, default: 'wandb'

# Resume a specific previous run
python src/models/ext/yolov11/train_detect.py \\
    --config <path_to_original_main_training_config.yaml> \\
    --resume_with <path/to/exact/run_folder_with_timestamp> \\
    --name <base_run_name> # Still required but overridden by resume_with
    [--wandb-dir <path_to_wandb_root>] # Optional, default: 'wandb'
```

**Arguments:**

- `--config`: Path to the main training configuration YAML file. This file must contain a `data` key pointing to the dataset-specific config file (e.g., `src/models/configs/datasets/voc_detect.yaml`). *Required.*
- `--project` (optional): Override the base project directory specified in the config file (or script default).
- `--name`: Base name for the training run. A timestamp (`_YYYYMMDD_HHMMSS`) will be automatically appended to create the actual run directory for *new* runs. For *resume* runs, this is still required syntactically but the actual run name is determined by the `--resume_with` path.
- `--resume_with` (optional): Path to the *exact* run directory (e.g., `runs/train/detect/my_run_20231027_103000`) to resume training *checkpoint* from. If provided, the `--name` argument's value is effectively ignored for naming, but still required. The script will look for `weights/last.pt` within this directory.
- `--wandb-dir` (optional): Path to the root directory containing WandB run folders (e.g., `./wandb`). Defaults to `wandb`. Used to automatically find the corresponding WandB run ID when resuming.

**WandB Integration:**

- The script integrates with WandB if it's enabled (`yolo settings wandb=True`).
- When resuming (`--resume_with`), the script automatically searches the `--wandb-dir` for a WandB run whose configuration matches the `name` of the directory specified in `--resume_with`.
- If a match is found, the script sets the `WANDB_RUN_ID` environment variable, allowing Ultralytics to resume logging to the correct existing WandB run.
- If no match is found, a warning is logged, and a new WandB run will be created if WandB is enabled.

**Example (Finetuning):**

```bash
# Start a new finetuning run
python src/models/ext/yolov11/train_detect.py \\
    --config src/models/configs/training/voc11_finetune_config.yaml \\
    --name voc11l_finetune_run1
# Output directory might be runs/train/detect/voc11l_finetune_run1_YYYYMMDD_HHMMSS/
```

**Example (Resuming a Specific Finetuning Run):**

```bash
# Assume the previous run was saved to runs/train/detect/voc11l_finetune_run1_20240801_120000/
python src/models/ext/yolov11/train_detect.py \\
    --config src/models/configs/training/voc11_finetune_config.yaml \\
    --resume_with runs/train/detect/voc11l_finetune_run1_20240801_120000 \\
    --name voc11l_finetune_run1 # Provide the base name
    # The script will automatically search for the matching WandB ID in ./wandb
    # If WandB logs are elsewhere, use --wandb-dir path/to/wandb/root
```

Training progress and results (checkpoints, metrics, logs) are saved to `<project>/<name_with_timestamp>/`.

### Training (Segmentation)

The `train_segment.py` script initiates training (finetuning or from scratch) for YOLOv11 segmentation models using parameters from a YAML configuration file. It now mirrors the detection training script structure, including argument handling and WandB integration.

**Configuration (`src/models/configs/training/finetune_segment_*.yaml`):**
- `model`: Path to the base segmentation model weights (`.pt`) or architecture YAML (`.yaml`) (e.g., `yolo11l-seg.pt`). *Required if `--model` is not provided via CLI for new runs.*
- `data`: Path (relative to project root) to the *dataset-specific* segmentation configuration YAML (e.g., `configs/yolov11/voc_segment.yaml`). This file should contain paths for `train`, `val`, `test` (optional), along with class `names`. *Required.*
- `project`: Base directory for saving runs (e.g., `runs/train/segment`). *Required.*
- `auto_resume` (optional): Boolean (`True` or `False`) to enable/disable automatic resuming of runs with the same base `--name` within the `project` directory. Defaults to `True`. Auto-resume is disabled if `--model` is provided.
- `pretrained`: Boolean (`True` for finetuning, `False` for scratch). Relevant only for *new* runs when loading weights via the `model` key here.
- Other keys correspond to `ultralytics.YOLO.train` arguments suitable for segmentation.

**Command-Line:**

```bash
# Start a new training run (or auto-resume if enabled and possible)
python src/models/ext/yolov11/train_segment.py \
    --config <path_to_main_training_config.yaml> \
    --name <your_base_run_name> \
    [--model <path_to_weights.pt>] \
    [--wandb-dir <path_to_wandb_root>] # Optional, default: 'wandb'
```

**Arguments:**

- `--config`: Path to the main training configuration YAML file (e.g., `configs/yolov11/finetune_segment_voc.yaml`). Must contain `project` and `data` keys. *Required.*
- `--name`: Base name for the training run (e.g., `voc11_seg_finetune_run1`). A timestamp is appended for new runs. This name is used with the `project` path to find previous runs for auto-resuming. *Required.*
- `--model` (optional): Path to a model file (`.pt`) to start a *new* training job from. This overrides the `model` key in the configuration file and *disables* the `auto_resume` feature for this specific invocation.
- `--wandb-dir` (optional): Path to the root directory containing WandB run folders (e.g., `./wandb`). Defaults to `wandb`. Relevant for organizing WandB logs; resume linking relies on checkpoint metadata.

**Auto-Resuming Behavior:**

- If `auto_resume: True` (default or set in config) and `--model` is *not* provided:
    1. The script checks for `<config.project>/last_run.log`.
    2. If the log exists and matches the current `project` and `--name`, it attempts to load the checkpoint path recorded (`last_ckpt_path`).
    3. If the checkpoint exists, training resumes from that state within the *original run directory* (e.g., `<project>/<name>_YYYYMMDD_HHMMSS`).
    4. If the log doesn't exist, doesn't match, or the checkpoint is missing, a *new run* is started with a timestamp appended to `--name`.
- If `auto_resume: False` or `--model` is provided, a *new run* is always started.

**Tracking File (`last_run.log`):**

- A file named `last_run.log` is created/updated in the `project` directory at the start of each training run (by Rank 0).
- It contains details of the *most recently started* run for a given project/name combination, including the actual run directory name and the full path to the `last.pt` checkpoint file.
- This file enables the auto-resume mechanism.

**WandB Integration:**

- The script integrates with WandB if it's enabled (`yolo settings wandb=True`).
- When resuming (via `auto_resume` finding a valid run), the underlying Ultralytics trainer automatically attempts to resume the corresponding WandB run using the ID stored in the checkpoint's metadata.
- The `--wandb-dir` argument helps organize log directories but doesn't directly influence the resume linking.

**Example (Fine-tune - New or Auto-Resume):**

```bash
# Start a new finetuning run OR resume the latest run named 'voc11_seg_finetune_run1'
# Behavior depends on 'auto_resume' in config and existence/content of last_run.log
python src/models/ext/yolov11/train_segment.py \
    --config configs/yolov11/finetune_segment_voc.yaml \
    --name voc11_seg_finetune_run1
```

**Example (Start New Run with Specific Weights):**

```bash
# Force a new run, starting from specific weights (disables auto-resume)
python src/models/ext/yolov11/train_segment.py \
    --config configs/yolov11/finetune_segment_voc.yaml \
    --name voc11_seg_finetune_run2_from_run1 \
    --model runs/train/segment/voc11_seg_finetune_run1_YYYYMMDD_HHMMSS/weights/best.pt
```

Training progress and results (checkpoints, metrics, logs) are saved to `<project>/<actual_run_name>/` (where `actual_run_name` includes the timestamp for new runs or is the resumed directory name).

### Evaluation (Detection)

The `evaluate_detect.py` script provides comprehensive evaluation of YOLOv11 detection models using custom metrics and visualization tools, integrating utilities for metric calculation and performance measurement.

**Configuration (`src/models/ext/yolov11/configs/evaluate_*.yaml`):**

This file configures the model path, evaluation dataset details (images, labels, class names), inference parameters (image size, confidence/IoU thresholds), metric settings, and output options. A key option is `save_results: True|False` which controls whether to save annotated images and prediction details in YOLO `.txt` format for each evaluated image into an `individual_results` subfolder.

For detailed configuration options, refer to the specific comments within `evaluate_default.yaml` or the dedicated documentation in [`evaluate.md`](./evaluate.md).

**Command-Line:**

```bash
python -m vibelab.models.ext.yolov11.evaluate_detect --config <path_to_config.yaml>
```