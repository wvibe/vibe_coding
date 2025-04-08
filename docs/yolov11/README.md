# YOLOv11 Experiments

This directory contains code and documentation related to experiments with YOLOv11 models using the Ultralytics library.

## Scope

- Initial focus: Object Detection (Inference, Finetuning) using VOC dataset.
- Future scope: Segmentation, COCO dataset integration, potential full training.

## Core Dependency

- [Ultralytics](https://github.com/ultralytics/ultralytics)
- Official YOLOv11 Docs: [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)

## Source Code

- Main scripts: [`src/models/ext/yolov11/`](../../src/models/ext/yolov11/)
- Configurations: [`src/models/ext/yolov11/configs/`](../../src/models/ext/yolov11/configs/)
- Tests: [`tests/models/ext/yolov11/`](../../tests/models/ext/yolov11/)

## Usage

### Prediction (Detection)

The `predict_detect.py` script runs inference using a YOLOv11 model based on parameters defined in a YAML configuration file. It supports command-line arguments for dataset selection and limited overrides.

**Configuration (`src/models/ext/yolov11/configs/predict_detect.yaml`):**

- `model`: Path to the model (`.pt`) or Ultralytics model name (e.g., `yolo11n.pt`).
- `project`: Base directory to save output runs (e.g., `runs/predict/detect`).
- Other optional keys correspond to `ultralytics.YOLO.predict` arguments (e.g., `conf`, `imgsz`, `save`, `save_txt`, `save_conf`, `device`).

**Command-Line:**

```bash
python -m src.models.ext.yolov11.predict_detect \
    --config <path_to_config.yaml> \
    --dataset <dataset_id> \
    --tag <split_tag> \
    --name <your_run_name> \
    [--device <device_id>] \
    [--save <True|False>] \
    [--show <True|False>] \
    [--sample_count <N>]
```

**Arguments:**

- `--config`: Path to the YAML configuration file. *Required.*
- `--dataset`: Dataset identifier (e.g., `voc`). Must have a corresponding `*_DETECT` environment variable defined in `.env` (e.g., `VOC_DETECT`). Default: `voc`.
- `--tag`: The specific split/tag for the dataset (e.g., `val2007`, `test2007`). The script looks for images in `${DATASET_BASE_PATH}/images/{tag}`. *Required.*
- `--name`: The name for this specific prediction run. This will be the name of the output directory created under the `project` specified in the config file. *Required.*
- `--device` (optional): Override the compute device specified in the config file. If omitted, the config value is used.
- `--save` (optional): Override whether to save annotated images (True/False). If omitted, the config value is used.
- `--show` (optional): Override whether to display results in a window (True/False). If omitted, the config value is used.
- `--sample_count` (optional): Randomly sample `N` images from the source directory. Processes all if omitted.

**Example (using default config on VOC test2007):**

```bash
python -m src.models.ext.yolov11.predict_detect \
    --config src/models/ext/yolov11/configs/predict_detect.yaml \
    --dataset voc \
    --tag test2007 \
    --name voc_test2007_detect_run1
```

**Example (using specific device and random sampling):**

```bash
python -m src.models.ext.yolov11.predict_detect \
    --config src/models/ext/yolov11/configs/predict_detect.yaml \
    --dataset voc \
    --tag val2007 \
    --name voc_val2007_detect_run2 \
    --device cpu \
    --save True \
    --sample_count 10
```

Results are saved to `<config.project>/<name>_<timestamp>/`.

### Prediction (Segmentation)

The `predict_segment.py` script runs instance segmentation inference using a YOLOv11 segmentation model.
It requires a YAML configuration file for model and inference settings, and command-line arguments
to specify the target dataset split and the output run name.

**Configuration (`src/models/ext/yolov11/configs/predict_segment.yaml`):**

- Defines model and inference parameters.
- `model`: Path to the segmentation model (`.pt`) or Ultralytics model name (e.g., `yolo11n-seg.pt`). *Required.*
- `project`: Base directory to save output runs (e.g., `runs/segment`). *Required.*
- Other optional keys correspond to `ultralytics.YOLO.predict` arguments for segmentation (e.g., `conf`, `iou`, `imgsz`, `max_det`, `device`, `save`, `save_txt`, `save_conf`, `save_crop`, `show`, `exist_ok`, `classes`, `agnostic_nms`, `retina_masks`, `boxes`). Refer to the YAML comments for defaults/details.
- **Note:** `source` and `name` are *not* set here; they are provided via command-line arguments.

**Command-Line:**

```bash
# Activate environment if needed (e.g., conda activate vbl)
# Ensure .env file is present at project root for dataset path resolution

python -m src.models.ext.yolov11.predict_segment \
    --config <path_to_config.yaml> \
    --dataset <dataset_id> \
    --tag <split_tag> \
    --name <your_run_name> \
    [--device <device_id>] \
    [--save <True|False>] \
    [--show <True|False>] \
    [--sample_count <N>]
```

**Arguments:**

- `--config`: Path to the YAML configuration file (e.g., `src/models/ext/yolov11/configs/predict_segment.yaml`). *Required.*
- `--dataset`: Dataset identifier (e.g., `voc`). Must have a corresponding `*_SEGMENT` environment variable defined in `.env` (e.g., `VOC_SEGMENT`). Default: `voc`.
- `--tag`: The specific split/tag for the dataset (e.g., `val2007`, `test2007`). The script looks for images in `${DATASET_BASE_PATH}/images/{tag}`. *Required.*
- `--name`: The name for this specific prediction run. This will be the name of the output directory created under the `project` specified in the config file. *Required.*
- `--device` (optional): Override the compute device specified in the config file. If omitted, the config value is used.
- `--save` (optional): Override whether to save annotated images (True/False). If omitted, the config value is used.
- `--show` (optional): Override whether to display results in a window (True/False). If omitted, the config value is used.
- `--sample_count` (optional): Randomly sample `N` images from the source directory. Processes all if omitted.

**Example (using default config on VOC val2007):**

```bash
python -m src.models.ext.yolov11.predict_segment \
    --config src/models/ext/yolov11/configs/predict_segment.yaml \
    --dataset voc \
    --tag val2007 \
    --name voc_val2007_predict_run1
```

**Example (using specific device and disabling saving):**

```bash
python -m src.models.ext.yolov11.predict_segment \
    --config src/models/ext/yolov11/configs/predict_segment.yaml \
    --dataset voc \
    --tag test2007 \
    --name voc_test2007_predict_run2 \
    --device cpu \
    --save False
```

**Output:**

- Results are saved to `<config.project>/<name>_<timestamp>/` (e.g., `runs/segment/voc_val2007_predict_run1_<timestamp>/`).
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
- `--name`: Base name for the training run. A timestamp (`_YYYYMMDD_HHMMSS`) will be automatically appended to create the actual run directory for *new* runs. For *resume* runs, this is still required syntactically but the actual run name is determined by the `--resume_with` path. *Required.*
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

The `train_segment.py` script initiates training (finetuning or from scratch) for YOLOv11 segmentation models using parameters from a YAML configuration file. It mirrors the detection training script structure. (Note: The updated resume logic and explicit `--wandb-id` requirement implemented for `train_detect.py` should be applied similarly to `train_segment.py` if consistent behavior is desired).

**Configuration (`src/models/ext/yolov11/configs/voc_segment_*.yaml`):**

- `model`: Path to the base segmentation model weights (`.pt`) or architecture YAML (`.yaml`) (e.g., `yolo11l-seg.pt`).
- `data`: Path to the segmentation dataset configuration YAML (e.g., `src/models/ext/yolov11/configs/voc_segment.yaml`).
- `project`: Base directory for segmentation training runs (e.g., `runs/train/segment`).
- `pretrained`: Boolean (`True` for finetuning, `False` for scratch).
- Other keys correspond to `ultralytics.YOLO.train` arguments suitable for segmentation.

**Command-Line:**

```bash
# Example assuming similar timestamping and resume logic might be added
python src/models/ext/yolov11/train_segment.py \
    --config <path_to_segment_training_config.yaml> \
    --name <your_run_name> \
    [--project <output_project_dir>] \
    # [--resume_with <path/to/exact/run_folder>] # If implemented
    # [--wandb_id <wandb_run_id>] # If implemented
```

**Example (Finetuning):**

```bash
python src/models/ext/yolov11/train_segment.py \
    --config src/models/ext/yolov11/configs/voc_segment_finetune.yaml \
    --name voc11l_segment_finetune_run1
```

**Example (Resuming Finetuning Run - Placeholder):**

```bash
# Example if --resume_with were implemented for segmentation
# python src/models/ext/yolov11/train_segment.py \\
#    --config src/models/ext/yolov11/configs/voc_segment_finetune.yaml \\
#    --resume_with runs/train/segment/voc11l_segment_finetune_run1_YYYYMMDD_HHMMSS
```

Training progress and results (checkpoints, metrics, logs) are saved to `<project>/<name_with_timestamp>/`. Similar to detection, a `wandb_info.txt` file may be created if WandB is used and the script is updated.

### Evaluation (Detection)

The `evaluate_detect.py` script provides comprehensive evaluation of YOLOv11 detection models using custom metrics and visualization tools, integrating utilities for metric calculation and performance measurement.

**Configuration (`src/models/ext/yolov11/configs/evaluate_*.yaml`):**

This file configures the model path, evaluation dataset details (images, labels, class names), inference parameters (image size, confidence/IoU thresholds), metric settings, and output options. A key option is `save_results: True|False` which controls whether to save annotated images and prediction details in YOLO `.txt` format for each evaluated image into an `individual_results` subfolder.

For detailed configuration options, refer to the specific comments within `evaluate_default.yaml` or the dedicated documentation in [`evaluate.md`](./evaluate.md).

**Command-Line:**

```bash
python -m src.models.ext.yolov11.evaluate_detect --config <path_to_config.yaml>
```

**Example (using default config):**

```bash
python -m src.models.ext.yolov11.evaluate_detect \
    --config src/models/ext/yolov11/configs/evaluate_default.yaml
```

Results (summary metrics JSON, plots, optional individual results) are saved to a timestamped directory within the project path defined in the configuration (e.g., `runs/evaluate/detect/expN`).

## Documentation

Detailed documentation is available in the following files:
- [Design Notes](./design.md): Architecture and design decisions
- [Evaluation Guide](./evaluate.md): Detailed evaluation metrics and configuration
- [Todo List](./todo.md): Project roadmap and task tracking