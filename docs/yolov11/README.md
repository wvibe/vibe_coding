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

The `predict_detect.py` script runs inference using a YOLOv11 model based on parameters defined in a YAML configuration file. It supports command-line overrides.

**Configuration (`src/models/ext/yolov11/configs/predict_*.yaml`):**

- `model`: Path to the model (`.pt`) or Ultralytics model name (e.g., `yolo11n.pt`).
- `source`: Path to the input image or directory.
- `project`: Base directory to save output runs (e.g., `runs/predict/detect`).
- `random_select`: (Optional) Number of images to randomly select if `source` is a directory.
- Other optional keys correspond to `ultralytics.YOLO.predict` arguments (e.g., `conf`, `imgsz`, `save`, `save_txt`, `save_conf`, `device`).

**Command-Line:**

```bash
python -m src.models.ext.yolov11.predict_detect \
    --config <path_to_config.yaml> \
    --name-prefix <your_run_prefix> \
    [--<param_to_override> <value>] # Optional overrides
```

**Example (using demo config):**

```bash
python -m src.models.ext.yolov11.predict_detect \
    --config src/models/ext/yolov11/configs/predict_detect.yaml \
    --name-prefix demo_voc_test
```

**Example (overriding model and confidence):**

```bash
python -m src.models.ext.yolov11.predict_detect \
    --config src/models/ext/yolov11/configs/predict_detect.yaml \
    --name-prefix demo_voc_yolo11s_conf50 \
    --model yolo11s.pt \
    --conf 0.5
```

Results are saved to `<config.project>/<name-prefix>_<timestamp>/`.

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

The `train_detect.py` script initiates training (finetuning or from scratch) using a YOLOv11 model based on parameters defined in a YAML configuration file.

**Configuration (`src/models/ext/yolov11/configs/voc_*.yaml`):**

- `model`: Path to the base model weights (`.pt`) for finetuning, or architecture YAML (`.yaml`) for scratch training (e.g., `yolo11l.pt` or `yolo11l.yaml`).
- `data`: Path to the dataset configuration YAML (e.g., `src/models/ext/yolov11/configs/voc_detect.yaml`).
- `project`: Base directory to save training runs (e.g., `runs/train/detect`).
- `pretrained`: Boolean (`True` for finetuning, `False` for scratch).
- Other keys correspond to `ultralytics.YOLO.train` arguments (e.g., `epochs`, `batch`, `imgsz`, `optimizer`, `lr0`, `device`, etc.).

**Command-Line:**

```bash
python src/models/ext/yolov11/train_detect.py \
    --config <path_to_training_config.yaml> \
    --name <your_run_name> \
    [--project <output_project_dir>] \
    [--resume] \
    [--wandb-id <wandb_run_id_to_resume>]
```

**Example (Finetuning):**

```bash
python src/models/ext/yolov11/train_detect.py \
    --config src/models/ext/yolov11/configs/voc_finetune.yaml \
    --name voc11l_finetune_run1
```

**Example (Training from Scratch):**

```bash
python src/models/ext/yolov11/train_detect.py \
    --config src/models/ext/yolov11/configs/voc_retrain.yaml \
    --name voc11l_scratch_run1
```

**Example (Resuming Finetuning Run):**

```bash
python src/models/ext/yolov11/train_detect.py \
    --config src/models/ext/yolov11/configs/voc_finetune.yaml \
    --name voc11l_finetune_run1 \
    --resume
```

Training progress and results (checkpoints, metrics, logs) are saved to `<project>/<name>/`.

### Training (Segmentation)

The `train_segment.py` script initiates training (finetuning or from scratch) for YOLOv11 segmentation models using parameters from a YAML configuration file. It mirrors the detection training script.

**Configuration (`src/models/ext/yolov11/configs/voc_segment_*.yaml`):**

- `model`: Path to the base segmentation model weights (`.pt`) or architecture YAML (`.yaml`) (e.g., `yolo11l-seg.pt`).
- `data`: Path to the segmentation dataset configuration YAML (e.g., `src/models/ext/yolov11/configs/voc_segment.yaml`).
- `project`: Base directory for segmentation training runs (e.g., `runs/train/segment`).
- `pretrained`: Boolean (`True` for finetuning, `False` for scratch).
- Other keys correspond to `ultralytics.YOLO.train` arguments suitable for segmentation.

**Command-Line:**

```bash
python src/models/ext/yolov11/train_segment.py \
    --config <path_to_segment_training_config.yaml> \
    --name <your_run_name> \
    [--project <output_project_dir>] \
    [--resume] \
    [--wandb-id <wandb_run_id_to_resume>]
```

**Example (Finetuning):**

```bash
python src/models/ext/yolov11/train_segment.py \
    --config src/models/ext/yolov11/configs/voc_segment_finetune.yaml \
    --name voc11l_segment_finetune_run1
```

**Example (Resuming Finetuning Run):**

```bash
python src/models/ext/yolov11/train_segment.py \
    --config src/models/ext/yolov11/configs/voc_segment_finetune.yaml \
    --name voc11l_segment_finetune_run1 \
    --resume
```

Training progress and results (checkpoints, metrics, logs) are saved to `<project>/<name>/`.

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