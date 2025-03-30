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
python src/models/ext/yolov11/predict_detect.py \
    --config <path_to_config.yaml> \
    --name-prefix <your_run_prefix> \
    [--<param_to_override> <value>] # Optional overrides
```

**Example (using demo config):**

```bash
python src/models/ext/yolov11/predict_detect.py \
    --config src/models/ext/yolov11/configs/predict_demo.yaml \
    --name-prefix demo_voc_test
```

**Example (overriding model and confidence):**

```bash
python src/models/ext/yolov11/predict_detect.py \
    --config src/models/ext/yolov11/configs/predict_demo.yaml \
    --name-prefix demo_voc_yolo11s_conf50 \
    --model yolo11s.pt \
    --conf 0.5
```

Results are saved to `<config.project>/<name-prefix>_<timestamp>/`.

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

### Evaluation (Detection)

The `evaluate_detect.py` script provides comprehensive evaluation of YOLOv11 models using custom metrics and visualization tools.

**Configuration (`src/models/ext/yolov11/configs/evaluate_*.yaml`):**

- `model`: Path to the model (`.pt`) or Ultralytics model name (e.g., `yolo11n.pt`).
- `dataset`: Configuration for evaluation data (image and label directories, class names).
- `evaluation_params`: Parameters for model inference (image size, batch size, etc.).
- `metrics`: Configuration for metric calculation (IoU thresholds, size ranges, etc.).
- `computation`: Options for measuring inference time and memory usage.
- `output`: Settings for output directory and formats.

For detailed configuration options, see [Evaluation Documentation](./evaluate.md).

**Command-Line:**

```bash
python src/models/ext/yolov11/evaluate_detect.py --config <path_to_config.yaml>
```

**Example:**

```bash
python src/models/ext/yolov11/evaluate_detect.py \
    --config src/models/ext/yolov11/configs/evaluate_default.yaml
```

Results (metrics, visualizations, inference statistics) are saved to the configured output directory.

## Documentation

Detailed documentation is available in the following files:
- [Design Notes](./design.md): Architecture and design decisions
- [Evaluation Guide](./evaluate.md): Detailed evaluation metrics and configuration
- [Todo List](./todo.md): Project roadmap and task tracking