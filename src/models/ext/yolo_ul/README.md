# Ultralytics YOLO Integration

This module provides a simplified integration with Ultralytics YOLO for object detection tasks.

## Features

- Simple command-line interface for training and inference
- YAML-based configuration for datasets and training parameters
- Support for common datasets (COCO, VOC)

## Usage

### Training a Model

Train a YOLO model:

```bash
# Simple training with dataset YAML
python train.py --model yolo11n.pt --data configs/voc.yaml --epochs 50

# Using additional training parameters from a YAML file
python train.py --model yolo11n.pt --data configs/coco.yaml --train configs/train_coco.yaml
```

### Running Inference

Run inference using a pre-trained model:

```bash
# Basic inference
python inference.py --model yolo11n.pt --source path/to/image.jpg --show

# Inference with dataset YAML for class names
python inference.py --model runs/train/exp/weights/best.pt --source path/to/image.jpg --data configs/coco.yaml --save
```

## Configuration Files

All configuration files are stored in the `configs` directory:

- **Dataset Configs**: Define dataset paths and classes (e.g., `coco.yaml`, `voc.yaml`)
- **Training Configs**: Define model training parameters (e.g., `train_coco.yaml`)

### Dataset YAML Structure

```yaml
# Dataset paths
path: /path/to/dataset
train: train2017     # Train images relative to 'path'
val: val2017         # Validation images relative to 'path'

# Classes
names:
  0: person
  1: bicycle
  # ... more classes
```

### Training YAML Structure

```yaml
# Training parameters
epochs: 100
batch: 16
imgsz: 640
device: 0

# Optimizer settings
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005

# ... other parameters
```

## Environment Variables

The module relies on environment variables set in `.env` file at project root:

- `VIBE_ROOT`: Root directory of the project
- `DATA_ROOT`: Root directory for datasets
- `CHECKPOINTS_ROOT`: Directory for model checkpoints
- `VOC_ROOT`, `VOC2007_DIR`, `VOC2012_DIR`, `COCO_ROOT`: Dataset-specific paths