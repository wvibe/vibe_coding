# YOLOv8 Integration

This document describes how to use YOLOv8 models from Ultralytics within the vibe_coding project for object detection tasks using the PASCAL VOC dataset.

## Overview

[YOLOv8](https://github.com/ultralytics/ultralytics) is the latest version of the YOLO (You Only Look Once) real-time object detection model developed by Ultralytics. This integration allows you to:

1. Run inference with pre-trained YOLOv8 models
2. Fine-tune YOLOv8 models on the PASCAL VOC dataset
3. Train YOLOv8 models from scratch

## Directory Structure

```
vibe_coding/
├── src/
│   └── models/
│       └── ext/
│           └── yolov8/
│               └── configs/        # YAML configuration files
│                   └── voc.yaml    # VOC dataset configuration
├── notebooks/
│   └── model/
│       └── ext/
│           └── yolov8/
│               └── yolov8_intro.ipynb  # Demo notebook
└── docs/
    └── yolov8/
        └── README.md               # This documentation
```

## Dataset Configuration

The `src/models/ext/yolov8/configs/voc.yaml` file defines the PASCAL VOC dataset configuration for YOLOv8. The file specifies:

- Dataset path (read from environment variable `VOC_ROOT`)
- Training, validation, and test splits
- Class names for the 20 object categories in PASCAL VOC

**Important:** This configuration relies on the `VOC_ROOT` environment variable being set to the correct path of your PASCAL VOC dataset. Ensure this variable is loaded (e.g., from the `.env` file) in your environment before running training or inference using this configuration file.

## Basic Usage

Make sure the `VOC_ROOT` environment variable is loaded before running these examples.

### Inference

To run inference on a single image:

```python
from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO('yolo8n.pt')

# Run inference on an image
results = model('path/to/image.jpg')

# Display results
results[0].show()
```

### Fine-tuning

To fine-tune a pretrained YOLOv8 model on the VOC dataset:

```python
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables (important for VOC_ROOT)
load_dotenv()

# Load a pretrained YOLOv8 model
model = YOLO('yolo8n.pt')

# Fine-tune the model on the VOC dataset
results = model.train(
    data='src/models/ext/yolov8/configs/voc.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='yolov8n_voc_finetuned'
)
```

### Training from Scratch

To train a YOLOv8 model from scratch:

```python
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables (important for VOC_ROOT)
load_dotenv()

# Create a new model from YAML configuration
model = YOLO('yolo8n.yaml')

# Train the model on the VOC dataset
results = model.train(
    data='src/models/ext/yolov8/configs/voc.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='yolov8n_voc_scratch'
)
```

## Demo Notebook

For a complete walkthrough, refer to the demo notebook at `notebooks/model/ext/yolov8/yolov8_intro.ipynb`. This notebook includes steps to load the environment variables.

## Benchmarking

A configurable tool is available to benchmark YOLOv8 and other object detection models on datasets like PASCAL VOC. It measures various accuracy (mAP, mAP_s/m/l, confusion matrix) and performance (inference time percentiles, model size, GPU memory) metrics.

For the detailed design and usage instructions, see the [Benchmark Design Document](./benchmark_design.md).

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PASCAL VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)