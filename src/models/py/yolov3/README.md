# YOLOv3 Implementation for Pascal VOC Dataset

This repository contains an implementation of YOLOv3 for object detection, optimized for the Pascal VOC dataset.

## Recent Improvements

The implementation has been significantly improved to address training stability and mAP performance issues:

### 1. Custom Anchor Boxes
- Generated using k-means clustering on VOC dataset bounding boxes
- Better match with the distribution of object sizes in VOC
- Organized into 3 scales for multi-scale detection

### 2. Improved Loss Function
- Fixed coordinate transformation between prediction and target
- Added IoU-based ignore mechanism for better objectness loss calculation
- Implemented scale-specific assignment based on object size
- Normalized losses by batch size for more stable training

### 3. Optimized Training Parameters
- Learning rate adjusted to 5e-4 with cosine annealing and warmup
- Gradient clipping to prevent exploding gradients
- Balanced loss weights (λ_coord=5.0, λ_noobj=0.5)
- Batch size increased to 16 for better gradient estimates

### 4. Code Structure Improvements
- Moved to proper Python package structure with src layout
- Replaced relative imports with absolute imports
- Simplified environment configuration
- Added proper package installation with setup.py

### 5. Data Processing Verification
- Added verification scripts to ensure data pipeline correctness
- Proper scale assignment for objects of different sizes
- Handling of dummy boxes to prevent false training signals

## Quick Start

```bash
# Install the package
pip install -e .

# Verify data pipeline and model loading
python -m src.models.py.yolov3.scripts.verify_pipeline

# Generate custom anchors for VOC dataset
python -m src.models.py.yolov3.scripts.generate_anchors voc custom

# Train with improved settings
bash src/models/py/yolov3/scripts/run_train_and_eval.sh

# Evaluate a trained model
bash src/models/py/yolov3/scripts/run_train_and_eval.sh --eval
```

## Model Overview

The model follows the YOLOv3 architecture with three main components:
1. **Backbone**: Darknet-53 pretrained on ImageNet
2. **Neck**: Feature Pyramid Network for multi-scale feature maps
3. **Heads**: Detection heads at 3 scales (13×13, 26×26, 52×52)

## Key Components
- `darknet.py`: Darknet-53 backbone implementation
- `neck.py`: FPN implementation for multi-scale features
- `head.py`: Detection heads for each scale
- `loss.py`: Improved loss function with proper anchor assignment
- `yolov3.py`: Full model implementation with prediction processing
- `config.py`: Configuration with optimized parameters for VOC
- `train.py`: Training loop with learning rate scheduling and validation

## Scripts
- `verify_pipeline.py`: Diagnostic script for data pipeline and weights
- `generate_anchors.py`: Script for generating custom anchor boxes
- `run_train_and_eval.sh`: Main training script with improved parameters
- `download_darknet_weights.py`: Script to download pretrained backbone weights

## Requirements
- PyTorch 2.0+
- torchvision
- Albumentations
- Weights & Biases (for logging)
- matplotlib, numpy, sklearn
- tqdm

## References
- YOLOv3 paper: "YOLOv3: An Incremental Improvement" by Joseph Redmon and Ali Farhadi
- Pascal VOC dataset: http://host.robots.ox.ac.uk/pascal/VOC/
- Detailed design document: [design.md](../../../../docs/yolov3/design.md)