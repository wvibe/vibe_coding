# YOLOv3 Technical Design Document

## Project Overview

This document outlines the technical design for implementing YOLOv3 based on the paper "YOLOv3: An Incremental Improvement" by Joseph Redmon and Ali Farhadi. The implementation uses PyTorch and is tested through training experiments with Pascal VOC and BDD100K datasets.

For information about recent improvements and usage instructions, see [YOLOv3 README](../../src/models/py/yolov3/README.md).

## 1. Project Structure

```
project_root/
├── src/                           # Source code directory
│   ├── data_loaders/              # Dataset utilities
│   │   ├── __init__.py
│   │   └── cv/                    # Computer Vision data loaders
│   │       ├── __init__.py
│   │       └── voc.py             # Pascal VOC dataset
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hf/                    # Huggingface models
│   │   │   ├── __init__.py
│   │   │   └── WT5/               # WT5 model implementation
│   │   └── py/                    # PyTorch models
│   │       ├── __init__.py
│   │       ├── transformer/       # Transformer implementation
│   │       └── yolov3/            # YOLOv3 implementation
│   │           ├── __init__.py
│   │           ├── config.py      # Model configuration
│   │           ├── darknet.py     # Darknet-53 backbone
│   │           ├── neck.py        # Feature pyramid network
│   │           ├── head.py        # Detection heads
│   │           ├── yolov3.py      # Full YOLOv3 model
│   │           ├── loss.py        # Loss functions
│   │           ├── train.py       # Training script
│   │           ├── inference.py   # Inference script
│   │           ├── evaluate.py    # Evaluation metrics and functions
│   │           ├── scripts/       # Utility scripts
│   │           ├── anchors/       # Generated anchor boxes
│   │           └── visualizations/ # Data pipeline visualizations
│   ├── notebooks/                 # Jupyter notebooks for verification
│   └── utils/                     # Shared utilities
│       ├── __init__.py
│       └── bbox/                  # Bounding box utilities
│           ├── __init__.py
│           └── bbox.py            # Bounding box operations
├── tests/                         # Centralized test directory
│   ├── data_loaders/
│   │   └── cv/                    # Tests for CV data loaders
│   └── models/
│       └── py/                    # Tests for PyTorch models
│           ├── transformer/       # Tests for transformer models
│           └── yolov3/            # Tests for YOLOv3 implementation
├── docs/                          # Documentation
│   └── yolov3/                    # YOLOv3 specific documentation
├── setup.py                       # Project setup script
├── .env                           # Environment variables
├── requirements.txt               # Project dependencies
└── wandb/                         # Weights & Biases logging directory
```

## 2. YOLOv3 Architecture Details

### 2.1 Darknet-53 Backbone Structure
```
Layer Type     Filters   Size/Stride   Output
---------------------------------------------
Convolutional  32        3×3/1         416×416
Convolutional  64        3×3/2         208×208
Residual Block 32,64     3×3           208×208
Convolutional  128       3×3/2         104×104
Residual Block 64,128    3×3           104×104 (×2)
Convolutional  256       3×3/2         52×52
Residual Block 128,256   3×3           52×52 (×8)
Convolutional  512       3×3/2         26×26
Residual Block 256,512   3×3           26×26 (×8)
Convolutional  1024      3×3/2         13×13
Residual Block 512,1024  3×3           13×13 (×4)
```

### 2.2 Detection System Details
- Each grid cell predicts 3 bounding boxes at each of 3 scales (13×13, 26×26, 52×52)
- Each bounding box prediction includes:
  - 4 coordinates (tx, ty, tw, th) using specialized encoding
  - 1 objectness score
  - C class probabilities (where C is the number of classes)

### 2.3 Loss Function Implementation
The YOLOv3 loss function consists of three components:
1. **Localization Loss**: MSE loss for bounding box coordinates
   ```
   L_loc = λ_coord * Σ 1^obj_ij * [(tx_ij - tx̂_ij)² + (ty_ij - tŷ_ij)² + (tw_ij - tŵ_ij)² + (th_ij - tĥ_ij)²]
   ```
2. **Objectness Loss**: Binary cross-entropy for objectness score
   ```
   L_obj = Σ 1^obj_ij * BCE(C_ij, Ĉ_ij) + λ_noobj * Σ 1^noobj_ij * BCE(C_ij, Ĉ_ij)
   ```
3. **Classification Loss**: Binary cross-entropy for class probabilities
   ```
   L_cls = Σ 1^obj_ij * Σ_c BCE(p_ij(c), p̂_ij(c))
   ```

## 3. Dataset Configuration

### 3.1 Pascal VOC Details
- **Size**: ~2GB
- **Images**: ~11K training images
- **Classes**: 20 object categories
- **Data Versions**:
  - VOC2007: 2,501 training images, 2,510 validation images
  - VOC2012: 5,717 training images, 5,823 validation images
- **Directory Structure**:
  ```
  VOCdevkit/
  ├── VOC2007/
  │   ├── Annotations/        # XML annotation files
  │   ├── ImageSets/Main/     # Text files with image IDs for splits
  │   ├── JPEGImages/         # Image files
  │   └── ...
  └── VOC2012/
      ├── Annotations/
      ├── ImageSets/Main/
      ├── JPEGImages/
      └── ...
  ```

### 3.2 BDD100K Details
- **Size**: ~7GB for the 10K subset
- **Images**: 10K images in the subset
- **Classes**: 10 object categories for detection
- **Annotations**: JSON format

## 4. Implementation Technical Specifications

### 4.1 Environment Configuration
- Environment variables defined in `.env` file:
  - `VIBE_ROOT`: Root directory for the entire project
  - `VHUB_ROOT`: Root directory for data and models outside Git
  - `DATA_ROOT`: Root directory for dataset storage
  - `VOC_ROOT`: Root directory for Pascal VOC datasets
  - `PRETRAINED_ROOT`: Directory for pretrained models
  - `CHECKPOINTS_ROOT`: Directory for training checkpoints
  - `DARKNET53_WEIGHTS`: Path to pretrained Darknet53 weights
  - `WANDB_API_KEY`: API key for Weights & Biases integration

### 4.2 Training Infrastructure
- **Two-stage training approach**:
  1. Freeze backbone and train detection heads
  2. Fine-tune entire network
- **Learning rate schedule**:
  - Cosine annealing from initial LR to LR/100
  - Lower learning rate after unfreezing backbone
- **Checkpointing**:
  - Regular checkpoints at specified intervals
  - Best model saved based on validation loss
  - Final model saved at the end of training

### 4.3 Experiment Tracking
- **Weights & Biases Integration**:
  - Track training/validation metrics in real-time
  - Log hyperparameters and configurations
  - Monitor model architecture and parameters
- **Logged Metrics**:
  - **Batch-level**: Loss, component losses, learning rate, batch time
  - **Epoch-level**: Average losses, epoch time, validation metrics
  - **Model**: Model architecture, parameter count, gradient flow
  - **Validation**: Loss metrics, mAP

### 4.4 Optimization Strategies
- **Efficient Non-Maximum Suppression**:
  - Class-aware NMS to avoid comparing boxes across different classes
  - Vectorized operations for faster processing on GPU/MPS
  - Early returns for edge cases to avoid unnecessary computation

- **Tiered Confidence Thresholds**:
  - Higher threshold (`conf_threshold=0.5`) for inference to reduce false positives
  - Lower threshold (`eval_conf_threshold=0.1`) for evaluation to improve recall

- **IoU Calculation Improvements**:
  - Direct arithmetic operations instead of tensor manipulation where appropriate
  - Minimized tensor creation and reshaping operations
  - Numerical stability improvements for edge cases

## 5. Testing Strategy

### 5.1 Test Organization
- **Centralized Test Directory**: All tests moved to root-level `tests/` directory
- **Mirror Project Structure**: Test directory hierarchy matches source code
- **Shared Test Utilities**: Common fixtures and helpers in `conftest.py`
- **Test Discovery**: Automatic test discovery using pytest conventions

### 5.2 Component Tests
- **test_darknet.py**: Test backbone features and outputs
- **test_yolov3.py**: Test model forward pass and output format
- **test_loss.py**: Test loss calculation and gradient flow
- **test_evaluate.py**: Test evaluation metrics and calculations
- **test_voc.py**: Test dataset loading and preprocessing

## 6. Experimental Design

### 6.1 Planned Experiments
- Initial training on Pascal VOC with default parameters
- Ablation studies on input resolution, data augmentation, and loss weights
- Cross-dataset validation (train on VOC, test on BDD100K)

### 6.2 Hardware Support
- **CUDA**: Primary GPU acceleration for NVIDIA GPUs
- **MPS**: Metal Performance Shaders support for Apple Silicon (M1/M2/M3) Macs
- **CPU**: Fallback for systems without GPU acceleration
- **Automatic device selection**: Runtime detection of available hardware

## 7. Dependencies and Requirements

- Python 3.12
- PyTorch and torchvision
- Huggingface Datasets
- OpenCV
- matplotlib, seaborn for visualization
- pytest for testing
- Weights & Biases for experiment tracking
- python-dotenv for environment management
- PIL for image processing

All dependencies are managed through:
- `requirements.txt`: Contains detailed dependency versions
- `setup.py`: Provides package installation and dependency definition

## 8. References

- YOLOv3 paper: "YOLOv3: An Incremental Improvement" by Joseph Redmon and Ali Farhadi
- Pascal VOC dataset: http://host.robots.ox.ac.uk/pascal/VOC/
- BDD100K dataset: https://www.bdd100k.com/