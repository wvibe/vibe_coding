# YOLOv3 Implementation and Training Experiment Design

## Project Overview

This document outlines the technical design for implementing YOLOv3 based on the paper "YOLOv3: An Incremental Improvement" by Joseph Redmon and Ali Farhadi. The implementation will use PyTorch and will be tested through training experiments with Pascal VOC and BDD100K datasets.

## 1. Project Structure

```
project_root/
├── data_loaders/                   # Dataset utilities (renamed from datasets)
│   ├── __init__.py
│   ├── images/                     # Image dataset loaders
│   │   ├── __init__.py
│   │   ├── voc.py                  # Pascal VOC dataset
│   │   └── bdd.py                  # BDD100K dataset
│   ├── object_detection/           # Object detection specific loaders
│   │   ├── __init__.py
│   │   ├── voc.py                  # VOC object detection format
│   │   └── bdd.py                  # BDD100K detection format
│   └── utils/                      # Common dataset utilities
│       ├── __init__.py
│       ├── augmentation.py         # Data augmentation functions
│       └── bbox.py                 # Bounding box utilities
├── models/
│   ├── vanilla/
│   │   ├── yolov3/                 # YOLOv3 implementation
│   │   │   ├── __init__.py
│   │   │   ├── config.py           # Model configuration
│   │   │   ├── darknet.py          # Darknet-53 backbone
│   │   │   ├── neck.py             # Feature pyramid network
│   │   │   ├── head.py             # Detection heads
│   │   │   ├── yolov3.py           # Full YOLOv3 model
│   │   │   ├── loss.py             # Loss functions
│   │   │   ├── train.py            # Training script
│   │   │   ├── inference.py        # Inference script
│   │   │   ├── model_outputs/      # Directory for model checkpoints
│   │   │   │   └── <run_name>/     # Run-specific subdirectories
│   │   │   └── tests/              # Unit tests for YOLOv3
│   │   │       ├── __init__.py
│   │   │       ├── test_darknet.py
│   │   │       ├── test_yolov3.py
│   │   │       └── test_loss.py
├── wandb/                          # Weights & Biases logging directory
├── notebooks/                      # Jupyter notebooks for verification
│   ├── yolov3/
│   │   ├── model_verification.ipynb   # Testing individual components
│   │   ├── training_playground.ipynb  # Experimental training
│   │   ├── inference_demo.ipynb       # Visualization of predictions
│   │   └── data_exploration.ipynb     # Dataset analysis
├── utils/                          # Shared utilities
│   ├── __init__.py
│   ├── visualization.py            # Visualization tools
│   └── metrics.py                  # Evaluation metrics
├── .env                            # Environment variables 
├── requirements.txt                # Project dependencies
└── doc/                            # Documentation
    └── yolov3/                     # YOLOv3 specific documentation
        ├── design.md               # This document
        ├── architecture.md         # Detailed architecture explanation
        └── experiments.md          # Experiment results
```

## 2. YOLOv3 Architecture

### 2.1 Key Features
- Multi-scale predictions (3 different scales)
- Darknet-53 backbone (better feature extraction than Darknet-19 used in YOLOv2)
- Feature pyramid network for detection at 3 scales
- Bounding box prediction with dimension priors (anchors)
- Class prediction using logistic regression instead of softmax
- Each grid cell predicts 3 bounding boxes at each scale

### 2.2 Network Components

#### 2.2.1 Darknet-53 Backbone
- 53 convolutional layers
- Residual connections
- No pooling layers (uses strided convolutions)
- Structure:
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

#### 2.2.2 Feature Pyramid Network
- Extracts features from 3 different scales of Darknet-53
- Routes and upsamples features to create a feature pyramid
- Scales: 13×13, 26×26, and 52×52

#### 2.2.3 Detection Heads
- Each detection head processes features at a specific scale
- Each grid cell predicts 3 bounding boxes
- Each bounding box prediction includes:
  - 4 coordinates (tx, ty, tw, th)
  - 1 objectness score
  - C class probabilities (where C is the number of classes)

### 2.3 Loss Function
The YOLOv3 loss function consists of three components:
1. **Localization Loss**: MSE loss for bounding box coordinates
2. **Objectness Loss**: Binary cross-entropy for objectness score
3. **Classification Loss**: Binary cross-entropy for class probabilities

## 3. Datasets

### 3.1 Pascal VOC
- **Size**: ~2GB
- **Images**: ~11K training images
- **Classes**: 20 object categories
- **Annotations**: XML format with bounding boxes
- **Splits**: Standard train/val split (trainval/test)
- **Data Versions**: 
  - VOC2007: 2,501 training images, 2,510 validation images
  - VOC2012: 5,717 training images, 5,823 validation images
  - Combined: Option to train on both for better performance
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

### 3.2 BDD100K
- **Size**: ~7GB for the 10K subset
- **Images**: 10K images in the subset
- **Classes**: 10 object categories for detection
- **Annotations**: JSON format
- **Features**: Contains driving scenes with varied conditions (day/night, clear/rainy)

## 4. Implementation Details

### 4.1 Environment Configuration
- Environment variables defined in `.env` file:
  - `DATA_ROOT`: Root directory for dataset storage
  - `WANDB_API_KEY`: API key for Weights & Biases integration
- Using `python-dotenv` to load environment variables
- Data paths and other configurable parameters centralized for easy management

### 4.2 Model Implementation
- **Backbone**: Implement Darknet-53 with configurable input size
- **Feature Pyramid**: Connect features from different scales of the backbone
- **Detection Heads**: Implement the detection logic for each scale
- **Forward Pass**: Process an image and output predictions in the required format
- **Loss Calculation**: Implement the compound loss function

### 4.3 Dataset Implementation
- Create dataset classes for Pascal VOC and BDD100K
- Support for both VOC2007 and VOC2012 datasets
- Option to combine datasets using PyTorch's `ConcatDataset`
- Implement XML parsing for VOC annotations
- Convert annotations to YOLOv3 format (normalized [x_center, y_center, width, height])
- Create data loaders with custom collate functions

### 4.4 Training Infrastructure
- **Run Management**: 
  - Each training run has a unique name (user-defined or timestamp-based)
  - Checkpoints stored in run-specific directories to prevent overwriting
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

### 4.5 Experiment Tracking
- **Weights & Biases Integration**:
  - Track training/validation metrics in real-time
  - Log hyperparameters and configurations
  - Monitor model architecture and parameters
  - Record resource usage (GPU memory, compute time)
- **Logged Metrics**:
  - **Batch-level**: Loss, component losses, learning rate, batch time
  - **Epoch-level**: Average losses, epoch time, validation metrics
  - **Model**: Model architecture, parameter count, gradient flow
  - **Validation**: Loss metrics, mAP (TODO)
- **Visualizations**:
  - Loss curves and learning rate schedule
  - Performance metrics over time
  - Resource utilization
  - Example predictions (TODO)

### 4.6 Evaluation
- Implement mAP calculation at IoU=0.5 (PASCAL VOC metric)
- Implement mAP at IoU=0.5:0.95 (COCO metric)
- Track precision, recall, and F1 score per class
- Visualize predictions and ground truth

## 5. Unit Tests

### 5.1 Component Tests
- **test_darknet.py**: Test Darknet-53 backbone
  - Test forward pass with different input sizes
  - Test feature extraction at specified layers
  - Test shape of output tensors

- **test_yolov3.py**: Test full YOLOv3 model
  - Test forward pass end-to-end
  - Test with batch processing
  - Test output format

- **test_loss.py**: Test loss function
  - Test loss calculation with dummy predictions and targets
  - Test gradient flow
  - Test each loss component individually

### 5.2 Integration Tests
- Test anchors generation
- Test bounding box transformations
- Test non-maximum suppression

## 6. Experiments

### 6.1 Initial Training
- Train on Pascal VOC with default parameters
- Evaluate performance on validation set
- Analyze common failure cases

### 6.2 Ablation Studies
- Effect of input resolution (416×416 vs 608×608)
- Impact of different data augmentation techniques
- Analysis of loss components weighting
- Effect of training on VOC2007, VOC2012, and combined datasets

### 6.3 Cross-Dataset Validation
- Train on Pascal VOC, test on BDD100K (domain transfer)
- Analyze performance differences between datasets

## 7. Timeline

1. **Week 1**: Model implementation and unit tests
2. **Week 2**: Dataset implementation and notebooks
3. **Week 3**: Initial training and evaluation
4. **Week 4**: Experiments and documentation

## 8. Expected Outcomes

1. A fully functional YOLOv3 implementation
2. Comprehensive documentation
3. Experimental results and analysis
4. Notebooks for visualization and testing
5. Trained models with proper evaluation metrics
6. Experiment tracking dashboards in Weights & Biases

## 9. Technologies and Dependencies

- Python 3.12
- PyTorch and torchvision
- Huggingface Datasets
- OpenCV
- matplotlib, seaborn for visualization
- pytest for testing
- Weights & Biases for experiment tracking 
- python-dotenv for environment management
- PIL for image processing 