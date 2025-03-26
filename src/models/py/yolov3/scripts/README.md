# YOLOv3 Scripts

This directory contains scripts for working with the YOLOv3 model implementation.

## Available Scripts

### `download_darknet_weights.py`
Downloads and converts pretrained Darknet53 weights to PyTorch format.
```bash
./download_darknet_weights.py
```

### `verify_pipeline.py`
Verifies the YOLOv3 data pipeline and weight loading functionality.
```bash
./verify_pipeline.py
```

### `generate_anchors.py`
Generates custom anchor boxes for the YOLOv3 model based on the dataset.
```bash
./generate_anchors.py [dataset_name] [timestamp]
```
Example:
```bash
./generate_anchors.py voc custom
```

### `run_train_and_eval.sh`
Runs the training and evaluation pipeline for YOLOv3.
```bash
./run_train_and_eval.sh
```

### `run_debug.sh`
Runs the training in debug mode with minimal data for quick testing.
```bash
./run_debug.sh
```

## Output Directories

- **Anchors**: `/src/models/py/yolov3/anchors/` - Contains generated anchor boxes and visualizations
- **Visualizations**: `/src/models/py/yolov3/visualizations/` - Contains data pipeline verification images