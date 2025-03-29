#!/bin/bash
# Script to run VOC2007 training with timestamped output directory
# Using MPS (Metal Performance Shaders) for Apple Silicon GPU acceleration

# Create shortened timestamp (YYMMDD_HHMM)
TIMESTAMP=$(date "+%y%m%d_%H%M")

# Set project directory with VOC prefix and timestamp
PROJECT_DIR="src/models/ext/yolo_ul/runs/train/VOC_${TIMESTAMP}"

# Print information
echo "Starting VOC2007 training with MPS acceleration..."
echo "Timestamp: ${TIMESTAMP}"
echo "Project directory: ${PROJECT_DIR}"

# Run the training
python src/models/ext/yolo_ul/train.py \
  --model yolo11n.pt \
  --data src/models/ext/yolo_ul/configs/voc2007_full.yaml \
  --train src/models/ext/yolo_ul/configs/train_quick.yaml \
  --project "${PROJECT_DIR}" \
  --epochs 3 \
  --batch 4 \
  --imgsz 640 \
  --device mps

echo "Training complete. Results saved to ${PROJECT_DIR}"