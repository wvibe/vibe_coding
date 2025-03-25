#!/bin/bash

# Simple training and evaluation script for YOLOv3 model
# This script trains a YOLOv3 model on Pascal VOC dataset for 10 epochs,
# evaluates it on the validation set, and saves the model to a custom folder.

# Change to project root directory to ensure .env file is accessible
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# Create timestamp for run name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="yolov3_pascal_voc_10epochs_${TIMESTAMP}"

# Create output directory
OUTPUT_DIR="${PROJECT_ROOT}/model_outputs/yolov3/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

# Print run information
echo "Starting training run: ${RUN_NAME}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Training for 10 epochs"

# Run the training script with desired parameters
python -m models.vanilla.yolov3.train \
  --epochs 10 \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --dataset voc \
  --year 2007 \
  --train-split train \
  --val-split val \
  --batch-size 8 \
  --checkpoint-interval 1 \
  --eval-interval 1 \
  --workers 4 \
  --iou-threshold 0.5 \
  --pretrained

# Print completion message
echo "Training completed. Model saved to ${OUTPUT_DIR}"