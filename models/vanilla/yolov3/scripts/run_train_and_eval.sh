#!/bin/bash

# Extended training and evaluation script for YOLOv3 model
# This script trains a YOLOv3 model on combined Pascal VOC 2007+2012 dataset for 100 epochs,
# evaluates it every 5 epochs on the validation set, logs metrics to WandB,
# and saves the model to a custom folder.

# Change to project root directory to ensure .env file is accessible
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

# Load environment variables
source .env

# Create timestamp for run name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="yolov3_pascal_voc_combined_100epochs_${TIMESTAMP}"

# Create output directory
OUTPUT_DIR="${PROJECT_ROOT}/models/vanilla/yolov3/model_outputs/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

# Print run information
echo "Starting extensive training run: ${RUN_NAME}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Training for 100 epochs using both VOC2007 and VOC2012 datasets"
echo "Using pretrained Darknet53 backbone weights"
echo "Logging metrics to WandB"
echo "Evaluating every 5 epochs"

# Check if Darknet53 weights exist
if [ ! -f "${DARKNET53_WEIGHTS}" ]; then
    echo "Darknet53 weights not found. Downloading..."
    ./models/vanilla/yolov3/scripts/download_darknet_weights.sh
else
    echo "Using existing Darknet53 weights at ${DARKNET53_WEIGHTS}"
fi

# Run the training script with desired parameters
python -m models.vanilla.yolov3.train \
  --epochs 100 \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --dataset voc \
  --year 2007,2012 \
  --train-split train \
  --val-split val \
  --batch-size 8 \
  --checkpoint-interval 5 \
  --eval-interval 5 \
  --workers 4 \
  --iou-threshold 0.5 \
  --pretrained \
  --wandb-project "yolov3_voc" \
  --wandb-entity "${WANDB_ENTITY}"

# Print completion message
echo "Training completed. Model saved to ${OUTPUT_DIR}"
echo "Check WandB dashboard for detailed metrics and visualizations"