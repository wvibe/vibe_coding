#!/bin/bash

# Extended training and evaluation script for YOLOv3 model with improved parameters
# This script trains a YOLOv3 model on Pascal VOC 2007+2012 dataset with optimized hyperparameters,
# evaluates it on the validation set, logs metrics to WandB, and saves the model.

# Change to project root directory to ensure .env file is accessible
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

# Load environment variables
source .env

# Create timestamp for run name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="yolov3_voc_improved_${TIMESTAMP}"

# Create output directory
OUTPUT_DIR="${PROJECT_ROOT}/models/vanilla/yolov3/model_outputs/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

# Print run information
echo "====== YOLOv3 Training with Improved Parameters ======"
echo "Run name: ${RUN_NAME}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Training on VOC2007 and VOC2012 with custom anchors and improved loss function"
echo "Using pretrained Darknet53 backbone weights"
echo "Logging metrics to WandB"

# Check if Darknet53 weights exist
if [ ! -f "${DARKNET53_WEIGHTS}" ]; then
    echo "Darknet53 weights not found. Downloading..."
    ./models/vanilla/yolov3/scripts/download_darknet_weights.sh
else
    echo "Using existing Darknet53 weights at ${DARKNET53_WEIGHTS}"
fi

# Run the training script with improved parameters
python -m models.vanilla.yolov3.train \
  --epochs 100 \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --dataset voc \
  --year 2007,2012 \
  --train-split train \
  --val-split val \
  --batch-size 16 \
  --learning-rate 5e-4 \
  --weight-decay 1e-4 \
  --optimizer adam \
  --lr-scheduler cosine \
  --warmup-epochs 3 \
  --min-lr-factor 0.01 \
  --lambda-coord 5.0 \
  --lambda-noobj 0.5 \
  --checkpoint-interval 5 \
  --eval-interval 5 \
  --workers 4 \
  --iou-threshold 0.5 \
  --pretrained \
  --grad-clip-norm 10.0 \
  --wandb-project "yolov3_voc" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-tags "improved,custom_anchors,fixed_loss"

# Print completion message
echo "Training completed. Model saved to ${OUTPUT_DIR}"
echo "Check WandB dashboard for detailed metrics and visualizations"

# Optional: Evaluate the final model on VOC2007 test set
if [ "$1" == "--eval" ]; then
    echo "Evaluating final model on VOC2007 test set..."
    FINAL_MODEL="${OUTPUT_DIR}/model_final.pth"

    python -m models.vanilla.yolov3.evaluate \
      --model-path "${FINAL_MODEL}" \
      --dataset voc \
      --year 2007 \
      --split test \
      --batch-size 16 \
      --workers 4 \
      --output-dir "${OUTPUT_DIR}/evaluation" \
      --visualize 10
fi