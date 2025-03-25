#!/bin/bash

# Debug script for YOLOv3 model
# This script runs a quick training cycle on a small subset of the dataset
# to verify that the training, validation and evaluation pipeline works properly

# Change to project root directory to ensure .env file is accessible
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

# Get subset percentage from command line parameter (default: 0.1 = 10%)
PERCENT=${1:-"0.1"}

# Print debug information
echo "=========================================================="
echo "YOLOv3 DEBUG MODE - Testing Pipeline on Small Dataset"
echo "=========================================================="
echo "- Running with ${PERCENT} of dataset (${PERCENT}% of all images)"
echo "- Using smaller input size for faster processing (224px)"
echo "- Using pretrained Darknet53 backbone weights"
echo "- Testing all components: train, validate, evaluate"
echo "=========================================================="

# Run the training script with subset percentage parameter and pretrained weights
python -m models.vanilla.yolov3.train \
  --epochs 1 \
  --batch-size 4 \
  --input-size 224 \
  --workers 2 \
  --run-name "debug_pretrained_${PERCENT}" \
  --subset-percent "${PERCENT}" \
  --pretrained \
  --fast-dev-run \
  --debug-mode \
  --no-wandb

# Print completion message
echo "Debug cycle completed."
echo "Usage:"
echo "  ./models/vanilla/yolov3/scripts/run_debug.sh         # Run with 10% of data (default)"
echo "  ./models/vanilla/yolov3/scripts/run_debug.sh 0.05    # Run with 5% of data"
echo "  ./models/vanilla/yolov3/scripts/run_debug.sh 0.01    # Run with 1% of data"