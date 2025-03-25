#!/bin/bash

# Debug script for YOLOv3 model
# This script runs a quick training cycle on a very small subset of the dataset
# to verify that the training, validation and evaluation pipeline works properly

# Change to project root directory to ensure .env file is accessible
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

# Print debug information
echo "=========================================================="
echo "YOLOv3 DEBUG MODE - Testing Pipeline on Tiny Dataset"
echo "=========================================================="
echo "- Running quick training cycle (1 epoch)"
echo "- Using only a tiny subset of the dataset (max 20 images)"
echo "- Using smaller input size for faster processing"
echo "- Testing all components: train, validate, evaluate"
echo "=========================================================="

# Run the training script with extreme debug parameters
python -m models.vanilla.yolov3.train \
  --epochs 1 \
  --batch-size 2 \
  --input-size 224 \
  --freeze-backbone \
  --freeze-epochs 0 \
  --workers 1 \
  --run-name "debug_run" \
  --max-images 20 \
  --fast-dev-run \
  --no-wandb

# Print completion message
echo "Debug cycle completed."