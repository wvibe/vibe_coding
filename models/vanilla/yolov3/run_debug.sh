#!/bin/bash

# Debug script for YOLOv3 model
# This script runs a quick training cycle on a small subset of the dataset
# to verify that the training, validation and evaluation pipeline works properly

# Change to project root directory to ensure .env file is accessible
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# Print debug information
echo "=========================================================="
echo "YOLOv3 DEBUG MODE - Testing Pipeline on Small Dataset"
echo "=========================================================="
echo "- Running quick training cycle (1 epoch)"
echo "- Using only 10% of the dataset"
echo "- Testing all components: train, validate, evaluate"
echo "=========================================================="

# Run the debug training script
python -m models.vanilla.yolov3.debug_train

# Print completion message
echo "Debug cycle completed."