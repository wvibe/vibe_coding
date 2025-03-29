#!/bin/bash
# Script to run COCO training with timestamped output directory
# Using MPS (Metal Performance Shaders) for Apple Silicon GPU acceleration

# Create shortened timestamp (YYMMDD_HHMM)
TIMESTAMP=$(date "+%y%m%d_%H%M")

# Set project directory with COCO prefix and timestamp
PROJECT_DIR="src/models/ext/yolo_ul/runs/train/COCO_${TIMESTAMP}"

# Print information
echo "Starting COCO 80-class training with MPS acceleration..."
echo "Timestamp: ${TIMESTAMP}"
echo "Project directory: ${PROJECT_DIR}"
echo "Using official yolov8n.pt pre-trained model (80 classes)"
echo "Training on COCO dataset (80 classes) for 20 epochs"
echo "Using batch size of 4 for better stability"

# Run the training with full 80-class COCO dataset using YOLOv8 pre-trained model
python src/models/ext/yolo_ul/train.py \
  --model yolov8n.pt \
  --data src/models/ext/yolo_ul/configs/coco_full.yaml \
  --train src/models/ext/yolo_ul/configs/train_full.yaml \
  --project "${PROJECT_DIR}" \
  --epochs 20 \
  --batch 4 \
  --imgsz 640 \
  --device mps

echo "Training complete. Results saved to ${PROJECT_DIR}"
echo "To view TensorBoard, run: tensorboard --logdir ${PROJECT_DIR}/train"