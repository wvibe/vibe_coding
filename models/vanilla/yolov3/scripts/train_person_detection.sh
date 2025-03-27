#!/bin/bash
# Train YOLOv3 for person detection using VOC dataset

# Get script directory and change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

# Output directory
OUTPUT_DIR="models/vanilla/yolov3/model_outputs/person_detection"
mkdir -p "${OUTPUT_DIR}"

# Configuration
BATCH_SIZE=16
EPOCHS=60
LEARNING_RATE=0.00001  # Reduced learning rate to prevent NaN losses
FREEZE_EPOCHS=30  # Number of epochs to freeze backbone
WANDB_ENABLED=false  # Disabled by default for debug runs
DEBUG_MODE=false  # Default to normal training mode

# Default to training mode
MODE="train"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --eval-only)
      MODE="eval"
      shift
      ;;
    --resume)
      RESUME="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --enable-wandb)
      WANDB_ENABLED=true
      shift
      ;;
    --debug)
      DEBUG_MODE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "=== Training YOLOv3 for Person Detection ==="
echo "Output directory: ${OUTPUT_DIR}"
echo "Mode: ${MODE}"
if [ -n "${RESUME}" ]; then
  echo "Resuming from: ${RESUME}"
fi
echo "Learning rate: ${LEARNING_RATE}"
echo "Debug mode: ${DEBUG_MODE}"

# Build command
CMD="python models/vanilla/yolov3/train.py \
  --class-name person \
  --year 2007,2012 \
  --batch-size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --output-dir ${OUTPUT_DIR} \
  --learning-rate ${LEARNING_RATE}"

# Add debug mode if enabled (faster training for testing)
if [ "${DEBUG_MODE}" = true ]; then
  CMD="${CMD} --fast-dev-run --max-images 32"
fi

# Add freeze backbone if needed
if [ "${FREEZE_EPOCHS}" -gt 0 ]; then
  CMD="${CMD} --freeze-backbone --freeze-epochs ${FREEZE_EPOCHS}"
fi

# Add mode-specific options
if [ "${MODE}" = "eval" ]; then
  CMD="${CMD} --eval-only"
fi

if [ -n "${RESUME}" ]; then
  CMD="${CMD} --resume ${RESUME}"
fi

# Enable/disable wandb
if [ "${WANDB_ENABLED}" = true ]; then
  # Simplified wandb config
  CMD="${CMD} --wandb-project yolov3-person-detection"
else
  CMD="${CMD} --no-wandb"
fi

# Print and execute command
echo "Running: ${CMD}"
echo
eval "${CMD}"

# Check exit status
if [ $? -eq 0 ]; then
  echo
  echo "Person detection ${MODE} completed successfully."
  echo "Results saved to: ${OUTPUT_DIR}"
else
  echo
  echo "Person detection ${MODE} failed."
  exit 1
fi