#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Default VOC root directory. Can be overridden by the first command-line argument.
DEFAULT_VOC_ROOT="$HOME/vibe/hub/datasets/VOC"
VOC_ROOT="${1:-$DEFAULT_VOC_ROOT}" # Use argument $1 if provided, otherwise use default

# VOC Dataset URLs
VOC2007_TRAINVAL_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
VOC2007_TEST_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
VOC2012_TRAINVAL_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

# --- Usage ---
echo "--------------------------------------------------"
echo "Pascal VOC Dataset Setup Script"
echo "--------------------------------------------------"
echo "Target VOC Root: $VOC_ROOT"
echo "Usage: $0 [target_voc_root_path]"
echo "If target_voc_root_path is not provided, it defaults to $DEFAULT_VOC_ROOT"
echo "--------------------------------------------------"
echo ""
read -p "Press Enter to continue, or Ctrl+C to cancel..."

# --- Step 1: Create Directory and Download ---
echo "[Step 1/3] Creating directory and downloading VOC data..."
mkdir -p "$VOC_ROOT"
cd "$VOC_ROOT"

echo "Downloading VOC2007 train/val..."
wget -c "$VOC2007_TRAINVAL_URL" # Use -c to continue partial downloads
VOC2007_TRAINVAL_TAR=$(basename "$VOC2007_TRAINVAL_URL")

echo "Downloading VOC2007 test..."
wget -c "$VOC2007_TEST_URL"
VOC2007_TEST_TAR=$(basename "$VOC2007_TEST_URL")

echo "Downloading VOC2012 train/val..."
wget -c "$VOC2012_TRAINVAL_URL"
VOC2012_TRAINVAL_TAR=$(basename "$VOC2012_TRAINVAL_URL")

echo "Downloads completed in $VOC_ROOT"
echo ""

# --- Step 2: Extraction ---
echo "[Step 2/3] Extracting VOCdevkit..."
echo "Extracting $VOC2007_TRAINVAL_TAR..."
tar xf "$VOC2007_TRAINVAL_TAR"
echo "Extracting $VOC2007_TEST_TAR..."
tar xf "$VOC2007_TEST_TAR"
echo "Extracting $VOC2012_TRAINVAL_TAR..."
tar xf "$VOC2012_TRAINVAL_TAR"

echo "Extraction complete. Removing tar files..."
rm "$VOC2007_TRAINVAL_TAR" "$VOC2007_TEST_TAR" "$VOC2012_TRAINVAL_TAR"
echo "VOCdevkit extracted and tar files removed."
echo ""

# --- Step 3: Create YOLO Directory Structure ---
echo "[Step 3/3] Creating YOLO directory structure..."
mkdir -p \
  "$VOC_ROOT/detect/images/"{train2007,val2007,test2007,train2012,val2012} \
  "$VOC_ROOT/detect/labels/"{train2007,val2007,test2007,train2012,val2012} \
  "$VOC_ROOT/segment/images/"{train2007,val2007,test2007,train2012,val2012} \
  "$VOC_ROOT/segment/labels/"{train2007,val2007,test2007,train2012,val2012}
echo "Created YOLO directory structure within $VOC_ROOT"
echo ""

# --- Completion ---
ORIGINAL_DIR=$(pwd -P) # Get the physical directory before cd-ing
cd - > /dev/null # Go back to original directory
echo "--------------------------------------------------"
echo "VOC Dataset setup steps 1-3 completed successfully!"
echo "Original VOCdevkit is in: $VOC_ROOT/VOCdevkit"
echo "YOLO structure created in: $VOC_ROOT/{detect,segment}"
echo "--------------------------------------------------"

exit 0