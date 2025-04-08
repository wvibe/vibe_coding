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

# --- Step 4: Convert to YOLO format ---
echo "[Step 4/5] Converting VOC data to YOLO format..."
echo "Converting detect images (2007, 2012 - train, val, test)..."
python -m src.utils.data_converter.voc2yolo_images \
    --task-type detect \
    --years 2007,2012 \
    --tags train,val,test \
    --voc-root "$VOC_ROOT"

echo "Converting detect labels (2007, 2012 - train, val, test)..."
python -m src.utils.data_converter.voc2yolo_detect_labels \
    --years 2007,2012 \
    --tags train,val,test \
    --voc-root "$VOC_ROOT"

echo "Converting segment images (2007, 2012 - train, val, test)..."
python -m src.utils.data_converter.voc2yolo_images \
    --task-type segment \
    --years 2007,2012 \
    --tags train,val,test \
    --voc-root "$VOC_ROOT"

echo "Converting segment labels (2007, 2012 - train, val, test)..."
python -m src.utils.data_converter.voc2yolo_segment_labels \
    --years 2007,2012 \
    --tags train,val,test \
    --voc-root "$VOC_ROOT"
echo "Conversion to YOLO format completed."
echo ""

# --- Step 5: Verify File Counts ---
echo "[Step 5/5] Verifying file counts..."
VERIFICATION_OUTPUT=""
ALL_MATCHED=true

TASKS=("detect" "segment")
SPLITS=("train2007" "val2007" "test2007" "train2012" "val2012")

for task in "${TASKS[@]}"; do
  VERIFICATION_OUTPUT+="
--- Task: $task ---"
  for split in "${SPLITS[@]}"; do
    IMG_DIR="$VOC_ROOT/$task/images/$split"
    LBL_DIR="$VOC_ROOT/$task/labels/$split"
    if [ -d "$IMG_DIR" ] && [ -d "$LBL_DIR" ]; then
      # Use find to count only files, robust against filenames with special characters
      IMG_COUNT=$(find "$IMG_DIR" -maxdepth 1 -type f | wc -l)
      LBL_COUNT=$(find "$LBL_DIR" -maxdepth 1 -type f | wc -l)
      VERIFICATION_OUTPUT+="
Split: $split - Images: $IMG_COUNT, Labels: $LBL_COUNT"
      if [ "$IMG_COUNT" -eq "$LBL_COUNT" ]; then
        VERIFICATION_OUTPUT+=" -> Match"
      else
        VERIFICATION_OUTPUT+=" -> MISMATCH"
        ALL_MATCHED=false
      fi
    else
       # Report missing directories more clearly
       if [ ! -d "$IMG_DIR" ] && [ ! -d "$LBL_DIR" ]; then
          VERIFICATION_OUTPUT+="
Split: $split - Directories not found ($IMG_DIR and $LBL_DIR)"
          # Consider if this should set ALL_MATCHED=false depending on expectations
       elif [ ! -d "$IMG_DIR" ]; then
          VERIFICATION_OUTPUT+="
Split: $split - Image directory missing: $IMG_DIR"
          ALL_MATCHED=false
       else # LBL_DIR missing
          VERIFICATION_OUTPUT+="
Split: $split - Label directory missing: $LBL_DIR"
          ALL_MATCHED=false
       fi
    fi
  done
  VERIFICATION_OUTPUT+="
"
done

echo -e "$VERIFICATION_OUTPUT" # Use -e to interpret newline characters

if $ALL_MATCHED; then
  echo "Verification complete: All file counts match."
else
  echo "Verification complete: File count MISMATCH detected." >&2 # Output mismatch to stderr
fi
echo ""


# --- Completion ---
ORIGINAL_DIR=$(pwd -P) # Get the physical directory before cd-ing
cd - > /dev/null # Go back to original directory
echo "--------------------------------------------------"
if $ALL_MATCHED; then
  echo "VOC Dataset setup, conversion, and verification completed successfully!"
else
  echo "VOC Dataset setup and conversion completed, BUT VERIFICATION FAILED (mismatched counts)!" >&2
fi
echo "Original VOCdevkit is in: $VOC_ROOT/VOCdevkit"
echo "YOLO data generated in: $VOC_ROOT/{detect,segment}"
echo "--------------------------------------------------"

if $ALL_MATCHED; then
  exit 0
else
  exit 1 # Exit with error if verification failed
fi