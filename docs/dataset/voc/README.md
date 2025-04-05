# Pascal VOC Dataset Explanation (Project Format)

This document clarifies the expected structure and format for the Pascal VOC dataset within this project, suitable for both detection and segmentation tasks.

## Location

- The root directory for the VOC dataset is expected to be `${VOC_ROOT}` (which points to `${DATASETS_ROOT}/VOC`).
  - This location is defined in the project's `.env` file (`${VHUB_ROOT}/datasets/VOC`)
- Task-specific roots are defined as:
  - Detection: `${VOC_DETECT}` (points to `${VOC_ROOT}/detect`)
  - Segmentation: `${VOC_SEGMENT}` (points to `${VOC_ROOT}/segment`)

## Structure (Revised)

The dataset should be organized as follows:

```
${VOC_ROOT}/
├── detect/                     # Detection-specific data
│   ├── images/
│   │   ├── train2007/          # From VOC2007 train+val
│   │   ├── val2007/            # From VOC2007 test (used for validation)
│   │   ├── test2007/           # From VOC2007 test
│   │   ├── train2012/          # From VOC2012 train
│   │   └── val2012/            # From VOC2012 val
│   └── labels/
│       ├── train2007/          # YOLO detection labels
│       ├── val2007/
│       ├── test2007/           # Often empty/unavailable
│       ├── train2012/
│       └── val2012/
├── segment/                    # Segmentation-specific data
│   ├── images/
│   │   ├── train2007/          # From VOC2007 train+val
│   │   ├── val2007/            # From VOC2007 test (used for validation)
│   │   ├── test2007/           # From VOC2007 test
│   │   ├── train2012/          # From VOC2012 train
│   │   └── val2012/            # From VOC2012 val
│   └── labels/
│       ├── train2007/          # YOLO segmentation labels
│       ├── val2007/
│       ├── test2007/           # Often empty/unavailable
│       ├── train2012/
│       └── val2012/
└── VOCdevkit/                  # Original downloaded content (raw data source)
    ├── VOC2007/
    │   ├── Annotations/
    │   ├── ImageSets/
    │   ├── JPEGImages/
    │   ├── SegmentationClass/
    │   └── SegmentationObject/
    └── VOC2012/
        └── ... (similar structure)
```

- **Images:** Original images are copied/symlinked into `detect/images/` and `segment/images/` split by year and type (train/val/test). *Note:* VOC test sets (`test2007`, `test2012`) often lack labels in the original dataset, so `val` splits are frequently used for validation/testing instead.
- **Detection Labels:** YOLO format bounding box labels (`.txt`) reside under `detect/labels/` mirroring the `images/` structure. Test set labels may be unavailable.
- **Segmentation Labels:** YOLO format segmentation labels (`.txt`, containing polygons) reside under `segment/labels/` mirroring the `images/` structure. Test set labels may be unavailable.
- **VOCdevkit:** The original downloaded structure serves as the source for the conversion scripts.

## Label Format

### Detection (`detect/labels/*.txt`)
- One `.txt` file per image.
- Each row represents one object:
  ```
  <class_index> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
  ```
- Coordinates are normalized (0.0 to 1.0).
- Class indices are zero-based (0-19 for VOC).

### Segmentation (`segment/labels/*.txt`)
- One `.txt` file per image.
- Each row represents one object instance:
  ```
  <class_index> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ... <xn_norm> <yn_norm>
  ```
- Contains the class index followed by normalized polygon coordinates.
- Polygons should have a minimum of 3 points (6 coordinates). Disconnected parts of the same instance are typically represented as a single complex polygon or multiple simple polygons on the same line (check converter implementation).

**Important Note on Segmentation Labels (Instance vs. Class ID):**

- The original VOC segmentation masks (`VOCdevkit/VOC*/SegmentationObject/*.png` and `SegmentationClass/*.png`) use **8-bit palettized PNG files** where:
  - Pixel values are indices into a predefined color palette.
  - Actual instance/class IDs are obtained by applying this palette during decoding.
  - Palette layout follows Pascal VOC specifications (different palettes for `SegmentationObject` vs `SegmentationClass`).
- ID encoding after palette application:
  - **Class IDs**: 1-20 in `SegmentationClass` masks (mapped to 0-19 in YOLO). Pixel value 0 is background.
  - **Instance IDs**: 1-254 in `SegmentationObject` masks. Pixel value 0 is background, 255 indicates difficult/void regions.
- Conversion process updates for `src/utils/data_converter/voc2yolo_segment_labels.py`:
    1. **Palette handling**: Script uses VOC's official color palettes to decode the PNGs to get actual Class and Instance IDs per pixel.
    2. **Handle difficult areas**: By default ignores pixels with value 255 (difficult regions) in the `SegmentationObject` mask. Use the `--with-difficult` flag during conversion to include them (often as a separate "difficult" class or merged with the object depending on the flag's implementation).
    3. **Instance extraction**:
        - Process all pixels with instance IDs 1-254 as valid instances.
        - Group all connected and disconnected pixel regions sharing the same instance ID.
        - Convert the combined region(s) for each instance into polygon coordinates.
    4. **Class determination**:
        - Primarily uses the corresponding `SegmentationClass` mask: The class ID (1-20) at the instance's pixel locations determines the object class (mapped to 0-19).
        - Fallback (if needed): XML bounding box matching (`Annotations/*.xml`) can be used as a secondary check or if class masks are ambiguous, matching the instance's bounding box to XML object boxes via IoU.
    5. The final YOLO label line combines the determined `class_index` (0-19) and the normalized polygon coordinates for the instance.

## Conversion Scripts

- `src/utils/data_converter/voc2yolo_detect_labels.py`: Converts original VOC XML annotations (`VOCdevkit`) into the YOLO bounding box format in `detect/labels/`.
- `src/utils/data_converter/voc2yolo_segment_labels.py`: Converts original VOC segmentation masks (`SegmentationObject`, `SegmentationClass`) and XML annotations (`Annotations`) from `VOCdevkit` into the YOLO polygon format in `segment/labels/`. Handles palette decoding, instance grouping, class assignment, and difficult regions.
- `src/utils/data_converter/voc2yolo_images.py`: Copies images from `VOCdevkit/.../JPEGImages` into the appropriate `detect/images/<split><year>` or `segment/images/<split><year>` directories. Uses the `--task-type` argument to determine whether to read image IDs from `ImageSets/Main` (for `detect`) or `ImageSets/Segmentation` (for `segment`).

## Dataset Status (Current Workspace)

- **VOCdevkit:** The `VOCdevkit` directory containing the raw VOC2007 and VOC2012 data is already present in the `${VOC_ROOT}` directory (`$HOME/vibe/hub/datasets/VOC`). Download and extraction steps can be skipped if using the existing data. The `scripts/setup_voc_dataset.sh` script can be used to re-download or set up on a new system.

## Usage Examples (Updated Paths)

### Detection Benchmark/Evaluation
```yaml
# Example using val2007 (often used instead of unlabeled test2007)
dataset:
  images_dir: "${VOC_DETECT}/images/val2007"
  labels_dir: "${VOC_DETECT}/labels/val2007"
  annotation_format: "yolo_txt"
  # ... other settings ...
```

### Segmentation Training (Hypothetical Ultralytics Config)
```yaml
path: ${VOC_SEGMENT} # Dataset root for segmentation task
train: images/train2012 # Relative path to training images
val: images/val2012   # Relative path to validation images

# Labels expected in: ${VOC_SEGMENT}/labels/train2012, ${VOC_SEGMENT}/labels/val2012

names:
  0: aeroplane
  1: bicycle
  # ... map all 20 classes ...
  19: tvmonitor
```

## Testing the Converters

Unit tests are provided for conversion scripts.

- **Location:** `tests/utils/data_converter/`
- **Files:** `test_voc2yolo_detect.py`, `test_voc2yolo_segment.py`, `test_voc_copy_images.py` (adjust filenames as needed)

To run the tests:
1. Ensure you are in the project's root directory (`/home/ubuntu/vibe/vibe_coding`).
2. Make sure the correct conda environment (`vbl`) is activated.
3. Run pytest targeting the specific directory:
   ```bash
   python -m pytest tests/utils/data_converter/ -v
   ```
This command will execute all tests within that directory and provide verbose output.