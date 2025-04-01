# Pascal VOC Dataset Explanation (Project Format)

This document clarifies the expected structure and format for the Pascal VOC dataset within this project, suitable for both detection and segmentation tasks.

## Location

- The root directory for the VOC dataset is expected to be `/home/wmu/vibe/hub/datasets/VOC`.
  - If using environment variables (e.g., in scripts or configs), use `$VOC_ROOT` which should point to this location.

## Structure

The dataset should be organized as follows:

```
/home/wmu/vibe/hub/datasets/VOC/
├── images/                      # Contains all JPG image files
│   ├── train2007/
│   │   ├── 000001.jpg
│   │   └── ...
│   ├── val2007/
│   │   └── ...
│   ├── test2007/                # Note: test sets might not have labels
│   │   └── ...
│   ├── train2012/
│   │   └── ...
│   └── val2012/
│       └── ...
├── labels_detect/               # Contains YOLO format bounding box labels
│   ├── train2007/
│   │   ├── 000001.txt
│   │   └── ...
│   ├── val2007/
│   │   └── ...
│   ├── train2012/
│   │   └── ...
│   └── val2012/
│       └── ...
├── labels_segment/              # Contains YOLO format segmentation labels (polygons)
│   ├── train2007/
│   │   ├── 000001.txt
│   │   └── ...
│   ├── val2007/
│   │   └── ...
│   ├── train2012/
│   │   └── ...
│   └── val2012/
│       └── ...
└── VOCdevkit/                   # Original VOC download structure (can be kept for reference)
    ├── VOC2007/
    │   ├── Annotations/
    │   ├── ImageSets/
    │   ├── JPEGImages/
    │   ├── SegmentationClass/
    │   └── SegmentationObject/
    └── VOC2012/
        └── ... (similar structure)
```

- **Images:** Original images are stored under `images/` split by year and type (train/val/test).
- **Detection Labels:** YOLO format bounding box labels (`.txt`) reside under `labels_detect/` mirroring the `images/` structure.
- **Segmentation Labels:** YOLO format segmentation labels (`.txt`, containing polygons) reside under `labels_segment/` mirroring the `images/` structure.
- **VOCdevkit:** The original downloaded structure can optionally be kept, but tools typically use the `images/` and `labels_*/` directories.

## Label Format

### Detection (`labels_detect/*.txt`)
- One `.txt` file per image.
- Each row represents one object:
  ```
  <class_index> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
  ```
- Coordinates are normalized (0.0 to 1.0).
- Class indices are zero-based.

### Segmentation (`labels_segment/*.txt`)
- One `.txt` file per image.
- Each row represents one object instance:
  ```
  <class_index> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ... <xn_norm> <yn_norm>
  ```
- Contains the class index followed by normalized polygon coordinates.
- Polygons should have a minimum of 3 points (6 coordinates).

**Important Note on Segmentation Labels (Instance vs. Class ID):**

- The original VOC segmentation masks (`VOCdevkit/VOC*/SegmentationObject/*.png`) use pixel values (1, 2, 3, ...) to identify different *object instances* within a single image. These pixel values are **Instance IDs**, not Class IDs.
- The actual class names ("person", "car", etc.) are only stored in the corresponding XML annotation file (`VOCdevkit/VOC*/Annotations/*.xml`), associated with bounding boxes.
- To generate the correct YOLO segmentation labels (`class_id polygon...`), the `segment_voc2yolo.py` script performs the following:
    1. Extracts each instance mask based on its pixel value (Instance ID).
    2. Converts the instance mask into polygon coordinates.
    3. Calculates the bounding box of the instance mask.
    4. Parses the XML to get all object bounding boxes and their class names.
    5. Matches the instance mask's bounding box to the XML object bounding boxes using Intersection over Union (IoU).
    6. If a match is found above a certain threshold (default 0.5), the class name from the matched XML object is retrieved.
    7. The class name is converted to a zero-based `class_index`.
    8. The final YOLO label line is written using the retrieved `class_index` and the calculated polygon coordinates.
- If an instance mask cannot be confidently matched to an XML object via IoU, it is skipped, and a warning is logged.

## Usage Examples

### Detection Benchmark (`detection_benchmark.yaml`)
```yaml
dataset:
  test_images_dir: "/home/wmu/vibe/hub/datasets/VOC/images/test2007" # Or path via $VOC_ROOT
  annotations_dir: "/home/wmu/vibe/hub/datasets/VOC/labels_detect/test2007" # Or path via $VOC_ROOT
  annotation_format: "yolo_txt"
  # ... other settings ...
```

### Segmentation Training (Hypothetical Ultralytics Config)
```yaml
path: /home/wmu/vibe/hub/datasets/VOC # Dataset root directory
train: images/train2012 # Relative path to training images
val: images/val2012   # Relative path to validation images

# Labels will be automatically inferred based on task (e.g., 'segment')
# Ultralytics expects labels in ../labels_segment/train2012, ../labels_segment/val2012

names:
  0: aeroplane
  1: bicycle
  # ... map all 20 classes ...
  19: tvmonitor
```

## Conversion Scripts

- `src/utils/data_converter/voc2yolo_detect_labels.py`: Converts original VOC XML annotations (from `VOCdevkit`) into the `labels_detect` YOLO bounding box format.
- `src/utils/data_converter/voc2yolo_segment_labels.py`: Converts original VOC segmentation masks (from `VOCdevkit`) and XML annotations into the `labels_segment` YOLO polygon format.
- `src/utils/data_converter/voc2yolo_images.py`: Copies original JPEG images (from `VOCdevkit/.../JPEGImages`) into the top-level `images/` directory, structured by `<tag><year>` (e.g., `images/train2012/`), based on the image IDs listed in the `VOCdevkit/.../ImageSets/Main/<tag>.txt` file for the specified years and tags.
    - Skips copying if the destination image already exists.
    - Supports an optional `--sample-count` argument to randomly select a subset of images from each split.
    - Example usage (sampling 100 images):
      ```bash
      python -m src.utils.data_converter.voc2yolo_images \
          --voc-root /path/to/VOC \
          --output-root /path/to/output \
          --years 2012 \
          --tags train \
          --sample-count 100
      ```

## Testing the Converters

Unit tests are provided for both conversion scripts to ensure core functionality.

- **Location:** `tests/utils/data_converter/`
- **Files:** `test_detect_voc2yolo.py`, `test_segment_voc2yolo.py`

To run the tests:
1. Ensure you are in the project's root directory (`/home/wmu/vibe/vibe_coding`).
2. Make sure the correct conda environment (`vbl`) is activated.
3. Run pytest targeting the specific directory:
   ```bash
   python -m pytest tests/utils/data_converter/ -v
   ```
This command will execute all tests within that directory and provide verbose output.