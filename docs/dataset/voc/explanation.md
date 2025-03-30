# Pascal VOC Dataset Explanation (Ultralytics Format)

This document clarifies the expected structure and format for the Pascal VOC dataset within this project, specifically when used in the Ultralytics format.

## Location

- The root directory for the VOC dataset is defined by the `VOC_ROOT` environment variable in the `.env` file.
- Based on the standard project `.env`, this typically resolves to `/home/wmu/vibe/hub/datasets/VOC`.

## Structure

Instead of using the raw `VOCdevkit` structure, tools within this project generally expect the dataset to be organized in the standard Ultralytics format:

```
<VOC_ROOT>/
├── images/
│   ├── train2007/
│   │   ├── 000001.jpg
│   │   └── ...
│   ├── val2007/
│   │   └── ...
│   ├── test2007/
│   │   └── ...
│   ├── train2012/
│   │   └── ...
│   └── val2012/
│       └── ...
├── labels/
│   ├── train2007/
│   │   ├── 000001.txt
│   │   └── ...
│   ├── val2007/
│   │   └── ...
│   ├── test2007/
│   │   └── ...
│   ├── train2012/
│   │   └── ...
│   └── val2012/
│       └── ...
├── downloads/      # (Typically ignored by tools)
└── VOCdevkit/      # (Typically ignored by tools)
```

- **Images:** Images (`.jpg`) reside in subdirectories under `images/` corresponding to the desired split (e.g., `images/test2007/`).
- **Labels:** Corresponding labels (`.txt`) must exist in subdirectories under `labels/` with the same split name (e.g., `labels/test2007/`).

## Label Format

- Labels **must** be in the Ultralytics YOLO format:
  - One `.txt` file per image.
  - Each row represents one bounding box: `<class_index> <x_center_norm> <y_center_norm> <width_norm> <height_norm>`
  - Coordinates are normalized (0.0 to 1.0).
  - Class indices are zero-based.

## Usage Example (Benchmark Configuration)

- When configuring tools like the detection benchmark (`detection_benchmark.yaml`), ensure the paths and format reflect this structure:
  ```yaml
  dataset:
    # Point to the specific split directory under images/
    test_images_dir: "/home/wmu/vibe/hub/datasets/VOC/images/test2007" # Example for test2007 split
    # Point to the corresponding split directory under labels/
    annotations_dir: "/home/wmu/vibe/hub/datasets/VOC/labels/test2007" # Example for test2007 split
    # Specify the correct format
    annotation_format: "yolo_txt"
    # ... other dataset settings ...
  ```