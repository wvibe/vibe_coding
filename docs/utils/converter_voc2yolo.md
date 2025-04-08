# VOC Data Conversion Utilities Design Notes

This document outlines the design choices and rationale behind the scripts in `src/utils/data_converter/`, which are used to convert the Pascal VOC dataset format into the YOLO format required by this project and manage related files.

---

## `voc2yolo_utils.py` - Common VOC Utilities

This module does not contain an executable script but provides shared constants and helper functions used by the other VOC conversion and visualization scripts.

### Purpose

To centralize common logic for:
- Defining standard VOC directory names and class lists.
- Parsing VOC XML annotation files.
- Constructing paths to various VOC dataset files (images, annotations, imagesets, masks).
- Constructing paths to the standardized output directories (`images/`, `labels_detect/`, `labels_segment/`).
- Reading image IDs from ImageSet files.

### Key Functions

- **Constants:** `VOC_CLASSES`, `VOC_CLASS_TO_ID`, standard directory names (`ANNOTATIONS_DIR`, `JPEG_IMAGES_DIR`, etc.), output directory names (`OUTPUT_IMAGES_DIR`, etc.).
- **`parse_voc_xml(xml_path)`:**
    - Reads a VOC XML file.
    - Extracts image dimensions (`width`, `height`).
    - Extracts all `<object>` details: class name (`name`), bounding box (`bbox`: `[xmin, ymin, xmax, ymax]`), and difficulty flag (`difficult`: 0 or 1).
    - Skips objects with unknown class names (not in `VOC_CLASSES`).
    - Performs basic validation on coordinates.
    - Returns a list of object dictionaries and the image dimensions.
- **Path Getters (`get_voc_dir`, `get_image_set_path`, `get_image_path`, `get_annotation_path`, `get_segm_inst_mask_path`, `get_segm_cls_mask_path`, `get_output_..._dir`)**:
    - Provide consistent methods for constructing absolute paths to specific files or directories within the VOC dataset structure (`VOCdevkit` or the processed output structure) based on root paths, year, and tag.
- **`read_image_ids(imageset_path, ...)`:**
    - Reads image IDs from a standard VOC ImageSet `.txt` file.
    - Handles files where lines might contain extra columns (common in VOC sets).
    - Includes optional random sampling logic (used by `voc2yolo_images.py`).

---

## `voc2yolo_images.py` - Image Copying Script

This script copies the original JPEG images from the `VOCdevkit` structure into the project's standardized `detect/images/` or `segment/images/` directories based on task type.

### Purpose

To gather all necessary images (e.g., for train/val splits across different years) for either a **detection** or **segmentation** task into the task-specific directory structure (`<output_root>/<task_type>/images/<tag><year>/`) expected by training frameworks and the project structure defined in `docs/dataset/voc/README.md`.

### Usage

```bash
# Corrected usage example with -m
python -m src.utils.data_converter.voc2yolo_images \
    --years <YEARS> \
    --tags <TAGS> \
    --task-type <detect|segment> \
    [--voc-root /path/to/VOC] \
    [--output-root /path/to/output] \
    [--sample-count N] \
    [--seed <SEED>]
```

- `--years`: Comma-separated years (e.g., `2007,2012`).
- `--tags`: Comma-separated tags (e.g., `train,val`).
- `--task-type`: **Required.** Specify `'detect'` or `'segment'`. This determines:
    - Which ImageSet list to read (`ImageSets/Main/` for `detect`, `ImageSets/Segmentation/` for `segment`).
    - The output directory structure (`<output_root>/detect/images/...` or `<output_root>/segment/images/...`).
- `--voc-root`: Path to VOC dataset root (contains `VOCdevkit`). Defaults to `$VOC_ROOT` environment variable or `./datasets/VOC`.
- `--output-root`: Root for saving the processed structure (`detect/`, `segment/`). Defaults to `--voc-root`.
- `--sample-count`: (Optional) Randomly sample N images *total* across all specified splits. If not set, copies all images.
- `--seed`: (Optional) Seed for random sampling (default: 42).

### Logic

1.  Parses arguments, including the new `--task-type`.
2.  Determines VOC root and output root paths.
3.  Reads image IDs for the specified year/tag combinations using `read_image_ids` from `voc2yolo_utils`, passing the `task_type` to `get_image_set_path` to ensure the correct `ImageSets/` subdirectory (`Main` or `Segmentation`) is used.
4.  Optionally applies random sampling across the collected IDs.
5.  For each selected image ID:
    - Constructs the source path (`VOCdevkit/<YEAR>/JPEGImages/<id>.jpg`).
    - Constructs the destination path using `get_output_image_dir` from `voc2yolo_utils`, passing the `task_type` to get the correct path: `<output_root>/<task_type>/images/<tag><year>/<id>.jpg`.
    - Creates the destination subdirectory (`<output_root>/<task_type>/images/<tag><year>/`) if needed.
    - Checks if the destination file already exists. If so, skips it.
    - If the destination doesn't exist, copies the image using `shutil.copy2`.
    - Tracks success, skip, and failure counts.
6.  Reports summary statistics.

### Input

- `VOCdevkit/<YEAR>/JPEGImages/` (Source images)
- `VOCdevkit/<YEAR>/ImageSets/Main/<tag>.txt` (if `task_type='detect'`)
- `VOCdevkit/<YEAR>/ImageSets/Segmentation/<tag>.txt` (if `task_type='segment'`)

### Output

- Creates images in `<output_root>/<task_type>/images/<tag><year>/` (e.g., `detect/images/train2007/`, `segment/images/val2012/`)

---

## `voc2yolo_detect_labels.py` - Detection Label Conversion Script

This script converts Pascal VOC XML annotations into YOLO detection format labels (bounding boxes).

### Purpose

To create `.txt` label files compatible with YOLO-based training and evaluation for object detection. Each file corresponds to an image and contains normalized bounding box coordinates and class indices for each object.

### Usage

```bash
python -m src.utils.data_converter.voc2yolo_detect_labels \
    --years <YEARS> \
    --tags <TAGS> \
    [--voc-root /path/to/VOC] \
    [--output-root /path/to/output] \
    [--sample-count N] \
    [--seed <SEED>]
```

- `--years`, `--tags`, `--voc-root`, `--output-root`: Similar to `voc2yolo_images.py`.
- `--sample-count`: (Optional) Randomly sample N images *total* across all specified splits. If not set, processes all images.
- `--seed`: (Optional) Seed for random sampling (default: 42).

### Logic

1.  Parses arguments.
2.  Determines VOC root and output root paths.
3.  Parses comma-separated lists for years and tags.
4.  Collects image IDs for each year/tag combination (from `ImageSets/Main/<tag>.txt`).
5.  Optionally applies random sampling across all collected IDs if `--sample-count` is specified.
6.  For each image ID:
    - Constructs the path to the corresponding XML file (`VOCdevkit/<YEAR>/Annotations/<id>.xml`).
    - Parses the XML using `parse_voc_xml` from `voc2yolo_utils` to get image dimensions and object list (`name`, `bbox`, `difficult`).
    - Constructs the output label file path using `get_output_detect_label_dir` from `voc2yolo_utils`: `<output_root>/detect/labels/<tag><year>/<id>.txt`.
    - Creates the output subdirectory if needed.
    - Checks if output file already exists. If so, skips it and increments skipped count.
    - Initializes an empty list for label lines.
    - For each object extracted from the XML:
        - Get the class name and look up its index using `VOC_CLASS_TO_ID`.
        - Get the bounding box `[xmin, ymin, xmax, ymax]`.
        - Convert the box coordinates to YOLO format (`center_x`, `center_y`, `width`, `height`) and normalize them using the image dimensions.
        - Format the YOLO label line: `<class_index> <cx_norm> <cy_norm> <w_norm> <h_norm>`.
        - Add the formatted line to the list.
    - Write all formatted label lines to the output `.txt` file.
7.  Reports summary statistics including success, skipped (already existing), and failed counts.

### Input

- `VOCdevkit/<YEAR>/Annotations/<id>.xml` (Source annotations)
- `VOCdevkit/<YEAR>/ImageSets/Main/<tag>.txt` (Lists of image IDs)

### Output

- Creates label files in `<output_root>/detect/labels/<tag><year>/` (e.g., `detect/labels/train2007/000001.txt`)
- Format: One object per line: `<class_index> <cx_norm> <cy_norm> <w_norm> <h_norm>` (space-separated, normalized coordinates).

---

## `voc2yolo_segment_labels.py` - Segmentation Label Conversion Script

This script converts Pascal VOC XML annotations and segmentation masks into YOLO segmentation format labels (polygons).

### Purpose

To create `.txt` label files compatible with YOLO-based training and evaluation for instance segmentation. Each file corresponds to an image and contains normalized polygon coordinates and the corresponding class index for each object instance.

### Usage

```bash
python -m src.utils.data_converter.voc2yolo_segment_labels \
    --years <YEARS> \
    --tags <TAGS> \
    [--voc-root /path/to/VOC] \
    [--output-root /path/to/output] \
    [--sample-count N] \
    [--no-connect-parts] \
    [--min-contour-area AREA] \
    [--seed <SEED>]
```

- `--years`, `--tags`, `--voc-root`, `--output-root`: Similar to `voc2yolo_detect_labels.py`.
- `--sample-count`: (Optional) Randomly sample N images *total* across all specified splits. If not set, processes all images.
- `--no-connect-parts`: (Optional) Disable the default behavior of connecting disconnected parts of the same instance into a single polygon. When this flag is not used (default behavior), all disconnected parts of the same instance ID will be joined with straight lines to form a single polygon.
- `--min-contour-area`: (Optional) Minimum area threshold for contours (in pixels squared). Contours smaller than this value will be filtered out. Only applicable when processing segmentation masks.
- `--seed`: (Optional) Seed for random sampling (default: 42).
- ~`--skip-difficult`~: (Removed - Difficulty is not directly handled as class is from class mask).
- ~`--iou-threshold`~: (Removed - Class matching uses class mask, not IoU with XML bbox).

### Logic

This conversion relies on matching instance IDs from `SegmentationObject` masks with class IDs from `SegmentationClass` masks. The script has been refactored to improve modularity:

1.  Parses arguments.
2.  Determines paths.
3.  Parses comma-separated lists for years and tags.
4.  Collects image IDs for each year/tag combination (from `ImageSets/Segmentation/<tag>.txt`).
5.  Optionally applies random sampling across all collected IDs if `--sample-count` is specified.
6.  Initializes a `VOC2YOLOConverter` for each year/tag combination.
7.  For each image ID:
    - Checks if output file already exists. If so, skips it and increments skipped count.
    - Constructs paths to instance mask (`SegmentationObject/<id>.png`) and class mask (`SegmentationClass/<id>.png`).
    - Loads the instance mask using PIL (to preserve palette indices).
    - Loads the class mask (e.g., using PIL or cv2).
    - Finds unique non-background/non-boundary pixel values in the instance mask; these are the instance IDs.
    - Initializes an empty list for label lines.
    - For each unique `instance_id` found in the instance mask:
        - Create a binary mask isolating only the pixels belonging to this instance.
        - **Crucially: Determine the Class:**
            - Find all pixels in the *class mask* that correspond to the instance's binary mask.
            - Determine the most frequent non-background/non-boundary class ID among these pixels using `scipy.stats.mode`.
            - If a valid modal class ID is found, convert it to the class name (from `VOC_CLASSES`) and then to the class index (from `VOC_CLASS_TO_ID`).
            - If no valid modal class ID can be found (e.g., instance only overlaps background/boundary), log a warning and skip this instance.
        - If class was determined:
            - Find contours (`cv2.findContours`) of the instance's binary mask.
            - If `--connect-parts` is enabled:
                - Connect all disconnected parts of the same instance into a single complex polygon using a nearest-neighbor approach. This creates connections between contours using straight lines between nearest points.
                - Contours smaller than `--min-contour-area` are filtered out if specified.
                - The resulting complex polygon maintains the detailed outline of each part while creating a single continuous boundary.
            - Otherwise (default behavior):
                - Process each contour separately, potentially resulting in multiple polygons for the same instance ID.
            - For each resulting polygon:
                - Check if contour area is above a minimum threshold.
                - Simplify the polygon slightly using `cv2.approxPolyDP`.
                - Ensure minimum number of points (e.g., 3).
                - Convert the contour points to a flat list `[x1, y1, x2, y2, ...]`.
                - Normalize the polygon coordinates using image dimensions.
                - Format the YOLO label line: `<class_index> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ...`.
                - Add the formatted line to the list.
    - Write all formatted label lines to the output `.txt` file using `get_output_segment_label_dir` from `voc2yolo_utils`: `<output_root>/segment/labels/<tag><year>/<id>.txt`.
8.  Reports summary statistics including success, skipped (already existing), and failed counts.

### Handling Disconnected Instance Parts

By default, the script connects disconnected parts of the same instance into a single complex polygon using the following approach:

1. **Contour Ordering**: Contours are ordered using a greedy nearest-neighbor algorithm that minimizes the total connection distance.
2. **Point-to-Point Matching**: For each pair of adjacent contours, the algorithm finds the closest points between them to create the connection.
3. **Single Complex Polygon**: The result is a single polygon that preserves the detailed outline of each part while creating a continuous boundary around all visible parts of the instance.

This approach is particularly useful for handling instances that appear as multiple disconnected regions in the image (e.g., an object partially occluded or split by another object).

If you prefer to have each part as a separate polygon, you can use the `--no-connect-parts` flag.

### Logging Behavior

The script implements intelligent logging that reduces verbosity for normal operations:

- Only logs unusual cases or potential issues, such as:
  - Mismatches between the number of unique instances and the number of generated polygons
  - When an instance generates multiple separate polygons (if `--connect-parts` is disabled)
  - When no instances are found or no output is written
- Debug-level logging is available for more detailed information
- Warning and error messages are always shown for invalid data or processing failures

### Input

- `VOCdevkit/<YEAR>/SegmentationObject/<id>.png` (Instance masks)
- `VOCdevkit/<YEAR>/SegmentationClass/<id>.png` (Class masks - Used for class lookup)
- `VOCdevkit/<YEAR>/ImageSets/Segmentation/<tag>.txt` (Lists of image IDs)
- ~`VOCdevkit/<YEAR>/Annotations/<id>.xml`~ (No longer directly used for class lookup in this script)

### Output

- Creates label files in `<output_root>/segment/labels/<tag><year>/`
- Format: One object instance per line: `<class_index> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ...` (space-separated, normalized coordinates).
- By default, each instance will be represented by a single line, even if it appears as multiple disconnected regions in the image. If you use the `--no-connect-parts` flag, each disconnected part will be represented as a separate polygon.