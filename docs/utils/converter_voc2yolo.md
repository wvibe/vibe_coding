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
- **Path Getters (`get_voc_dir`, `get_image_set_path`, `get_image_path`, `get_annotation_path`, `get_segmentation_mask_path`, `get_output_..._dir`)**:
    - Provide consistent methods for constructing absolute paths to specific files or directories within the VOC dataset structure (`VOCdevkit` or the processed output structure) based on root paths, year, and tag.
- **`read_image_ids(imageset_path, ...)`:**
    - Reads image IDs from a standard VOC ImageSet `.txt` file.
    - Handles files where lines might contain extra columns (common in VOC sets).
    - Includes optional random sampling logic (used by `voc2yolo_images.py`).

---

## `voc2yolo_images.py` - Image Copying Script

This script copies the original JPEG images from the `VOCdevkit` structure into the project's standardized `images/` directory.

### Purpose

To gather all necessary images (e.g., for train/val splits across different years) into a single top-level directory structure (`images/<tag><year>/`) expected by training frameworks like Ultralytics and the project's visualization tools.

### Usage

```bash
python -m src.utils.data_converter.voc2yolo_images \
    --years <YEARS> \
    --tags <TAGS> \
    [--voc-root /path/to/VOC] \
    [--output-root /path/to/output] \
    [--sample-count N]
```

- `--years`: Comma-separated years (e.g., `2007,2012`).
- `--tags`: Comma-separated tags (e.g., `train,val`). Uses `ImageSets/Main/<tag>.txt` to determine which images belong to the split.
- `--voc-root`: Path to VOC dataset root (contains `VOCdevkit`). Defaults to `$VOC_ROOT`.
- `--output-root`: Root for saving the `images/` structure. Defaults to `--voc-root`.
- `--sample-count`: (Optional) Randomly sample N images *total* across all specified splits.

### Logic

1.  Parses arguments.
2.  Determines VOC root and output root paths.
3.  Reads image IDs for the specified year/tag combinations using `read_image_ids` from `voc2yolo_utils` (reading from `ImageSets/Main/<tag>.txt`).
4.  Optionally applies random sampling across the collected IDs.
5.  For each selected image ID:
    - Constructs the source path (`VOCdevkit/<YEAR>/JPEGImages/<id>.jpg`).
    - Constructs the destination path (`<output_root>/images/<tag><year>/<id>.jpg`).
    - Creates the destination subdirectory (`<output_root>/images/<tag><year>/`) if needed.
    - Checks if the destination file already exists. If so, skips it and increments a counter.
    - If the destination doesn't exist, copies the image using `shutil.copy2` (preserves metadata).
    - Tracks success, skip, and failure counts.
6.  Reports summary statistics (copied, skipped, failed).

### Input

- `VOCdevkit/<YEAR>/JPEGImages/` (Source images)
- `VOCdevkit/<YEAR>/ImageSets/Main/<tag>.txt` (Lists of image IDs for detection splits)

### Output

- Creates images in `<output_root>/images/<tag><year>/` (e.g., `images/train2007/`, `images/val2012/`)

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
    [--skip-difficult]
```

- `--years`, `--tags`, `--voc-root`, `--output-root`: Similar to `voc2yolo_images.py`.
- `--skip-difficult`: (Optional flag) If set, objects marked as 'difficult' in the XML will not be included in the output label files.

### Logic

1.  Parses arguments.
2.  Determines VOC root and output root paths.
3.  Reads image IDs for the specified year/tag combinations (from `ImageSets/Main/<tag>.txt`).
4.  For each image ID:
    - Constructs the path to the corresponding XML file (`VOCdevkit/<YEAR>/Annotations/<id>.xml`).
    - Parses the XML using `parse_voc_xml` from `voc2yolo_utils` to get image dimensions and object list (`name`, `bbox`, `difficult`).
    - Constructs the output label file path (`<output_root>/labels_detect/<tag><year>/<id>.txt`).
    - Creates the output subdirectory if needed.
    - Initializes an empty list for label lines.
    - For each object extracted from the XML:
        - If `--skip-difficult` is set and the object is difficult, continue to the next object.
        - Get the class name and look up its index using `VOC_CLASS_TO_ID`.
        - Get the bounding box `[xmin, ymin, xmax, ymax]`.
        - Convert the box coordinates to YOLO format (`center_x`, `center_y`, `width`, `height`) and normalize them using the image dimensions.
        - Format the YOLO label line: `<class_index> <cx_norm> <cy_norm> <w_norm> <h_norm>`.
        - Add the formatted line to the list.
    - Write all formatted label lines to the output `.txt` file.
5.  Reports summary statistics.

### Input

- `VOCdevkit/<YEAR>/Annotations/<id>.xml` (Source annotations)
- `VOCdevkit/<YEAR>/ImageSets/Main/<tag>.txt` (Lists of image IDs)

### Output

- Creates label files in `<output_root>/labels_detect/<tag><year>/` (e.g., `labels_detect/train2007/000001.txt`)
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
    [--skip-difficult] \
    [--iou-threshold 0.5]
```

- `--years`, `--tags`, `--voc-root`, `--output-root`, `--skip-difficult`: Similar to `voc2yolo_detect_labels.py`.
- `--iou-threshold`: (Optional) The IoU threshold used to match mask instances to XML bounding boxes to determine the class label (default: 0.5).

### Logic

This conversion is more complex because the VOC segmentation masks (`SegmentationObject`) identify object *instances* by pixel value (1, 2, 3...), but don't directly contain class information. The class information is only in the XML linked via bounding boxes.

1.  Parses arguments.
2.  Determines paths.
3.  Reads image IDs (using `ImageSets/Segmentation/<tag>.txt`).
4.  For each image ID:
    - Constructs paths to XML (`Annotations`), segmentation mask (`SegmentationObject`), and optionally the image (`JPEGImages` - needed for dimensions if XML fails).
    - Parses the XML using `parse_voc_xml` to get image dimensions and a list of ground truth objects (`name`, `bbox`, `difficult`). If XML parsing fails, attempt to get dimensions from the image file.
    - Loads the segmentation mask PNG (`cv2.imread` with `IMREAD_GRAYSCALE`).
    - Finds unique non-zero pixel values in the mask; these are the instance IDs.
    - Initializes an empty list for label lines.
    - For each unique `instance_id` found in the mask:
        - Create a binary mask isolating only the pixels belonging to this instance.
        - Find contours (`cv2.findContours`) of this binary mask.
        - Select the largest contour (usually the main object outline).
        - **Crucially: Determine the Class:**
            - Calculate the bounding box of the instance contour.
            - Compare this instance bounding box to all ground truth bounding boxes *of the same instance ID (inferred positionally or via matching - check exact logic)* **Correction**: Compare instance bbox to *all* GT boxes from XML using IoU.
            - Find the GT box from the XML with the highest IoU overlap with the instance box.
            - If the highest IoU is above `--iou-threshold`:
                - Get the class name associated with that best-matching GT box.
                - Get the `difficult` status of that GT box.
            - Else (no match above threshold): Log a warning and skip this instance (cannot determine class).
        - If class was determined and (not `--skip-difficult` or the object is not difficult):
            - Convert the instance contour points to a flat list `[x1, y1, x2, y2, ...]`. Simplify the polygon slightly if needed (`cv2.approxPolyDP`). Ensure minimum number of points (e.g., 3).
            - Normalize the polygon coordinates using image dimensions.
            - Get the class index using `VOC_CLASS_TO_ID`.
            - Format the YOLO label line: `<class_index> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ...`.
            - Add the formatted line to the list.
    - Write all formatted label lines to the output `.txt` file (`<output_root>/labels_segment/<tag><year>/<id>.txt`).
5.  Reports summary statistics.

### Input

- `VOCdevkit/<YEAR>/Annotations/<id>.xml` (For class names, difficult status, GT boxes)
- `VOCdevkit/<YEAR>/SegmentationObject/<id>.png` (Instance masks)
- `VOCdevkit/<YEAR>/ImageSets/Segmentation/<tag>.txt` (Lists of image IDs)

### Output

- Creates label files in `<output_root>/labels_segment/<tag><year>/`
- Format: One object instance per line: `<class_index> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ...` (space-separated, normalized coordinates).