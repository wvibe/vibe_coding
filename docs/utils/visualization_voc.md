# VOC Visualization Utilities Design Notes

This document describes the visualization scripts used to inspect Pascal VOC dataset annotations and the corresponding YOLO-formatted labels.

---

## `vocdev_detect_viz.py` - Visualize VOC Detection Ground Truth

This script reads original Pascal VOC XML annotation files and draws the ground truth bounding boxes and class labels directly onto the corresponding JPEG images from the `VOCdevkit`.

### Purpose

To visually verify the correctness and content of the original VOC detection annotations before or after conversion. Allows inspection of individual images or batch processing to generate visual samples.

### Usage

```bash
python -m src.utils.visualization.vocdev_detect_viz \
    --year <YEARS> \
    --tag <TAGS> \
    [--image-id <ID>] \
    [--sample-count N] \
    [--voc-root /path/to/VOC] \
    [--output-root /path/to/output] \
    [--output-subdir visual_detect] \
    [--show-difficult] \
    [--percentiles 0.25,0.5,0.75] \
    [--seed 42]
```

-   `--year`, `--tag`: Specify the VOC dataset year(s) and split tag(s) (e.g., '2007', 'train,val'). Uses `ImageSets/Main/<tag>.txt`.
-   `--image-id`: Visualize a single specific image ID. This enables interactive display mode (shows the image in a window).
-   `--sample-count`: Randomly sample N images from the specified splits for batch processing. If not set or <= 0, processes all images in the splits. Batch mode saves images instead of displaying them.
-   `--voc-root`: Path to the root directory containing `VOCdevkit`. Defaults to `$VOC_ROOT` environment variable.
-   `--output-root`: Root directory where visualizations will be saved (in batch mode). Defaults to `--voc-root`.
-   `--output-subdir`: Subdirectory within `--output-root` to save images (default: `visual_detect`). Output structure will be `<output_root>/<output_subdir>/<tag><year>/<image_id>.png`.
-   `--show-difficult`: If set, adds a '*' marker to the label of objects marked as 'difficult' in the XML.
-   `--percentiles`: (Optional) Calculate and report statistics (box count, class count, difficult count per image) using these percentiles (e.g., '0.5,0.9'). Reports averages if not specified.
-   `--seed`: Random seed for sampling.

### Logic

1.  Parses arguments and sets up paths.
2.  Determines the list of target image IDs based on `--image-id` or `--year`/`--tag` (reading from `ImageSets/Main/<tag>.txt`). Applies sampling if requested.
3.  Iterates through the target image IDs.
4.  For each ID:
    -   Constructs paths to the JPEG image (`JPEGImages`) and XML annotation (`Annotations`).
    -   Loads the image using OpenCV.
    -   Parses the XML using `voc2yolo_utils.parse_voc_xml` to get object details (class name, bounding box, difficult flag).
    -   Draws bounding boxes and labels onto a copy of the image using `common.image_annotate.draw_box`. Colors are assigned based on `VOC_CLASSES`. Difficult markers are added if requested.
    -   If in single-image mode (`--image-id`), displays the image using `cv2.imshow`.
    -   If in batch mode, saves the visualized image to the specified output directory.
5.  Reports summary statistics about processed images and annotation counts (average or percentiles).

### Input

-   `VOCdevkit/<YEAR>/JPEGImages/<id>.jpg`
-   `VOCdevkit/<YEAR>/Annotations/<id>.xml`
-   `VOCdevkit/<YEAR>/ImageSets/Main/<tag>.txt`

### Output

-   Displays image via OpenCV window (single image mode).
-   Saves visualized images to `<output_root>/<output_subdir>/<tag><year>/` (batch mode).
-   Logs processing and annotation statistics to the console.

---

## `vocdev_segment_viz.py` - Visualize VOC Segmentation Ground Truth

This script reads original Pascal VOC instance segmentation mask files (`SegmentationObject/*.png`) and overlays the masks onto the corresponding JPEG images from the `VOCdevkit`. It determines the object class name for each instance by looking up the corresponding pixels in the class segmentation mask (`SegmentationClass/*.png`).

### Purpose

To visually verify the original VOC instance segmentation masks and confirm the class label associated with each instance using the corresponding class segmentation map. Allows inspection of individual images or batch processing for generating visual samples. Also includes analysis of class mask value distributions.

### Usage

```bash
python -m src.utils.visualization.vocdev_segment_viz \
    --year <YEARS> \
    --tag <TAGS> \
    [--image-id <ID>] \
    [--sample-count N] \
    [--voc-root /path/to/VOC] \
    [--output-root /path/to/output] \
    [--output-subdir visual_segment] \
    [--percentiles 0.25,0.5,0.75] \
    [--show-difficult] \
    [--seed 42]
```

-   `--year`, `--tag`: Specify the VOC dataset year(s) and split tag(s) (e.g., '2007', 'train,val'). Uses `ImageSets/Segmentation/<tag>.txt` to find relevant image IDs.
-   `--image-id`: Visualize a single specific image ID. Enables interactive display mode.
-   `--sample-count`: Randomly sample N images from the specified splits for batch processing. Batch mode saves images.
-   `--voc-root`: Path to the root directory containing `VOCdevkit`. Defaults to `$VOC_ROOT`.
-   `--output-root`: Root directory for saving visualizations (batch mode). Defaults to the `VOCdevkit` directory within `--voc-root`.
-   `--output-subdir`: Subdirectory within `--output-root` to save images (default: `visual_segment`). Structure: `<output_root>/<output_subdir>/<tag><year>/<image_id>.png`.
-   `--percentiles`: (Optional) Calculate and report statistics on instance counts per image using these percentiles. Reports averages if not specified.
-   `--show-difficult`: (Optional) Visualize boundaries/difficult regions (value 255) with 'Unk' label. In VOC segmentation, 255 typically marks object boundaries or difficult-to-segment pixels.
-   `--seed`: Random seed for sampling.

### Logic

1.  Parses arguments and sets up paths.
2.  Determines the list of target image IDs based on `--image-id` or `--year`/`--tag` (reading from `ImageSets/Segmentation/<tag>.txt`). Applies sampling.
3.  Initializes a dictionary to track the distribution of class mask values across images.
4.  Iterates through target image IDs.
5.  For each ID:
    -   Constructs paths to the JPEG image (`JPEGImages`), the instance segmentation mask (`SegmentationObject`), and the class segmentation mask (`SegmentationClass`).
    -   Loads the image using OpenCV.
    -   Loads both segmentation masks (`.png` files) using `PIL.Image.open` and converts to NumPy arrays. This is crucial for correctly reading palette-based PNGs used in VOC. If loading fails for either mask, an error is logged, and subsequent steps requiring that mask might be skipped.
    -   Analyzes the loaded class mask to update the `class_mask_value_distribution` statistics.
    -   Finds unique non-zero pixel values (instance IDs) in the instance mask. By default, value 255 (boundary/difficult) is excluded unless `--show-difficult` is used.
    -   For each valid instance ID:
        -   Creates a binary mask for the specific instance using the instance mask (`SegmentationObject`).
        -   If the instance ID is 255 and `--show-difficult` is enabled, uses the label "Unk.255".
        -   Otherwise (for instance IDs 1-N):
            -   Identifies the pixels belonging to this instance in the instance mask.
            -   Looks up the values of these corresponding pixels in the class mask (`SegmentationClass`).
            -   Determines the most frequent valid class ID (ignoring 0 for background and 255 for void/boundary) among these pixels using `scipy.stats.mode`.
            -   Maps the determined class ID to a class name using `VOC_CLASSES` (defaults to "Unknown" if no valid class ID found or mapping fails).
            -   Formats a label as `{class_name}.{instance_id}` (e.g., `car.1`).
        -   Overlays the binary *instance* mask onto a copy of the image using `common.image_annotate.overlay_mask` with the generated label, a unique color per instance ID, and an alpha value of `0.3` for transparency.
    -   If in single-image mode, displays the image.
    -   If in batch mode, saves the visualized image.
6.  Reports summary statistics about processed images, instance counts (average or percentiles), and the distribution of class mask values (counts for values 1-20, warning for values 21-254).

### Input

-   `VOCdevkit/<YEAR>/JPEGImages/<id>.jpg`
-   `VOCdevkit/<YEAR>/SegmentationObject/<id>.png`
-   `VOCdevkit/<YEAR>/SegmentationClass/<id>.png` # Used for determining class names and distribution stats
-   `VOCdevkit/<YEAR>/ImageSets/Segmentation/<tag>.txt`

### Output

-   Displays image via OpenCV window (single image mode).
-   Saves visualized images to `<output_root>/<output_subdir>/<tag><year>/` (batch mode).
-   Logs processing, instance statistics, and class mask value distribution statistics to the console.

### Future Considerations / TODO

-   The `main` and `report_statistics` functions currently have slightly high complexity (`C901` lint warning). They could optionally be refactored into smaller helper functions in the future for improved readability and maintainability.

### Implementation Notes

- Functions have been refactored to reduce complexity warnings:
  - `main()` function was simplified by extracting the image processing loop into `_process_images()`
  - `report_statistics()` was broken down into helper functions `_report_instance_statistics()` and `_report_class_distributions()`
  - A common `_initialize_stats()` function was added to centralize statistics tracking structure

---

## `yolo_detect_viz.py` - Visualize YOLO Detection Labels

This script reads YOLO-formatted detection label files (`detect/labels/<tag><year>/*.txt`) and draws the corresponding bounding boxes and class labels onto the associated images (`detect/images/<tag><year>/*.jpg`).

### Purpose

To visually verify the correctness of the generated YOLO detection labels after running the `voc2yolo_detect_labels.py` conversion script. Allows inspection of individual images or batch processing to generate visual samples of the YOLO labels.

### Usage

```bash
python -m src.utils.visualization.yolo_detect_viz \
    --year <YEARS> \
    --tag <TAGS> \
    [--image-id <ID>] \
    [--sample-count N] \
    [--voc-root /path/to/VOC] \
    [--output-root /path/to/output] \
    [--output-subdir detect/visual] \
    [--percentiles 0.25,0.5,0.75] \
    [--seed 42]
```

-   `--year`, `--tag`: Specify the dataset year(s) and split tag(s) (e.g., '2012', 'train,val'). Used to locate the `detect/images/<tag><year>/` and `detect/labels/<tag><year>/` directories.
-   `--image-id`: Visualize a single specific image ID. Enables interactive display mode.
-   `--sample-count`: Randomly sample N images from the specified splits for batch processing. Batch mode saves images instead of displaying them.
-   `--voc-root`: Path to the root directory containing `detect/images/` and `detect/labels/`. Defaults to `$VOC_ROOT` environment variable.
-   `--output-root`: Root directory where visualizations will be saved (batch mode). Defaults to `--voc-root`.
-   `--output-subdir`: Subdirectory within `--output-root` to save images (default: `detect/visual`). Output structure: `<output_root>/<output_subdir>/<tag><year>/<image_id>.png`.
-   `--percentiles`: (Optional) Calculate and report statistics (box count, class count per image) using these percentiles. Reports averages if not specified.
-   `--seed`: Random seed for sampling.

### Logic

1.  Parses arguments and sets up paths (input `detect/images/`, `detect/labels/`, output `detect/visual/`).
2.  Determines the list of target image IDs based on `--image-id` or by scanning `detect/labels/<tag><year>/` for `.txt` files and checking for corresponding `.jpg` files in `detect/images/<tag><year>/`. Applies sampling if requested.
3.  Iterates through the target image IDs.
4.  For each ID:
    -   Constructs paths to the JPEG image (`detect/images/...`) and YOLO label file (`detect/labels/...`).
    -   Loads the image using OpenCV to get dimensions.
    -   Parses the YOLO label file using `parse_yolo_detection_label`:
        - Reads each line (`<class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>`).
        - Denormalizes coordinates using image dimensions.
        - Converts coordinates to `[xmin, ymin, xmax, ymax]` pixel format.
        - Looks up class name using `VOC_CLASSES`.
    -   Draws bounding boxes and labels onto a copy of the image using `common.image_annotate.draw_box` and `get_color`.
    -   If in single-image mode (`--image-id`), displays the image using `cv2.imshow`.
    -   If in batch mode, saves the visualized image to `<output_root>/detect/visual/<tag><year>/<image_id>.png`.
5.  Reports summary statistics about processed images and annotation counts (average or percentiles).

### Input

-   `<voc_root>/detect/images/<tag><year>/<id>.jpg`
-   `<voc_root>/detect/labels/<tag><year>/<id>.txt`

### Output

-   Displays image via OpenCV window (single image mode).
-   Saves visualized images to `<output_root>/detect/visual/<tag><year>/` (batch mode).
-   Logs processing and annotation statistics to the console.

---

## `yolo_segment_viz.py` - Visualize YOLO Segmentation Labels

This script reads YOLO-formatted segmentation label files (`segment/labels/<tag><year>/*.txt`) and draws the corresponding polygons and class labels onto the associated images (`segment/images/<tag><year>/*.jpg`).

### Purpose

To visually verify the correctness of the generated YOLO segmentation labels after running the `voc2yolo_segment_labels.py` conversion script. Allows inspection of individual images or batch processing to generate visual samples of the YOLO segmentation polygons.

### Usage

```bash
python -m src.utils.visualization.yolo_segment_viz \
    --years <YEARS> \
    --tags <TAGS> \
    [--image-id <ID>] \
    [--sample-count N] \
    [--voc-root /path/to/VOC] \
    [--output-root /path/to/output] \
    [--output-subdir segment/visual] \
    [--fill-polygons] \
    [--alpha 0.3] \
    [--percentiles 0.25,0.5,0.75] \
    [--seed 42]
```

-   `--years`, `--tags`: Specify the dataset year(s) and split tag(s) (e.g., '2012', 'train,val'). Used to locate the `segment/images/<tag><year>/` and `segment/labels/<tag><year>/` directories.
-   `--image-id`: Visualize a single specific image ID. Enables interactive display mode.
-   `--sample-count`: Randomly sample N images from the specified splits for batch processing. Batch mode saves images instead of displaying them.
-   `--voc-root`: Path to the root directory containing `segment/images/` and `segment/labels/`. Defaults to `$VOC_ROOT` environment variable.
-   `--output-root`: Root directory where visualizations will be saved (batch mode). Defaults to `--voc-root`.
-   `--output-subdir`: Subdirectory within `--output-root` to save images (default: `segment/visual`). Output structure: `<output_root>/<output_subdir>/<tag><year>/<image_id>.png`.
-   `--fill-polygons`: If set, fills polygons with semi-transparent color. Otherwise, only draws polygon outlines.
-   `--alpha`: Alpha (transparency) value for filled polygons when `--fill-polygons` is used. (default: 0.3).
-   `--percentiles`: (Optional) Calculate and report statistics (polygon count, class count, points per polygon) using these percentiles. Reports averages if not specified.
-   `--seed`: Random seed for sampling.

### Logic

1.  Parses arguments and sets up paths (input `segment/images/`, `segment/labels/`, output `segment/visual/`).
2.  Determines the list of target image IDs based on `--image-id` or by scanning `segment/labels/<tag><year>/` for `.txt` files and checking for corresponding `.jpg` files in `segment/images/<tag><year>/`. Applies sampling if requested.
3.  Iterates through the target image IDs.
4.  For each ID:
    -   Constructs paths to the JPEG image (`segment/images/...`) and YOLO label file (`segment/labels/...`).
    -   Loads the image using OpenCV to get dimensions.
    -   Parses the YOLO label file using `parse_yolo_segmentation_label`:
        - Reads each line (`<class_id> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ...`).
        - Validates coordinates are within [0-1] range and form valid polygons (at least 3 points).
        - Converts normalized coordinates to pixel coordinates.
        - Looks up class name using `VOC_CLASSES`.
    -   For each polygon, either:
        - Draws polygon outlines using `draw_polygon` if `--fill-polygons` is not set.
        - Creates a binary mask from the polygon and uses `overlay_mask` with the specified alpha transparency if `--fill-polygons` is set.
    -   If in single-image mode (`--image-id`), displays the image using `cv2.imshow`.
    -   If in batch mode, saves the visualized image to `<output_root>/segment/visual/<tag><year>/<image_id>.png`.
5.  Reports summary statistics about processed images, annotation counts, and polygon point counts (average or percentiles).

### Input

-   `<voc_root>/segment/images/<tag><year>/<id>.jpg`
-   `<voc_root>/segment/labels/<tag><year>/<id>.txt`

### Output

-   Displays image via OpenCV window (single image mode).
-   Saves visualized images to `<output_root>/segment/visual/<tag><year>/` (batch mode).
-   Logs processing and annotation statistics to the console.

### Testing

The script is thoroughly tested in `tests/utils/visualization/test_yolo_segment_viz.py` with unit tests covering:

1. Path handling and directory setup
2. Target image list building (both single image and batch modes)
3. YOLO segmentation label parsing (including handling of invalid coordinates, invalid class IDs, and polygons with too few points)
4. Image processing and visualization (both outline and filled polygon modes)
5. Statistics reporting (both percentiles and averages)

Tests use mock objects to avoid actual filesystem operations and validate edge cases without requiring actual images or labels.