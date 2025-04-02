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
-   `--output-subdir`: Subdirectory within `--output-root` to save images (default: `visual_detect`). Output structure will be `<output_root>/<output_subdir>/<tag><year>/<image_id>_voc_detect.png`.
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

This script reads original Pascal VOC instance segmentation mask files (`SegmentationObject/*.png`) and overlays the masks onto the corresponding JPEG images from the `VOCdevkit`. It also attempts to associate each instance mask with its object class name by referencing the corresponding XML annotation file (`Annotations/*.xml`) using Intersection over Union (IoU) matching.

### Purpose

To visually verify the original VOC instance segmentation masks and check the alignment between instance masks and object annotations from the XML files. Allows inspection of individual images or batch processing for generating visual samples.

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
    [--seed 42]
```

-   `--year`, `--tag`: Specify the VOC dataset year(s) and split tag(s) (e.g., '2007', 'train,val'). Uses `ImageSets/Segmentation/<tag>.txt` to find relevant image IDs.
-   `--image-id`: Visualize a single specific image ID. Enables interactive display mode.
-   `--sample-count`: Randomly sample N images from the specified splits for batch processing. Batch mode saves images.
-   `--voc-root`: Path to the root directory containing `VOCdevkit`. Defaults to `$VOC_ROOT`.
-   `--output-root`: Root directory for saving visualizations (batch mode). Defaults to the `VOCdevkit` directory within `--voc-root`.
-   `--output-subdir`: Subdirectory within `--output-root` to save images (default: `visual_segment`). Structure: `<output_root>/<output_subdir>/<tag><year>/<image_id>_voc_segment.png`.
-   `--percentiles`: (Optional) Calculate and report statistics on instance counts per image using these percentiles. Reports averages if not specified.
-   `--seed`: Random seed for sampling.

### Logic

1.  Parses arguments and sets up paths.
2.  Determines the list of target image IDs based on `--image-id` or `--year`/`--tag` (reading from `ImageSets/Segmentation/<tag>.txt`). Applies sampling.
3.  Iterates through target image IDs.
4.  For each ID:
    -   Constructs paths to the JPEG image (`JPEGImages`), segmentation mask (`SegmentationObject`), and XML annotation (`Annotations`).
    -   Loads the image and the grayscale segmentation mask using OpenCV.
    -   Loads and parses the XML annotation using `voc2yolo_utils.parse_voc_xml`.
    -   Finds unique non-zero pixel values (instance IDs) in the mask.
    -   For each instance ID:
        -   Creates a binary mask for the specific instance.
        -   Calculates the bounding box `[xmin, ymin, xmax, ymax]` of the binary mask.
        -   Compares this bounding box to all object bounding boxes from the parsed XML using `common.iou.calculate_iou`.
        -   Identifies the XML object with the highest IoU, if the IoU is >= 0.5.
        -   Retrieves the class name from the matched XML object (defaults to "Unknown" if no match or XML is missing/invalid).
        -   Formats a label as `{class_name}.{instance_id}` (e.g., `car.1`).
        -   Overlays the binary mask onto a copy of the image using `common.image_annotate.overlay_mask` with the generated label, a unique color per instance ID, and an alpha value of `0.3` for transparency.
    -   If in single-image mode, displays the image.
    -   If in batch mode, saves the visualized image.
5.  Reports summary statistics about processed images and instance counts.

### Input

-   `VOCdevkit/<YEAR>/JPEGImages/<id>.jpg`
-   `VOCdevkit/<YEAR>/SegmentationObject/<id>.png`
-   `VOCdevkit/<YEAR>/Annotations/<id>.xml` (Optional, for class names)
-   `VOCdevkit/<YEAR>/ImageSets/Segmentation/<tag>.txt`

### Output

-   Displays image via OpenCV window (single image mode).
-   Saves visualized images to `<output_root>/<output_subdir>/<tag><year>/` (batch mode).
-   Logs processing and instance statistics to the console.

---

## `yolo_detect_viz.py` - Visualize YOLO Detection Labels

This script reads YOLO-formatted detection label files (`labels_detect/<tag><year>/*.txt`) and draws the corresponding bounding boxes and class labels onto the associated images (`images/<tag><year>/*.jpg`).

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
    [--output-subdir visual_detect] \
    [--percentiles 0.25,0.5,0.75] \
    [--seed 42]
```

-   `--year`, `--tag`: Specify the dataset year(s) and split tag(s) (e.g., '2012', 'train,val'). Used to locate the `images/<tag><year>/` and `labels_detect/<tag><year>/` directories.
-   `--image-id`: Visualize a single specific image ID. Enables interactive display mode.
-   `--sample-count`: Randomly sample N images from the specified splits for batch processing. Batch mode saves images instead of displaying them.
-   `--voc-root`: Path to the root directory containing `images/` and `labels_detect/`. Defaults to `$VOC_ROOT` environment variable.
-   `--output-root`: Root directory where visualizations will be saved (batch mode). Defaults to `--voc-root`.
-   `--output-subdir`: Subdirectory within `--output-root` to save images (default: `visual_detect`). Output structure: `<output_root>/<output_subdir>/<tag><year>/<image_id>.png`.
-   `--percentiles`: (Optional) Calculate and report statistics (box count, class count per image) using these percentiles. Reports averages if not specified.
-   `--seed`: Random seed for sampling.

### Logic

1.  Parses arguments and sets up paths (input `images/`, `labels_detect/`, output `visual_detect/`).
2.  Determines the list of target image IDs based on `--image-id` or by scanning `labels_detect/<tag><year>/` for `.txt` files and checking for corresponding `.jpg` files in `images/<tag><year>/`. Applies sampling if requested.
3.  Iterates through the target image IDs.
4.  For each ID:
    -   Constructs paths to the JPEG image (`images/...`) and YOLO label file (`labels_detect/...`).
    -   Loads the image using OpenCV to get dimensions.
    -   Parses the YOLO label file using `parse_yolo_detection_label`:
        - Reads each line (`<class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>`).
        - Denormalizes coordinates using image dimensions.
        - Converts coordinates to `[xmin, ymin, xmax, ymax]` pixel format.
        - Looks up class name using `VOC_CLASSES`.
    -   Draws bounding boxes and labels onto a copy of the image using `common.image_annotate.draw_box` and `get_color`.
    -   If in single-image mode (`--image-id`), displays the image using `cv2.imshow`.
    -   If in batch mode, saves the visualized image to `<output_root>/visual_detect/<tag><year>/<image_id>.png`.
5.  Reports summary statistics about processed images and annotation counts (average or percentiles).

### Input

-   `<voc_root>/images/<tag><year>/<id>.jpg`
-   `<voc_root>/labels_detect/<tag><year>/<id>.txt`

### Output

-   Displays image via OpenCV window (single image mode).
-   Saves visualized images to `<output_root>/visual_detect/<tag><year>/` (batch mode).
-   Logs processing and annotation statistics to the console.

---

## Future Visualization Scripts (Planned)

The following visualization scripts are planned based on the [VOC Dataset Conversion TODO](./../dataset/voc/TODO.md):

1.  **`vocdev_segment_viz.py`:**
    -   **Purpose:** Visualize the original VOC instance segmentation masks (`SegmentationObject`) overlaid on the images. This script would read the `.png` mask files and potentially the XML for context (though masks themselves don't store class names directly in VOC). It would help verify the raw segmentation data.
    -   **Status:** Implemented.

2.  **`yolo_detect_viz.py`:**
    -   **Purpose:** Visualize the generated YOLO *detection* labels (`labels_detect/`). This script would read the `.txt` files containing `<class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>` lines, denormalize the coordinates based on the corresponding image dimensions, and draw the resulting bounding boxes and class labels onto the images stored in the project's `images/` directory. This is crucial for verifying the correctness of the `voc2yolo_detect_labels.py` conversion.
    -   **Status:** Implemented.

3.  **`yolo_segment_viz.py`:**
    -   **Purpose:** Visualize the generated YOLO *segmentation* labels (`labels_segment/`). This script would read the `.txt` files containing `<class_id> <x1_norm> <y1_norm> ...` lines, denormalize the polygon points based on image dimensions, and draw the resulting filled polygons and class labels onto the images stored in the project's `images/` directory. This is essential for verifying the `voc2yolo_segment_labels.py` conversion.
    -   **Status:** Not yet implemented.