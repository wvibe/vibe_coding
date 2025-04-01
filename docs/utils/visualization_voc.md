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

## Future Visualization Scripts (Planned)

The following visualization scripts are planned based on the [VOC Dataset Conversion TODO](./../dataset/voc/TODO.md):

1.  **`vocdev_segment_viz.py`:**
    -   **Purpose:** Visualize the original VOC instance segmentation masks (`SegmentationObject`) overlaid on the images. This script would read the `.png` mask files and potentially the XML for context (though masks themselves don't store class names directly in VOC). It would help verify the raw segmentation data.
    -   **Status:** Not yet implemented.

2.  **`yolo_detect_viz.py`:**
    -   **Purpose:** Visualize the generated YOLO *detection* labels (`labels_detect/`). This script would read the `.txt` files containing `<class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>` lines, denormalize the coordinates based on the corresponding image dimensions, and draw the resulting bounding boxes and class labels onto the images stored in the project's `images/` directory. This is crucial for verifying the correctness of the `voc2yolo_detect_labels.py` conversion.
    -   **Status:** Not yet implemented.

3.  **`yolo_segment_viz.py`:**
    -   **Purpose:** Visualize the generated YOLO *segmentation* labels (`labels_segment/`). This script would read the `.txt` files containing `<class_id> <x1_norm> <y1_norm> ...` lines, denormalize the polygon points based on image dimensions, and draw the resulting filled polygons and class labels onto the images stored in the project's `images/` directory. This is essential for verifying the `voc2yolo_segment_labels.py` conversion.
    -   **Status:** Not yet implemented.