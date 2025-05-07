# Common Utilities Design Notes

This document outlines the design choices and rationale behind the functions in the `vibelab.utils.common/` directory, which contain general-purpose utilities used across different parts of the project.

---

## `bbox.py` - Bounding Box Utilities

This module combines bounding box format conversion and Intersection over Union (IoU) calculation functions, providing a comprehensive set of utilities for working with bounding boxes.

### Format Conversion Functions

#### `xywh_to_xyxy(box)`

- **Purpose:** Converts a box from center coordinates (`[center_x, center_y, width, height]`) to corner coordinates (`[x_min, y_min, x_max, y_max]`).
- **Logic:** Uses basic arithmetic: `x_min = cx - w/2`, `y_min = cy - h/2`, `x_max = cx + w/2`, `y_max = cy + h/2`.
- **Input/Output:** Can handle single boxes (list/tuple) or multiple boxes (NumPy array `[N, 4]` or PyTorch tensor).

#### `xyxy_to_xywh(box)`

- **Purpose:** Converts a box from corner coordinates (`[x_min, y_min, x_max, y_max]`) to center coordinates (`[center_x, center_y, width, height]`).
- **Logic:** Uses basic arithmetic: `width = x_max - x_min`, `height = y_max - y_min`, `center_x = x_min + width/2`, `center_y = y_min + height/2`.
- **Input/Output:** Can handle single boxes (list/tuple) or multiple boxes (NumPy array `[N, 4]` or PyTorch tensor).

#### `normalize_boxes(boxes, image_width, image_height)`

- **Purpose:** Normalizes pixel coordinates (usually `[x_min, y_min, x_max, y_max]`) to be relative to image dimensions (range 0.0 to 1.0).
- **Logic:** Divides x-coordinates by `image_width` and y-coordinates by `image_height`.
- **Input:** A single box or a NumPy array/PyTorch tensor `[N, 4]` of boxes, plus image dimensions.
- **Output:** Box(es) with coordinates normalized to the [0, 1] range.

#### `denormalize_boxes(boxes, image_width, image_height)`

- **Purpose:** Converts normalized coordinates (range 0.0 to 1.0) back to absolute pixel coordinates.
- **Logic:** Multiplies normalized x-coordinates by `image_width` and y-coordinates by `image_height`.
- **Input:** A single box or a NumPy array/PyTorch tensor `[N, 4]` of normalized boxes, plus image dimensions.
- **Output:** Box(es) with absolute pixel coordinates.

### IoU Calculation Functions

#### `calculate_iou(box1, box2)`

- **Purpose:** Calculates the IoU between two individual bounding boxes.
- **Input:** Takes two boxes, `box1` and `box2`, expected in the format `[x_min, y_min, x_max, y_max]`.
- **Logic:**
    1.  Calculates the coordinates of the intersection rectangle (`inter_x_min`, `inter_y_min`, `inter_x_max`, `inter_y_max`).
    2.  Computes the area of the intersection. If `inter_x_max <= inter_x_min` or `inter_y_max <= inter_y_min`, the intersection area is 0.
    3.  Calculates the area of each individual box (`box1_area`, `box2_area`).
    4.  Computes the union area: `union_area = box1_area + box2_area - intersection_area`.
    5.  Calculates IoU: `iou = intersection_area / union_area`.
    6.  Handles division by zero (if `union_area` is 0) by returning 0.0.
- **Output:** Returns the IoU score as a float between 0.0 and 1.0.

#### `calculate_iou_matrix(boxes1, boxes2)`

- **Purpose:** Efficiently calculates the pairwise IoU between two sets of bounding boxes.
- **Input:** Takes two sets of boxes, `boxes1` (shape `[N, 4]`) and `boxes2` (shape `[M, 4]`), as NumPy arrays. Boxes are expected in `[x_min, y_min, x_max, y_max]` format.
- **Logic:**
    1.  Uses vectorized NumPy operations for efficiency.
    2.  Expands dimensions of the input arrays to enable broadcasting.
    3.  Calculates the intersection coordinates for all pairs simultaneously.
    4.  Calculates intersection areas for all pairs.
    5.  Calculates individual box areas for `boxes1` and `boxes2`.
    6.  Calculates union areas for all pairs.
    7.  Computes the IoU matrix by dividing the intersection area matrix by the union area matrix.
    8.  Handles potential division by zero.
- **Output:** Returns an `[N, M]` NumPy array where `matrix[i, j]` is the IoU between `boxes1[i]` and `boxes2[j]`.

---

## `image_annotate.py` - Image Annotation Utilities

This module provides tools for drawing annotations like bounding boxes, polygons, masks, and labels onto images.

### Core Concepts

- **Inplace Modification:** All drawing functions (`draw_box`, `draw_polygon`, `overlay_mask`) modify the input image array *inplace* for efficiency.
- **Pixel Coordinates:** Drawing functions expect coordinates (boxes, points) in absolute pixel values relative to the image dimensions.
- **Label Flexibility:** Handles various label formats including simple strings, instance IDs, class names, and confidence scores.
- **Color Consistency:** Provides utilities to generate consistent colors based on class or instance IDs, or random colors.
- **Text Handling:** Includes logic for label formatting, score display, and text truncation to avoid overflow.

### Data Classes

- **`LabelInfo`**: Container for label text + optional score + optional max_length for truncation. Formats as `"text (score)"` with truncation applied.
- **`InstanceLabel`**: Container for instance ID + optional class name. Formats as `"Inst <id> (<class_name>)"` or `"Inst <id>"`.

### Helper Functions

- **`get_color(idx=None)`**: Returns a consistent BGR color from a palette if `idx` is given, otherwise a random color.
- **`get_color_map(num_classes)`**: Generates a `dict` mapping class indices `0..N-1` to consistent colors.
- **`yolo_to_pixel_coords(...)`**: Converts normalized YOLO coords (box `[cx,cy,w,h]` or polygon `[x1,y1,...]`) to pixel coordinates.
- **`get_text_size(...)`**: Calculates pixel width/height of text using `cv2.getTextSize`.
- **`format_label(...)`**: Central function to format label strings. Takes `label` (str, LabelInfo, InstanceLabel, dict), optional `score`, optional `max_length`. Handles precedence (explicit args > internal values) and applies truncation/score formatting.
- **`_format_label_from_dict(...)`**: Private helper for `format_label` to parse dict inputs.
- **`_draw_label_near_point(...)`**: Private helper to draw formatted text with background near an anchor point, handling offscreen adjustments.

### Drawing Functions

- **`draw_box(image, box, label, ...)`**: Draws a box (`[x1,y1,x2,y2]` pixels) and label. Determines color, draws rect, formats/truncates label (using `format_label` and `max_label_width_ratio`), draws label background and text (handling placement).
- **`draw_polygon(image, points, label, ...)`**: Draws a polygon outline (`[(x,y),...]` pixels) and label. Determines color, draws polyline, formats/truncates label, calls `_draw_label_near_point` to draw the label near the first point.
- **`overlay_mask(image, mask, label=None, ...)`**: Overlays a semi-transparent colored mask (binary 0/1 or 0/255 array). Determines color, blends mask using `cv2.addWeighted`, optionally formats/truncates label, finds anchor point (top of largest contour), calls `_draw_label_near_point` to draw label.

---

## `mask.py` - Mask Operations and Utilities

This module provides utility functions for mask operations, especially for working with polygons, contours, and binary masks. It's particularly useful for converting binary masks to polygon representations suitable for YOLO format data and calculating IoU between masks.

### Key Constants

- `DEFAULT_MIN_CONTOUR_AREA`: Minimum area (in pixels) for a contour to be considered valid (default: 1.0).
- `DEFAULT_POLYGON_APPROX_TOLERANCE`: Default epsilon value for polygon approximation relative to arc length (default: 0.01).

### Main Public Functions

#### `mask_to_yolo_polygons(binary_mask, img_shape, connect_parts=False, min_contour_area=DEFAULT_MIN_CONTOUR_AREA, polygon_approx_tolerance=DEFAULT_POLYGON_APPROX_TOLERANCE)`

- **Purpose:** Converts a binary mask to YOLO polygon format.
- **Input:**
  - `binary_mask`: Binary mask array (0/1 or 0/255 values)
  - `img_shape`: Image dimensions (height, width)
  - `connect_parts`: Whether to try connecting separated mask parts/fragments into a single polygon
  - `min_contour_area`: Minimum area for a contour to be considered (filters tiny noise)
  - `polygon_approx_tolerance`: Controls how much to simplify contours
- **Logic:**
  1. Finds and simplifies contours in the binary mask using OpenCV
  2. If `connect_parts` is True, attempts to stitch multiple contours into a single polygon
  3. Normalizes pixel coordinates to [0,1] range and flattens to YOLO format
- **Output:** Tuple containing:
  - List of polygons in YOLO format (`[x1, y1, x2, y2, ...]`), each polygon normalized to [0,1] range
  - Error string or None if successful

#### `mask_to_yolo_polygons_verified(binary_mask, img_shape, iou_threshold=0.95, min_contour_area=0.0, polygon_approx_tolerance=0.0)`

- **Purpose:** Converts a binary mask to YOLO polygons and verifies the accuracy through IoU checking.
- **Input:**
  - `binary_mask`: Binary mask array (0/1 or 0/255 values)
  - `img_shape`: Image dimensions (height, width)
  - `iou_threshold`: Minimum IoU required between original and reconstructed mask (default: 0.95)
  - `min_contour_area`: Minimum contour area to keep (default: 0.0, no filtering)
  - `polygon_approx_tolerance`: Controls simplification (default: 0.0, no simplification)
- **Logic:**
  1. Performs the conversion to YOLO polygons
  2. Reconstructs a mask from the polygons
  3. Calculates IoU between original and reconstructed mask
  4. Only returns the polygons if IoU meets the threshold
- **Output:** Tuple containing:
  - List of polygons in YOLO format if IoU threshold is met, otherwise empty list
  - Error string or None if successful

#### `polygons_to_mask(polygons, img_shape, normalized=False)`

- **Purpose:** Convert polygons to a binary mask. Unified function that handles different polygon formats.
- **Input:**
  - `polygons`: List of polygons in one of three formats:
    1. List of normalized YOLO coordinates [x1,y1,x2,y2,...] (if normalized=True)
    2. List of numpy arrays with pixel coordinates (shape (N,1,2) or (N,2))
    3. Single polygon as List of (x,y) pixel coordinate tuples
  - `img_shape`: Dimensions of the target mask (height, width)
  - `normalized`: Whether the coordinates are normalized [0.0-1.0] (YOLO format) or already in pixel coordinates
- **Logic:** Uses OpenCV's `fillPoly` to create a binary mask from the polygon points, handling different input formats
- **Output:** Boolean numpy array where True indicates pixels inside the polygon(s)

#### `calculate_mask_iou(mask1, mask2)`

- **Purpose:** Calculate the IoU (Intersection over Union) between two binary masks.
- **Input:** Two boolean numpy arrays of the same shape
- **Logic:** Computes logical AND for intersection and logical OR for union, then calculates the ratio
- **Output:** IoU value between 0.0 and 1.0

### Helper Functions

#### `_preprocess_binary_mask(binary_mask, img_shape)`

- **Purpose:** Validate and preprocess a binary mask for contour finding.
- **Logic:** Checks for valid dimensions, handles different data types, and normalizes to uint8 with values 0/255.
- **Error Handling:** Raises ValueError for invalid image shape or mask dimensions
- **Output:** Preprocessed uint8 mask with values 0/255

#### `_find_and_simplify_contours(binary_mask, min_contour_area, polygon_approx_tolerance)`

- **Purpose:** Extract, filter, and simplify contours from a binary mask.
- **Logic:** Uses OpenCV's `findContours` and `approxPolyDP` to extract and simplify contours, filtering out small contours.
- **Output:** List of simplified contour arrays, each suitable for polygon representation.

#### `_connect_contours_stitched(simplified_contours)`

- **Purpose:** Implements a sophisticated algorithm to stitch multiple separate contours into a single coherent polygon.
- **Logic:**
  1. Calculates centroids of contours and sorts them top-to-bottom, then left-to-right
  2. Identifies closest connection points between consecutive contours
  3. Traverses each contour from entry to exit point, building a connected path
  4. Creates bridge points to connect contours
- **Output:** Tuple containing:
  - A single numpy array containing the vertices of the stitched polygon, or None if stitching fails
  - Error string explaining failure reason, or None if successful

#### `_normalize_and_flatten_polygons(polygons_pixels, img_shape)`

- **Purpose:** Convert pixel-coordinate polygons to normalized YOLO format.
- **Logic:**
  1. Clamps coordinates to image bounds
  2. Normalizes coordinates to [0,1] range based on image dimensions
  3. Flattens coordinates to YOLO format `[x1, y1, x2, y2, ...]`
  4. Performs validity checks (at least 3 points after processing)
- **Error Handling:** Raises ValueError for invalid image shape
- **Output:** List of polygon coordinates in flattened YOLO format, normalized to [0,1] range

---

## `label_match.py` - Instance Matching Utilities

This module provides algorithms for matching instances between datasets based on IoU (Intersection over Union) metrics. It implements both an optimal matching algorithm using the Hungarian method and a greedy matching approach.

### Constants

- `_INVALID_IOU_PLACEHOLDER`: A large negative value (-1e6) used to mark invalid/below-threshold pairs in the IoU matrix.

### Main Public Function

#### `match_instances(dataset_a, dataset_b, compute_iou_fn, iou_cutoff=0.5, use_hungarian=True)`

- **Purpose:** Match instances between two datasets using either Hungarian (optimal) or Greedy algorithm.
- **Input:**
  - `dataset_a`, `dataset_b`: Lists of objects to match (e.g., bounding boxes, masks)
  - `compute_iou_fn`: Function that computes IoU between two objects (returns float in range [0, 1])
  - `iou_cutoff`: Minimum IoU value to consider a valid match (default: 0.5)
  - `use_hungarian`: If True, use Hungarian algorithm (optimal), otherwise use Greedy (default: True)
- **Logic:**
  1. Builds an IoU matrix between all pairs from dataset_a and dataset_b
  2. Applies the selected matching algorithm (Hungarian or Greedy)
  3. Returns matched pairs and unmatched indices
- **Output:** Tuple containing:
  - List of matched pairs as (index_a, index_b) tuples
  - List of unmatched indices from dataset_a
  - List of unmatched indices from dataset_b

### Helper Functions

#### `_build_iou_matrix(dataset_a, dataset_b, compute_iou_fn, iou_cutoff)`

- **Purpose:** Constructs an IoU matrix between all pairs of objects from the two datasets.
- **Logic:** Computes IoU for each pair, keeping values that meet the cutoff and replacing others with `_INVALID_IOU_PLACEHOLDER`.
- **Output:** NumPy array of shape (len(dataset_a), len(dataset_b)) containing IoU values.

#### `_match_instances_hungarian(iou_matrix, iou_cutoff)`

- **Purpose:** Finds optimal matching using the Hungarian algorithm.
- **Logic:**
  1. Pads the matrix with dummy rows/columns to handle non-square matrices
  2. Uses SciPy's `linear_sum_assignment` with `maximize=True` to find optimal assignment
  3. Filters out matches involving dummy rows/columns and pairs below the IoU cutoff
- **Output:** Tuple containing matched pairs, mask for matched rows, and mask for matched columns.

#### `_match_instances_greedy(iou_matrix, iou_cutoff)`

- **Purpose:** Performs greedy matching based on highest IoU value first.
- **Logic:**
  1. Creates a flattened list of valid (row, column, iou) triples
  2. Sorts in descending order of IoU
  3. Greedily assigns matches, skipping already matched rows and columns
- **Output:** Tuple containing matched pairs, mask for matched rows, and mask for matched columns.

---

## `stats.py` - Statistical Calculation and Formatting Utilities

This module provides functions for calculating and displaying summary statistics from numerical data. It's designed to be easy to use for creating statistical tables from datasets with consistent formatting.

### Constants

- `DEFAULT_NA_VALUE`: The default value to display when a statistic cannot be computed (default: "N/A").

### Main Public Functions

#### `calculate_numeric_summary(values, metrics)`

- **Purpose:** Calculates requested summary statistics for a list of numeric values, handling None/NaN values gracefully.
- **Input:**
  - `values`: A list of numbers (int/float) that may contain None or NaN values
  - `metrics`: A list of strings specifying the statistics to calculate (e.g., "count", "mean", "p50", "max")
- **Logic:**
  1. Filters out None and NaN values from the input list
  2. For empty lists, returns 0 for "count" and None for all other metrics
  3. Uses NumPy for efficient statistical calculations
  4. Handles different metric types: basic stats (mean, min, max), custom (count), percentiles (p50, p90)
  5. Gracefully handles errors in calculation (e.g., for invalid metrics or computation issues)
- **Output:** A tuple containing:
  - A dictionary mapping each requested metric name (lowercase) to its calculated value
  - A boolean indicating whether all requested metrics were successfully calculated

#### `format_statistics_table(data_dict, format_string)`

- **Purpose:** Generates a formatted table of statistics from a dictionary of numerical data lists using the provided format string.
- **Input:**
  - `data_dict`: A dictionary where keys are identifiers/labels and values are lists of numbers
  - `format_string`: A Python f-string-like format template (e.g., `"{key:<10} {count:>5} {mean:>6.1f}"`) specifying:
    - What statistics to calculate (metrics in the placeholders)
    - How to format each column (width, alignment, precision)
    - Order of columns in the table
- **Logic:**
  1. Parses the format string to identify required metrics and their format specifications
  2. Validates that `{key}` is the first placeholder
  3. Generates a header row with capitalized column names
  4. For each key-value list pair:
     - Calculates requested statistics using `calculate_numeric_summary`
     - Formats the row according to the format string
     - Handles type mismatches (e.g., integer format specifier with float values)
     - Provides fallback formatting for errors
  5. Returns a list of formatted strings representing the table
- **Output:** A list of strings representing the table: [header, divider, row1, row2, ...]

### Helper Functions

#### `_parse_format_string(format_string)`

- **Purpose:** Parses a format string to extract placeholder names, format specifications, and required metrics.
- **Logic:** Uses regex to extract placeholder info, validates that 'key' is present and is the first placeholder.
- **Output:** A tuple of (ordered_placeholders, format_specs, required_metrics).

#### `_generate_header_and_divider(format_string, ordered_placeholders)`

- **Purpose:** Generates the header and divider lines for the table based on the format string.
- **Logic:** Creates a header using capitalized placeholder names, preserving alignment and width but removing type specifiers.
- **Output:** A tuple of (header, divider) strings.

#### `_create_row_format_string(format_string, calculated_stats)`

- **Purpose:** Adapts the format string to handle special cases like None values or type mismatches.
- **Logic:** Modifies format specifiers based on the actual data (e.g., removing format specifiers for None values, adapting integer specifiers for float values).
- **Output:** A modified format string that can be safely used with the data.

#### `_format_row(item_key, calculated_stats, format_string, ordered_placeholders)`

- **Purpose:** Formats a single row of the table using the calculated statistics and format string.
- **Logic:** Prepares a data dictionary, handles None values by replacing with DEFAULT_NA_VALUE, applies the format string using Python's string formatting.
- **Output:** A formatted string representing a single row of the table.

### Usage Example

```python
# Calculate statistics for selected metrics
data = [1.2, 3.4, None, 5.6, 7.8]
stats_dict, success = calculate_numeric_summary(data, ["count", "mean", "min", "max", "p50"])
# stats_dict = {'count': 4, 'mean': 4.5, 'min': 1.2, 'max': 7.8, 'p50': 4.5}
# success = True

# Format a table for multiple data sets
data_dict = {
    "apples": [10, 12, 11, 13],
    "bananas": [5, 6, 5, 7, 6],
    "cherries": [100, 150]
}
format_string = "{key:<10} {count:>5} {mean:>7.1f} {min:>5} {max:>5}"
table_lines = format_statistics_table(data_dict, format_string)
# ['Key        Count   Mean    Min   Max',
#  '------------------------------------',
#  'apples         4    11.5     10    13',
#  'bananas        5     5.8      5     7',
#  'cherries       2   125.0    100   150']
```