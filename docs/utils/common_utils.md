# Common Utilities Design Notes

This document outlines the design choices and rationale behind the functions in the `src/utils/common/` directory, which contain general-purpose utilities used across different parts of the project.

---

## `iou.py` - Intersection over Union Utilities

This module provides functions for calculating the Intersection over Union (IoU) between bounding boxes, a fundamental operation in object detection evaluation.

### `calculate_iou(box1, box2)`

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

### `calculate_iou_matrix(boxes1, boxes2)`

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

## `bbox_format.py` - Bounding Box Format Conversion

This module provides utilities to convert bounding box coordinates between different common formats.

### `cxcywh_to_xyxy(box)`

- **Purpose:** Converts a box from center coordinates (`[center_x, center_y, width, height]`) to corner coordinates (`[x_min, y_min, x_max, y_max]`).
- **Logic:** Uses basic arithmetic: `x_min = cx - w/2`, `y_min = cy - h/2`, `x_max = cx + w/2`, `y_max = cy + h/2`.
- **Input/Output:** Can handle single boxes (list/tuple) or multiple boxes (NumPy array `[N, 4]`).

### `xyxy_to_cxcywh(box)`

- **Purpose:** Converts a box from corner coordinates (`[x_min, y_min, x_max, y_max]`) to center coordinates (`[center_x, center_y, width, height]`).
- **Logic:** Uses basic arithmetic: `width = x_max - x_min`, `height = y_max - y_min`, `center_x = x_min + width/2`, `center_y = y_min + height/2`.
- **Input/Output:** Can handle single boxes (list/tuple) or multiple boxes (NumPy array `[N, 4]`).

### `normalize_boxes(boxes, image_width, image_height)`

- **Purpose:** Normalizes pixel coordinates (usually `[x_min, y_min, x_max, y_max]`) to be relative to image dimensions (range 0.0 to 1.0).
- **Logic:** Divides x-coordinates by `image_width` and y-coordinates by `image_height`.
- **Input:** A single box or a NumPy array `[N, 4]` of boxes, plus image dimensions.
- **Output:** Box(es) with coordinates normalized to the [0, 1] range.

### `denormalize_boxes(boxes, image_width, image_height)`

- **Purpose:** Converts normalized coordinates (range 0.0 to 1.0) back to absolute pixel coordinates.
- **Logic:** Multiplies normalized x-coordinates by `image_width` and y-coordinates by `image_height`.
- **Input:** A single box or a NumPy array `[N, 4]` of normalized boxes, plus image dimensions.
- **Output:** Box(es) with absolute pixel coordinates.

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