# src/utils/common/image_annotate.py

"""Common utilities for drawing annotations (boxes, polygons, masks, labels) on images."""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Default color palette (BGR format)
# Using a simple palette for now, can be expanded or made more sophisticated
DEFAULT_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (192, 192, 192),
    (128, 128, 128),
    (128, 0, 0),
    (128, 128, 0),
    (0, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (0, 0, 128),
]

# --- Data Classes for Labels ---


@dataclass
class LabelInfo:
    """Container for label information with optional score and truncation."""

    text: str
    score: Optional[float] = None
    max_length: Optional[int] = None

    def __str__(self) -> str:
        """Format label with optional score and truncation."""
        text = self.text
        if self.max_length and len(text) > self.max_length:
            if self.max_length > 3:
                text = text[: self.max_length - 3] + "..."
            else:
                text = text[: self.max_length]  # Handle very small max_length

        if self.score is not None:
            return f"{text} ({self.score:.2f})"
        return text


@dataclass
class InstanceLabel:
    """Container for instance label with optional class information."""

    instance_id: int
    class_name: Optional[str] = None

    def __str__(self) -> str:
        """Format instance label with optional class name."""
        if self.class_name:
            return f"Inst {self.instance_id} ({self.class_name})"
        return f"Inst {self.instance_id}"


# --- Helper Functions ---


def get_color(idx: Optional[int] = None) -> Tuple[int, int, int]:
    """Get a BGR color. Returns a specific color if idx is provided, else random."""
    if idx is None:
        # Return a random BGR color
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # Return a color from the palette based on index
    return DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]


def get_color_map(num_classes: int) -> Dict[int, Tuple[int, int, int]]:
    """Generate a color map for a given number of classes."""
    return {i: get_color(i) for i in range(num_classes)}


def yolo_to_pixel_coords(
    coords: Union[List[float], Tuple[float, ...]],
    image_width: int,
    image_height: int,
) -> List[int]:
    """Convert YOLO format coordinates (normalized cx, cy, w, h or polygon points) to pixel coordinates."""
    if len(coords) == 4:  # Bounding box [cx, cy, w, h]
        cx, cy, w, h = coords
        x_center = cx * image_width
        y_center = cy * image_height
        box_w = w * image_width
        box_h = h * image_height
        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)
        return [x1, y1, x2, y2]
    elif len(coords) > 4 and len(coords) % 2 == 0:  # Polygon [x1, y1, x2, y2, ...]
        pixel_points = []
        for i in range(0, len(coords), 2):
            px = int(coords[i] * image_width)
            py = int(coords[i + 1] * image_height)
            pixel_points.extend([px, py])
        return pixel_points
    else:
        raise ValueError(
            "Invalid coordinate format for YOLO conversion. Expected [cx, cy, w, h] or [x1, y1, x2, y2, ...]"
        )


def get_text_size(
    text: str,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
    font_thickness: int = 1,
) -> Tuple[int, int]:
    """Calculate text size in pixels."""
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font_face, font_scale, font_thickness
    )
    # Add baseline to height for better background box calculation
    return text_width, text_height + baseline


def _format_label_from_dict(label_dict: Dict) -> str:
    """Helper to format core text from a dictionary label."""
    instance_id = label_dict.get("instance_id")
    class_name = label_dict.get("class_name")
    if instance_id is not None:
        core_text = f"Inst {instance_id}"
        if class_name:
            core_text += f" ({class_name})"
    elif class_name is not None:
        core_text = class_name
    else:
        core_text = "Label"  # Default fallback
    return core_text


def format_label(
    label: Union[str, LabelInfo, InstanceLabel, Dict],
    score: Optional[float] = None,
    max_length: Optional[int] = None,
) -> str:
    """Format label string from various input types with optional truncation."""
    # 1. Extract base text, initial score, and internal max_length
    core_text = ""
    initial_score = None
    internal_max_length = None

    if isinstance(label, str):
        core_text = label
    elif isinstance(label, LabelInfo):
        core_text = label.text
        initial_score = label.score
        internal_max_length = label.max_length
    elif isinstance(label, InstanceLabel):
        core_text = str(label)  # Already formatted by __str__
    elif isinstance(label, dict):
        core_text = _format_label_from_dict(label)
    else:
        raise ValueError(f"Unsupported label type: {type(label)}")

    # 2. Determine final score and max_length (explicit parameters override)
    final_score = score if score is not None else initial_score
    final_max_length = max_length if max_length is not None else internal_max_length

    # 3. Apply truncation *only to the core_text* based on final_max_length
    truncated_text = core_text
    # Apply truncation only if a max_length is active and text exceeds it
    if final_max_length is not None and len(core_text) > final_max_length:
        if final_max_length > 3:
            # Truncate the core text
            truncated_text = core_text[: final_max_length - 3] + "..."
        else:
            # Handle very small max_length, just truncate sharply
            truncated_text = core_text[:final_max_length]

    # 4. Append final score if it exists
    if final_score is not None:
        return f"{truncated_text} ({final_score:.2f})"
    else:
        return truncated_text  # Return potentially truncated text


# --- Private Helper for Label Drawing ---


def _draw_label_near_point(
    image: np.ndarray,
    text: str,
    point: Tuple[int, int],  # Anchor point (x, y)
    color: Tuple[int, int, int],
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
    font_thickness: int = 1,
) -> None:
    """Draws a text label with background near a given point (inplace)."""
    img_h, img_w = image.shape[:2]
    text_w, text_h = get_text_size(text, font_face, font_scale, font_thickness)

    # Calculate initial label position (e.g., centered above the point)
    label_x = point[0] - text_w // 2
    label_y = point[1] - text_h - 5  # 5 pixels above point

    label_bg_x1 = label_x - 2  # Add padding
    label_bg_y1 = label_y - 2
    label_bg_x2 = label_x + text_w + 2
    label_bg_y2 = label_y + text_h + 2

    # Adjust if label goes offscreen
    if label_bg_y1 < 0:
        label_bg_y1 = point[1] + 5  # Move below point
        label_bg_y2 = label_bg_y1 + text_h + 4
    if label_bg_y2 > img_h:
        label_bg_y2 = img_h
        label_bg_y1 = img_h - text_h - 4
    if label_bg_x1 < 0:
        label_bg_x1 = 0
        label_bg_x2 = label_bg_x1 + text_w + 4
    if label_bg_x2 > img_w:
        label_bg_x2 = img_w
        label_bg_x1 = img_w - text_w - 4

    # Draw background
    cv2.rectangle(image, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, cv2.FILLED)

    # Draw text
    text_color = (255, 255, 255)
    # Calculate text baseline position within the background box
    text_base_y = (
        label_bg_y1
        + text_h
        - (text_h - cv2.getTextSize(text, font_face, font_scale, font_thickness)[0][1]) // 2
    )
    cv2.putText(
        image,
        text,
        (label_bg_x1 + 2, text_base_y),  # Add padding for x
        font_face,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )


# --- Drawing Functions ---


def draw_box(
    image: np.ndarray,
    box: Union[List[float], Tuple[float, ...]],  # Expects pixel coords [x1, y1, x2, y2]
    label: Union[str, LabelInfo, InstanceLabel, Dict],
    color: Optional[Tuple[int, int, int]] = None,
    score: Optional[float] = None,
    thickness: int = 2,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    max_label_width_ratio: float = 0.8,  # Max label width as ratio of image width
) -> np.ndarray:
    """Draw a bounding box and label on an image.

    Args:
        image: OpenCV image (BGR format)
        box: Box coordinates in pixel format [x1, y1, x2, y2]
        label: Label information (see format_label for supported types)
        color: Optional BGR color tuple. If None, generates a color based on label/instance.
        score: Optional confidence score (overrides score in LabelInfo if provided)
        thickness: Box line thickness
        font_face: OpenCV font face
        font_scale: Text label font scale
        font_thickness: Text label font thickness
        max_label_width_ratio: Max label width as ratio of image width for truncation.

    Returns:
        Image with box and label drawn (modifies the input image inplace)
    """
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    # Determine color
    if color is None:
        # Try to get consistent color if label is instance or class index
        idx = None
        if isinstance(label, InstanceLabel):
            idx = label.instance_id
        elif isinstance(label, dict) and "instance_id" in label:
            idx = label["instance_id"]
        elif isinstance(label, dict) and "class_id" in label:
            idx = label["class_id"]
        color = get_color(idx)

    # Draw the bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Format label and handle truncation
    final_label_text = format_label(label, score)
    max_pixel_width = int(img_w * max_label_width_ratio)
    text_w, text_h = get_text_size(final_label_text, font_face, font_scale, font_thickness)

    if text_w > max_pixel_width:
        # Estimate max characters based on average width
        avg_char_width = text_w / len(final_label_text)
        max_chars = int(max_pixel_width / avg_char_width)
        final_label_text = format_label(label, score, max_length=max_chars)
        text_w, text_h = get_text_size(final_label_text, font_face, font_scale, font_thickness)

    # Calculate label position
    label_bg_x1 = x1
    label_bg_y1 = y1 - text_h - thickness  # Position above the box
    label_bg_x2 = x1 + text_w
    label_bg_y2 = y1 - thickness

    # Adjust if label goes offscreen (top or left)
    if label_bg_y1 < 0:
        label_bg_y1 = y2 + thickness
        label_bg_y2 = y2 + text_h + thickness
    if label_bg_x2 > img_w:
        label_bg_x1 = img_w - text_w
        label_bg_x2 = img_w
    if label_bg_x1 < 0:
        label_bg_x1 = 0
        label_bg_x2 = text_w

    # Draw filled background for label
    cv2.rectangle(image, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, cv2.FILLED)

    # Draw label text (white on colored background)
    text_color = (255, 255, 255)
    text_y_pos = (
        label_bg_y1
        + text_h
        - (text_h - cv2.getTextSize(final_label_text, font_face, font_scale, font_thickness)[0][1])
        // 2
    )  # Center vertically
    cv2.putText(
        image,
        final_label_text,
        (
            label_bg_x1,
            text_y_pos
            - (
                text_h
                - cv2.getTextSize(final_label_text, font_face, font_scale, font_thickness)[0][1]
            )
            // 2,
        ),  # Adjust based on baseline
        font_face,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    return image


def draw_polygon(
    image: np.ndarray,
    points: Union[
        List[Tuple[int, int]], np.ndarray
    ],  # Expects pixel coords [(x1,y1), (x2,y2), ...]
    label: Union[str, LabelInfo, InstanceLabel, Dict],
    color: Optional[Tuple[int, int, int]] = None,
    score: Optional[float] = None,
    thickness: int = 2,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    max_label_width_ratio: float = 0.8,
) -> np.ndarray:
    """Draw a polygon and label on an image."""
    if not isinstance(points, np.ndarray):
        pts_np = np.array(points, dtype=np.int32)
    else:
        pts_np = points.astype(np.int32)
    pts_np = pts_np.reshape((-1, 1, 2))

    img_h, img_w = image.shape[:2]

    # Determine color
    if color is None:
        idx = None
        if isinstance(label, InstanceLabel):
            idx = label.instance_id
        elif isinstance(label, dict) and "instance_id" in label:
            idx = label["instance_id"]
        elif isinstance(label, dict) and "class_id" in label:
            idx = label["class_id"]
        color = get_color(idx)

    # Draw the polygon outline
    cv2.polylines(image, [pts_np], isClosed=True, color=color, thickness=thickness)

    # Format label and handle truncation
    final_label_text = format_label(label, score)
    max_pixel_width = int(img_w * max_label_width_ratio)
    text_w, text_h = get_text_size(final_label_text, font_face, font_scale, font_thickness)

    if text_w > max_pixel_width:
        avg_char_width = text_w / len(final_label_text)
        # Avoid division by zero
        if avg_char_width > 0:
            max_chars = int(max_pixel_width / avg_char_width)
            max_chars = max(1, max_chars)
        else:
            max_chars = 1  # Fallback
        final_label_text = format_label(label, score, max_length=max_chars)
        # text_w, text_h = get_text_size(final_label_text, font_face, font_scale, font_thickness)

    # Use the helper to draw the label near the first point
    anchor_point = tuple(pts_np[0][0])  # First point of the polygon
    _draw_label_near_point(
        image, final_label_text, anchor_point, color, font_face, font_scale, font_thickness
    )
    # --- Removed complex label drawing logic ---

    return image


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,  # Expects binary mask (0/255 or 0/1)
    label: Optional[
        Union[str, LabelInfo, InstanceLabel, Dict]
    ] = None,  # Label is optional for masks
    color: Optional[Tuple[int, int, int]] = None,
    score: Optional[float] = None,
    alpha: float = 0.4,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    max_label_width_ratio: float = 0.8,
) -> np.ndarray:
    """Overlay a mask on an image with transparency and optional label."""
    img_h, img_w = image.shape[:2]

    # Determine color
    if color is None:
        idx = None
        if isinstance(label, InstanceLabel):
            idx = label.instance_id
        elif isinstance(label, dict) and "instance_id" in label:
            idx = label["instance_id"]
        elif isinstance(label, dict) and "class_id" in label:
            idx = label["class_id"]
        color = get_color(idx)

    # Ensure mask is boolean or 0/1
    if mask.dtype != bool and mask.max() > 1:
        bool_mask = mask.astype(bool)
    else:
        bool_mask = mask > 0

    # Create colored overlay
    colored_mask = np.zeros_like(image, dtype=image.dtype)
    colored_mask[bool_mask] = color

    # Blend the mask with the image
    cv2.addWeighted(colored_mask, alpha, image, 1.0 - alpha, 0, image)

    # Draw label if provided
    if label is not None:
        final_label_text = format_label(label, score)
        max_pixel_width = int(img_w * max_label_width_ratio)
        text_w, text_h = get_text_size(final_label_text, font_face, font_scale, font_thickness)

        if text_w > max_pixel_width:
            avg_char_width = text_w / len(final_label_text)
            # Avoid division by zero if avg_char_width is zero
            if avg_char_width > 0:
                max_chars = int(max_pixel_width / avg_char_width)
                # Ensure max_chars is reasonable
                max_chars = max(1, max_chars)
            else:
                max_chars = 1  # Fallback if text_w was 0 but somehow > max_pixel_width?
            final_label_text = format_label(label, score, max_length=max_chars)
            # Recalculate text_w, text_h might not be necessary if _draw_label handles it

        # Find contour to place label near (e.g., top-most point)
        contours, _ = cv2.findContours(
            bool_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            # Find top-most point of the largest contour
            cnt = max(contours, key=cv2.contourArea)
            topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])

            # Use the helper to draw the label
            _draw_label_near_point(
                image, final_label_text, topmost, color, font_face, font_scale, font_thickness
            )
            # --- Removed complex label drawing logic from here ---

    return image
