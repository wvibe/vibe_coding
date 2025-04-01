# tests/utils/common/test_image_annotate.py

"""Tests for the image annotation utilities."""

import numpy as np
import pytest

# Assume the module is importable; adjust path if needed in a real environment
from src.utils.common.image_annotate import (
    DEFAULT_COLORS,
    InstanceLabel,
    LabelInfo,
    draw_box,
    draw_polygon,
    format_label,
    get_color,
    get_color_map,
    get_text_size,
    overlay_mask,
    yolo_to_pixel_coords,
)

# --- Fixtures ---


@pytest.fixture
def dummy_image() -> np.ndarray:
    """Create a blank 100x200 BGR image."""
    return np.zeros((100, 200, 3), dtype=np.uint8)


# --- Test Helper Functions ---


def test_get_color():
    """Test color generation."""
    # Test getting color by index
    color0 = get_color(0)
    assert isinstance(color0, tuple) and len(color0) == 3
    assert color0 == DEFAULT_COLORS[0]

    color1 = get_color(1)
    assert color1 == DEFAULT_COLORS[1]
    assert color0 != color1

    # Test index wrapping
    color_wrap = get_color(len(DEFAULT_COLORS))
    assert color_wrap == DEFAULT_COLORS[0]

    # Test random color
    random_color = get_color(None)
    assert isinstance(random_color, tuple) and len(random_color) == 3


def test_get_color_map():
    """Test color map generation."""
    num_classes = 5
    color_map = get_color_map(num_classes)
    assert isinstance(color_map, dict)
    assert len(color_map) == num_classes
    assert 0 in color_map and 4 in color_map
    assert color_map[0] == get_color(0)
    assert color_map[1] == get_color(1)


def test_yolo_to_pixel_coords():
    """Test YOLO coordinate conversion."""
    img_w, img_h = 200, 100

    # Test box [cx, cy, w, h]
    yolo_box = [0.5, 0.5, 0.2, 0.4]  # Center, 20% width, 40% height
    pixel_box = yolo_to_pixel_coords(yolo_box, img_w, img_h)
    # cx=100, cy=50, w=40, h=40
    # x1=100-20=80, y1=50-20=30, x2=100+20=120, y2=50+20=70
    assert pixel_box == [80, 30, 120, 70]

    # Test polygon [x1, y1, x2, y2, ...]
    yolo_poly = [0.1, 0.1, 0.9, 0.1, 0.5, 0.9]  # Triangle
    pixel_poly = yolo_to_pixel_coords(yolo_poly, img_w, img_h)
    # x1=20, y1=10, x2=180, y2=10, x3=100, y3=90
    assert pixel_poly == [20, 10, 180, 10, 100, 90]

    # Test invalid format
    with pytest.raises(ValueError):
        yolo_to_pixel_coords([0.1, 0.2, 0.3], img_w, img_h)  # Odd number
    with pytest.raises(ValueError):
        yolo_to_pixel_coords([], img_w, img_h)  # Empty


def test_get_text_size():
    """Test text size calculation."""
    text = "Test Label"
    width, height = get_text_size(text, font_scale=0.5, font_thickness=1)
    assert isinstance(width, int) and width > 0
    assert isinstance(height, int) and height > 0

    width_l, height_l = get_text_size(text, font_scale=1.0, font_thickness=2)
    assert width_l > width
    assert height_l > height


def test_format_label():
    """Test label formatting logic."""
    # Simple string
    assert format_label("person") == "person"

    # String with score
    assert format_label("person", score=0.95) == "person (0.95)"

    # LabelInfo
    li = LabelInfo("car", score=0.88)
    assert format_label(li) == "car (0.88)"

    # LabelInfo with score override
    assert format_label(li, score=0.77) == "car (0.77)"

    # LabelInfo with max_length internal
    li_trunc = LabelInfo("longlabeltext", score=0.8, max_length=10)
    assert format_label(li_trunc) == "longlab... (0.80)"

    # LabelInfo with max_length override
    assert format_label(li_trunc, max_length=5) == "lo... (0.80)"

    # InstanceLabel
    il = InstanceLabel(instance_id=5, class_name="dog")
    assert format_label(il) == "Inst 5 (dog)"

    # InstanceLabel without class
    il_no_class = InstanceLabel(instance_id=10)
    assert format_label(il_no_class) == "Inst 10"

    # InstanceLabel with score
    assert format_label(il, score=0.6) == "Inst 5 (dog) (0.60)"

    # Dictionary format
    d = {"class_name": "cat", "instance_id": 3}
    assert format_label(d) == "Inst 3 (cat)"
    assert format_label(d, score=0.5) == "Inst 3 (cat) (0.50)"

    # Dictionary format - class only
    d_class = {"class_name": "tree"}
    assert format_label(d_class) == "tree"

    # Truncation with simple string
    assert format_label("verylonglabel", max_length=10) == "verylon..."

    # Truncation edge case (max_length <= 3)
    assert format_label("verylonglabel", max_length=3) == "ver"

    # Unsupported type
    with pytest.raises(ValueError):
        format_label(123)


# --- Test Drawing Functions (Basic Checks) ---
# These tests mainly check if the functions run without errors and modify the image.
# Visual inspection is often needed for precise drawing verification.


def test_draw_box(dummy_image):  # Use the fixture
    """Test drawing a box."""
    # Test basic drawing
    img_to_draw_on = dummy_image.copy()
    img_before = img_to_draw_on.copy()
    box = [20, 30, 80, 70]  # x1, y1, x2, y2
    label = "TestBox"
    img_after = draw_box(img_to_draw_on, box, label)

    assert img_after is img_to_draw_on  # Check if modified inplace
    assert not np.array_equal(img_before, img_after)  # Check image was modified
    # Check if some pixels changed color (simple check)
    assert np.sum(img_after[30:70, 20:80]) > 0

    # Test with LabelInfo and score
    img_to_draw_on_2 = dummy_image.copy()
    img_before2 = img_to_draw_on_2.copy()
    li = LabelInfo("Car", 0.9)
    img_after2 = draw_box(img_to_draw_on_2, box, li)
    assert not np.array_equal(img_before2, img_after2)

    # Test with truncation (visually verify label if needed)
    img_to_draw_on_3 = dummy_image.copy()
    img_before3 = img_to_draw_on_3.copy()
    long_label = "A Very Long Label That Will Definitely Need To Be Truncated"
    img_after3 = draw_box(img_to_draw_on_3, box, long_label, max_label_width_ratio=0.1)
    assert not np.array_equal(img_before3, img_after3)

    # Verify that drawing different things results in different images
    assert not np.array_equal(img_after2, img_after3)


def test_draw_polygon(dummy_image):
    """Test drawing a polygon."""
    img_before = dummy_image.copy()
    # Pixel coordinates for a triangle
    points = [(20, 10), (180, 10), (100, 90)]
    label = InstanceLabel(1, "Triangle")
    img_after = draw_polygon(dummy_image, points, label)

    assert img_after is dummy_image
    assert not np.array_equal(img_before, img_after)
    # Check if some pixels along the expected path changed
    assert np.sum(img_after[10:11, 20:180]) > 0  # Top line


def test_overlay_mask(dummy_image):
    """Test overlaying a mask."""
    img_before = dummy_image.copy()
    mask = np.zeros((100, 200), dtype=np.uint8)
    # Create a rectangular mask region
    mask[40:60, 80:120] = 255
    label = {"class_name": "MaskRegion", "class_id": 5}

    img_after = overlay_mask(dummy_image, mask, label=label, alpha=0.5)

    assert img_after is dummy_image
    assert not np.array_equal(img_before, img_after)
    # Check if the masked area has changed color
    assert np.any(img_after[40:60, 80:120] != img_before[40:60, 80:120])

    # Test without label
    img_before2 = dummy_image.copy()
    img_after2 = overlay_mask(dummy_image, mask, label=None, alpha=0.5)
    assert not np.array_equal(img_before2, img_after2)
