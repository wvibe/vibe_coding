#!/usr/bin/env python3
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

# Add src to path to allow direct import for testing
# sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
# Import the functions/classes to be tested from the renamed files
from src.utils.data_converter.voc2yolo_detect_labels import (
    apply_sampling_across_splits,
    convert_annotation,
    convert_box,
)

# Import VOC_CLASSES from the correct new utility file
from src.utils.data_converter.voc2yolo_utils import VOC_CLASSES

# --- Test convert_box --- #


def test_convert_box():
    """Test the VOC bounding box to YOLO format conversion."""
    img_size = (640, 480)  # width, height
    # VOC box: xmin, ymin, xmax, ymax (as list)
    voc_box = [100.0, 120.0, 300.0, 360.0]
    # Expected YOLO: x_center_norm, y_center_norm, width_norm, height_norm
    expected_yolo = (
        (100 + 300) / 2 / 640,  # x_center = (xmin+xmax)/2
        (120 + 360) / 2 / 480,  # y_center = (ymin+ymax)/2
        (300 - 100) / 640,  # width = xmax-xmin
        (360 - 120) / 480,  # height = ymax-ymin
    )
    # Expected: (0.3125, 0.5, 0.3125, 0.5)

    result_yolo = convert_box(img_size, voc_box)
    assert result_yolo == pytest.approx(expected_yolo)


# --- Test sampling function --- #


def test_apply_sampling_across_splits():
    """Test the sampling function that selects a subset of image IDs."""
    # Setup test data
    all_image_ids_map = {
        ("2007", "train"): ["img1", "img2", "img3", "img4", "img5"],
        ("2007", "val"): ["img6", "img7", "img8"],
        ("2012", "train"): ["img9", "img10"],
    }
    total_ids = 10

    # Test no sampling (sample_count=None)
    result = apply_sampling_across_splits(all_image_ids_map, None, total_ids, seed=42)
    assert result == all_image_ids_map

    # Test sampling more than available (should return all)
    result = apply_sampling_across_splits(all_image_ids_map, 20, total_ids, seed=42)
    assert result == all_image_ids_map

    # Test sampling a subset
    result = apply_sampling_across_splits(all_image_ids_map, 5, total_ids, seed=42)
    assert sum(len(ids) for ids in result.values()) == 5

    # Test sampling with consistent seed
    result1 = apply_sampling_across_splits(all_image_ids_map, 3, total_ids, seed=42)
    result2 = apply_sampling_across_splits(all_image_ids_map, 3, total_ids, seed=42)
    # Both results should be the same with the same seed
    assert result1 == result2


# --- Fixtures for convert_annotation --- #


@pytest.fixture
def temp_dir_fixture():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_good_xml():
    """Create a valid sample XML ElementTree object."""
    root = ET.Element("annotation")
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = "100"
    ET.SubElement(size, "height").text = "100"

    # Object 1 (valid)
    obj1 = ET.SubElement(root, "object")
    ET.SubElement(obj1, "name").text = "person"
    bbox1 = ET.SubElement(obj1, "bndbox")
    ET.SubElement(bbox1, "xmin").text = "10"
    ET.SubElement(bbox1, "ymin").text = "20"
    ET.SubElement(bbox1, "xmax").text = "30"
    ET.SubElement(bbox1, "ymax").text = "40"

    # Object 2 (valid)
    obj2 = ET.SubElement(root, "object")
    ET.SubElement(obj2, "name").text = "car"
    bbox2 = ET.SubElement(obj2, "bndbox")
    ET.SubElement(bbox2, "xmin").text = "50"
    ET.SubElement(bbox2, "ymin").text = "60"
    ET.SubElement(bbox2, "xmax").text = "70"
    ET.SubElement(bbox2, "ymax").text = "80"

    # Object 3 (unknown class)
    obj3 = ET.SubElement(root, "object")
    ET.SubElement(obj3, "name").text = "unknown_thing"
    bbox3 = ET.SubElement(obj3, "bndbox")
    ET.SubElement(bbox3, "xmin").text = "1"
    ET.SubElement(bbox3, "ymin").text = "1"
    ET.SubElement(bbox3, "xmax").text = "5"
    ET.SubElement(bbox3, "ymax").text = "5"

    return ET.ElementTree(root)


@pytest.fixture
def sample_bad_xml_missing_size():
    """Create XML missing the size tag."""
    root = ET.Element("annotation")
    obj1 = ET.SubElement(root, "object")
    ET.SubElement(obj1, "name").text = "person"
    # ... bbox ...
    return ET.ElementTree(root)


@pytest.fixture
def sample_bad_xml_invalid_coords():
    """Create XML with invalid bbox coordinates."""
    root = ET.Element("annotation")
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = "100"
    ET.SubElement(size, "height").text = "100"
    obj1 = ET.SubElement(root, "object")
    ET.SubElement(obj1, "name").text = "person"
    bbox1 = ET.SubElement(obj1, "bndbox")
    ET.SubElement(bbox1, "xmin").text = "110"  # Invalid: > width
    ET.SubElement(bbox1, "ymin").text = "20"
    ET.SubElement(bbox1, "xmax").text = "130"
    ET.SubElement(bbox1, "ymax").text = "40"
    return ET.ElementTree(root)


# --- Test convert_annotation --- #


def test_convert_annotation_success(temp_dir_fixture, sample_good_xml):
    """Test successful conversion of a valid XML file."""
    xml_path = temp_dir_fixture / "good.xml"
    output_txt_path = temp_dir_fixture / "good.txt"
    sample_good_xml.write(xml_path, encoding="utf-8")

    # Pass the directory, not the full file path
    result = convert_annotation(xml_path, temp_dir_fixture)

    assert result is True
    # Check if the expected output file exists in the directory
    assert output_txt_path.exists()

    with open(output_txt_path, "r") as f:
        lines = f.readlines()

    assert len(lines) == 2  # parse_voc_xml filters unknown classes, so only 2 objects remain

    # Check line 1 (person) - Use VOC_CLASS_TO_ID from utils indirectly
    person_id = VOC_CLASSES.index("person")
    expected_box1 = convert_box((100, 100), [10.0, 20.0, 30.0, 40.0])
    assert lines[0].strip() == f"{person_id} {' '.join(map(str, expected_box1))}"

    # Check line 2 (car)
    car_id = VOC_CLASSES.index("car")
    expected_box2 = convert_box((100, 100), [50.0, 60.0, 70.0, 80.0])
    assert lines[1].strip() == f"{car_id} {' '.join(map(str, expected_box2))}"

    # Test for skipping existing files by running the conversion again
    result = convert_annotation(xml_path, temp_dir_fixture)
    assert result is True  # Still returns success for an existing file


def test_convert_annotation_missing_size(temp_dir_fixture, sample_bad_xml_missing_size):
    """Test XML conversion failure when size tag is missing."""
    xml_path = temp_dir_fixture / "bad_size.xml"
    output_txt_path = temp_dir_fixture / "bad_size.txt"
    sample_bad_xml_missing_size.write(xml_path, encoding="utf-8")

    # Pass the directory
    result = convert_annotation(xml_path, temp_dir_fixture)

    assert result is False
    # Output file should not be created
    assert not output_txt_path.exists()


def test_convert_annotation_invalid_coords(temp_dir_fixture, sample_bad_xml_invalid_coords):
    """Test XML conversion with invalid coordinates (should skip object)."""
    xml_path = temp_dir_fixture / "bad_coords.xml"
    output_txt_path = temp_dir_fixture / "bad_coords.txt"
    sample_bad_xml_invalid_coords.write(xml_path, encoding="utf-8")

    # Pass the directory
    result = convert_annotation(xml_path, temp_dir_fixture)

    assert result is True  # Conversion itself succeeds, even if objects are skipped
    # Check if the expected output file exists in the directory
    assert output_txt_path.exists()

    with open(output_txt_path, "r") as f:
        lines = f.readlines()

    assert len(lines) == 0  # The only object had invalid coords, so file should be empty


def test_convert_annotation_file_not_found(temp_dir_fixture):
    """Test XML conversion failure when input file doesn't exist."""
    xml_path = temp_dir_fixture / "nonexistent.xml"
    output_txt_path = temp_dir_fixture / "nonexistent.txt"

    # Pass the directory
    result = convert_annotation(xml_path, temp_dir_fixture)

    assert result is False
    assert not output_txt_path.exists()
