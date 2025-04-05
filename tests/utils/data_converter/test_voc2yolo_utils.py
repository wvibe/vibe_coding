import sys
from pathlib import Path

import pytest

# Ensure src directory is in sys.path for imports
project_root = Path(__file__).resolve().parents[3]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Use absolute import based on the updated sys.path
from utils.data_converter import voc2yolo_utils

# Define some constants for testing
# TEST_ROOT = Path("/tmp/voc_test_root") # Unused
# TEST_OUT_ROOT = Path("/tmp/output_root") # Unused
# YEAR = "2007" # Unused
# TAG = "train" # Unused

# --- Test Data ---
MOCK_VOC_ROOT = Path("/mock/voc")
MOCK_OUTPUT_ROOT = Path("/mock/output")
TEST_YEARS = ["2007", "2012"]
TEST_TAGS = ["train", "val", "test"]
TEST_TASKS = ["detect", "segment"]
TEST_IMAGE_ID = "000001"

# --- Tests for Path Getters ---


@pytest.mark.parametrize("year", TEST_YEARS)
@pytest.mark.parametrize("tag", TEST_TAGS)
@pytest.mark.parametrize("task_type", TEST_TASKS)
def test_get_output_image_dir(year, tag, task_type):
    """Verify correct output image directory path construction."""
    expected = MOCK_OUTPUT_ROOT / task_type / voc2yolo_utils.OUTPUT_IMAGES_DIR / f"{tag}{year}"
    actual = voc2yolo_utils.get_output_image_dir(MOCK_OUTPUT_ROOT, task_type, year, tag)
    assert actual == expected


@pytest.mark.parametrize("year", TEST_YEARS)
@pytest.mark.parametrize("tag", TEST_TAGS)
def test_get_output_detect_label_dir(year, tag):
    """Verify correct output detection label directory path construction."""
    expected = (
        MOCK_OUTPUT_ROOT
        / voc2yolo_utils.OUTPUT_DETECT_DIR_NAME
        / voc2yolo_utils.OUTPUT_LABELS_SUBDIR
        / f"{tag}{year}"
    )
    actual = voc2yolo_utils.get_output_detect_label_dir(MOCK_OUTPUT_ROOT, year, tag)
    assert actual == expected


@pytest.mark.parametrize("year", TEST_YEARS)
@pytest.mark.parametrize("tag", TEST_TAGS)
def test_get_output_segment_label_dir(year, tag):
    """Verify correct output segmentation label directory path construction."""
    expected = (
        MOCK_OUTPUT_ROOT
        / voc2yolo_utils.OUTPUT_SEGMENT_DIR_NAME
        / voc2yolo_utils.OUTPUT_LABELS_SUBDIR
        / f"{tag}{year}"
    )
    actual = voc2yolo_utils.get_output_segment_label_dir(MOCK_OUTPUT_ROOT, year, tag)
    assert actual == expected


@pytest.mark.parametrize("year", TEST_YEARS)
@pytest.mark.parametrize("tag", TEST_TAGS)
@pytest.mark.parametrize("task_type", TEST_TASKS)
def test_get_image_set_path(year, tag, task_type):
    """Verify correct image set path construction based on task type."""
    voc_dir = voc2yolo_utils.get_voc_dir(MOCK_VOC_ROOT, year)
    expected_subdir = (
        voc2yolo_utils.MAIN_DIR if task_type == "detect" else voc2yolo_utils.SEGMENTATION_DIR
    )
    # Construct expected path relative to the year-specific voc_dir
    expected = (
        voc_dir  # Start with /mock/voc/VOCdevkit/VOC<year>
        # No longer need / voc2yolo_utils.VOCDEVKIT_DIR / f"VOC{year}"
        / voc2yolo_utils.IMAGESETS_DIR
        / expected_subdir
        / f"{tag}.txt"
    )
    actual = voc2yolo_utils.get_image_set_path(voc_dir, task_type, tag)
    assert actual == expected


@pytest.mark.parametrize("year", TEST_YEARS)
def test_get_image_path(year):
    """Verify correct image source path construction."""
    voc_dir = voc2yolo_utils.get_voc_dir(MOCK_VOC_ROOT, year)
    expected = voc_dir / voc2yolo_utils.JPEG_IMAGES_DIR / f"{TEST_IMAGE_ID}.jpg"
    actual = voc2yolo_utils.get_image_path(voc_dir, TEST_IMAGE_ID)
    assert actual == expected


@pytest.mark.parametrize("year", TEST_YEARS)
def test_get_annotation_path(year):
    """Verify correct annotation source path construction."""
    voc_dir = voc2yolo_utils.get_voc_dir(MOCK_VOC_ROOT, year)
    expected = voc_dir / voc2yolo_utils.ANNOTATIONS_DIR / f"{TEST_IMAGE_ID}.xml"
    actual = voc2yolo_utils.get_annotation_path(voc_dir, TEST_IMAGE_ID)
    assert actual == expected


@pytest.mark.parametrize("year", TEST_YEARS)
def test_get_segm_inst_mask_path(year):
    """Verify correct instance segmentation mask source path construction."""
    voc_dir = voc2yolo_utils.get_voc_dir(MOCK_VOC_ROOT, year)
    expected = voc_dir / voc2yolo_utils.SEGMENTATION_OBJECT_DIR / f"{TEST_IMAGE_ID}.png"
    actual = voc2yolo_utils.get_segm_inst_mask_path(voc_dir, TEST_IMAGE_ID)
    assert actual == expected


@pytest.mark.parametrize("year", TEST_YEARS)
def test_get_segm_cls_mask_path(year):
    """Verify correct class segmentation mask source path construction."""
    voc_dir = voc2yolo_utils.get_voc_dir(MOCK_VOC_ROOT, year)
    expected = voc_dir / voc2yolo_utils.SEGMENTATION_CLASS_DIR / f"{TEST_IMAGE_ID}.png"
    actual = voc2yolo_utils.get_segm_cls_mask_path(voc_dir, TEST_IMAGE_ID)
    assert actual == expected


# --- Tests for File Readers ---


def test_read_image_ids_simple(tmp_path):
    """Test reading simple image IDs from a file."""
    imageset_file = tmp_path / "test_set.txt"
    ids_to_write = ["000001", "000002", "000003"]
    imageset_file.write_text("\n".join(ids_to_write))

    read_ids = voc2yolo_utils.read_image_ids(imageset_file)
    assert read_ids == ids_to_write


def test_read_image_ids_with_extra_columns(tmp_path):
    """Test reading image IDs when lines have extra columns (like VOC sets)."""
    imageset_file = tmp_path / "test_set_extra.txt"
    content = """
000001  1
000002 -1
000003  1
""".strip()
    imageset_file.write_text(content)
    expected_ids = ["000001", "000002", "000003"]
    read_ids = voc2yolo_utils.read_image_ids(imageset_file)
    assert read_ids == expected_ids


def test_read_image_ids_empty_file(tmp_path):
    """Test reading from an empty image set file."""
    imageset_file = tmp_path / "empty_set.txt"
    imageset_file.touch()
    read_ids = voc2yolo_utils.read_image_ids(imageset_file)
    assert read_ids == []


def test_read_image_ids_file_not_found():
    """Test reading from a non-existent file."""
    non_existent_file = Path("/path/to/non/existent/file.txt")
    # Assert that FileNotFoundError is raised
    with pytest.raises(FileNotFoundError):
        voc2yolo_utils.read_image_ids(non_existent_file)
    # assert read_ids == []  # Should return empty list and log error (manual check)


# --- Tests for XML Parsing ---


def test_parse_voc_xml_basic(tmp_path):
    """Test parsing a basic valid VOC XML file."""
    xml_content = """
<annotation>
	<folder>VOC2007</folder>
	<filename>000001.jpg</filename>
	<size>
		<width>353</width>
		<height>500</height>
		<depth>3</depth>
	</size>
	<object>
		<name>dog</name>
		<pose>Left</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>48</xmin>
			<ymin>240</ymin>
			<xmax>195</xmax>
			<ymax>371</ymax>
		</bndbox>
	</object>
	<object>
		<name>person</name>
		<pose>Left</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>8</xmin>
			<ymin>12</ymin>
			<xmax>352</xmax>
			<ymax>498</ymax>
		</bndbox>
	</object>
</annotation>
    """.strip()
    xml_file = tmp_path / "000001.xml"
    xml_file.write_text(xml_content)

    objects, img_size = voc2yolo_utils.parse_voc_xml(xml_file)

    assert img_size == (353, 500)
    assert len(objects) == 2
    assert objects[0] == {"name": "dog", "bbox": [48, 240, 195, 371], "difficult": 0}
    assert objects[1] == {"name": "person", "bbox": [8, 12, 352, 498], "difficult": 0}


def test_parse_voc_xml_difficult(tmp_path):
    """Test parsing an object marked as difficult."""
    xml_content = """
<annotation>
	<size><width>100</width><height>100</height></size>
	<object>
		<name>car</name>
		<difficult>1</difficult>
		<bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>
	</object>
</annotation>
    """.strip()
    xml_file = tmp_path / "difficult.xml"
    xml_file.write_text(xml_content)
    objects, _ = voc2yolo_utils.parse_voc_xml(xml_file)
    assert len(objects) == 1
    assert objects[0]["difficult"] == 1


def test_parse_voc_xml_unknown_class(tmp_path):
    """Test parsing an object with a class name not in VOC_CLASSES."""
    xml_content = """
<annotation>
	<size><width>100</width><height>100</height></size>
	<object>
		<name>unknown_thing</name>
		<difficult>0</difficult>
		<bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>
	</object>
</annotation>
    """.strip()
    xml_file = tmp_path / "unknown.xml"
    xml_file.write_text(xml_content)
    objects, _ = voc2yolo_utils.parse_voc_xml(xml_file)
    assert len(objects) == 0  # Should skip the unknown class


def test_parse_voc_xml_missing_file():
    """Test parsing a non-existent XML file."""
    non_existent_file = Path("/path/to/non/existent/file.xml")
    objects, img_size = voc2yolo_utils.parse_voc_xml(non_existent_file)
    assert objects is None
    assert img_size is None


def test_parse_voc_xml_invalid_xml(tmp_path):
    """Test parsing an invalid/malformed XML file."""
    xml_content = "<annotation><size><width>100</size>"  # Malformed
    xml_file = tmp_path / "invalid.xml"
    xml_file.write_text(xml_content)
    objects, img_size = voc2yolo_utils.parse_voc_xml(xml_file)
    assert objects is None
    assert img_size is None
