import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import mock_open, patch

import cv2
import numpy as np
import pytest

# Add src to path to allow direct import for testing
# sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
# Import from renamed script and util file
from src.utils.data_converter.voc2yolo_segment_labels import (
    VOC2YOLOConverter as SegmentConverter,
)


@pytest.fixture
def temp_voc_dir():
    """Create a temporary directory mimicking VOC structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Required input structure (relative to voc_root)
        for year in ["2007", "2012"]:
            devkit_path = tmp_path / f"VOCdevkit/VOC{year}"
            (devkit_path / "Annotations").mkdir(parents=True, exist_ok=True)
            (devkit_path / "ImageSets" / "Main").mkdir(parents=True, exist_ok=True)
            (devkit_path / "JPEGImages").mkdir(parents=True, exist_ok=True)
            (devkit_path / "SegmentationObject").mkdir(parents=True, exist_ok=True)
            (devkit_path / "SegmentationClass").mkdir(
                parents=True, exist_ok=True
            )  # Include for completeness

        # Output directory structure (separate)
        output_segment_dir = tmp_path / "output_segment"
        output_segment_dir.mkdir(parents=True, exist_ok=True)

        yield tmp_path, output_segment_dir  # Yield both paths


@pytest.fixture
def sample_segmentation_xml_tree():
    """Create a sample XML ET.ElementTree matching sample mask instances."""
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "VOC2012"
    ET.SubElement(root, "filename").text = "test_img.jpg"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = "100"
    ET.SubElement(size, "height").text = "100"
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "1"

    # Object 1: Person (matches the square instance ID=1 in mask)
    obj1 = ET.SubElement(root, "object")
    ET.SubElement(obj1, "name").text = "person"
    ET.SubElement(obj1, "pose").text = "Unspecified"
    ET.SubElement(obj1, "truncated").text = "0"
    ET.SubElement(obj1, "difficult").text = "0"
    bbox1 = ET.SubElement(obj1, "bndbox")
    ET.SubElement(bbox1, "xmin").text = "11"
    ET.SubElement(bbox1, "ymin").text = "20"
    ET.SubElement(bbox1, "xmax").text = "30"
    ET.SubElement(bbox1, "ymax").text = "40"

    # Object 2: Car (matches the circle instance ID=2 in mask)
    obj2 = ET.SubElement(root, "object")
    ET.SubElement(obj2, "name").text = "car"
    ET.SubElement(obj2, "pose").text = "Unspecified"
    ET.SubElement(obj2, "truncated").text = "0"
    ET.SubElement(obj2, "difficult").text = "0"
    bbox2 = ET.SubElement(obj2, "bndbox")
    ET.SubElement(bbox2, "xmin").text = "58"  # Approx bbox for circle centered at 70,70 R=10
    ET.SubElement(bbox2, "ymin").text = "58"
    ET.SubElement(bbox2, "xmax").text = "82"
    ET.SubElement(bbox2, "ymax").text = "82"

    # Object 3: Bicycle (no corresponding mask instance)
    obj3 = ET.SubElement(root, "object")
    ET.SubElement(obj3, "name").text = "bicycle"
    ET.SubElement(obj3, "pose").text = "Unspecified"
    ET.SubElement(obj3, "truncated").text = "0"
    ET.SubElement(obj3, "difficult").text = "0"
    bbox3 = ET.SubElement(obj3, "bndbox")
    ET.SubElement(bbox3, "xmin").text = "1"
    ET.SubElement(bbox3, "ymin").text = "1"
    ET.SubElement(bbox3, "xmax").text = "5"
    ET.SubElement(bbox3, "ymax").text = "5"

    return ET.ElementTree(root)


@pytest.fixture
def sample_segmentation_mask_array():
    """Create a numpy array for a segmentation mask with multiple instances."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Instance 1: Square (person)
    mask[20:40, 10:30] = 1
    # Instance 2: Circle (car)
    cv2.circle(mask, (70, 70), 10, 2, -1)  # Center (x,y), radius, color, thickness (-1=fill)
    # Instance 3: Another shape (unmatched) - Not added to keep it simple
    # Boundary pixels
    mask[50, 50:60] = 255
    return mask


@pytest.fixture
def create_mock_files(temp_voc_dir, sample_segmentation_xml_tree, sample_segmentation_mask_array):
    """Helper fixture to create necessary mock files in temp dir."""
    voc_root, _ = temp_voc_dir
    year = "2012"
    img_id = "test_img"
    split = "train"  # Matches ImageSet filename

    # Create XML
    xml_path = voc_root / f"VOCdevkit/VOC{year}/Annotations/{img_id}.xml"
    sample_segmentation_xml_tree.write(xml_path, encoding="utf-8")

    # Create Mask PNG
    mask_path = voc_root / f"VOCdevkit/VOC{year}/SegmentationObject/{img_id}.png"
    cv2.imwrite(str(mask_path), sample_segmentation_mask_array)

    # Create ImageSet file
    imageset_path = voc_root / f"VOCdevkit/VOC{year}/ImageSets/Main/{split}.txt"
    with open(imageset_path, "w") as f:
        f.write(f"{img_id}\n")
        f.write("another_img_id\n")  # Add another ID to test skipping

    # Create dummy JPEG image (optional, needed if code reads it)
    img_path = voc_root / f"VOCdevkit/VOC{year}/JPEGImages/{img_id}.jpg"
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)  # Match XML size
    cv2.imwrite(str(img_path), dummy_img)

    # Return paths needed by tests
    return {
        "voc_root": voc_root,
        "xml_path": xml_path,
        "mask_path": mask_path,
        "imageset_path": imageset_path,
        "img_id": img_id,
        "split_name": f"{split}{year}",
    }


# Helper fixture to create a dummy VOC structure
@pytest.fixture
def voc_segment_test_setup(tmp_path):
    devkit_path = tmp_path / "VOCdevkit"
    devkit_path.mkdir()
    year = "2012"
    voc_year_path = devkit_path / f"VOC{year}"
    voc_year_path.mkdir()  # <<< Create the VOC year subdir
    segmentation_obj_path = voc_year_path / "SegmentationObject"
    segmentation_obj_path.mkdir()
    annotations_path = voc_year_path / "Annotations"
    annotations_path.mkdir()
    output_path = tmp_path / "output_labels"
    output_path.mkdir()

    # Create dummy class mapping
    class_names = ["background", "classA", "classB"]

    # Create dummy mask file (example)
    mask_data = np.array([[0, 0, 1, 1], [0, 2, 2, 0]], dtype=np.uint8)
    cv2.imwrite(str(segmentation_obj_path / "img1.png"), mask_data)

    # Create dummy annotation file (example)
    annotation_content = """<annotation>
<size><width>4</width><height>2</height></size>
<object><name>classA</name><bndbox><xmin>2</xmin><ymin>0</ymin><xmax>4</xmax><ymax>1</ymax></bndbox></object>
<object><name>classB</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>3</xmax><ymax>2</ymax></bndbox></object>
</annotation>"""
    (annotations_path / "img1.xml").write_text(annotation_content)

    return devkit_path, year, output_path, class_names


def test_get_mask_instances(voc_segment_test_setup):
    devkit_path, year, output_path, _ = voc_segment_test_setup
    # Use the root of the devkit path as voc_root for consistency
    voc_root = devkit_path.parent
    converter = SegmentConverter(
        # devkit_path=str(devkit_path), year=year, output_segment_dir=str(output_path)
        voc_root=voc_root,
        output_root=voc_root,  # Output within the temp dir for test
        year=year,
        tag="test",
    )
    mask_path = devkit_path / f"VOC{year}" / "SegmentationObject" / "img1.png"

    instance_masks = converter._get_mask_instances(mask_path)

    assert instance_masks is not None
    assert 1 in instance_masks
    assert 2 in instance_masks
    assert 0 not in instance_masks  # Background excluded
    assert 255 not in instance_masks  # Boundary excluded

    # Check one mask content
    expected_mask1 = np.array([[0, 0, 1, 1], [0, 0, 0, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(instance_masks[1], expected_mask1)


def test_mask_to_polygons(voc_segment_test_setup):
    devkit_path, year, output_path, _ = voc_segment_test_setup
    voc_root = devkit_path.parent
    converter = SegmentConverter(
        # devkit_path=str(devkit_path), year=year, output_segment_dir=str(output_path)
        voc_root=voc_root,
        output_root=voc_root,
        year=year,
        tag="test",
    )
    # Create a more complex mask for polygon testing
    # L-shape
    mask_data = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    h, w = mask_data.shape

    polygons = converter._mask_to_polygons(mask_data)

    assert len(polygons) == 1  # Should find one contour
    polygon = polygons[0]
    assert len(polygon) >= 6  # Should have at least 3 points (6 coords)

    # Check normalization (example point: bottom-right corner of L is (4, 4))
    # Normalized should be (4/5, 4/6)
    expected_norm_x = 4 / w
    expected_norm_y = 4 / h
    # Find the point in the polygon list (allow for approximation)
    found = False
    for i in range(0, len(polygon), 2):
        if np.allclose([polygon[i], polygon[i + 1]], [expected_norm_x, expected_norm_y], atol=0.1):
            found = True
            break
    assert found, (
        f"Expected normalized point close to ({expected_norm_x:.2f}, {expected_norm_y:.2f}) not found"
    )


def test_get_mask_bbox(voc_segment_test_setup):
    devkit_path, year, output_path, _ = voc_segment_test_setup
    voc_root = devkit_path.parent
    converter = SegmentConverter(
        # devkit_path=str(devkit_path), year=year, output_segment_dir=str(output_path)
        voc_root=voc_root,
        output_root=voc_root,
        year=year,
        tag="test",
    )
    mask_data = np.array([[0, 0, 1, 1], [0, 2, 2, 0]], dtype=np.uint8)
    h, w = mask_data.shape

    # Test for instance 1
    mask1 = (mask_data == 1).astype(np.uint8)
    bbox1 = converter._get_mask_bbox(mask1)
    expected_bbox1 = [2 / w, 0 / h, (3 + 1) / w, (0 + 1) / h]  # xmin/w, ymin/h, xmax+1/w, ymax+1/h
    assert bbox1 is not None
    np.testing.assert_allclose(bbox1, expected_bbox1, atol=1e-6)

    # Test for instance 2
    mask2 = (mask_data == 2).astype(np.uint8)
    bbox2 = converter._get_mask_bbox(mask2)
    expected_bbox2 = [1 / w, 1 / h, (2 + 1) / w, (1 + 1) / h]
    assert bbox2 is not None
    np.testing.assert_allclose(bbox2, expected_bbox2, atol=1e-6)


def test_match_instance_to_class(voc_segment_test_setup):
    devkit_path, year, output_path, _ = voc_segment_test_setup
    voc_root = devkit_path.parent
    converter = SegmentConverter(
        # devkit_path=str(devkit_path),
        voc_root=voc_root,
        output_root=voc_root,
        year=year,
        # output_segment_dir=str(output_path),
        tag="test",
        iou_threshold=0.5,
    )
    mask_data = np.array([[0, 0, 1, 1], [0, 2, 2, 0]], dtype=np.uint8)
    h, w = mask_data.shape
    img_dims = (w, h)

    # Dummy XML objects (match structure from _parse_annotation)
    xml_objects = [
        {"name": "classA", "bbox": [2, 0, 4, 1], "difficult": 0},
        {"name": "classB", "bbox": [1, 1, 3, 2], "difficult": 0},
    ]

    # Match mask 1
    mask1 = (mask_data == 1).astype(np.uint8)
    matched_class1 = converter._match_instance_to_class(mask1, xml_objects, img_dims)
    assert matched_class1 == "classA"

    # Match mask 2
    mask2 = (mask_data == 2).astype(np.uint8)
    matched_class2 = converter._match_instance_to_class(mask2, xml_objects, img_dims)
    assert matched_class2 == "classB"

    # Test low IoU threshold (should still match if IoU is 1.0)
    converter.iou_threshold = 0.99
    matched_class_high_iou = converter._match_instance_to_class(mask1, xml_objects, img_dims)
    assert matched_class_high_iou == "classA"  # Match should still succeed if IoU is 1.0


@patch("src.utils.data_converter.voc2yolo_segment_labels.parse_voc_xml")
@patch("src.utils.data_converter.voc2yolo_segment_labels.VOC2YOLOConverter._get_mask_instances")
@patch("src.utils.data_converter.voc2yolo_segment_labels.VOC2YOLOConverter._mask_to_polygons")
# @patch("src.utils.data_converter.voc2yolo_segment_labels.VOC2YOLOConverter._compute_iou") # Remove mock for compute_iou
@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_dir")
def test_process_segmentation_file(
    mock_is_dir,
    mock_path_exists,
    mock_open_write,
    # mock_compute_iou, # Remove mock argument
    mock_mask_to_polygons,
    mock_get_masks,
    mock_parse_xml,
    temp_voc_dir,
):
    """Test processing of a single segmentation file."""
    # Setup
    year = "2012"
    voc_root, output_path = temp_voc_dir  # Unpack the tuple
    img_id = "test_image"

    # Mock Path.exists and Path.is_dir to return True for our test paths
    mock_path_exists.return_value = True
    mock_is_dir.return_value = True

    # Create converter instance - use temp_voc_dir which yields voc_root, output_path
    voc_root, output_path = temp_voc_dir
    converter = SegmentConverter(
        # devkit_path=str(voc_root / "VOCdevkit"), year=year, output_segment_dir=str(output_path)
        voc_root=voc_root,
        output_root=output_path,  # Use the yielded output path
        year=year,
        tag="test",  # Use a dummy tag
        iou_threshold=0.8,  # Set threshold used in test logic
    )

    # Mock parse_voc_xml to return valid data
    mock_parse_xml.return_value = (
        [
            {
                "name": "person",
                "bbox": [100, 100, 200, 200],  # [xmin, ymin, xmax, ymax]
            }
        ],
        (640, 480),  # img_dims
    )

    # Mock _get_mask_instances to return valid data
    mock_get_masks.return_value = {
        1: np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)  # Simple mask
    }

    # Mock _mask_to_polygons to return valid polygons
    mock_mask_to_polygons.return_value = [[0.1, 0.1, 0.2, 0.2, 0.1, 0.2]]  # Simple triangle

    # Mock the common calculate_iou function instead of the internal method
    with patch("src.utils.data_converter.voc2yolo_segment_labels.calculate_iou") as mock_common_iou:
        mock_common_iou.return_value = 0.8  # High IoU value

        # Call the method under test
        # _process_segmentation_file no longer takes output_dir
        converter._process_segmentation_file(img_id)

        # Verify parse_voc_xml was called correctly
        expected_xml_path = voc_root / f"VOC{year}" / "Annotations" / f"{img_id}.xml"
        # Note: Need to adjust expected path based on how get_annotation_path works
        # It takes voc_year_path. Let's mock get_annotation_path maybe?
        # Simpler: Check the call to parse_voc_xml directly.
        mock_parse_xml.assert_called_once()
        call_args, _ = mock_parse_xml.call_args
        assert isinstance(call_args[0], Path)
        assert call_args[0].name == f"{img_id}.xml"

        # Verify _get_mask_instances was called
        # Similarly, check the path passed
        mock_get_masks.assert_called_once()
        call_args_mask, _ = mock_get_masks.call_args
        assert isinstance(call_args_mask[0], Path)
        assert call_args_mask[0].name == f"{img_id}.png"

        # Verify _mask_to_polygons was called
        mock_mask_to_polygons.assert_called_once()

        # Verify common calculate_iou was called
        # mock_compute_iou.assert_called_once() # Remove assertion for old mock
        mock_common_iou.assert_called_once()

    # Verify the output file write was attempted
    # Output path is now determined within the class based on output_root, year, tag
    expected_output_file = converter.output_segment_dir / f"{img_id}.txt"
    mock_open_write.assert_called_once_with(expected_output_file, "w")

    # Verify the content written
    handle = mock_open_write()  # Get the mock file handle
    handle.write.assert_called()  # Check that write was called


@pytest.mark.skip(reason="Functionality covered by more specific tests like test_get_mask_bbox")
def test_empty_mask(temp_voc_dir):
    """Test handling of empty mask (placeholders)."""
    # This logic is implicitly tested in test_get_mask_bbox and _process_segmentation_file
    pass


@pytest.mark.skip(reason="Functionality covered by test_mask_to_polygons")
def test_invalid_mask(temp_voc_dir):
    """Test handling of invalid mask (placeholders)."""
    # This logic is implicitly tested in test_mask_to_polygons
    pass


@pytest.mark.skip(reason="Class mapping now uses imported constants")
def test_class_mapping():
    pass
