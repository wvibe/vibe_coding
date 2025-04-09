"""
Unit tests for Pascal VOC dataset loader using pytest
"""

import os
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.utils.data_loaders.cv.voc import PascalVOCDataset


@pytest.fixture
def voc_root():
    """Get mock VOC dataset root path"""
    return "/mock/data/VOCdevkit"


@pytest.fixture
def mock_image():
    """Create mock image data"""
    return Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))


@pytest.fixture
def mock_annotation():
    """Create mock annotation data"""
    return """<?xml version="1.0" ?>
        <annotation>
            <size>
                <width>400</width>
                <height>300</height>
                <depth>3</depth>
            </size>
            <object>
                <name>car</name>
                <bndbox>
                    <xmin>100</xmin>
                    <ymin>50</ymin>
                    <xmax>300</xmax>
                    <ymax>150</ymax>
                </bndbox>
            </object>
            <object>
                <name>person</name>
                <bndbox>
                    <xmin>50</xmin>
                    <ymin>30</ymin>
                    <xmax>100</xmax>
                    <ymax>100</ymax>
                </bndbox>
            </object>
        </annotation>
    """


@pytest.fixture
def voc_paths(voc_root):
    """Get VOC dataset paths based on mock root"""
    return {
        "2007": {
            "root": os.path.join(voc_root, "VOC2007"),
            "splits": os.path.join(voc_root, "VOC2007", "ImageSets", "Main"),
            "images": os.path.join(voc_root, "VOC2007", "JPEGImages"),
            "annotations": os.path.join(voc_root, "VOC2007", "Annotations"),
        },
        "2012": {
            "root": os.path.join(voc_root, "VOC2012"),
            "splits": os.path.join(voc_root, "VOC2012", "ImageSets", "Main"),
            "images": os.path.join(voc_root, "VOC2012", "JPEGImages"),
            "annotations": os.path.join(voc_root, "VOC2012", "Annotations"),
        },
    }


def test_init_single_year(voc_root, voc_paths):
    """Test dataset initialization with single year"""
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="000001\n000002\n")):
            dataset = PascalVOCDataset(years=["2007"], split_file="train.txt", data_dir=voc_root)

            assert len(dataset.image_info) == 2
            assert dataset.image_info[0]["year"] == "2007"
            assert dataset.image_info[0]["id"] == "000001"
            assert dataset.image_info[0]["image_path"] == os.path.join(
                voc_paths["2007"]["images"], "000001.jpg"
            )
            assert dataset.image_info[0]["annotation_path"] == os.path.join(
                voc_paths["2007"]["annotations"], "000001.xml"
            )


def test_init_multiple_years(voc_root, voc_paths):
    """Test dataset initialization with multiple years"""
    split_files = {
        os.path.join(voc_paths["2007"]["splits"], "train.txt"): "000001\n000002\n",
        os.path.join(voc_paths["2012"]["splits"], "train.txt"): "000003\n000004\n",
    }

    def mock_exists(path):
        return path in split_files or path.endswith((".jpg", ".xml"))

    def mock_open_file(filename, *args, **kwargs):
        if filename in split_files:
            return mock_open(read_data=split_files[filename])()
        return mock_open(read_data="")()

    with patch("os.path.exists", side_effect=mock_exists):
        with patch("builtins.open", create=True) as mock_file:
            mock_file.side_effect = mock_open_file

            dataset = PascalVOCDataset(
                years=["2007", "2012"], split_file="train.txt", data_dir=voc_root
            )

            assert len(dataset.image_info) == 4
            assert dataset.image_info[0]["year"] == "2007"
            assert dataset.image_info[2]["year"] == "2012"


def test_invalid_year(voc_root):
    """Test initialization with invalid year"""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            PascalVOCDataset(years=["invalid"], split_file="train.txt", data_dir=voc_root)


def test_getitem(mock_image, mock_annotation, voc_root, voc_paths):
    """Test __getitem__ method"""
    with patch("PIL.Image.open", return_value=mock_image):
        with patch("xml.etree.ElementTree.parse") as mock_parse:
            mock_tree = MagicMock()
            mock_tree.getroot.return_value = ET.fromstring(mock_annotation)
            mock_parse.return_value = mock_tree

            with patch("os.path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="000001\n")):
                    dataset = PascalVOCDataset(
                        years=["2007"], split_file="val.txt", data_dir=voc_root
                    )
                    sample = dataset[0]

                    # Check sample structure
                    assert isinstance(sample, dict)
                    assert "image" in sample
                    assert "boxes" in sample
                    assert "labels" in sample
                    assert "image_id" in sample

                    # Check image ID format
                    assert sample["image_id"] == "2007/000001"

                    # Check tensor shapes and types
                    assert isinstance(sample["image"], torch.Tensor)
                    assert sample["image"].shape == (3, 416, 416)  # After transforms
                    assert isinstance(sample["boxes"], torch.Tensor)
                    assert isinstance(sample["labels"], torch.Tensor)
                    assert len(sample["boxes"].shape) == 2  # [num_boxes, 4]
                    assert len(sample["labels"].shape) == 1  # [num_boxes]

                    # Check number of objects
                    assert len(sample["boxes"]) == 2
                    assert len(sample["labels"]) == 2

                    # Check label values (car=6, person=14 in VOC classes)
                    assert sample["labels"].tolist() == [6, 14]


def test_sample_pct(voc_root):
    """Test using sample percentage of dataset"""
    # Create mock data with 100 image IDs
    mock_ids = [f"{i:06d}" for i in range(100)]
    mock_data = "\n".join(mock_ids)

    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=mock_data)):
            # Test with 10% subset
            dataset_10p = PascalVOCDataset(
                years=["2007"], split_file="train.txt", data_dir=voc_root, sample_pct=0.1
            )

            # Test with 50% subset
            dataset_50p = PascalVOCDataset(
                years=["2007"], split_file="train.txt", data_dir=voc_root, sample_pct=0.5
            )

            # Verify dataset sizes are approximately correct
            assert len(dataset_10p.image_info) == 10
            assert len(dataset_50p.image_info) == 50

            # Verify datasets are different (random subset)
            ids_10p = set(info["id"] for info in dataset_10p.image_info)
            ids_50p = set(info["id"] for info in dataset_50p.image_info)
            assert ids_10p.issubset(ids_50p) or not ids_10p.issubset(ids_50p)


def test_invalid_sample_pct(voc_root):
    """Test with invalid sample percentage values"""
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="000001\n000002\n")):
            # Test with negative value
            with pytest.raises(ValueError):
                PascalVOCDataset(
                    years=["2007"], split_file="train.txt", data_dir=voc_root, sample_pct=-0.1
                )

            # Test with too large value
            with pytest.raises(ValueError):
                PascalVOCDataset(
                    years=["2007"], split_file="train.txt", data_dir=voc_root, sample_pct=1.1
                )


def test_class_specific_file(voc_root, voc_paths):
    """Test loading class-specific dataset file"""
    # Mock class-specific file for 'person' class with positive and negative samples
    # Format: "<image_id> <label>" where label is 1 (present), -1 (difficult), 0 (not present)
    person_train_data = "000001 1\n000002 0\n000003 1\n000004 -1\n000005 1\n"

    class_split_file = os.path.join(voc_paths["2007"]["splits"], "person_train.txt")

    def mock_exists(path):
        return path == class_split_file or path.endswith((".jpg", ".xml"))

    def mock_open_file(filename, *args, **kwargs):
        if filename == class_split_file:
            return mock_open(read_data=person_train_data)()
        return mock_open(read_data="")()

    with patch("os.path.exists", side_effect=mock_exists):
        with patch("builtins.open", create=True) as mock_file:
            mock_file.side_effect = mock_open_file

            # Create dataset with person class-specific file
            dataset = PascalVOCDataset(
                years=["2007"], split_file="person_train.txt", data_dir=voc_root
            )

            # Should only include images with label 1 (class present)
            assert len(dataset.image_info) == 3
            assert dataset.image_info[0]["id"] == "000001"
            assert dataset.image_info[1]["id"] == "000003"
            assert dataset.image_info[2]["id"] == "000005"


def test_combined_class_and_sampling(voc_root, voc_paths):
    """Test using both class-specific file and sampling percentage"""
    # Create mock data with 100 image IDs with class present (label 1)
    mock_ids = [f"{i:06d} 1" for i in range(100)]
    mock_data = "\n".join(mock_ids)

    class_split_file = os.path.join(voc_paths["2007"]["splits"], "person_train.txt")

    def mock_exists(path):
        return path == class_split_file or path.endswith((".jpg", ".xml"))

    def mock_open_file(filename, *args, **kwargs):
        if filename == class_split_file:
            return mock_open(read_data=mock_data)()
        return mock_open(read_data="")()

    with patch("os.path.exists", side_effect=mock_exists):
        with patch("builtins.open", create=True) as mock_file:
            mock_file.side_effect = mock_open_file

            # Create dataset with person class and 20% sample
            dataset = PascalVOCDataset(
                years=["2007"], split_file="person_train.txt", data_dir=voc_root, sample_pct=0.2
            )

            # Should include 20% of the 100 images
            assert len(dataset.image_info) == 20


def test_sample_pct_edge_cases(voc_root):
    """Test edge cases for sample_pct (0.0 and 1.0)"""
    # Create mock data with 100 image IDs
    mock_ids = [f"{i:06d}" for i in range(100)]
    mock_data = "\n".join(mock_ids)

    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=mock_data)):
            # Test with 100% (should include all images)
            dataset_100p = PascalVOCDataset(
                years=["2007"], split_file="train.txt", data_dir=voc_root, sample_pct=1.0
            )
            assert len(dataset_100p.image_info) == 100

            # Test with minimum value
            dataset_min = PascalVOCDataset(
                years=["2007"], split_file="train.txt", data_dir=voc_root, sample_pct=0.01
            )
            assert len(dataset_min.image_info) == 1


def test_collate_function(voc_root, mock_image, mock_annotation):
    """Test the collate_fn for creating batches"""
    with patch("PIL.Image.open", return_value=mock_image):
        with patch("xml.etree.ElementTree.parse") as mock_parse:
            mock_tree = MagicMock()
            mock_tree.getroot.return_value = ET.fromstring(mock_annotation)
            mock_parse.return_value = mock_tree

            with patch("os.path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="000001\n000002\n")):
                    dataset = PascalVOCDataset(
                        years=["2007"], split_file="val.txt", data_dir=voc_root
                    )

                    # Create samples manually
                    samples = [dataset[0], dataset[1]]

                    # Test the collate function
                    batch = dataset.collate_fn(samples)

                    # Check batch structure
                    assert "images" in batch
                    assert "boxes" in batch
                    assert "labels" in batch
                    assert "image_ids" in batch

                    # Check batch shapes and types
                    assert isinstance(batch["images"], torch.Tensor)
                    assert batch["images"].shape == (
                        2,
                        3,
                        416,
                        416,
                    )  # [batch_size, channels, height, width]
                    assert isinstance(batch["boxes"], list)
                    assert len(batch["boxes"]) == 2
                    assert isinstance(batch["labels"], list)
                    assert len(batch["labels"]) == 2
                    assert isinstance(batch["image_ids"], list)
                    assert len(batch["image_ids"]) == 2


def test_empty_split_file(voc_root):
    """Test handling of empty split files"""
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="")):
            dataset = PascalVOCDataset(years=["2007"], split_file="empty.txt", data_dir=voc_root)

            # Dataset should be empty but initialized
            assert len(dataset.image_info) == 0

            # Check that __len__ works with empty dataset
            assert len(dataset) == 0


def test_missing_image_handling(voc_root, mock_annotation):
    """Test handling of missing images"""

    # Create a mock for Image.open that raises FileNotFoundError
    def mock_image_open_error(path):
        raise FileNotFoundError(f"Mock file not found: {path}")

    with patch("PIL.Image.open", side_effect=mock_image_open_error):
        with patch("xml.etree.ElementTree.parse") as mock_parse:
            mock_tree = MagicMock()
            mock_tree.getroot.return_value = ET.fromstring(mock_annotation)
            mock_parse.return_value = mock_tree

            with patch("os.path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="000001\n")):
                    dataset = PascalVOCDataset(
                        years=["2007"], split_file="train.txt", data_dir=voc_root
                    )

                    # Accessing the item should raise FileNotFoundError
                    with pytest.raises(FileNotFoundError):
                        _ = dataset[0]


def test_malformed_xml(voc_root, mock_image):
    """Test handling of malformed XML annotations"""
    malformed_xml = """<?xml version="1.0" ?>
        <annotation>
            <size>
                <width>400</width>
                <height>300</height>
                <depth>3</depth>
            </size>
            <object>
                <!-- Missing name tag -->
                <bndbox>
                    <xmin>100</xmin>
                    <ymin>50</ymin>
                    <xmax>300</xmax>
                    <ymax>150</ymax>
                </bndbox>
            </object>
        </annotation>
    """

    with patch("PIL.Image.open", return_value=mock_image):
        with patch("xml.etree.ElementTree.parse") as mock_parse:
            mock_tree = MagicMock()
            mock_tree.getroot.return_value = ET.fromstring(malformed_xml)
            mock_parse.return_value = mock_tree

            with patch("os.path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="000001\n")):
                    dataset = PascalVOCDataset(
                        years=["2007"], split_file="train.txt", data_dir=voc_root
                    )

                    # Should handle the malformed XML gracefully by using default values
                    sample = dataset[0]

                    # Should have empty box list
                    assert sample["boxes"].shape[0] == 1  # Should have the default box
                    assert sample["labels"].shape[0] == 1  # Should have the default label


def test_custom_image_size(voc_root, mock_image, mock_annotation):
    """Test custom image size configuration"""
    with patch("PIL.Image.open", return_value=mock_image):
        with patch("xml.etree.ElementTree.parse") as mock_parse:
            mock_tree = MagicMock()
            mock_tree.getroot.return_value = ET.fromstring(mock_annotation)
            mock_parse.return_value = mock_tree

            with patch("os.path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="000001\n")):
                    # Create dataset with custom image size
                    custom_size = 608  # Common YOLO size
                    dataset = PascalVOCDataset(
                        years=["2007"],
                        split_file="val.txt",
                        data_dir=voc_root,
                        img_size=custom_size,
                    )

                    # Get sample and check shape
                    sample = dataset[0]
                    assert sample["image"].shape == (3, custom_size, custom_size)

                    # Make sure the transforms use the correct size
                    assert dataset.transform.transforms[0].max_size == custom_size
                    assert dataset.transform.transforms[1].min_height == custom_size
                    assert dataset.transform.transforms[1].min_width == custom_size
