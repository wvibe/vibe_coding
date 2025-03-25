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

from data_loaders.object_detection.voc import PascalVOCDataset


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
            dataset = PascalVOCDataset(years=["2007"], split="train", data_dir=voc_root)

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
                years=["2007", "2012"], split="train", data_dir=voc_root
            )

            assert len(dataset.image_info) == 4
            assert dataset.image_info[0]["year"] == "2007"
            assert dataset.image_info[2]["year"] == "2012"


def test_invalid_year(voc_root):
    """Test initialization with invalid year"""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            PascalVOCDataset(years=["invalid"], split="train", data_dir=voc_root)


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
                        years=["2007"], split="val", data_dir=voc_root
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


def test_collate_fn(voc_root):
    """Test collate_fn for batching samples"""
    mock_samples = [
        {
            "image": torch.randn(3, 416, 416),
            "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            "labels": torch.tensor([1]),
            "image_id": "2007/000001",
        },
        {
            "image": torch.randn(3, 416, 416),
            "boxes": torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]]),
            "labels": torch.tensor([2, 3]),
            "image_id": "2007/000002",
        },
    ]

    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="000001\n000002\n")):
            dataset = PascalVOCDataset(years=["2007"], split="train", data_dir=voc_root)
            batch = dataset.collate_fn(mock_samples)

            # Check batch structure
            assert isinstance(batch, dict)
            assert "images" in batch
            assert "boxes" in batch
            assert "labels" in batch
            assert "image_ids" in batch

            # Check batch shapes and types
            assert isinstance(batch["images"], torch.Tensor)
            assert batch["images"].shape == (2, 3, 416, 416)
            assert isinstance(batch["boxes"], list)
            assert isinstance(batch["labels"], list)
            assert isinstance(batch["image_ids"], list)
            assert len(batch["boxes"]) == 2
            assert len(batch["labels"]) == 2
            assert len(batch["image_ids"]) == 2
