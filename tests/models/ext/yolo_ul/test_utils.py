"""
Tests for Ultralytics YOLO utility functions
"""

import os

# Update the import path to access the module from the src directory
from src.models.ext.yolo_ul.utils import (
    COCO_ROOT,
    DATA_ROOT,
    VOC2007_DIR,
    VOC2012_DIR,
    VOC_ROOT,
    load_yaml_config,
    resolve_source_path,
)


class TestSourcePathResolution:
    """Test cases for source path resolution functionality"""

    def test_absolute_path(self):
        """Test handling of absolute paths"""
        abs_path = "/absolute/path/to/image.jpg"
        path = resolve_source_path(abs_path)
        assert path == abs_path

    def test_url_path(self):
        """Test handling of URL paths"""
        url = "http://example.com/image.jpg"
        path = resolve_source_path(url)
        assert path == url

    def test_voc2007_prefix(self, monkeypatch):
        """Test handling of VOC2007 prefix"""
        dummy_path = "/tmp/dummy_voc2007"
        monkeypatch.setattr('src.models.ext.yolo_ul.utils.VOC2007_DIR', dummy_path)
        relative_path = "JPEGImages/000001.jpg"
        source = f"VOC2007/{relative_path}"
        path = resolve_source_path(source)
        assert path == os.path.join(dummy_path, relative_path)

    def test_voc2012_prefix(self, monkeypatch):
        """Test handling of VOC2012 prefix"""
        dummy_path = "/tmp/dummy_voc2012"
        monkeypatch.setattr('src.models.ext.yolo_ul.utils.VOC2012_DIR', dummy_path)
        relative_path = "JPEGImages/000001.jpg"
        source = f"VOC2012/{relative_path}"
        path = resolve_source_path(source)
        assert path == os.path.join(dummy_path, relative_path)

    def test_coco_prefix(self, monkeypatch):
        """Test handling of COCO prefix"""
        dummy_path = "/tmp/dummy_coco"
        monkeypatch.setattr('src.models.ext.yolo_ul.utils.COCO_ROOT', dummy_path)
        relative_path = "train2017/000000000001.jpg"
        source = f"COCO/{relative_path}"
        path = resolve_source_path(source)
        assert path == os.path.join(dummy_path, relative_path)

    def test_vocdevkit_prefix(self, monkeypatch):
        """Test handling of VOCdevkit prefix"""
        dummy_path = "/tmp/dummy_voc_root"
        monkeypatch.setattr('src.models.ext.yolo_ul.utils.VOC_ROOT', dummy_path)
        relative_path = "VOC2007/JPEGImages/000001.jpg"
        source = f"VOCdevkit/{relative_path}"
        path = resolve_source_path(source)
        assert path == os.path.join(dummy_path, relative_path)

    def test_relative_path(self):
        """Test handling of relative paths"""
        relative_path = "images/test.jpg"
        path = resolve_source_path(relative_path)
        assert path == os.path.join(DATA_ROOT, relative_path)


class TestYamlConfig:
    """Test cases for YAML configuration loading"""

    def test_load_yaml_config(self, tmp_path):
        """Test loading a YAML configuration file"""
        # Create a temporary YAML file
        test_config = {"names": ["person", "car", "bicycle"], "nc": 3, "path": "some/path"}

        config_path = tmp_path / "test_config.yaml"

        import yaml

        with open(config_path, "w") as f:
            yaml.dump(test_config, f)

        # Load the config
        loaded_config = load_yaml_config(config_path)

        # Verify the loaded config matches
        assert loaded_config["names"] == test_config["names"]
        assert loaded_config["nc"] == test_config["nc"]
        assert loaded_config["path"] == test_config["path"]
