import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Assuming predict_segment.py is importable relative to project root
# This might need adjustment based on actual test runner setup
from src.models.ext.yolov11.predict_segment import (
    _extract_and_average_times,
    construct_source_path,
    process_source,
)

# --- Tests for construct_source_path ---


@patch("src.models.ext.yolov11.predict_segment.os.getenv")
@patch("src.models.ext.yolov11.predict_segment.Path.is_dir")
@patch("src.models.ext.yolov11.predict_segment.Path.iterdir")
def test_construct_source_path_success(mock_iterdir, mock_is_dir, mock_getenv):
    """Test successful path construction."""
    mock_getenv.return_value = "/fake/voc/base"
    mock_is_dir.return_value = True
    mock_iterdir.return_value = [Path("/fake/voc/base/images/val2007/img1.jpg")]  # Not empty
    dataset_id = "voc"
    tag = "val2007"
    expected_path = Path("/fake/voc/base/images/val2007")

    result_path = construct_source_path(dataset_id, tag)

    mock_getenv.assert_called_once_with("VOC_SEGMENT")
    assert result_path == expected_path
    mock_is_dir.assert_called_once()


@patch("src.models.ext.yolov11.predict_segment.os.getenv")
def test_construct_source_path_unknown_dataset(mock_getenv, caplog):
    """Test error handling for unknown dataset ID."""
    mock_getenv.return_value = None  # Simulate unknown dataset mapping
    dataset_id = "unknown_ds"
    tag = "test"

    with pytest.raises(SystemExit):
        construct_source_path(dataset_id, tag)

    assert f"Unknown dataset identifier '{dataset_id}'" in caplog.text
    mock_getenv.assert_not_called()  # getenv shouldn't be called if mapping fails


@patch("src.models.ext.yolov11.predict_segment.os.getenv")
def test_construct_source_path_env_var_not_set(mock_getenv, caplog):
    """Test error handling when the required environment variable is not set."""
    mock_getenv.return_value = None
    dataset_id = "voc"
    tag = "val2007"

    with pytest.raises(SystemExit):
        construct_source_path(dataset_id, tag)

    assert "Environment variable 'VOC_SEGMENT' for dataset 'voc' is not set" in caplog.text
    mock_getenv.assert_called_once_with("VOC_SEGMENT")


@patch("src.models.ext.yolov11.predict_segment.os.getenv")
@patch("src.models.ext.yolov11.predict_segment.Path.is_dir")
def test_construct_source_path_not_a_directory(mock_is_dir, mock_getenv, caplog):
    """Test error handling when the constructed path is not a directory."""
    mock_getenv.return_value = "/fake/path"
    mock_is_dir.return_value = False
    dataset_id = "voc"
    tag = "test"

    with pytest.raises(SystemExit):
        construct_source_path(dataset_id, tag)

    expected_path = Path("/fake/path/images/test")
    assert f"Constructed source path is not a valid directory: {expected_path}" in caplog.text
    mock_getenv.assert_called_once_with("VOC_SEGMENT")
    mock_is_dir.assert_called_once()


@patch("src.models.ext.yolov11.predict_segment.os.getenv")
@patch("src.models.ext.yolov11.predict_segment.Path.is_dir")
@patch("src.models.ext.yolov11.predict_segment.Path.iterdir")
def test_construct_source_path_empty_directory(mock_iterdir, mock_is_dir, mock_getenv, caplog):
    """Test warning when the source directory is empty."""
    mock_getenv.return_value = "/empty/dir"
    mock_is_dir.return_value = True
    mock_iterdir.return_value = []  # Empty directory
    dataset_id = "voc"
    tag = "empty_tag"
    expected_path = Path("/empty/dir/images/empty_tag")

    caplog.set_level(logging.WARNING)
    result_path = construct_source_path(dataset_id, tag)

    assert result_path == expected_path
    assert f"Source directory is empty: {expected_path}" in caplog.text
    mock_getenv.assert_called_once_with("VOC_SEGMENT")
    mock_is_dir.assert_called_once()
    mock_iterdir.assert_called_once()


# --- Tests for process_source ---


@patch("src.models.ext.yolov11.predict_segment.Path.glob")
def test_process_source_no_sampling(mock_glob):
    """Test processing source without sampling (sample_count=None)."""
    source_dir = Path("/fake/source")
    # Mock glob results
    img1 = source_dir / "image1.jpg"
    img2 = source_dir / "image2.png"
    non_img = source_dir / "readme.txt"
    mock_glob.return_value = [img1, img2, non_img]

    result = process_source(source_dir, None)

    assert result == str(source_dir)
    mock_glob.assert_called_once_with("*")


@patch("src.models.ext.yolov11.predict_segment.Path.glob")
def test_process_source_sample_count_zero(mock_glob):
    """Test processing source with sample_count=0 (should process all)."""
    source_dir = Path("/fake/source")
    img1 = source_dir / "image1.jpg"
    mock_glob.return_value = [img1]

    result = process_source(source_dir, 0)

    assert result == str(source_dir)
    mock_glob.assert_called_once_with("*")


@patch("src.models.ext.yolov11.predict_segment.Path.glob")
@patch("src.models.ext.yolov11.predict_segment.random.sample")
def test_process_source_with_sampling(mock_random_sample, mock_glob):
    """Test processing source with random sampling."""
    source_dir = Path("/fake/source")
    img1 = source_dir / "image1.jpg"
    img2 = source_dir / "image2.PNG"
    img3 = source_dir / "image3.jpeg"
    img4 = source_dir / "image4.bmp"
    non_img = source_dir / "notes.txt"
    all_files = [img1, img2, img3, img4, non_img]
    valid_images = [img1, img2, img3, img4]
    mock_glob.return_value = all_files

    sample_count = 2
    # Mock random.sample to return a specific subset
    mock_random_sample.return_value = [img2, img4]

    result = process_source(source_dir, sample_count)

    expected_result = [str(img2), str(img4)]
    assert result == expected_result
    mock_glob.assert_called_once_with("*")
    # Check that random.sample was called with the list of valid images and count
    mock_random_sample.assert_called_once_with(valid_images, sample_count)


@patch("src.models.ext.yolov11.predict_segment.Path.glob")
@patch("src.models.ext.yolov11.predict_segment.random.sample")
def test_process_source_sample_more_than_found(mock_random_sample, mock_glob, caplog):
    """Test sampling when sample_count exceeds the number of found images."""
    source_dir = Path("/fake/source")
    img1 = source_dir / "image1.jpg"
    img2 = source_dir / "image2.png"
    valid_images = [img1, img2]
    mock_glob.return_value = valid_images
    sample_count = 5
    num_found = len(valid_images)

    # random.sample will be called with k=num_found in this case
    mock_random_sample.return_value = valid_images  # It samples all available

    caplog.set_level(logging.WARNING)
    result = process_source(source_dir, sample_count)

    expected_result = [str(img1), str(img2)]
    assert result == expected_result
    assert f"Requested {sample_count} images, but only found {num_found}" in caplog.text
    mock_glob.assert_called_once_with("*")
    # random.sample is called with the number found when count > found
    mock_random_sample.assert_called_once_with(valid_images, num_found)


@patch("src.models.ext.yolov11.predict_segment.Path.glob")
def test_process_source_no_images_found(mock_glob, caplog):
    """Test processing source when no image files are found."""
    source_dir = Path("/fake/source")
    non_img1 = source_dir / "readme.md"
    non_img2 = source_dir / ".DS_Store"
    mock_glob.return_value = [non_img1, non_img2]

    caplog.set_level(logging.WARNING)
    result = process_source(source_dir, None)

    assert result == str(source_dir)  # Returns dir path if no images
    assert f"No image files found in source directory: {source_dir}" in caplog.text
    mock_glob.assert_called_once_with("*")


# --- Tests for _extract_and_average_times ---


def create_mock_result(preprocess=0.0, inference=0.0, postprocess=0.0):
    """Helper to create a mock result object with a speed dictionary."""
    mock = Mock()
    mock.speed = {
        "preprocess": preprocess,
        "inference": inference,
        "postprocess": postprocess,
    }
    return mock


def test_extract_average_times_valid():
    """Test averaging times with valid speed dictionaries in all results."""
    results = [
        create_mock_result(1.0, 10.0, 2.0),  # total 13.0
        create_mock_result(1.5, 11.0, 2.5),  # total 15.0
        create_mock_result(0.5, 9.0, 1.5),  # total 11.0
    ]
    expected_avg = {
        "preprocess": (1.0 + 1.5 + 0.5) / 3,
        "inference": (10.0 + 11.0 + 9.0) / 3,
        "postprocess": (2.0 + 2.5 + 1.5) / 3,
        "total": (13.0 + 15.0 + 11.0) / 3,
    }
    expected_count = 3

    avg_times, valid_count = _extract_and_average_times(results)

    assert valid_count == expected_count
    assert avg_times == pytest.approx(expected_avg)


def test_extract_average_times_some_invalid():
    """Test averaging when some results lack valid speed info (inf=0 or missing)."""
    results = [
        create_mock_result(1.0, 10.0, 2.0),  # Valid (inf > 0)
        Mock(),  # Missing speed attr
        create_mock_result(1.5, 0.0, 2.5),  # Invalid (inf = 0)
        create_mock_result(0.5, 9.0, 1.5),  # Valid
    ]
    expected_avg = {
        "preprocess": (1.0 + 0.5) / 2,
        "inference": (10.0 + 9.0) / 2,
        "postprocess": (2.0 + 1.5) / 2,
        "total": (13.0 + 11.0) / 2,  # (1+10+2) + (0.5+9+1.5) / 2
    }
    expected_count = 2  # Only two valid results

    avg_times, valid_count = _extract_and_average_times(results)

    assert valid_count == expected_count
    assert avg_times == pytest.approx(expected_avg)


def test_extract_average_times_missing_keys():
    """Test averaging when speed keys (pre/post) are missing (should default to 0)."""
    mock1 = Mock()
    mock1.speed = {"inference": 10.0}  # Missing pre/post
    mock2 = Mock()
    mock2.speed = {"preprocess": 1.0, "inference": 12.0}  # Missing post
    results = [mock1, mock2]

    expected_avg = {
        "preprocess": (0.0 + 1.0) / 2,
        "inference": (10.0 + 12.0) / 2,
        "postprocess": (0.0 + 0.0) / 2,
        "total": (10.0 + 13.0) / 2,
    }
    expected_count = 2

    avg_times, valid_count = _extract_and_average_times(results)

    assert valid_count == expected_count
    assert avg_times == pytest.approx(expected_avg)


def test_extract_average_times_no_valid_results(caplog):
    """Test case where no results have valid speed information."""
    results = [
        Mock(),  # Missing speed
        create_mock_result(1.5, 0.0, 2.5),  # inf = 0
    ]

    caplog.set_level(logging.WARNING)
    avg_times, valid_count = _extract_and_average_times(results)

    assert valid_count == 0
    assert avg_times is None
    assert "Could not extract any valid speed information from results." in caplog.text


def test_extract_average_times_empty_input():
    """Test averaging with an empty results list."""
    results = []
    avg_times, valid_count = _extract_and_average_times(results)

    assert valid_count == 0
    assert avg_times is None  # Should return None if count is 0
