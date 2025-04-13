import csv
import os
import random
import tempfile
from typing import Any, Dict

import pytest

# Import the converter module for proper global state access
from src.dataops.cov_segm import converter as converter_module
from src.dataops.cov_segm.converter import _get_sampled_mapping_info, load_mapping_config


@pytest.fixture
def reset_stats_counters():
    """Reset the global stats_counters before and after each test."""
    # Reset before test
    converter_module.stats_counters = {
        "total_samples": 0,
        "total_conversations": 0,
        "total_phrases": 0,
        "processed_samples": 0,
        "skipped_samples_load_error": 0,
        "skipped_samples_no_mapping": 0,
        "skipped_samples_sampling": 0,
        "skipped_samples_zero_masks": 0,
        "mask_convert_no_polygon": 0,
        "mask_convert_multiple_polygons": 0,
        "skipped_existing_labels": 0,
        "skipped_existing_images": 0,
        "generated_annotations": 0,
        "images_with_annotations": set(),
        "copied_images": 0,
        "class_annotations_per_image": {},
    }
    yield
    # Reset after test
    converter_module.stats_counters = {
        "total_samples": 0,
        "total_conversations": 0,
        "total_phrases": 0,
        "processed_samples": 0,
        "skipped_samples_load_error": 0,
        "skipped_samples_no_mapping": 0,
        "skipped_samples_sampling": 0,
        "skipped_samples_zero_masks": 0,
        "mask_convert_no_polygon": 0,
        "mask_convert_multiple_polygons": 0,
        "skipped_existing_labels": 0,
        "skipped_existing_images": 0,
        "generated_annotations": 0,
        "images_with_annotations": set(),
        "copied_images": 0,
        "class_annotations_per_image": {},
    }


@pytest.fixture
def sample_phrase_map() -> Dict[str, Dict[str, Any]]:
    """Create a sample phrase map for testing."""
    return {
        "car": {
            "class_id": 0,
            "class_name": "Car",
            "sampling_ratio": 1.0,  # Always include
        },
        "truck": {
            "class_id": 1,
            "class_name": "Truck",
            "sampling_ratio": 0.5,  # 50% chance to include
        },
        "bus": {
            "class_id": 2,
            "class_name": "Bus",
            "sampling_ratio": 0.0,  # Never include
        },
        "vehicle": {  # Added for test_get_sampled_mapping_info_multiple_phrases
            "class_id": 3,
            "class_name": "Vehicle",
            "sampling_ratio": 1.0,  # Always include
        },
    }


def test_get_sampled_mapping_info_match_found(sample_phrase_map, reset_stats_counters):
    """Test _get_sampled_mapping_info when a match is found."""
    random.seed(42)

    # Create a conversation with phrases as list of dicts
    conversation = {"phrases": [{"text": "car"}, {"text": "vehicle"}, {"text": "automobile"}]}

    result = _get_sampled_mapping_info(conversation, sample_phrase_map)

    assert result is not None
    mapping_info, matched_phrase = result
    assert mapping_info["class_id"] == 0
    assert mapping_info["class_name"] == "Car"
    assert matched_phrase == "car"
    assert converter_module.stats_counters["skipped_samples_no_mapping"] == 0
    assert converter_module.stats_counters["skipped_samples_sampling"] == 0


def test_get_sampled_mapping_info_multiple_phrases(sample_phrase_map, reset_stats_counters):
    """Test _get_sampled_mapping_info with multiple phrases, first one not matched."""
    random.seed(42)

    # Create a conversation with phrases as list of dicts
    conversation = {"phrases": [{"text": "unknown"}, {"text": "vehicle"}, {"text": "automobile"}]}

    result = _get_sampled_mapping_info(conversation, sample_phrase_map)

    assert result is not None
    mapping_info, matched_phrase = result
    assert mapping_info["class_id"] == 3
    assert mapping_info["class_name"] == "Vehicle"
    assert matched_phrase == "vehicle"
    assert converter_module.stats_counters["skipped_samples_no_mapping"] == 0
    assert converter_module.stats_counters["skipped_samples_sampling"] == 0


def test_get_sampled_mapping_info_sampling_skip(
    sample_phrase_map, reset_stats_counters, monkeypatch
):
    """Test _get_sampled_mapping_info where match is found but sampling skips it."""
    monkeypatch.setattr(random, "random", lambda: 0.9)  # Higher than truck's 0.5

    # Create a conversation with phrases as list of dicts
    conversation = {"phrases": [{"text": "truck"}]}

    result = _get_sampled_mapping_info(conversation, sample_phrase_map)

    assert result is None
    assert converter_module.stats_counters["skipped_samples_no_mapping"] == 0
    assert converter_module.stats_counters["skipped_samples_sampling"] == 1


def test_get_sampled_mapping_info_no_match(sample_phrase_map, reset_stats_counters):
    """Test _get_sampled_mapping_info when no phrases match."""
    # Create a conversation with phrases as list of dicts
    conversation = {"phrases": [{"text": "bicycle"}, {"text": "motorcycle"}, {"text": "scooter"}]}

    result = _get_sampled_mapping_info(conversation, sample_phrase_map)

    assert result is None
    assert converter_module.stats_counters["skipped_samples_no_mapping"] == 1
    assert converter_module.stats_counters["skipped_samples_sampling"] == 0


def test_get_sampled_mapping_info_empty_phrases(sample_phrase_map, reset_stats_counters):
    """Test _get_sampled_mapping_info with empty phrases list."""
    # Create a conversation with empty phrases
    conversation = {"phrases": []}

    # Call function - should return None (no phrases to match)
    result = _get_sampled_mapping_info(conversation, sample_phrase_map)

    # Check result
    assert result is None

    # Check correct counter was incremented
    assert converter_module.stats_counters["skipped_samples_no_mapping"] == 1
    assert converter_module.stats_counters["skipped_samples_sampling"] == 0


def test_load_mapping_config_valid_csv():
    """Test loading a valid mapping config CSV."""
    # Create a temporary CSV file with valid mapping data
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp_file:
        writer = csv.writer(tmp_file)
        writer.writerow(["yolo_class_id", "yolo_class_name", "sampling_ratio", "hf_phrase"])
        writer.writerow(["0", "Car", "1.0", "car"])
        writer.writerow(["1", "Person", "0.8", "person"])
        tmp_path = tmp_file.name

    try:
        # Test the function
        phrase_map, class_names = load_mapping_config(tmp_path)

        # Check results
        assert len(phrase_map) == 2  # car and person phrases
        assert len(class_names) == 2  # Car and Person classes

        assert phrase_map["car"]["class_id"] == 0
        assert phrase_map["car"]["class_name"] == "Car"
        assert phrase_map["car"]["sampling_ratio"] == 1.0

        assert phrase_map["person"]["class_id"] == 1
        assert phrase_map["person"]["class_name"] == "Person"
        assert phrase_map["person"]["sampling_ratio"] == 0.8

        assert class_names[0] == "Car"
        assert class_names[1] == "Person"
    finally:
        # Clean up
        os.unlink(tmp_path)


def test_load_mapping_config_missing_columns():
    """Test loading a mapping config with missing required columns."""
    # Create a temporary CSV file with missing columns
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp_file:
        writer = csv.writer(tmp_file)
        writer.writerow(["yolo_class_id", "yolo_class_name", "hf_phrase"])  # Missing sampling_ratio
        writer.writerow(["0", "Car", "car"])
        tmp_path = tmp_file.name

    try:
        # Test the function - should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            load_mapping_config(tmp_path)

        # Check error message
        assert "Missing required columns" in str(excinfo.value)
        assert "sampling_ratio" in str(excinfo.value)
    finally:
        # Clean up
        os.unlink(tmp_path)


def test_load_mapping_config_invalid_sampling_ratio():
    """Test loading a mapping config with invalid sampling ratio."""
    # Create a temporary CSV file with invalid sampling ratio
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp_file:
        writer = csv.writer(tmp_file)
        writer.writerow(["yolo_class_id", "yolo_class_name", "sampling_ratio", "hf_phrase"])
        writer.writerow(["0", "Car", "2.0", "car"])  # 2.0 is > 1.0
        # Add another valid row to prevent the "No valid mappings" error
        writer.writerow(["1", "Person", "0.8", "person"])
        tmp_path = tmp_file.name

    try:
        # Test the function - should load only the valid row
        phrase_map, class_names = load_mapping_config(tmp_path)

        # Check that only the valid row was loaded
        assert len(phrase_map) == 1  # Only the person phrase
        assert len(class_names) == 1  # Only the Person class
        assert class_names[1] == "Person"
        assert "car" not in phrase_map  # Car should be skipped due to invalid sampling ratio

    finally:
        # Clean up
        os.unlink(tmp_path)


def test_load_mapping_config_empty_csv():
    """Test loading an empty mapping config."""
    # Create a temporary empty CSV file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp_file:
        writer = csv.writer(tmp_file)
        writer.writerow(["yolo_class_id", "yolo_class_name", "sampling_ratio", "hf_phrase"])
        # No data rows
        tmp_path = tmp_file.name

    try:
        # Test the function - should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            load_mapping_config(tmp_path)

        # Check error message
        assert "No valid mappings found" in str(excinfo.value)
    finally:
        # Clean up
        os.unlink(tmp_path)
