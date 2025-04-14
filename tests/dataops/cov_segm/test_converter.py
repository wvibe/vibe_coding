import csv
import os
import random
import tempfile
from typing import Any, Dict

import pytest

# Import the converter module for proper global state access
from src.dataops.cov_segm import converter as converter_module
from src.dataops.cov_segm.converter import _get_sampled_mapping_info, load_mapping_config

# Import the data models
from src.dataops.cov_segm.datamodel import ClsSegment, Phrase


@pytest.fixture
def reset_stats_counters():
    """Reset the global stats_counters before and after each test."""
    # Reset before test
    converter_module.stats_counters = {
        "total_samples": 0,
        "processed_samples": 0,  # Samples successfully loaded by load_sample
        "segments_loaded": 0,
        "skipped_samples_load_error": 0,
        "skipped_segments_no_mapping": 0,  # Segments skipped
        "skipped_segments_sampling": 0,  # Segments skipped
        "skipped_segments_zero_masks": 0,  # Segments skipped
        "segments_processed": 0,  # Segments processed
        "masks_skipped_invalid": 0,
        "masks_for_annotation": 0,
        "mask_convert_no_polygon": 0,
        "mask_convert_multiple_polygons": 0,
        "skipped_existing_labels": 0,  # Labels skipped due to already existing
        "skipped_existing_images": 0,  # Images skipped due to already existing
        "generated_annotations": 0,
        "images_with_annotations": set(),
        "copied_images": 0,
        "class_masks_processed_per_segment": {},  # Dict mapping class_id -> List[int] (counts per segment)
    }
    yield
    # Reset after test
    converter_module.stats_counters = {
        "total_samples": 0,
        "processed_samples": 0,  # Samples successfully loaded by load_sample
        "segments_loaded": 0,
        "skipped_samples_load_error": 0,
        "skipped_segments_no_mapping": 0,  # Segments skipped
        "skipped_segments_sampling": 0,  # Segments skipped
        "skipped_segments_zero_masks": 0,  # Segments skipped
        "segments_processed": 0,  # Segments processed
        "masks_skipped_invalid": 0,
        "masks_for_annotation": 0,
        "mask_convert_no_polygon": 0,
        "mask_convert_multiple_polygons": 0,
        "skipped_existing_labels": 0,  # Labels skipped due to already existing
        "skipped_existing_images": 0,  # Images skipped due to already existing
        "generated_annotations": 0,
        "images_with_annotations": set(),
        "copied_images": 0,
        "class_masks_processed_per_segment": {},  # Dict mapping class_id -> List[int] (counts per segment)
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


@pytest.fixture
def sample_segment_with_phrases() -> Dict[str, ClsSegment]:
    """Create sample ClsSegment objects with different phrases for testing."""
    # Create a dictionary of segments with different phrases
    segments = {}

    # Segment with a "car" phrase
    car_phrase = Phrase(id=1, text="car", type="object")
    segments["car"] = ClsSegment(
        phrases=[car_phrase], type="vehicle", visible_masks=[], full_masks=[]
    )

    # Segment with a "truck" phrase
    truck_phrase = Phrase(id=2, text="truck", type="object")
    segments["truck"] = ClsSegment(
        phrases=[truck_phrase], type="vehicle", visible_masks=[], full_masks=[]
    )

    # Segment with "bus" phrase
    bus_phrase = Phrase(id=3, text="bus", type="object")
    segments["bus"] = ClsSegment(
        phrases=[bus_phrase], type="vehicle", visible_masks=[], full_masks=[]
    )

    # Segment with "unknown" and "vehicle" phrases (for multiple phrase test)
    unknown_phrase = Phrase(id=4, text="unknown", type="object")
    vehicle_phrase = Phrase(id=5, text="vehicle", type="object")
    segments["multiple"] = ClsSegment(
        phrases=[unknown_phrase, vehicle_phrase], type="vehicle", visible_masks=[], full_masks=[]
    )

    # Segment with phrases that don't match any mappings
    bicycle_phrase = Phrase(id=6, text="bicycle", type="object")
    motorcycle_phrase = Phrase(id=7, text="motorcycle", type="object")
    segments["no_match"] = ClsSegment(
        phrases=[bicycle_phrase, motorcycle_phrase], type="vehicle", visible_masks=[], full_masks=[]
    )

    # Segment with empty phrases list
    segments["empty"] = ClsSegment(phrases=[], type="unknown", visible_masks=[], full_masks=[])

    return segments


def test_get_sampled_mapping_info_match_found(
    sample_phrase_map, sample_segment_with_phrases, reset_stats_counters
):
    """Test _get_sampled_mapping_info when a match is found."""
    random.seed(42)

    # Get the segment with a "car" phrase
    segment = sample_segment_with_phrases["car"]

    # Call the function
    result = _get_sampled_mapping_info(segment, sample_phrase_map)

    assert result is not None
    mapping_info, matched_phrase = result
    assert mapping_info["class_id"] == 0
    assert mapping_info["class_name"] == "Car"
    assert matched_phrase == "car"
    assert converter_module.stats_counters["skipped_segments_no_mapping"] == 0
    assert converter_module.stats_counters["skipped_segments_sampling"] == 0


def test_get_sampled_mapping_info_multiple_phrases(
    sample_phrase_map, sample_segment_with_phrases, reset_stats_counters
):
    """Test _get_sampled_mapping_info with multiple phrases, first one not matched."""
    random.seed(42)

    # Get the segment with "unknown" and "vehicle" phrases
    segment = sample_segment_with_phrases["multiple"]

    result = _get_sampled_mapping_info(segment, sample_phrase_map)

    assert result is not None
    mapping_info, matched_phrase = result
    assert mapping_info["class_id"] == 3
    assert mapping_info["class_name"] == "Vehicle"
    assert matched_phrase == "vehicle"
    assert converter_module.stats_counters["skipped_segments_no_mapping"] == 0
    assert converter_module.stats_counters["skipped_segments_sampling"] == 0


def test_get_sampled_mapping_info_sampling_skip(
    sample_phrase_map, sample_segment_with_phrases, reset_stats_counters, monkeypatch
):
    """Test _get_sampled_mapping_info where match is found but sampling skips it."""
    monkeypatch.setattr(random, "random", lambda: 0.9)  # Higher than truck's 0.5

    # Get the segment with a "truck" phrase
    segment = sample_segment_with_phrases["truck"]

    result = _get_sampled_mapping_info(segment, sample_phrase_map)

    assert result is None
    assert converter_module.stats_counters["skipped_segments_no_mapping"] == 0
    assert converter_module.stats_counters["skipped_segments_sampling"] == 1


def test_get_sampled_mapping_info_global_sampling(
    sample_phrase_map, sample_segment_with_phrases, reset_stats_counters, monkeypatch
):
    """Test _get_sampled_mapping_info with global_sample_ratio applied."""
    monkeypatch.setattr(random, "random", lambda: 0.4)  # Lower than car's 1.0 but higher than 0.3

    # Get the segment with a "car" phrase
    segment = sample_segment_with_phrases["car"]

    # Set global_sample_ratio to 0.3 (30%)
    result = _get_sampled_mapping_info(segment, sample_phrase_map, global_sample_ratio=0.3)

    # Should be skipped because 0.4 > (1.0 * 0.3)
    assert result is None
    assert converter_module.stats_counters["skipped_segments_no_mapping"] == 0
    assert converter_module.stats_counters["skipped_segments_sampling"] == 1


def test_get_sampled_mapping_info_no_match(
    sample_phrase_map, sample_segment_with_phrases, reset_stats_counters
):
    """Test _get_sampled_mapping_info when no phrases match."""
    # Get the segment with phrases that don't match any mappings
    segment = sample_segment_with_phrases["no_match"]

    result = _get_sampled_mapping_info(segment, sample_phrase_map)

    assert result is None
    assert converter_module.stats_counters["skipped_segments_no_mapping"] == 1
    assert converter_module.stats_counters["skipped_segments_sampling"] == 0


def test_get_sampled_mapping_info_empty_phrases(
    sample_phrase_map, sample_segment_with_phrases, reset_stats_counters
):
    """Test _get_sampled_mapping_info with empty phrases list."""
    # Get the segment with empty phrases
    segment = sample_segment_with_phrases["empty"]

    # Call function - should return None (no phrases to match)
    result = _get_sampled_mapping_info(segment, sample_phrase_map)

    # Check result
    assert result is None

    # Check correct counter was incremented
    assert converter_module.stats_counters["skipped_segments_no_mapping"] == 1
    assert converter_module.stats_counters["skipped_segments_sampling"] == 0


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
