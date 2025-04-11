import json
import logging
from unittest.mock import MagicMock, patch

import pytest  # Import pytest

# Assuming src directory is in PYTHONPATH or adjust as needed
from src.dataops.cov_segm.analyzer import (
    aggregate_phrase_stats,
    calculate_summary_stats,  # Import the new function
)

# Import ConversationItem for type hinting mocks, adjust path if needed

# Configure logging for tests (optional, but can be helpful)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Mock Data ---

# Mock raw dataset rows - These now primarily need a 'conversations' key with JSON
MOCK_RAW_ROW_1 = {
    "id": "sample1",
    "conversations": (
        '[{"phrases": [{"text": "cat"}], "instance_masks": [{}, {}], '  # Line break
        '"instance_full_masks": [{}]}]'
    ),
}
MOCK_RAW_ROW_2 = {
    "id": "sample2",
    "conversations": (
        '[{"phrases": [{"text": "dog"}], "instance_masks": [{}], "instance_full_masks": []}, '
        '{"phrases": [{"text": "cat"}], "instance_masks": [{}, {}, {}], "instance_full_masks": [{}, {}]}]'
    ),
}
MOCK_RAW_ROW_3 = {
    "id": "sample3",
    "conversations": (
        '[{"phrases": [{"text": "cat"}], "instance_masks": [{}], "instance_full_masks": []}, '
        '{"phrases": [{"text": "tree"}], "instance_masks": [{}, {}], "instance_full_masks": [{}]}, '
        '{"phrases": [{"text": "cat"}], "instance_masks": [{}, {}], "instance_full_masks": [{}]}]'
    ),
}
MOCK_RAW_ROW_4 = {
    "id": "sample4",
    "conversations": (
        '[{"phrases": [], "instance_masks": [{}]}, '
        '{"phrases": [{"text": null}], "instance_masks": [{}]}, '
        '{"phrases": [{"text": "dog"}], "instance_masks": [{}, {}], "instance_full_masks": [{}]}]'
    ),
}

# Simplified JSON string for the 'object' sample
# Note: Structure inside masks doesn't matter for counting, only list length
MOCK_CONVERSATIONS_JSON_OBJ = json.dumps(
    [
        {
            "phrases": [{"text": "object"}],
            "instance_masks": [{}, {}, {}, {}, {}],  # 5 visible masks
            "instance_full_masks": [{}, {}, {}],  # 3 full masks
        }
    ]
)
MOCK_RAW_ROW_OBJECT = {"id": "sample_obj", "conversations": MOCK_CONVERSATIONS_JSON_OBJ}

MOCK_RAW_ROW_INVALID_JSON = {"id": "invalid_json", "conversations": '{"key": value}'}
MOCK_RAW_ROW_SCHEMA_ERROR = {
    "id": "schema_err",
    "conversations": '[{"phrases": [{"text": "bad"}], "invalid_field": true}]',
}  # Valid JSON, bad Pydantic schema
MOCK_RAW_ROW_NO_CONVO_KEY = {"id": "no_convo"}  # Missing 'conversations' key

# --- Mock Pydantic Objects (Mimicking parse_conversations output) ---
# Using simple MagicMock with attributes is often sufficient for tests
# You don't necessarily need to instantiate real Pydantic models unless complex validation is tested


class MockConversationItem:
    def __init__(self, phrases_list, instance_masks_list, instance_full_masks_list):
        # Simulate Pydantic structure
        self.phrases = [MagicMock(text=p.get("text")) for p in phrases_list] if phrases_list else []
        self.instance_masks = instance_masks_list if instance_masks_list is not None else []
        self.instance_full_masks = (
            instance_full_masks_list if instance_full_masks_list is not None else []
        )


# Mock outputs for parse_conversations based on input JSON strings
MOCK_PARSED_CONVO_1 = [MockConversationItem([{"text": "cat"}], [{}, {}], [{}])]
MOCK_PARSED_CONVO_2 = [
    MockConversationItem([{"text": "dog"}], [{}], []),
    MockConversationItem([{"text": "cat"}], [{}, {}, {}], [{}, {}]),
]
MOCK_PARSED_CONVO_3 = [
    MockConversationItem([{"text": "cat"}], [{}], []),
    MockConversationItem([{"text": "tree"}], [{}, {}], [{}]),
    MockConversationItem([{"text": "cat"}], [{}, {}], [{}]),  # Phrase 'cat' appears again
]
MOCK_PARSED_CONVO_4 = [
    MockConversationItem([], [{}], []),  # No phrases
    MockConversationItem([{"text": None}], [{}], []),  # Phrase text is None
    MockConversationItem([{"text": "dog"}], [{}, {}], [{}]),  # Valid dog phrase
]
MOCK_PARSED_CONVO_OBJ = [
    MockConversationItem(
        [{"text": "object"}], [{}, {}, {}, {}, {}], [{}, {}, {}]
    )  # 5 visible, 3 full
]

# --- Mock parse_conversations Function ---


def mock_parse_conversations_logic(json_string):
    logger.debug(f"Mock parse_conversations called with string: {json_string[:50]}...")
    if json_string == MOCK_RAW_ROW_1["conversations"]:
        return MOCK_PARSED_CONVO_1
    elif json_string == MOCK_RAW_ROW_2["conversations"]:
        return MOCK_PARSED_CONVO_2
    elif json_string == MOCK_RAW_ROW_3["conversations"]:
        return MOCK_PARSED_CONVO_3
    elif json_string == MOCK_RAW_ROW_4["conversations"]:
        return MOCK_PARSED_CONVO_4
    elif json_string == MOCK_RAW_ROW_OBJECT["conversations"]:
        return MOCK_PARSED_CONVO_OBJ
    elif json_string == MOCK_RAW_ROW_INVALID_JSON["conversations"]:
        raise json.JSONDecodeError("Simulated invalid JSON", "", 0)
    elif json_string == MOCK_RAW_ROW_SCHEMA_ERROR["conversations"]:
        # Simulate a Pydantic validation error (e.g., ValueError)
        raise ValueError("Simulated Pydantic schema error")
    else:
        logger.warning("Mock parse_conversations received unexpected JSON")
        return []  # Default to empty list for unknown strings


# --- Test Functions ---


@patch(
    "src.dataops.cov_segm.analyzer.parse_conversations",
    side_effect=mock_parse_conversations_logic,
)
def test_basic_aggregation(mock_parse):
    """Test aggregation with multiple valid samples and phrases."""
    dataset = [MOCK_RAW_ROW_1, MOCK_RAW_ROW_2]
    # Unpack the tuple result
    result_dict, processed_count = aggregate_phrase_stats(dataset)

    assert processed_count == 2
    assert "cat" in result_dict
    assert "dog" in result_dict
    assert len(result_dict) == 2

    # Check 'cat' stats (appears in sample1 and sample2)
    assert result_dict["cat"]["appearance_count"] == 2
    assert result_dict["cat"]["sample_ids"] == ["sample1", "sample2"]
    assert result_dict["cat"]["total_visible_mask_count"] == 2 + 3
    assert result_dict["cat"]["visible_mask_counts_per_image"] == [2, 3]
    assert result_dict["cat"]["total_full_mask_count"] == 1 + 2
    assert result_dict["cat"]["full_mask_counts_per_image"] == [1, 2]

    # Check 'dog' stats (appears only in sample2)
    assert result_dict["dog"]["appearance_count"] == 1
    assert result_dict["dog"]["sample_ids"] == ["sample2"]
    assert result_dict["dog"]["total_visible_mask_count"] == 1
    assert result_dict["dog"]["visible_mask_counts_per_image"] == [1]
    assert result_dict["dog"]["total_full_mask_count"] == 0
    assert result_dict["dog"]["full_mask_counts_per_image"] == [0]


@patch(
    "src.dataops.cov_segm.analyzer.parse_conversations",
    side_effect=mock_parse_conversations_logic,
)
def test_phrase_deduplication_per_sample(mock_parse):
    """Test that a phrase appearing multiple times in one sample is counted correctly (once)."""
    dataset = [MOCK_RAW_ROW_3]
    result_dict, processed_count = aggregate_phrase_stats(dataset)

    assert processed_count == 1
    assert "cat" in result_dict
    assert "tree" in result_dict
    assert len(result_dict) == 2

    # 'cat' appeared in 2 conversation items, but only 1 sample
    assert result_dict["cat"]["appearance_count"] == 1
    assert result_dict["cat"]["sample_ids"] == ["sample3"]
    # Should count masks only from the *first* time it sees 'cat' in the sample
    assert result_dict["cat"]["total_visible_mask_count"] == 1
    assert result_dict["cat"]["visible_mask_counts_per_image"] == [1]
    assert result_dict["cat"]["total_full_mask_count"] == 0
    assert result_dict["cat"]["full_mask_counts_per_image"] == [0]

    # 'tree' stats
    assert result_dict["tree"]["appearance_count"] == 1
    assert result_dict["tree"]["sample_ids"] == ["sample3"]
    assert result_dict["tree"]["total_visible_mask_count"] == 2
    assert result_dict["tree"]["visible_mask_counts_per_image"] == [2]
    assert result_dict["tree"]["total_full_mask_count"] == 1
    assert result_dict["tree"]["full_mask_counts_per_image"] == [1]


@patch(
    "src.dataops.cov_segm.analyzer.parse_conversations",
    side_effect=mock_parse_conversations_logic,
)
def test_empty_input(mock_parse):
    """Test with an empty dataset iterable."""
    dataset = []
    result_dict, processed_count = aggregate_phrase_stats(dataset)
    assert result_dict == {}
    assert processed_count == 0


@patch(
    "src.dataops.cov_segm.analyzer.parse_conversations",
    side_effect=mock_parse_conversations_logic,
)
def test_missing_conversations_key(mock_parse):
    """Test row missing the 'conversations' key."""
    dataset = [MOCK_RAW_ROW_1, MOCK_RAW_ROW_NO_CONVO_KEY, MOCK_RAW_ROW_2]
    result_dict, processed_count = aggregate_phrase_stats(dataset, verbose=True)

    # Check processed count reflects skipped row
    assert processed_count == 2  # MOCK_RAW_ROW_NO_CONVO_KEY is skipped before parsing
    assert len(result_dict) == 2
    assert "cat" in result_dict
    assert "dog" in result_dict
    assert mock_parse.call_count == 2


@patch(
    "src.dataops.cov_segm.analyzer.parse_conversations",
    side_effect=mock_parse_conversations_logic,
)
def test_invalid_json_string(mock_parse):
    """Test row with invalid JSON in 'conversations' field."""
    dataset = [MOCK_RAW_ROW_1, MOCK_RAW_ROW_INVALID_JSON, MOCK_RAW_ROW_2]
    result_dict, processed_count = aggregate_phrase_stats(dataset, verbose=True)

    # Check processed count reflects skipped row
    assert processed_count == 2  # Invalid JSON row is skipped during parsing
    assert len(result_dict) == 2
    assert "cat" in result_dict
    assert "dog" in result_dict
    assert mock_parse.call_count == 3  # Parse is attempted


@patch(
    "src.dataops.cov_segm.analyzer.parse_conversations",
    side_effect=mock_parse_conversations_logic,
)
def test_pydantic_schema_error(mock_parse):
    """Test row with valid JSON but invalid schema for ConversationItem."""
    dataset = [MOCK_RAW_ROW_1, MOCK_RAW_ROW_SCHEMA_ERROR, MOCK_RAW_ROW_2]
    result_dict, processed_count = aggregate_phrase_stats(dataset, verbose=True)

    # Check processed count reflects skipped row
    assert processed_count == 2  # Schema error row is skipped during parsing
    assert len(result_dict) == 2
    assert "cat" in result_dict
    assert "dog" in result_dict
    assert mock_parse.call_count == 3  # Parse is attempted


@patch(
    "src.dataops.cov_segm.analyzer.parse_conversations",
    side_effect=mock_parse_conversations_logic,
)
def test_missing_phrase_data_in_item(mock_parse):
    """Test samples with missing/empty phrase data within an item."""
    dataset = [MOCK_RAW_ROW_4]
    result_dict, processed_count = aggregate_phrase_stats(dataset, verbose=True)

    assert processed_count == 1
    assert len(result_dict) == 1
    assert "dog" in result_dict
    assert result_dict["dog"]["appearance_count"] == 1
    assert result_dict["dog"]["sample_ids"] == ["sample4"]
    assert result_dict["dog"]["total_visible_mask_count"] == 2
    assert result_dict["dog"]["visible_mask_counts_per_image"] == [2]
    assert result_dict["dog"]["total_full_mask_count"] == 1
    assert result_dict["dog"]["full_mask_counts_per_image"] == [1]


@patch(
    "src.dataops.cov_segm.analyzer.parse_conversations",
    side_effect=mock_parse_conversations_logic,
)
def test_aggregation_with_object_phrase(mock_parse):
    """Test aggregation specifically for the 'object' phrase based on the simplified sample."""
    dataset = [MOCK_RAW_ROW_OBJECT]
    result_dict, processed_count = aggregate_phrase_stats(dataset)

    assert processed_count == 1
    assert len(result_dict) == 1
    assert "object" in result_dict
    assert result_dict["object"]["appearance_count"] == 1
    assert result_dict["object"]["sample_ids"] == ["sample_obj"]
    assert result_dict["object"]["total_visible_mask_count"] == 5
    assert result_dict["object"]["visible_mask_counts_per_image"] == [5]
    assert result_dict["object"]["total_full_mask_count"] == 3
    assert result_dict["object"]["full_mask_counts_per_image"] == [3]


# --- Tests for skip_zero_masks ---

# Mock data where a phrase has zero masks
MOCK_RAW_ROW_ZERO_MASKS = {
    "id": "sample_zero",
    "conversations": '[{"phrases": [{"text": "sky"}], "instance_masks": [], "instance_full_masks": []}, '  # Zero masks
    '{"phrases": [{"text": "cat"}], "instance_masks": [{}], "instance_full_masks": [{}]}]',  # Non-zero masks
}

MOCK_PARSED_ZERO_MASKS = [
    MockConversationItem([{"text": "sky"}], [], []),  # Sky has 0 visible, 0 full
    MockConversationItem([{"text": "cat"}], [{}], [{}]),  # Cat has 1 visible, 1 full
]


def mock_parse_logic_with_zero(json_string):
    # Extend the main mock logic
    if json_string == MOCK_RAW_ROW_ZERO_MASKS["conversations"]:
        return MOCK_PARSED_ZERO_MASKS
    else:
        # Fallback to the original logic for other inputs
        return mock_parse_conversations_logic(json_string)


@patch("src.dataops.cov_segm.analyzer.parse_conversations", side_effect=mock_parse_logic_with_zero)
def test_skip_zero_masks_false(mock_parse):
    """Test aggregation when skip_zero_masks is False (default)."""
    dataset = [MOCK_RAW_ROW_ZERO_MASKS]
    result_dict, processed_count = aggregate_phrase_stats(dataset, skip_zero_masks=False)

    assert processed_count == 1
    assert len(result_dict) == 2  # Both 'sky' and 'cat' should be present
    assert "sky" in result_dict
    assert "cat" in result_dict

    # 'sky' should have 0 counts but still appear
    assert result_dict["sky"]["appearance_count"] == 1
    assert result_dict["sky"]["total_visible_mask_count"] == 0
    assert result_dict["sky"]["visible_mask_counts_per_image"] == [0]
    assert result_dict["sky"]["total_full_mask_count"] == 0
    assert result_dict["sky"]["full_mask_counts_per_image"] == [0]

    # 'cat' should have normal counts
    assert result_dict["cat"]["appearance_count"] == 1
    assert result_dict["cat"]["total_visible_mask_count"] == 1
    assert result_dict["cat"]["visible_mask_counts_per_image"] == [1]
    assert result_dict["cat"]["total_full_mask_count"] == 1
    assert result_dict["cat"]["full_mask_counts_per_image"] == [1]


@patch("src.dataops.cov_segm.analyzer.parse_conversations", side_effect=mock_parse_logic_with_zero)
def test_skip_zero_masks_true(mock_parse):
    """Test aggregation when skip_zero_masks is True."""
    dataset = [MOCK_RAW_ROW_ZERO_MASKS]
    result_dict, processed_count = aggregate_phrase_stats(dataset, skip_zero_masks=True)

    assert processed_count == 1
    assert len(result_dict) == 1  # Only 'cat' should be present
    assert "cat" in result_dict
    assert "sky" not in result_dict  # 'sky' should be skipped

    # Check 'cat' stats (should be unchanged)
    assert result_dict["cat"]["appearance_count"] == 1
    assert result_dict["cat"]["total_visible_mask_count"] == 1
    assert result_dict["cat"]["visible_mask_counts_per_image"] == [1]
    assert result_dict["cat"]["total_full_mask_count"] == 1
    assert result_dict["cat"]["full_mask_counts_per_image"] == [1]


# --- Tests for calculate_summary_stats ---

SAMPLE_AGG_DATA = {
    "cat": {
        "appearance_count": 2,
        "sample_ids": ["s1", "s2"],
        "total_visible_mask_count": 5,
        "visible_mask_counts_per_image": [2, 3],
        "total_full_mask_count": 3,
        "full_mask_counts_per_image": [1, 2],
    },
    "dog": {
        "appearance_count": 1,
        "sample_ids": ["s2"],
        "total_visible_mask_count": 1,
        "visible_mask_counts_per_image": [1],
        "total_full_mask_count": 0,
        "full_mask_counts_per_image": [],
    },
    "sky": {
        "appearance_count": 1,
        "sample_ids": ["s3"],
        "total_visible_mask_count": 0,
        "visible_mask_counts_per_image": [0],
        "total_full_mask_count": 0,
        "full_mask_counts_per_image": [0],
    },
}
TOTAL_PROCESSED_FOR_SUMMARY = 3  # Assume 3 samples were processed in total


def test_calculate_summary_stats():
    """Test the summary statistics calculation."""
    percentiles_to_test = [0.5, 0.9]
    summary = calculate_summary_stats(
        SAMPLE_AGG_DATA, TOTAL_PROCESSED_FOR_SUMMARY, percentiles=percentiles_to_test
    )

    assert len(summary) == 3
    # Check sorting (cat appears most, then dog/sky)
    assert summary[0]["phrase"] == "cat"
    assert summary[1]["phrase"] in ["dog", "sky"]
    assert summary[2]["phrase"] in ["dog", "sky"]
    assert summary[1]["phrase"] != summary[2]["phrase"]

    # Check stats for 'cat'
    cat_stats = next(s for s in summary if s["phrase"] == "cat")
    assert cat_stats["appearance_count"] == 2
    assert cat_stats["appearance_percentage"] == pytest.approx((2 / 3) * 100)
    assert cat_stats["avg_visible_masks_per_image"] == pytest.approx(2.5)  # (2+3)/2
    assert cat_stats["avg_full_masks_per_image"] == pytest.approx(1.5)  # (1+2)/2
    assert cat_stats["visible_mask_percentiles"] == {0.5: 2.5, 0.9: 2.9}
    assert cat_stats["full_mask_percentiles"] == {0.5: 1.5, 0.9: 1.9}

    # Check stats for 'dog'
    dog_stats = next(s for s in summary if s["phrase"] == "dog")
    assert dog_stats["appearance_count"] == 1
    assert dog_stats["appearance_percentage"] == pytest.approx((1 / 3) * 100)
    assert dog_stats["avg_visible_masks_per_image"] == pytest.approx(1.0)
    assert dog_stats["avg_full_masks_per_image"] == pytest.approx(0.0)  # Mean of empty is 0
    assert dog_stats["visible_mask_percentiles"] == {
        0.5: 1.0,
        0.9: 1.0,
    }  # Percentiles of single item
    assert dog_stats["full_mask_percentiles"] == {}  # Percentiles of empty list

    # Check stats for 'sky' (zero masks)
    sky_stats = next(s for s in summary if s["phrase"] == "sky")
    assert sky_stats["appearance_count"] == 1
    assert sky_stats["appearance_percentage"] == pytest.approx((1 / 3) * 100)
    assert sky_stats["avg_visible_masks_per_image"] == pytest.approx(0.0)
    assert sky_stats["avg_full_masks_per_image"] == pytest.approx(0.0)
    assert sky_stats["visible_mask_percentiles"] == {0.5: 0.0, 0.9: 0.0}
    assert sky_stats["full_mask_percentiles"] == {0.5: 0.0, 0.9: 0.0}


def test_calculate_summary_stats_empty_input():
    """Test summary calculation with empty aggregated data."""
    summary = calculate_summary_stats({}, total_processed_samples=0, percentiles=[0.5])
    assert summary == []
