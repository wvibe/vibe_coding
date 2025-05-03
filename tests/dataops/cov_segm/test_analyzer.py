import json
import logging
from typing import List
from unittest.mock import MagicMock, patch

# Assuming src directory is in PYTHONPATH or adjust as needed
from vibelab.dataops.cov_segm.analyzer import _aggregate_stats_from_metadata
from vibelab.dataops.cov_segm.datamodel import ConversationItem

# Configure logging for tests (optional, but can be helpful)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Mock Data ---

# Mock raw dataset rows - Need 'conversations' key with JSON
MOCK_RAW_ROW_1 = {
    "id": "sample1",
    "conversations": json.dumps(
        [
            {
                "phrases": [{"text": "cat"}, {"text": " tabby"}],  # 1 alt phrase
                "instance_masks": [{}, {}],  # 2 visible
                "instance_full_masks": [{}],  # 1 full
            }
        ]
    ),
}
MOCK_RAW_ROW_2 = {
    "id": "sample2",
    "conversations": json.dumps(
        [
            {
                "phrases": [{"text": "dog"}],  # 0 alt phrases
                "instance_masks": [{}],  # 1 visible
                "instance_full_masks": [],  # 0 full
            },
            {
                "phrases": [
                    {"text": "cat"},
                    {"text": " ginger"},
                    {"text": " fluffy"},
                ],  # 2 alt phrases
                "instance_masks": [{}, {}, {}],  # 3 visible
                "instance_full_masks": [{}, {}],  # 2 full
            },
        ]
    ),
}
MOCK_RAW_ROW_3_DUPLICATE_PHRASE = {
    "id": "sample3",
    "conversations": json.dumps(
        [
            {
                "phrases": [{"text": "cat"}],  # First appearance
                "instance_masks": [{}],  # 1 visible
                "instance_full_masks": [],  # 0 full
            },
            {
                "phrases": [{"text": "tree"}],
                "instance_masks": [{}, {}],  # 2 visible
                "instance_full_masks": [{}],  # 1 full
            },
            {
                "phrases": [{"text": "cat"}],  # Second appearance in same sample
                "instance_masks": [{}, {}, {}],  # 3 visible (SHOULD BE IGNORED for stats)
                "instance_full_masks": [{}],  # 1 full (SHOULD BE IGNORED for stats)
            },
        ]
    ),
}
MOCK_RAW_ROW_4_EMPTY_NULL = {
    "id": "sample4",
    "conversations": json.dumps(
        [
            {"phrases": [], "instance_masks": [{}]},  # No phrases
            {"phrases": [{"text": None}], "instance_masks": [{}]},  # Null phrase
            {
                "phrases": [{"text": "dog"}],  # Valid dog
                "instance_masks": [{}, {}],  # 2 visible
                "instance_full_masks": [{}],  # 1 full
            },
        ]
    ),
}
MOCK_RAW_ROW_5_ZERO_MASKS = {
    "id": "sample5",
    "conversations": json.dumps(
        [
            {
                "phrases": [{"text": "sky"}],  # Zero masks
                "instance_masks": [],
                "instance_full_masks": [],
            },
            {
                "phrases": [{"text": "cat"}],  # Non-zero masks
                "instance_masks": [{}],
                "instance_full_masks": [{}],
            },
        ]
    ),
}
MOCK_RAW_ROW_INVALID_JSON = {"id": "invalid_json", "conversations": '{"key": value}'}
MOCK_RAW_ROW_SCHEMA_ERROR = {
    "id": "schema_err",
    "conversations": '[{"phrases": [{"text": "bad"}], "invalid_field": true}]',
}
MOCK_RAW_ROW_NO_CONVO_KEY = {"id": "no_convo"}


# --- Mock Pydantic Objects (Representing parse_conversations output) ---


def create_mock_convo(
    phrases_texts: List[str | None], visible_mask_count: int, full_mask_count: int
) -> ConversationItem:
    mock_item = MagicMock(spec=ConversationItem)
    mock_item.phrases = [MagicMock(text=t) for t in phrases_texts]
    # Simulate Pydantic list access for masks - Just set the attribute directly
    mock_item.instance_masks = [MagicMock()] * visible_mask_count
    mock_item.instance_full_masks = [MagicMock()] * full_mask_count
    # Removed the problematic attach_mock calls below
    # mock_item.attach_mock(MagicMock(return_value=mock_item.instance_masks), "instance_masks")
    # mock_item.attach_mock(
    #     MagicMock(return_value=mock_item.instance_full_masks), "instance_full_masks"
    # )
    return mock_item


# Mock outputs for parse_conversations based on input JSON strings
MOCK_PARSED_CONVO_1 = [create_mock_convo(["cat", " tabby"], 2, 1)]
MOCK_PARSED_CONVO_2 = [
    create_mock_convo(["dog"], 1, 0),
    create_mock_convo(["cat", " ginger", " fluffy"], 3, 2),
]
MOCK_PARSED_CONVO_3 = [
    create_mock_convo(["cat"], 1, 0),
    create_mock_convo(["tree"], 2, 1),
    create_mock_convo(["cat"], 3, 1),  # Phrase 'cat' appears again
]
MOCK_PARSED_CONVO_4 = [
    create_mock_convo([], 1, 0),  # No phrases
    create_mock_convo([None], 1, 0),  # Phrase text is None
    create_mock_convo(["dog"], 2, 1),  # Valid dog phrase
]
MOCK_PARSED_CONVO_5_ZERO = [
    create_mock_convo(["sky"], 0, 0),  # Sky has 0 visible, 0 full
    create_mock_convo(["cat"], 1, 1),  # Cat has 1 visible, 1 full
]

# --- Mock parse_conversations Function ---


def mock_parse_conversations_logic(json_string):
    logger.debug(f"Mock parse_conversations called with string: {json_string[:50]}...")
    if json_string == MOCK_RAW_ROW_1["conversations"]:
        return MOCK_PARSED_CONVO_1
    elif json_string == MOCK_RAW_ROW_2["conversations"]:
        return MOCK_PARSED_CONVO_2
    elif json_string == MOCK_RAW_ROW_3_DUPLICATE_PHRASE["conversations"]:
        return MOCK_PARSED_CONVO_3
    elif json_string == MOCK_RAW_ROW_4_EMPTY_NULL["conversations"]:
        return MOCK_PARSED_CONVO_4
    elif json_string == MOCK_RAW_ROW_5_ZERO_MASKS["conversations"]:
        return MOCK_PARSED_CONVO_5_ZERO
    elif json_string == MOCK_RAW_ROW_INVALID_JSON["conversations"]:
        raise json.JSONDecodeError("Simulated invalid JSON", "", 0)
    elif json_string == MOCK_RAW_ROW_SCHEMA_ERROR["conversations"]:
        raise ValueError("Simulated Pydantic schema error")  # Or TypeError
    else:
        logger.warning("Mock parse_conversations received unexpected JSON")
        return []  # Default to empty list for unknown strings


# --- Test Functions for _aggregate_stats_from_metadata ---


@patch(
    "vibelab.dataops.cov_segm.analyzer.parse_conversations",
    side_effect=mock_parse_conversations_logic,
)
def test_aggregate_basic(mock_parse):
    """Test aggregation with multiple valid samples and phrases."""
    dataset = [MOCK_RAW_ROW_1, MOCK_RAW_ROW_2]
    agg_stats, processed, convos, valid_convos = _aggregate_stats_from_metadata(
        dataset, skip_zero_masks=False
    )

    assert processed == 2
    assert convos == 1 + 2  # 1 in sample1, 2 in sample2
    assert valid_convos == 1 + 2  # All convos have > 0 masks

    assert "cat" in agg_stats
    assert "dog" in agg_stats
    assert len(agg_stats) == 2

    # Check 'cat' stats (appears once in sample1, once in sample2)
    cat_data = agg_stats["cat"]
    assert cat_data["appearance_count"] == 2
    assert cat_data["visible_mask_counts"] == [2, 3]  # from sample1, sample2
    assert cat_data["full_mask_counts"] == [1, 2]  # from sample1, sample2
    assert cat_data["alternative_phrase_counts"] == [1, 2]  # from sample1, sample2

    # Check 'dog' stats (appears only in sample2)
    dog_data = agg_stats["dog"]
    assert dog_data["appearance_count"] == 1
    assert dog_data["visible_mask_counts"] == [1]
    assert dog_data["full_mask_counts"] == [0]
    assert dog_data["alternative_phrase_counts"] == [0]


@patch(
    "vibelab.dataops.cov_segm.analyzer.parse_conversations",
    side_effect=mock_parse_conversations_logic,
)
def test_aggregate_phrase_deduplication_per_sample(mock_parse):
    """Test that a phrase appearing multiple times in one sample is counted only once."""
    dataset = [MOCK_RAW_ROW_3_DUPLICATE_PHRASE]
    agg_stats, processed, convos, valid_convos = _aggregate_stats_from_metadata(
        dataset, skip_zero_masks=False
    )

    assert processed == 1
    assert convos == 3
    assert valid_convos == 3  # All have > 0 masks

    assert "cat" in agg_stats
    assert "tree" in agg_stats
    assert len(agg_stats) == 2

    # 'cat' appeared in 2 conversation items, but stats should only be from the *first* time
    cat_data = agg_stats["cat"]
    assert cat_data["appearance_count"] == 1
    assert cat_data["visible_mask_counts"] == [1]  # From first appearance
    assert cat_data["full_mask_counts"] == [0]  # From first appearance
    assert cat_data["alternative_phrase_counts"] == [0]  # From first appearance

    # 'tree' stats
    tree_data = agg_stats["tree"]
    assert tree_data["appearance_count"] == 1
    assert tree_data["visible_mask_counts"] == [2]
    assert tree_data["full_mask_counts"] == [1]
    assert tree_data["alternative_phrase_counts"] == [0]


@patch(
    "vibelab.dataops.cov_segm.analyzer.parse_conversations",
    side_effect=mock_parse_conversations_logic,
)
def test_aggregate_empty_input(mock_parse):
    """Test with an empty dataset iterable."""
    dataset = []
    agg_stats, processed, convos, valid_convos = _aggregate_stats_from_metadata(
        dataset, skip_zero_masks=False
    )
    assert agg_stats == {}
    assert processed == 0
    assert convos == 0
    assert valid_convos == 0
    assert mock_parse.call_count == 0


@patch(
    "vibelab.dataops.cov_segm.analyzer.parse_conversations",
    side_effect=mock_parse_conversations_logic,
)
def test_aggregate_missing_conversations_key(mock_parse):
    """Test row missing the 'conversations' key."""
    dataset = [MOCK_RAW_ROW_1, MOCK_RAW_ROW_NO_CONVO_KEY, MOCK_RAW_ROW_2]
    agg_stats, processed, convos, valid_convos = _aggregate_stats_from_metadata(
        dataset, skip_zero_masks=False
    )

    # Check processed count reflects skipped row
    assert processed == 2  # MOCK_RAW_ROW_NO_CONVO_KEY is skipped
    assert len(agg_stats) == 2
    assert "cat" in agg_stats
    assert "dog" in agg_stats
    assert mock_parse.call_count == 2  # Parse not called for skipped row


@patch(
    "vibelab.dataops.cov_segm.analyzer.parse_conversations",
    side_effect=mock_parse_conversations_logic,
)
def test_aggregate_invalid_json_string(mock_parse):
    """Test row with invalid JSON in 'conversations' field."""
    dataset = [MOCK_RAW_ROW_1, MOCK_RAW_ROW_INVALID_JSON, MOCK_RAW_ROW_2]
    agg_stats, processed, convos, valid_convos = _aggregate_stats_from_metadata(
        dataset, skip_zero_masks=False
    )

    assert processed == 2  # Invalid JSON row is skipped
    assert len(agg_stats) == 2
    assert "cat" in agg_stats
    assert "dog" in agg_stats
    # Parse *is* attempted for the invalid row before failing
    assert mock_parse.call_count == 3


@patch(
    "vibelab.dataops.cov_segm.analyzer.parse_conversations",
    side_effect=mock_parse_conversations_logic,
)
def test_aggregate_pydantic_schema_error(mock_parse):
    """Test row with valid JSON but invalid schema for ConversationItem."""
    dataset = [MOCK_RAW_ROW_1, MOCK_RAW_ROW_SCHEMA_ERROR, MOCK_RAW_ROW_2]
    agg_stats, processed, convos, valid_convos = _aggregate_stats_from_metadata(
        dataset, skip_zero_masks=False
    )

    assert processed == 2  # Schema error row is skipped
    assert len(agg_stats) == 2
    assert "cat" in agg_stats
    assert "dog" in agg_stats
    # Parse *is* attempted for the schema error row before failing
    assert mock_parse.call_count == 3


@patch(
    "vibelab.dataops.cov_segm.analyzer.parse_conversations",
    side_effect=mock_parse_conversations_logic,
)
def test_aggregate_missing_phrase_data_in_item(mock_parse):
    """Test samples with missing/empty/null phrase data within an item."""
    dataset = [MOCK_RAW_ROW_4_EMPTY_NULL]
    agg_stats, processed, convos, valid_convos = _aggregate_stats_from_metadata(
        dataset, skip_zero_masks=False
    )

    assert processed == 1
    assert convos == 3
    assert valid_convos == 3  # All have > 0 masks

    assert len(agg_stats) == 1
    assert "dog" in agg_stats
    dog_data = agg_stats["dog"]
    assert dog_data["appearance_count"] == 1
    assert dog_data["visible_mask_counts"] == [2]
    assert dog_data["full_mask_counts"] == [1]
    assert dog_data["alternative_phrase_counts"] == [0]


@patch(
    "vibelab.dataops.cov_segm.analyzer.parse_conversations",
    side_effect=mock_parse_conversations_logic,
)
def test_aggregate_skip_zero_masks_false(mock_parse):
    """Test aggregation when skip_zero_masks is False (default)."""
    dataset = [MOCK_RAW_ROW_5_ZERO_MASKS]
    agg_stats, processed, convos, valid_convos = _aggregate_stats_from_metadata(
        dataset, skip_zero_masks=False
    )

    assert processed == 1
    assert convos == 2
    assert valid_convos == 1  # Only 'cat' convo is valid

    assert len(agg_stats) == 2  # Both 'sky' and 'cat' should be present
    assert "sky" in agg_stats
    assert "cat" in agg_stats

    # 'sky' should have 0 counts but still appear
    sky_data = agg_stats["sky"]
    assert sky_data["appearance_count"] == 1
    assert sky_data["visible_mask_counts"] == [0]
    assert sky_data["full_mask_counts"] == [0]
    assert sky_data["alternative_phrase_counts"] == [0]

    # 'cat' should have normal counts
    cat_data = agg_stats["cat"]
    assert cat_data["appearance_count"] == 1
    assert cat_data["visible_mask_counts"] == [1]
    assert cat_data["full_mask_counts"] == [1]
    assert cat_data["alternative_phrase_counts"] == [0]


@patch(
    "vibelab.dataops.cov_segm.analyzer.parse_conversations",
    side_effect=mock_parse_conversations_logic,
)
def test_aggregate_skip_zero_masks_true(mock_parse):
    """Test aggregation when skip_zero_masks is True."""
    dataset = [MOCK_RAW_ROW_5_ZERO_MASKS]
    agg_stats, processed, convos, valid_convos = _aggregate_stats_from_metadata(
        dataset, skip_zero_masks=True
    )

    assert processed == 1
    assert convos == 2
    assert valid_convos == 1  # Only 'cat' convo is valid

    assert len(agg_stats) == 1  # Only 'cat' should be present
    assert "cat" in agg_stats
    assert "sky" not in agg_stats  # 'sky' should be skipped

    # Check 'cat' stats (should be unchanged)
    cat_data = agg_stats["cat"]
    assert cat_data["appearance_count"] == 1
    assert cat_data["visible_mask_counts"] == [1]
    assert cat_data["full_mask_counts"] == [1]
    assert cat_data["alternative_phrase_counts"] == [0]


@patch(
    "vibelab.dataops.cov_segm.analyzer.parse_conversations",
    side_effect=mock_parse_conversations_logic,
)
def test_aggregate_all_skipped(mock_parse):
    """Test when all samples are skipped due to errors."""
    dataset = [MOCK_RAW_ROW_INVALID_JSON, MOCK_RAW_ROW_NO_CONVO_KEY]
    agg_stats, processed, convos, valid_convos = _aggregate_stats_from_metadata(
        dataset, skip_zero_masks=False
    )

    assert processed == 0
    assert convos == 0
    assert valid_convos == 0
    assert agg_stats == {}
    assert mock_parse.call_count == 1  # Called for invalid json, not for missing key


@patch(
    "vibelab.dataops.cov_segm.analyzer.parse_conversations",
    side_effect=mock_parse_conversations_logic,
)
def test_aggregate_valid_and_skipped(mock_parse):
    """Test mix of valid and skipped samples."""
    dataset = [MOCK_RAW_ROW_1, MOCK_RAW_ROW_INVALID_JSON, MOCK_RAW_ROW_2]
    agg_stats, processed, convos, valid_convos = _aggregate_stats_from_metadata(
        dataset, skip_zero_masks=False
    )

    assert processed == 2  # Only valid rows counted
    assert convos == 1 + 2  # From valid rows
    assert valid_convos == 1 + 2  # From valid rows
    assert len(agg_stats) == 2
    assert "cat" in agg_stats
    assert "dog" in agg_stats
    assert mock_parse.call_count == 3  # Attempted parse on invalid
