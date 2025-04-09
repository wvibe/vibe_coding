# Modules to test
# Type hints and helpers
# Basic logging setup for tests if needed (optional)
import io
import json
import logging
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from src.dataops.cov_segm import datamodel as dm
from src.dataops.cov_segm import loader

logging.basicConfig(level=logging.DEBUG)  # Show debug logs during tests
logger = logging.getLogger(__name__)

# --- Test Fixtures (if needed later) ---

# --- Test Functions ---

# == Tests for parse_conversations ==

VALID_CONVERSATION_ITEM = {
    "phrases": [{"id": 1, "text": "prompt1", "type": "PhraseType.RDP_PROMPT"}],
    "image_uri": {"jpg": "s3://bucket/image.jpg", "format": "RGB_8B_HW3"},
    "instance_masks": [
        {
            "column": "mask_0",
            "positive_value": 1,
            "image_uri": {"jpg": "s3://bucket/mask.jpg", "format": "RGB_8B_HW3"},
        }
    ],
    "type": "ConversationType.SEG_QA",
}

VALID_CONVERSATIONS_JSON = json.dumps([VALID_CONVERSATION_ITEM])

INVALID_JSON_STRING = '{"phrases": [}}'  # Malformed JSON

JSON_WITH_SCHEMA_ERROR = json.dumps(
    [
        {
            # Missing 'image_uri'
            "phrases": [{"id": 1, "text": "prompt1", "type": "PhraseType.RDP_PROMPT"}],
            "instance_masks": [{"column": "mask_0", "positive_value": 1}],
            "type": "ConversationType.SEG_QA",
        }
    ]
)

JSON_WITH_TYPE_ERROR = json.dumps(
    [
        {
            "phrases": [{"id": "not_an_int", "text": "prompt1", "type": "PhraseType.RDP_PROMPT"}],
            "image_uri": {"jpg": "s3://bucket/image.jpg", "format": "RGB_8B_HW3"},
            "instance_masks": [{"column": "mask_0", "positive_value": 1}],
            "type": "ConversationType.SEG_QA",
        }
    ]
)


def test_parse_conversations_valid():
    """Test parsing a valid JSON string matching the Pydantic models."""
    result = loader.parse_conversations(VALID_CONVERSATIONS_JSON)
    assert isinstance(result, list)
    assert len(result) == 1
    # Check if the result is a list of Pydantic models (ConversationItem)
    assert isinstance(result[0], dm.ConversationItem)
    assert result[0].image_uri.jpg == "s3://bucket/image.jpg"
    assert result[0].phrases[0].text == "prompt1"
    assert result[0].instance_masks[0].column == "mask_0"


def test_parse_conversations_invalid_json():
    """Test parsing an invalid JSON string."""
    with pytest.raises(ValueError, match="Invalid JSON"):
        loader.parse_conversations(INVALID_JSON_STRING)


def test_parse_conversations_schema_error_missing_field():
    """Test parsing valid JSON but with missing required fields (schema error)."""
    with pytest.raises(ValueError, match="Invalid conversation data structure"):
        loader.parse_conversations(JSON_WITH_SCHEMA_ERROR)


def test_parse_conversations_schema_error_wrong_type():
    """Test parsing valid JSON but with incorrect data types (schema error)."""
    with pytest.raises(ValueError, match="Invalid conversation data structure"):
        loader.parse_conversations(JSON_WITH_TYPE_ERROR)


def test_parse_conversations_empty_string():
    """Test parsing an empty string."""
    with pytest.raises(ValueError, match="Invalid JSON"):
        loader.parse_conversations("")


def test_parse_conversations_empty_list_json():
    """Test parsing valid JSON representing an empty list."""
    result = loader.parse_conversations("[]")
    assert isinstance(result, list)
    assert len(result) == 0


# == Tests for _load_mask ==


@pytest.fixture
def mock_pil_image():
    """Creates a simple mock PIL Image."""
    return Image.new("P", (10, 10))  # Simple 10x10 palette image


@pytest.fixture
def mock_numpy_array():
    """Creates a simple mock NumPy array."""
    return np.zeros((10, 10), dtype=np.uint8)


def test_load_mask_pil_direct(mock_pil_image):
    """Test loading a PIL image directly from a column."""
    row = {"mask_0": mock_pil_image}
    mask_info = dm.InstanceMask(
        column="mask_0", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
    )
    result = loader._load_mask(mask_info, row)
    assert result is not None
    assert isinstance(result["mask"], Image.Image)
    assert result["mask"] == mock_pil_image
    assert result["positive_value"] == 1
    assert result["source"] == "mask_0"


def test_load_mask_numpy_direct(mock_numpy_array):
    """Test loading a NumPy array directly from a column (should be converted to PIL)."""
    row = {"mask_0": mock_numpy_array}
    mask_info = dm.InstanceMask(
        column="mask_0", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
    )
    result = loader._load_mask(mask_info, row)
    assert result is not None
    assert isinstance(result["mask"], Image.Image)
    assert result["mask"].size == (mock_numpy_array.shape[1], mock_numpy_array.shape[0])
    assert result["positive_value"] == 1
    assert result["source"] == "mask_0"


def test_load_mask_pil_from_rest(mock_pil_image):
    """Test loading a PIL image from the 'masks_rest' list."""
    row = {"masks_rest": [mock_pil_image, Image.new("P", (5, 5))]}
    # Test index 0
    mask_info_0 = dm.InstanceMask(
        column="masks_rest/0", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
    )
    mask = loader._load_mask(mask_info_0, row)
    assert mask is not None
    assert mask["mask"].size == (10, 10)
    # Test index 1
    mask_info_1 = dm.InstanceMask(
        column="masks_rest/1", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
    )
    mask = loader._load_mask(mask_info_1, row)
    assert mask is not None
    assert mask["mask"].size == (5, 5)


def test_load_mask_numpy_from_rest(mock_numpy_array):
    """Test loading a NumPy array from the 'masks_rest' list."""
    row = {"masks_rest": [mock_numpy_array]}
    mask_info = dm.InstanceMask(
        column="masks_rest/0", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
    )
    mask = loader._load_mask(mask_info, row)
    assert mask is not None
    assert isinstance(mask["mask"], Image.Image)
    assert mask["mask"].size == (mock_numpy_array.shape[1], mock_numpy_array.shape[0])


def test_load_mask_missing_column():
    """Test behavior when the specified column key is missing."""
    row = {"some_other_key": 123}
    mask_info = dm.InstanceMask(
        column="mask_0", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
    )
    mask = loader._load_mask(mask_info, row)
    assert mask is None


def test_load_mask_rest_key_missing():
    """Test behavior when 'masks_rest' key is missing."""
    row = {"mask_0": Image.new("P", (1, 1))}
    mask_info = dm.InstanceMask(
        column="masks_rest/0", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
    )
    mask = loader._load_mask(mask_info, row)
    assert mask is None


def test_load_mask_rest_index_out_of_bounds():
    """Test behavior when the index for 'masks_rest' is invalid."""
    row = {"masks_rest": [Image.new("P", (1, 1))]}
    mask_info = dm.InstanceMask(
        column="masks_rest/1", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
    )
    mask = loader._load_mask(mask_info, row)
    assert mask is None


def test_load_mask_invalid_type_in_column():
    """Test behavior when the column contains an unexpected data type."""
    row = {"mask_0": "not_an_image"}
    mask_info = dm.InstanceMask(
        column="mask_0", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
    )
    mask = loader._load_mask(mask_info, row)
    assert mask is None


def test_load_mask_invalid_type_in_rest():
    """Test behavior when 'masks_rest' contains an unexpected data type."""
    row = {"masks_rest": ["not_an_image"]}
    mask_info = dm.InstanceMask(
        column="masks_rest/0", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
    )
    mask = loader._load_mask(mask_info, row)
    assert mask is None


def test_load_mask_none_in_column():
    """Test behavior when the column value is None."""
    row = {"mask_0": None}
    mask_info = dm.InstanceMask(
        column="mask_0", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
    )
    mask = loader._load_mask(mask_info, row)
    assert mask is None


def test_load_mask_none_in_rest():
    """Test behavior when 'masks_rest' contains None."""
    row = {"masks_rest": [None]}
    mask_info = dm.InstanceMask(
        column="masks_rest/0", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
    )
    mask = loader._load_mask(mask_info, row)
    assert mask is None


# == Tests for load_sample ==


@pytest.fixture
def mock_s3_image():
    """Creates mock PNG image bytes for S3 fetch."""
    img = Image.new("RGB", (100, 50))  # Different size for distinction
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()  # Return bytes


@pytest.fixture(scope="function")
def mock_input_row(mock_pil_image):
    """Creates a mock Hugging Face dataset row dictionary."""
    conversations_data = [
        {
            "phrases": [{"id": 1, "text": "object1", "type": "PhraseType.RDP_PROMPT"}],
            "image_uri": {"jpg": "s3://test-bucket/main.jpg", "format": "RGB_8B_HW3"},
            "instance_masks": [
                {
                    "column": "mask_0",
                    "positive_value": 1,
                    "image_uri": {"jpg": "s3://test-bucket/imask.jpg", "format": "RGB_8B_HW3"},
                }
            ],
            "instance_full_masks": [
                {
                    "column": "masks_rest/0",
                    "positive_value": 1,
                    "image_uri": {"jpg": "s3://test-bucket/fmask.jpg", "format": "RGB_8B_HW3"},
                }
            ],
            "type": "ConversationType.SEG_QA",
        },
        {
            "phrases": [{"id": 2, "text": "object2", "type": "PhraseType.RDP_PROMPT"}],
            "image_uri": {"jpg": "s3://test-bucket/main.jpg", "format": "RGB_8B_HW3"},
            # No masks for this item
            "instance_masks": [],
            "instance_full_masks": [],
            "type": "ConversationType.SEG_QA",
        },
    ]
    return {
        "id": "test_sample_123",
        "conversations": json.dumps(conversations_data),
        "mask_0": mock_pil_image,  # 10x10 palette image
        "mask_1": None,  # Example of unused mask column
        "masks_rest": [
            Image.new("P", (20, 20))  # Different size for full mask
        ],
    }


@patch("src.dataops.cov_segm.loader.fetch_s3_uri")
def test_load_sample_success(mock_fetch, mock_pil_image):
    """Test successful loading and processing of a sample."""
    # Create test row directly
    conversations_data = [
        {
            "phrases": [{"id": 1, "text": "object1", "type": "PhraseType.RDP_PROMPT"}],
            "image_uri": {"jpg": "s3://test-bucket/main.jpg", "format": "RGB_8B_HW3"},
            "instance_masks": [
                {
                    "column": "mask_0",
                    "positive_value": 1,
                    "image_uri": {"jpg": "s3://test-bucket/imask.jpg", "format": "RGB_8B_HW3"},
                }
            ],
            "instance_full_masks": [
                {
                    "column": "masks_rest/0",
                    "positive_value": 1,
                    "image_uri": {"jpg": "s3://test-bucket/fmask.jpg", "format": "RGB_8B_HW3"},
                }
            ],
            "type": "ConversationType.SEG_QA",
        },
        {
            "phrases": [{"id": 2, "text": "object2", "type": "PhraseType.RDP_PROMPT"}],
            "image_uri": {"jpg": "s3://test-bucket/main.jpg", "format": "RGB_8B_HW3"},
            "instance_masks": [],
            "instance_full_masks": [],
            "type": "ConversationType.SEG_QA",
        },
    ]

    # Full mask image (different size to distinguish)
    full_mask_img = Image.new("P", (20, 20))

    test_row = {
        "id": "test_sample_123",
        "conversations": json.dumps(conversations_data),
        "mask_0": mock_pil_image,  # 10x10 palette image
        "masks_rest": [full_mask_img],
    }

    # Configure the mock S3 fetcher with image bytes
    mock_image = Image.new("RGB", (100, 50))
    buffer = io.BytesIO()
    mock_image.save(buffer, format="PNG")
    mock_image_bytes = buffer.getvalue()
    mock_fetch.return_value = mock_image_bytes

    # Call the function
    result = loader.load_sample(test_row)

    # Check S3 fetch call
    mock_fetch.assert_called_once_with("s3://test-bucket/main.jpg")

    # Check overall structure
    assert isinstance(result, dict)
    assert result["id"] == "test_sample_123"
    assert result["image"] is not None
    assert "processed_conversations" in result
    assert isinstance(result["processed_conversations"], list)
    assert len(result["processed_conversations"]) == 2

    # Check first conversation item
    item1 = result["processed_conversations"][0]
    assert item1["phrases"][0]["text"] == "object1"
    assert len(item1["processed_instance_masks"]) == 1
    assert item1["processed_instance_masks"][0]["mask"].size == (10, 10)  # From mock_pil_image
    assert item1["processed_instance_masks"][0]["positive_value"] == 1
    assert item1["processed_instance_masks"][0]["source"] == "mask_0"
    assert len(item1["processed_full_masks"]) == 1
    assert item1["processed_full_masks"][0]["mask"].size == (20, 20)  # From full_mask_img
    assert item1["processed_full_masks"][0]["positive_value"] == 1
    assert item1["processed_full_masks"][0]["source"] == "masks_rest/0"

    # Check second conversation item (no masks)
    item2 = result["processed_conversations"][1]
    assert item2["phrases"][0]["text"] == "object2"
    assert len(item2["processed_instance_masks"]) == 0
    assert len(item2["processed_full_masks"]) == 0


@patch("src.dataops.cov_segm.loader.fetch_s3_uri")
def test_load_sample_s3_fetch_fails(mock_fetch, mock_input_row):
    """Test behavior when S3 image fetching fails."""
    mock_fetch.return_value = None  # Simulate fetch failure

    # load_sample logs error and returns None if image load fails
    result = loader.load_sample(mock_input_row)
    assert result is None

    mock_fetch.assert_called_once_with("s3://test-bucket/main.jpg")


def test_load_sample_parse_fails(mock_input_row):
    """Test behavior when conversations JSON parsing fails."""
    mock_input_row["conversations"] = INVALID_JSON_STRING  # Use invalid JSON

    # load_sample catches the ValueError from parse_conversations and returns None
    result = loader.load_sample(mock_input_row)
    assert result is None


def test_load_sample_missing_conversations_key(mock_input_row):
    """Test behavior when the 'conversations' key is missing from the row."""
    del mock_input_row["conversations"]

    # The function should log an error and return None
    result = loader.load_sample(mock_input_row)
    assert result is None


@patch("src.dataops.cov_segm.loader.fetch_s3_uri")
def test_load_sample_missing_id_key(mock_fetch, mock_pil_image):
    """Test behavior when the 'id' key is missing (should still work but log warning)."""
    # Create a minimal valid row directly in the test instead of using fixture
    conversations_data = [
        {
            "phrases": [{"id": 1, "text": "test", "type": "PhraseType.RDP_PROMPT"}],
            "image_uri": {"jpg": "s3://test-bucket/test.jpg", "format": "RGB_8B_HW3"},
            "instance_masks": [],
            "instance_full_masks": [],
            "type": "ConversationType.SEG_QA",
        }
    ]

    # Create test row WITHOUT an id key
    row = {
        "conversations": json.dumps(conversations_data),
        "mask_0": mock_pil_image,
    }

    # Create image bytes for mock
    img = Image.new("RGB", (1, 1))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    mock_fetch.return_value = buffer.getvalue()

    # Call the function
    result = loader.load_sample(row)

    # Verify result
    assert result is not None, "load_sample returned None unexpectedly"
    assert result["id"] == "unknown_id", "Missing id should default to 'unknown_id'"
