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
    # Mocking resolve_mask_path to simulate direct PIL Image data
    with patch("src.dataops.cov_segm.loader._resolve_mask_path") as mock_resolve:
        mock_resolve.return_value = (mock_pil_image, True)
        row = {}  # Not used as resolve is mocked
        mask_info = dm.InstanceMask(
            column="mask_0", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
        )
        result = loader._load_mask(mask_info, row)

    assert result is not None
    assert isinstance(result["mask"], Image.Image)
    assert result["mask"] == mock_pil_image
    assert result["positive_value"] == 1
    assert result["source"] == "mask_0"
    # Geometry might be calculated - check they are present
    assert "pixel_area" in result
    assert "width" in result
    assert "height" in result


def test_load_mask_numpy_direct(mock_numpy_array):
    """Test loading a NumPy array directly from a column (should be converted to PIL)."""
    # Mocking resolve_mask_path to simulate direct numpy array data
    with patch("src.dataops.cov_segm.loader._resolve_mask_path") as mock_resolve:
        mock_resolve.return_value = (mock_numpy_array, True)
        row = {}
        mask_info = dm.InstanceMask(
            column="mask_0", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
        )
        result = loader._load_mask(mask_info, row)

    assert result is not None
    assert isinstance(result["mask"], Image.Image)
    assert result["mask"].size == (mock_numpy_array.shape[1], mock_numpy_array.shape[0])
    assert result["positive_value"] == 1
    assert result["source"] == "mask_0"
    assert "pixel_area" in result
    assert "width" in result
    assert "height" in result


@patch("src.dataops.cov_segm.loader._resolve_mask_path")
def test_load_mask_invalid_type_in_column(mock_resolve):
    """Test behavior when the column contains an unexpected data type."""
    mock_resolve.return_value = ("not_an_image", True)
    row = {}
    mask_info = dm.InstanceMask(
        column="mask_0", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
    )
    mask = loader._load_mask(mask_info, row)
    assert mask is None


@patch("src.dataops.cov_segm.loader._resolve_mask_path")
def test_load_mask_none_in_column(mock_resolve):
    """Test behavior when the column value is None."""
    mock_resolve.return_value = (None, True)
    row = {}
    mask_info = dm.InstanceMask(
        column="mask_0", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
    )
    mask = loader._load_mask(mask_info, row)
    assert mask is None


# == Tests for _resolve_mask_path ==


def test_resolve_mask_path_direct_hit():
    row = {"mask_0": 123}
    value, success = loader._resolve_mask_path("mask_0", row)
    assert success is True
    assert value == 123


def test_resolve_mask_path_direct_miss():
    row = {"mask_0": 123}
    value, success = loader._resolve_mask_path("mask_1", row)
    assert success is False
    assert value is None


def test_resolve_mask_path_nested_hit():
    row = {"masks_rest": [10, 20, 30]}
    value, success = loader._resolve_mask_path("masks_rest/1", row)
    assert success is True
    assert value == 20


def test_resolve_mask_path_nested_base_miss():
    row = {"other_key": [10, 20, 30]}
    value, success = loader._resolve_mask_path("masks_rest/1", row)
    assert success is False
    assert value is None


def test_resolve_mask_path_nested_index_out_of_bounds():
    row = {"masks_rest": [10, 20, 30]}
    value, success = loader._resolve_mask_path("masks_rest/3", row)
    assert success is False
    assert value is None


def test_resolve_mask_path_nested_index_not_int():
    row = {"masks_rest": [10, 20, 30]}
    value, success = loader._resolve_mask_path("masks_rest/abc", row)
    assert success is False
    assert value is None


def test_resolve_mask_path_nested_not_list():
    row = {"masks_rest": {"0": 10}}
    value, success = loader._resolve_mask_path("masks_rest/0", row)
    assert success is False
    assert value is None


# == Tests for _load_mask (including geometry) ==

# (Keep existing fixtures: mock_pil_image, mock_numpy_array)


def test_load_mask_pil_direct(mock_pil_image):
    """Test loading a PIL image directly from a column."""
    # Mocking resolve_mask_path to simulate direct PIL Image data
    with patch("src.dataops.cov_segm.loader._resolve_mask_path") as mock_resolve:
        mock_resolve.return_value = (mock_pil_image, True)
        row = {}  # Not used as resolve is mocked
        mask_info = dm.InstanceMask(
            column="mask_0", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
        )
        result = loader._load_mask(mask_info, row)

    assert result is not None
    assert isinstance(result["mask"], Image.Image)
    assert result["mask"] == mock_pil_image
    assert result["positive_value"] == 1
    assert result["source"] == "mask_0"
    # Geometry might be calculated - check they are present
    assert "pixel_area" in result
    assert "width" in result
    assert "height" in result


def test_load_mask_numpy_direct(mock_numpy_array):
    """Test loading a NumPy array directly from a column (should be converted to PIL)."""
    # Mocking resolve_mask_path to simulate direct numpy array data
    with patch("src.dataops.cov_segm.loader._resolve_mask_path") as mock_resolve:
        mock_resolve.return_value = (mock_numpy_array, True)
        row = {}
        mask_info = dm.InstanceMask(
            column="mask_0", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
        )
        result = loader._load_mask(mask_info, row)

    assert result is not None
    assert isinstance(result["mask"], Image.Image)
    assert result["mask"].size == (mock_numpy_array.shape[1], mock_numpy_array.shape[0])
    assert result["positive_value"] == 1
    assert result["source"] == "mask_0"
    assert "pixel_area" in result
    assert "width" in result
    assert "height" in result


@patch("src.dataops.cov_segm.loader._resolve_mask_path")
def test_load_mask_invalid_type_in_column(mock_resolve):
    """Test behavior when the column contains an unexpected data type."""
    mock_resolve.return_value = ("not_an_image", True)
    row = {}
    mask_info = dm.InstanceMask(
        column="mask_0", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
    )
    mask = loader._load_mask(mask_info, row)
    assert mask is None


@patch("src.dataops.cov_segm.loader._resolve_mask_path")
def test_load_mask_none_in_column(mock_resolve):
    """Test behavior when the column value is None."""
    mock_resolve.return_value = (None, True)
    row = {}
    mask_info = dm.InstanceMask(
        column="mask_0", positive_value=1, image_uri=dm.ImageURI(jpg="dummy", format="dummy")
    )
    mask = loader._load_mask(mask_info, row)
    assert mask is None


# (Keep test_load_mask_geometry)
# (Keep test_load_mask_geometry_calc_exception)


# == Tests for _process_mask_metadata ==
@patch("src.dataops.cov_segm.loader._load_mask")
def test_process_mask_metadata_calls_load_mask(mock_load_mask):
    """Verify _process_mask_metadata correctly calls _load_mask."""
    mock_mask_data = {"key": "value"}  # Dummy return value
    mock_load_mask.return_value = mock_mask_data

    mask_info = dm.InstanceMask(
        column="mask_path/1", positive_value=2, image_uri=dm.ImageURI(jpg="d", format="d")
    )
    hf_row = {"data": "sample"}

    result = loader._process_mask_metadata(mask_info, hf_row)

    assert result == mock_mask_data
    mock_load_mask.assert_called_once_with(mask_info, hf_row)


def test_process_mask_metadata_no_column():
    """Test _process_mask_metadata returns None if mask_info has no column."""
    mask_info = dm.InstanceMask(
        column="",
        positive_value=1,
        image_uri=dm.ImageURI(jpg="d", format="d"),  # Empty column
    )
    hf_row = {"data": "sample"}
    result = loader._process_mask_metadata(mask_info, hf_row)
    assert result is None


# == Tests for _load_image_from_uri ==
@patch("src.dataops.cov_segm.loader.is_s3_uri")
@patch("src.dataops.cov_segm.loader.fetch_s3_uri")
def test_load_image_from_uri_success(mock_fetch, mock_is_s3):
    """Test successful image loading from S3."""
    mock_is_s3.return_value = True
    # Create dummy image bytes (e.g., small PNG)
    img = Image.new("RGB", (10, 10))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    mock_fetch.return_value = img_byte_arr.getvalue()

    uri = "s3://good/uri.png"
    result = loader._load_image_from_uri(uri)

    mock_is_s3.assert_called_once_with(uri)
    mock_fetch.assert_called_once_with(uri)
    assert isinstance(result, Image.Image)
    assert result.size == (10, 10)


@patch("src.dataops.cov_segm.loader.is_s3_uri")
def test_load_image_from_uri_invalid_uri(mock_is_s3):
    """Test loading with an invalid (non-S3) URI."""
    mock_is_s3.return_value = False
    uri = "http://not/s3/uri.jpg"
    result = loader._load_image_from_uri(uri)
    mock_is_s3.assert_called_once_with(uri)
    assert result is None


@patch("src.dataops.cov_segm.loader.is_s3_uri")
@patch("src.dataops.cov_segm.loader.fetch_s3_uri")
def test_load_image_from_uri_fetch_fails(mock_fetch, mock_is_s3):
    """Test when fetching S3 content returns None."""
    mock_is_s3.return_value = True
    mock_fetch.return_value = None  # Simulate fetch failure
    uri = "s3://fetch/fails.jpg"
    result = loader._load_image_from_uri(uri)
    mock_is_s3.assert_called_once_with(uri)
    mock_fetch.assert_called_once_with(uri)
    assert result is None


@patch("src.dataops.cov_segm.loader.is_s3_uri")
@patch("src.dataops.cov_segm.loader.fetch_s3_uri")
@patch("PIL.Image.open")
def test_load_image_from_uri_pil_error(mock_pil_open, mock_fetch, mock_is_s3):
    """Test when PIL.Image.open raises an error."""
    mock_is_s3.return_value = True
    mock_fetch.return_value = b"some_bytes"  # Simulate successful fetch
    mock_pil_open.side_effect = IOError("PIL cannot open")  # Simulate PIL error

    uri = "s3://pil/error.jpg"
    result = loader._load_image_from_uri(uri)

    mock_is_s3.assert_called_once_with(uri)
    mock_fetch.assert_called_once_with(uri)
    mock_pil_open.assert_called_once()
    assert result is None


# == Tests for load_sample (Updates needed) ==


@pytest.fixture
def mock_s3_image():
    """Creates mock PNG image bytes for S3 fetch."""
    # This fixture is still useful for mocking the return of _load_image_from_uri
    return Image.new("RGB", (100, 50))  # Return PIL Image directly


# (Keep test_load_sample_success_updated)
# (Keep test_load_sample_mask_processing_fails)


# Keep failure tests but ensure mocks are correct
@patch("src.dataops.cov_segm.loader.parse_conversations")
@patch("src.dataops.cov_segm.loader._load_image_from_uri")  # Mock this layer
def test_load_sample_s3_fetch_fails(mock_load_image, mock_parse_conv):
    """Test behavior when S3 image fetching fails (via _load_image_from_uri)."""
    # Setup valid conversation parsing
    conv_item = dm.ConversationItem(
        phrases=[dm.Phrase(id=1, text="obj", type="t")],
        image_uri=dm.ImageURI(jpg="s3://main.jpg", format="f"),
        type="SEG",
    )
    mock_parse_conv.return_value = [conv_item]

    # Simulate image load failure
    mock_load_image.return_value = None

    input_row = {"conversations": json.dumps([conv_item.model_dump()])}

    result = loader.load_sample(input_row)
    assert result is None
    mock_parse_conv.assert_called_once()
    mock_load_image.assert_called_once_with("s3://main.jpg")


@patch("src.dataops.cov_segm.loader.parse_conversations")  # Mock parse_conversations
def test_load_sample_parse_fails(mock_parse_conv):
    """Test behavior when conversations JSON parsing fails."""
    # Simulate parsing failure
    mock_parse_conv.side_effect = ValueError("Simulated parse error")

    input_row = {"conversations": INVALID_JSON_STRING}

    result = loader.load_sample(input_row)
    assert result is None
    mock_parse_conv.assert_called_once_with(INVALID_JSON_STRING)


def test_load_sample_missing_conversations_key():
    """Test behavior when the 'conversations' key is missing from the row."""
    input_row = {"id": "no_convo_key"}
    result = loader.load_sample(input_row)
    assert result is None


@patch("src.dataops.cov_segm.loader.parse_conversations")
@patch("src.dataops.cov_segm.loader._load_image_from_uri")
@patch("src.dataops.cov_segm.loader._process_mask_metadata")
def test_load_sample_missing_id_key(
    mock_process_mask, mock_load_img, mock_parse_conv, mock_s3_image
):
    """Test behavior when the 'id' key is missing (should default)."""
    # Setup for successful load otherwise
    mock_load_img.return_value = mock_s3_image
    conv_item = dm.ConversationItem(
        phrases=[dm.Phrase(id=1, text="obj", type="t")],
        image_uri=dm.ImageURI(jpg="s3://main.jpg", format="f"),
        type="SEG",
        instance_masks=None,
        instance_full_masks=None,  # No masks for simplicity
    )
    mock_parse_conv.return_value = [conv_item]
    mock_process_mask.return_value = None  # No masks to process

    # Row missing 'id'
    input_row = {"conversations": json.dumps([conv_item.model_dump()])}

    result = loader.load_sample(input_row)

    assert result is not None
    assert result["id"] == "unknown_id"  # Check for default ID
    assert result["image"] == mock_s3_image
    assert len(result["processed_conversations"]) == 1
    mock_process_mask.assert_not_called()  # No masks defined in conv_item
