# Modules to test
# Type hints and helpers
# Basic logging setup for tests if needed (optional)
import io
import json
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from vibelab.dataops.cov_segm import loader

# Import necessary components from datamodel
from vibelab.dataops.cov_segm.datamodel import (
    ClsSegment,
    ConversationItem,
    ImageURI,
    InstanceMask,
    Phrase,
    SegmMask,
    SegmSample,
)

logging.basicConfig(level=logging.DEBUG)  # Show debug logs during tests
logger = logging.getLogger(__name__)

# --- Fixtures --- #


@pytest.fixture
def mock_pil_image_fixture():
    """Creates a simple mock PIL Image (10x10)."""
    return Image.new("L", (10, 10), color=1)  # Grayscale for easier mask testing


@pytest.fixture
def mock_numpy_array_fixture():
    """Creates a simple mock NumPy array (10x10)."""
    return np.ones((10, 10), dtype=np.uint8)


@pytest.fixture
def mock_valid_segm_mask_fixture():
    """Creates a mock SegmMask instance that is valid."""
    mask = MagicMock(spec=SegmMask)
    mask.is_valid = True
    mask.pixel_area = 50  # Example value
    mask.source_info = InstanceMask(
        column="mock_col", positive_value=1, image_uri=ImageURI(jpg="d", format="d")
    )
    return mask


@pytest.fixture
def mock_invalid_segm_mask_fixture():
    """Creates a mock SegmMask instance that is invalid."""
    mask = MagicMock(spec=SegmMask)
    mask.is_valid = False
    mask.source_info = InstanceMask(
        column="mock_col_invalid", positive_value=1, image_uri=ImageURI(jpg="d", format="d")
    )
    return mask


@pytest.fixture
def simple_instance_mask_list():
    """Provides a simple list of InstanceMask metadata."""
    return [
        InstanceMask(column="mask_0", positive_value=1, image_uri=ImageURI(jpg="d1", format="f1")),
        InstanceMask(
            column="masks_rest/0", positive_value=2, image_uri=ImageURI(jpg="d2", format="f2")
        ),
    ]


@pytest.fixture
def simple_hf_row_fixture(mock_pil_image_fixture):
    """Provides a simplified hf_row dict for testing helpers."""
    return {
        "mask_0": mock_pil_image_fixture,  # Direct access
        "masks_rest": [Image.new("L", (5, 5)), Image.new("L", (8, 8))],  # Nested access
        "image_0": Image.new("RGB", (20, 20)),  # For testing non-mask refs
        "other_col": "some_string",
    }


@pytest.fixture
def basic_conv_item_fixture():
    """Provides a basic valid ConversationItem."""
    return ConversationItem(
        phrases=[Phrase(id=1, text="obj", type="t")],
        image_uri=ImageURI(jpg="s3://main.jpg", format="f"),
        instance_masks=[
            InstanceMask(column="mask_0", positive_value=1, image_uri=ImageURI(jpg="d", format="d"))
        ],
        instance_full_masks=None,
        type="SEG",
    )


@pytest.fixture
def full_hf_row_fixture(basic_conv_item_fixture, mock_pil_image_fixture):
    """Provides a more complete hf_row for load_sample tests."""
    conv_items_list = [basic_conv_item_fixture.model_dump()]
    return {
        "id": "sample_123",
        "conversations": json.dumps(conv_items_list),
        "mask_0": mock_pil_image_fixture,  # Ensure mask data exists
        # Add other columns if needed by tests
    }


# --- Test Functions --- #

# == Tests for parse_conversations (Keep as is) ==

VALID_CONVERSATION_ITEM_DICT = {
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

VALID_CONVERSATIONS_JSON = json.dumps([VALID_CONVERSATION_ITEM_DICT])
INVALID_JSON_STRING = '{"phrases": [}}'  # Malformed JSON
JSON_WITH_SCHEMA_ERROR = json.dumps(
    [
        {
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
    result = loader.parse_conversations(VALID_CONVERSATIONS_JSON)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], ConversationItem)
    assert result[0].image_uri.jpg == "s3://bucket/image.jpg"
    assert result[0].phrases[0].text == "prompt1"
    assert result[0].instance_masks[0].column == "mask_0"


def test_parse_conversations_invalid_json():
    with pytest.raises(ValueError, match="Invalid JSON"):
        loader.parse_conversations(INVALID_JSON_STRING)


def test_parse_conversations_schema_error_missing_field():
    with pytest.raises(ValueError, match="Invalid conversation data structure"):
        loader.parse_conversations(JSON_WITH_SCHEMA_ERROR)


def test_parse_conversations_schema_error_wrong_type():
    with pytest.raises(ValueError, match="Invalid conversation data structure"):
        loader.parse_conversations(JSON_WITH_TYPE_ERROR)


def test_parse_conversations_empty_string():
    with pytest.raises(ValueError, match="Invalid JSON"):
        loader.parse_conversations("")


def test_parse_conversations_empty_list_json():
    result = loader.parse_conversations("[]")
    assert isinstance(result, list)
    assert len(result) == 0


# == Tests for _resolve_reference_path (Updated) ==


def test_resolve_reference_path_direct_hit(simple_hf_row_fixture):
    value, success = loader._resolve_reference_path("mask_0", simple_hf_row_fixture)
    assert success is True
    assert isinstance(value, Image.Image)
    assert value == simple_hf_row_fixture["mask_0"]


def test_resolve_reference_path_direct_miss(simple_hf_row_fixture):
    value, success = loader._resolve_reference_path("non_existent_mask", simple_hf_row_fixture)
    assert success is False
    assert value is None


def test_resolve_reference_path_nested_hit(simple_hf_row_fixture):
    value, success = loader._resolve_reference_path("masks_rest/1", simple_hf_row_fixture)
    assert success is True
    assert isinstance(value, Image.Image)
    assert value == simple_hf_row_fixture["masks_rest"][1]


def test_resolve_reference_path_nested_base_miss(simple_hf_row_fixture):
    value, success = loader._resolve_reference_path("images_rest/0", simple_hf_row_fixture)
    assert success is False  # 'images_rest' key doesn't exist
    assert value is None


def test_resolve_reference_path_nested_index_out_of_bounds(simple_hf_row_fixture):
    value, success = loader._resolve_reference_path("masks_rest/5", simple_hf_row_fixture)
    assert success is False
    assert value is None


def test_resolve_reference_path_nested_index_not_int(simple_hf_row_fixture):
    value, success = loader._resolve_reference_path("masks_rest/abc", simple_hf_row_fixture)
    assert success is False
    assert value is None


def test_resolve_reference_path_nested_not_list(simple_hf_row_fixture):
    # Test case where base column exists but isn't a list
    row = {"masks_rest": {"0": Image.new("L", (1, 1))}}
    value, success = loader._resolve_reference_path("masks_rest/0", row)
    assert success is False
    assert value is None


def test_resolve_reference_path_invalid_path_format(simple_hf_row_fixture):
    value, success = loader._resolve_reference_path("masks_rest/", simple_hf_row_fixture)
    assert success is False
    assert value is None
    value, success = loader._resolve_reference_path("masks_rest/0/1", simple_hf_row_fixture)
    assert success is False  # Only splits once
    assert value is None


# == Tests for _process_mask_list (New) ==


@patch("vibelab.dataops.cov_segm.loader._resolve_reference_path")
@patch("vibelab.dataops.cov_segm.loader.SegmMask")
def test_process_mask_list_success(
    mock_segmmask_class,
    mock_resolve,
    simple_instance_mask_list,
    simple_hf_row_fixture,
    mock_valid_segm_mask_fixture,
    mock_pil_image_fixture,
):
    """Test successful processing of a list of mask metadata."""
    # Mock _resolve to return valid image data for the columns in simple_instance_mask_list
    mock_resolve.side_effect = [
        (simple_hf_row_fixture["mask_0"], True),  # For mask_0
        (simple_hf_row_fixture["masks_rest"][0], True),  # For masks_rest/0
    ]
    # Mock SegmMask constructor to return a valid mock mask
    mock_segmmask_class.return_value = mock_valid_segm_mask_fixture

    result = loader._process_mask_list(
        simple_instance_mask_list, simple_hf_row_fixture, "row1", 0, "visible"
    )

    assert len(result) == 2
    assert all(isinstance(m, MagicMock) for m in result)  # Check they are our mocks
    assert all(m.is_valid for m in result)
    assert mock_resolve.call_count == 2
    assert mock_segmmask_class.call_count == 2
    # Check SegmMask was called with correct args (using call_args_list)
    # First call args:
    call_args1 = mock_segmmask_class.call_args_list[0][1]  # kwargs of first call
    assert call_args1["instance_mask_info"] == simple_instance_mask_list[0]
    assert call_args1["raw_mask_data"] == simple_hf_row_fixture["mask_0"]
    # Second call args:
    call_args2 = mock_segmmask_class.call_args_list[1][1]
    assert call_args2["instance_mask_info"] == simple_instance_mask_list[1]
    assert call_args2["raw_mask_data"] == simple_hf_row_fixture["masks_rest"][0]


def test_process_mask_list_empty_or_none_input(simple_hf_row_fixture):
    """Test with None or empty list input."""
    result_none = loader._process_mask_list(None, simple_hf_row_fixture, "row1", 0, "visible")
    assert result_none == []
    result_empty = loader._process_mask_list([], simple_hf_row_fixture, "row1", 0, "visible")
    assert result_empty == []


@patch("vibelab.dataops.cov_segm.loader._resolve_reference_path")
@patch("vibelab.dataops.cov_segm.loader.SegmMask")
def test_process_mask_list_resolve_fails(
    mock_segmmask_class,
    mock_resolve,
    simple_instance_mask_list,
    simple_hf_row_fixture,
    mock_valid_segm_mask_fixture,
):
    """Test when _resolve_reference_path fails for one item."""
    mock_resolve.side_effect = [
        (simple_hf_row_fixture["mask_0"], True),
        (None, False),  # Fail for masks_rest/0
    ]
    mock_segmmask_class.return_value = mock_valid_segm_mask_fixture

    result = loader._process_mask_list(
        simple_instance_mask_list, simple_hf_row_fixture, "row1", 0, "visible"
    )

    assert len(result) == 1  # Only the first mask should be processed
    assert mock_resolve.call_count == 2
    mock_segmmask_class.assert_called_once()  # Only called for the successful resolve


@patch("vibelab.dataops.cov_segm.loader._resolve_reference_path")
@patch("vibelab.dataops.cov_segm.loader.SegmMask")
def test_process_mask_list_invalid_data_type(
    mock_segmmask_class,
    mock_resolve,
    simple_instance_mask_list,
    simple_hf_row_fixture,
    mock_valid_segm_mask_fixture,
):
    """Test when _resolve returns an unsupported data type."""
    mock_resolve.side_effect = [
        (simple_hf_row_fixture["mask_0"], True),
        ("not_an_image_or_array", True),  # Wrong type for masks_rest/0
    ]
    mock_segmmask_class.return_value = mock_valid_segm_mask_fixture

    result = loader._process_mask_list(
        simple_instance_mask_list, simple_hf_row_fixture, "row1", 0, "visible"
    )

    assert len(result) == 1
    assert mock_resolve.call_count == 2
    mock_segmmask_class.assert_called_once()


@patch("vibelab.dataops.cov_segm.loader._resolve_reference_path")
@patch("vibelab.dataops.cov_segm.loader.SegmMask")
def test_process_mask_list_segmmask_invalid(
    mock_segmmask_class,
    mock_resolve,
    simple_instance_mask_list,
    simple_hf_row_fixture,
    mock_invalid_segm_mask_fixture,
):
    """Test when SegmMask instance is created but is_valid is False."""
    mock_resolve.return_value = (simple_hf_row_fixture["mask_0"], True)  # Always succeed resolve
    mock_segmmask_class.return_value = mock_invalid_segm_mask_fixture  # Always return invalid mask

    result = loader._process_mask_list(
        simple_instance_mask_list, simple_hf_row_fixture, "row1", 0, "visible"
    )

    assert len(result) == 0  # No valid masks should be returned
    assert mock_resolve.call_count == 2
    assert mock_segmmask_class.call_count == 2  # Constructor called twice


@patch("vibelab.dataops.cov_segm.loader._resolve_reference_path")
@patch("vibelab.dataops.cov_segm.loader.SegmMask")
def test_process_mask_list_segmmask_exception(
    mock_segmmask_class, mock_resolve, simple_instance_mask_list, simple_hf_row_fixture
):
    """Test when SegmMask instantiation raises an exception."""
    mock_resolve.return_value = (simple_hf_row_fixture["mask_0"], True)  # Always succeed resolve
    mock_segmmask_class.side_effect = ValueError("Test exception")  # Raise error on instantiation

    result = loader._process_mask_list(
        simple_instance_mask_list, simple_hf_row_fixture, "row1", 0, "visible"
    )

    assert len(result) == 0
    assert mock_resolve.call_count == 2
    assert mock_segmmask_class.call_count == 2  # Constructor called twice


def test_process_mask_list_no_column(simple_hf_row_fixture):
    """Test InstanceMask metadata missing the 'column' field."""
    mask_list = [InstanceMask(column="", positive_value=1, image_uri=ImageURI(jpg="d", format="d"))]
    result = loader._process_mask_list(mask_list, simple_hf_row_fixture, "row1", 0, "visible")
    assert result == []


# == Tests for _load_image_from_uri (Keep as is, maybe simplify fixture) ==
@pytest.fixture
def mock_image_bytes():
    """Creates dummy PNG image bytes."""
    img = Image.new("RGB", (10, 10))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


@patch("vibelab.dataops.cov_segm.loader.is_s3_uri")
@patch("vibelab.dataops.cov_segm.loader.fetch_s3_uri")
def test_load_image_from_uri_success(mock_fetch, mock_is_s3, mock_image_bytes):
    mock_is_s3.return_value = True
    mock_fetch.return_value = mock_image_bytes
    uri = "s3://good/uri.png"
    result = loader._load_image_from_uri(uri)
    mock_is_s3.assert_called_once_with(uri)
    mock_fetch.assert_called_once_with(uri)
    assert isinstance(result, Image.Image)
    assert result.size == (10, 10)


@patch("vibelab.dataops.cov_segm.loader.is_s3_uri")
def test_load_image_from_uri_invalid_uri(mock_is_s3):
    mock_is_s3.return_value = False
    uri = "http://not/s3/uri.jpg"
    result = loader._load_image_from_uri(uri)
    mock_is_s3.assert_called_once_with(uri)
    assert result is None


@patch("vibelab.dataops.cov_segm.loader.is_s3_uri")
@patch("vibelab.dataops.cov_segm.loader.fetch_s3_uri")
def test_load_image_from_uri_fetch_fails(mock_fetch, mock_is_s3):
    mock_is_s3.return_value = True
    mock_fetch.return_value = None
    uri = "s3://fetch/fails.jpg"
    result = loader._load_image_from_uri(uri)
    mock_is_s3.assert_called_once_with(uri)
    mock_fetch.assert_called_once_with(uri)
    assert result is None


@patch("vibelab.dataops.cov_segm.loader.is_s3_uri")
@patch("vibelab.dataops.cov_segm.loader.fetch_s3_uri")
@patch("PIL.Image.open")
def test_load_image_from_uri_pil_error(mock_pil_open, mock_fetch, mock_is_s3, mock_image_bytes):
    mock_is_s3.return_value = True
    mock_fetch.return_value = mock_image_bytes
    mock_pil_open.side_effect = IOError("PIL cannot open")
    uri = "s3://pil/error.jpg"
    result = loader._load_image_from_uri(uri)
    mock_is_s3.assert_called_once_with(uri)
    mock_fetch.assert_called_once_with(uri)
    mock_pil_open.assert_called_once()
    assert result is None


# == Tests for load_sample (Refactored) ==


@patch("vibelab.dataops.cov_segm.loader.parse_conversations")
@patch("vibelab.dataops.cov_segm.loader._load_image_from_uri")
@patch("vibelab.dataops.cov_segm.loader._process_mask_list")
def test_load_sample_success(
    mock_process_list,
    mock_load_image,
    mock_parse_conv,
    full_hf_row_fixture,
    basic_conv_item_fixture,
    mock_pil_image_fixture,
    mock_valid_segm_mask_fixture,
):
    """Test successful loading and processing of a sample."""
    # Mock dependencies
    mock_parse_conv.return_value = [basic_conv_item_fixture]  # Return the parsed item
    mock_load_image.return_value = mock_pil_image_fixture  # Return the mock main image
    # Mock _process_mask_list to return one valid mask for visible, none for full
    mock_process_list.side_effect = [[mock_valid_segm_mask_fixture], []]

    # Call the function under test
    result = loader.load_sample(full_hf_row_fixture)

    # Assertions
    assert isinstance(result, SegmSample)
    assert result.id == "sample_123"
    assert result.image == mock_pil_image_fixture
    assert len(result.segments) == 1
    assert isinstance(result.segments[0], ClsSegment)

    # Check segment details
    segment = result.segments[0]
    assert segment.phrases == basic_conv_item_fixture.phrases
    assert segment.type == basic_conv_item_fixture.type
    assert len(segment.visible_masks) == 1
    assert segment.visible_masks[0] == mock_valid_segm_mask_fixture
    assert segment.full_masks == []

    # Verify mocks were called correctly
    mock_parse_conv.assert_called_once_with(full_hf_row_fixture["conversations"])
    mock_load_image.assert_called_once_with(basic_conv_item_fixture.image_uri.jpg)
    assert mock_process_list.call_count == 2
    # Check calls to _process_mask_list
    mock_process_list.assert_any_call(
        basic_conv_item_fixture.instance_masks, full_hf_row_fixture, "sample_123", 0, "visible"
    )
    mock_process_list.assert_any_call(
        basic_conv_item_fixture.instance_full_masks, full_hf_row_fixture, "sample_123", 0, "full"
    )


@patch("vibelab.dataops.cov_segm.loader.parse_conversations")
@patch("vibelab.dataops.cov_segm.loader._load_image_from_uri")
@patch("vibelab.dataops.cov_segm.loader._process_mask_list")
def test_load_sample_no_masks_processed(
    mock_process_list,
    mock_load_image,
    mock_parse_conv,
    full_hf_row_fixture,
    basic_conv_item_fixture,
    mock_pil_image_fixture,
):
    """Test successful load when no masks are found/processed."""
    mock_parse_conv.return_value = [basic_conv_item_fixture]
    mock_load_image.return_value = mock_pil_image_fixture
    mock_process_list.return_value = []  # Simulate no valid masks found

    result = loader.load_sample(full_hf_row_fixture)

    assert isinstance(result, SegmSample)
    assert result.id == "sample_123"
    assert len(result.segments) == 1
    assert result.segments[0].visible_masks == []
    assert result.segments[0].full_masks == []
    assert mock_process_list.call_count == 2


@patch("vibelab.dataops.cov_segm.loader.parse_conversations")
def test_load_sample_parse_fails(mock_parse_conv, full_hf_row_fixture):
    """Test behavior when conversations JSON parsing fails."""
    mock_parse_conv.side_effect = ValueError("Simulated parse error")
    result = loader.load_sample(full_hf_row_fixture)
    assert result is None
    mock_parse_conv.assert_called_once_with(full_hf_row_fixture["conversations"])


@patch("vibelab.dataops.cov_segm.loader.parse_conversations")
@patch("vibelab.dataops.cov_segm.loader._load_image_from_uri")
def test_load_sample_image_load_fails(
    mock_load_image, mock_parse_conv, full_hf_row_fixture, basic_conv_item_fixture
):
    """Test behavior when main image loading fails."""
    mock_parse_conv.return_value = [basic_conv_item_fixture]
    mock_load_image.return_value = None  # Simulate failure
    result = loader.load_sample(full_hf_row_fixture)
    assert result is None
    mock_load_image.assert_called_once_with(basic_conv_item_fixture.image_uri.jpg)


@patch("vibelab.dataops.cov_segm.loader.parse_conversations")
@patch("vibelab.dataops.cov_segm.loader._load_image_from_uri")
def test_load_sample_mismatched_uris(
    mock_load_image, mock_parse_conv, full_hf_row_fixture, mock_pil_image_fixture
):
    """Test when conversation items have different image URIs."""
    item1 = ConversationItem(
        phrases=[Phrase(id=1, text="obj1", type="t")],
        image_uri=ImageURI(jpg="s3://main1.jpg", format="f"),
        type="SEG",
    )
    item2 = ConversationItem(
        phrases=[Phrase(id=2, text="obj2", type="t")],
        image_uri=ImageURI(jpg="s3://main2.jpg", format="f"),
        type="SEG",
    )  # Different URI
    mock_parse_conv.return_value = [item1, item2]
    mock_load_image.return_value = mock_pil_image_fixture  # Assume first load succeeds

    row = {"id": "mismatch", "conversations": json.dumps([item1.model_dump(), item2.model_dump()])}
    result = loader.load_sample(row)
    assert result is None
    mock_load_image.assert_called_once_with(item1.image_uri.jpg)  # Only called for the first item


@patch("vibelab.dataops.cov_segm.loader.parse_conversations")
@patch("vibelab.dataops.cov_segm.loader._load_image_from_uri")
@patch("vibelab.dataops.cov_segm.loader._process_mask_list")
def test_load_sample_no_valid_segments(
    mock_process_list, mock_load_image, mock_parse_conv, mock_pil_image_fixture
):
    """Test when no segments can be created (e.g., empty phrases)."""
    item_no_phrase = ConversationItem(
        phrases=[],  # Empty phrases
        image_uri=ImageURI(jpg="s3://main.jpg", format="f"),
        type="SEG",
    )
    mock_parse_conv.return_value = [item_no_phrase]
    mock_load_image.return_value = mock_pil_image_fixture
    mock_process_list.return_value = []  # No masks either

    row = {"id": "no_segments", "conversations": json.dumps([item_no_phrase.model_dump()])}
    result = loader.load_sample(row)
    assert result is None  # Should return None as no segments were added


def test_load_sample_missing_id(full_hf_row_fixture):
    """Test when 'id' key is missing."""
    del full_hf_row_fixture["id"]
    result = loader.load_sample(full_hf_row_fixture)
    assert result is None


def test_load_sample_missing_conversations():
    """Test when 'conversations' key is missing."""
    row = {"id": "no_conv"}
    result = loader.load_sample(row)
    assert result is None
