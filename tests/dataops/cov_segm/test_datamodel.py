import pytest
from PIL import Image
from pydantic import ValidationError

# Import the models and typeddicts to test
from src.dataops.cov_segm.datamodel import (
    ConversationItem,
    ImageURI,
    InstanceMask,
    Phrase,
    ProcessedConversationItem,
    ProcessedCovSegmSample,
    ProcessedMask,
)

# --- Pydantic Model Tests ---


# Phrase Tests
def test_phrase_valid():
    data = {"id": 1, "text": "hello", "type": "greeting"}
    phrase = Phrase(**data)
    assert phrase.id == 1
    assert phrase.text == "hello"
    assert phrase.type == "greeting"


def test_phrase_invalid_type():
    data = {"id": "not_an_int", "text": "hello", "type": "greeting"}
    with pytest.raises(ValidationError):
        Phrase(**data)


def test_phrase_missing_field():
    data = {"id": 1, "type": "greeting"}  # Missing 'text'
    with pytest.raises(ValidationError):
        Phrase(**data)


# ImageURI Tests
def test_image_uri_valid():
    data = {"jpg": "s3://bucket/img.jpg", "format": "RGB"}
    uri = ImageURI(**data)
    assert uri.jpg == "s3://bucket/img.jpg"
    assert uri.format == "RGB"


def test_image_uri_missing_field():
    data = {"format": "RGB"}  # Missing 'jpg'
    with pytest.raises(ValidationError):
        ImageURI(**data)


# InstanceMask Tests
def test_instance_mask_valid():
    data = {
        "column": "mask_0",
        "image_uri": {"jpg": "s3://bucket/mask.jpg", "format": "L"},
        "positive_value": 1,
    }
    mask = InstanceMask(**data)
    assert mask.column == "mask_0"
    assert isinstance(mask.image_uri, ImageURI)
    assert mask.image_uri.jpg == "s3://bucket/mask.jpg"
    assert mask.positive_value == 1


def test_instance_mask_invalid_nested_model():
    data = {
        "column": "mask_0",
        "image_uri": {"jpg": "s3://bucket/mask.jpg"},  # Missing 'format' in ImageURI
        "positive_value": 1,
    }
    with pytest.raises(ValidationError):
        InstanceMask(**data)


def test_instance_mask_missing_field():
    data = {
        "image_uri": {"jpg": "s3://bucket/mask.jpg", "format": "L"},
        "positive_value": 1,
    }  # Missing 'column'
    with pytest.raises(ValidationError):
        InstanceMask(**data)


# ConversationItem Tests
def test_conversation_item_valid_required_only():
    data = {
        "phrases": [{"id": 1, "text": "hi", "type": "greet"}],
        "image_uri": {"jpg": "s3://main.jpg", "format": "RGB"},
        "type": "QA",
    }
    item = ConversationItem(**data)
    assert len(item.phrases) == 1
    assert isinstance(item.phrases[0], Phrase)
    assert isinstance(item.image_uri, ImageURI)
    assert item.instance_masks is None  # Optional field default
    assert item.instance_full_masks is None  # Optional field default
    assert item.type == "QA"


def test_conversation_item_valid_with_optional():
    data = {
        "phrases": [{"id": 1, "text": "hi", "type": "greet"}],
        "image_uri": {"jpg": "s3://main.jpg", "format": "RGB"},
        "instance_masks": [
            {
                "column": "mask_inst",
                "image_uri": {"jpg": "s3://mask_i.jpg", "format": "L"},
                "positive_value": 1,
            }
        ],
        "instance_full_masks": [
            {
                "column": "mask_full",
                "image_uri": {"jpg": "s3://mask_f.jpg", "format": "P"},
                "positive_value": 2,
            }
        ],
        "type": "SEG",
    }
    item = ConversationItem(**data)
    assert len(item.instance_masks) == 1
    assert isinstance(item.instance_masks[0], InstanceMask)
    assert item.instance_masks[0].column == "mask_inst"
    assert len(item.instance_full_masks) == 1
    assert isinstance(item.instance_full_masks[0], InstanceMask)
    assert item.instance_full_masks[0].column == "mask_full"
    assert item.type == "SEG"


def test_conversation_item_invalid_phrase_list():
    data = {
        "phrases": [{"id": "bad", "text": "hi", "type": "greet"}],  # Invalid id type
        "image_uri": {"jpg": "s3://main.jpg", "format": "RGB"},
        "type": "QA",
    }
    with pytest.raises(ValidationError):
        ConversationItem(**data)


def test_conversation_item_missing_required():
    data = {
        "phrases": [{"id": 1, "text": "hi", "type": "greet"}],
        # Missing "image_uri"
        "type": "QA",
    }
    with pytest.raises(ValidationError):
        ConversationItem(**data)


# --- TypedDict Tests (Basic Checks) ---


# ProcessedMask Tests
def test_processed_mask_structure():
    # Create a dummy PIL Image
    dummy_mask_img = Image.new("L", (10, 10))
    mask: ProcessedMask = {
        "mask": dummy_mask_img,
        "positive_value": 1,
        "source": "mask_0",
        "pixel_area": 50,
        "width": 8,
        "height": 7,
    }
    # Basic checks for key existence and rough types
    assert "mask" in mask and isinstance(mask["mask"], Image.Image)
    assert "positive_value" in mask and isinstance(mask["positive_value"], int)
    assert "source" in mask and isinstance(mask["source"], str)
    assert "pixel_area" in mask and (
        mask["pixel_area"] is None or isinstance(mask["pixel_area"], int)
    )
    assert "width" in mask and (mask["width"] is None or isinstance(mask["width"], int))
    assert "height" in mask and (mask["height"] is None or isinstance(mask["height"], int))


# ProcessedConversationItem Tests
def test_processed_conversation_item_structure():
    dummy_mask_img = Image.new("L", (5, 5))
    processed_mask: ProcessedMask = {
        "mask": dummy_mask_img,
        "positive_value": 1,
        "source": "mask_0",
        "pixel_area": 10,
        "width": 3,
        "height": 4,
    }
    item: ProcessedConversationItem = {
        "phrases": [{"id": 1, "text": "obj", "type": "obj"}],
        "type": "SEG",
        "processed_instance_masks": [processed_mask],
        "processed_full_masks": [],
    }
    assert "phrases" in item and isinstance(item["phrases"], list)
    assert "type" in item and isinstance(item["type"], str)
    assert "processed_instance_masks" in item and isinstance(item["processed_instance_masks"], list)
    assert "processed_full_masks" in item and isinstance(item["processed_full_masks"], list)
    if item["processed_instance_masks"]:
        assert isinstance(
            item["processed_instance_masks"][0], dict
        )  # Check it's a dict (like ProcessedMask)


# ProcessedCovSegmSample Tests
def test_processed_cov_segm_sample_structure():
    dummy_main_img = Image.new("RGB", (20, 20))
    dummy_mask_img = Image.new("L", (5, 5))
    processed_mask: ProcessedMask = {
        "mask": dummy_mask_img,
        "positive_value": 1,
        "source": "mask_0",
        "pixel_area": 10,
        "width": 3,
        "height": 4,
    }
    processed_item: ProcessedConversationItem = {
        "phrases": [{"id": 1, "text": "obj", "type": "obj"}],
        "type": "SEG",
        "processed_instance_masks": [processed_mask],
        "processed_full_masks": [],
    }
    sample: ProcessedCovSegmSample = {
        "id": "sample_123",
        "image": dummy_main_img,
        "processed_conversations": [processed_item],
    }
    assert "id" in sample and isinstance(sample["id"], str)
    assert "image" in sample and isinstance(sample["image"], Image.Image)
    assert "processed_conversations" in sample and isinstance(
        sample["processed_conversations"], list
    )
    if sample["processed_conversations"]:
        assert isinstance(
            sample["processed_conversations"][0], dict
        )  # Check it's a dict (like ProcessedConversationItem)
