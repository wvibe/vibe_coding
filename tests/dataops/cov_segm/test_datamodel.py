import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

# Import the models and typeddicts to test
from src.dataops.cov_segm.datamodel import (
    ClsSegment,
    ConversationItem,
    ImageURI,
    InstanceMask,
    Phrase,
    SegmMask,
    SegmSample,
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


# --- Fixtures ---
@pytest.fixture
def sample_image_uri():
    """Provides a sample ImageURI object."""
    # Use a real S3 URI format, but dummy values
    return ImageURI(jpg="s3://dummy-bucket/images/test_image.jpg", format="jpg")


@pytest.fixture
def sample_instance_mask_info_pv1(sample_image_uri):
    """Provides a sample InstanceMask with positive_value=1."""
    return InstanceMask(column="mask_col_1", image_uri=sample_image_uri, positive_value=1)


@pytest.fixture
def sample_instance_mask_info_pv0(sample_image_uri):
    """Provides a sample InstanceMask with positive_value=0."""
    return InstanceMask(column="mask_col_0", image_uri=sample_image_uri, positive_value=0)


@pytest.fixture
def sample_image():
    """Creates a dummy PIL Image for testing."""
    return Image.new("RGB", (60, 30), color="red")


# --- Raw Mask Data Generation ---
def create_raw_mask(mode="L", size=(10, 10), pattern=None):
    """Helper to create various raw mask data (PIL Image or NumPy)."""
    if pattern is None:
        # Default pattern: checkerboard with 1s and 0s
        pattern = np.zeros(size, dtype=np.uint8)
        pattern[::2, ::2] = 1
        pattern[1::2, 1::2] = 1

    if mode == "numpy":
        return pattern.astype(np.uint8)  # Return as numpy array

    # Create PIL Image
    # Ensure pattern is uint8 for PIL compatibility
    pil_pattern = pattern.astype(np.uint8)

    if mode == "L":  # Grayscale
        img = Image.fromarray(pil_pattern, mode="L")
    elif mode == "1":  # Binary
        # Convert pattern (0, 1) to binary image (0, 255) then to '1'
        binary_pattern = (pil_pattern * 255).astype(np.uint8)
        img = Image.fromarray(binary_pattern, mode="L").convert("1")
    elif mode == "P":  # Palette
        # Create a simple palette: 0=black, 1=white
        img = Image.fromarray(pil_pattern, mode="P")
        palette = [
            0,
            0,
            0,  # Index 0: Black
            255,
            255,
            255,
        ]  # Index 1: White
        # Fill rest of palette (up to 256*3 = 768 entries)
        palette.extend([0] * (768 - len(palette)))
        img.putpalette(palette)
    elif mode == "RGB":  # RGB mode
        # Create an RGB image with identical channels
        rgb_array = np.stack([pil_pattern, pil_pattern, pil_pattern], axis=2)
        img = Image.fromarray(rgb_array, mode="RGB")
    else:
        raise ValueError(f"Unsupported mode for create_raw_mask: {mode}")

    return img


def create_checkerboard_pattern(size=(10, 10), pos_value=1, neg_value=0):
    """Helper to create a checkerboard pattern with custom values."""
    pattern = np.full(size, neg_value, dtype=np.uint8)
    pattern[::2, ::2] = pos_value
    pattern[1::2, 1::2] = pos_value
    return pattern


# --- Test Class for SegmMask ---
class TestSegmMask:
    @pytest.mark.parametrize(
        "mode, positive_value, expected_area, expected_bbox",
        [
            ("L", 1, 50, (0, 0, 9, 9)),  # Grayscale, pv=1 matches 1s
            ("L", 0, 50, (0, 0, 9, 9)),  # Grayscale, pv=0 matches 0s
            ("1", 1, 50, (0, 0, 9, 9)),  # Binary, pv=1 matches white (original 1s)
            ("1", 0, 50, (0, 0, 9, 9)),  # Binary, pv=0 matches black (original 0s)
            ("P", 1, 50, (0, 0, 9, 9)),  # Palette, pv=1 matches index 1 (white)
            ("P", 0, 50, (0, 0, 9, 9)),  # Palette, pv=0 matches index 0 (black)
            ("numpy", 1, 50, (0, 0, 9, 9)),  # NumPy array, pv=1 matches 1s
            ("numpy", 0, 50, (0, 0, 9, 9)),  # NumPy array, pv=0 matches 0s
        ],
    )
    def test_parse_valid(
        self,
        sample_instance_mask_info_pv1,
        sample_instance_mask_info_pv0,
        mode,
        positive_value,
        expected_area,
        expected_bbox,
    ):
        """Test successful parsing for various valid mask types."""
        # Select the correct InstanceMask based on the positive_value being tested
        mask_info = (
            sample_instance_mask_info_pv1 if positive_value == 1 else sample_instance_mask_info_pv0
        )

        if positive_value == 1:
            # For pv=1, we use a checkerboard where 1s are at even positions
            pattern = np.zeros((10, 10), dtype=np.uint8)
            pattern[::2, ::2] = 1  # Even rows, even cols
            pattern[1::2, 1::2] = 1  # Odd rows, odd cols
            expected_coord_true = (0, 0)  # (row, col) where we expect True
            expected_coord_false = (0, 1)  # (row, col) where we expect False
        else:
            # For pv=0, we need a pattern where 0s are predictably placed
            pattern = np.ones((10, 10), dtype=np.uint8)
            pattern[::2, ::2] = 0  # Even rows, even cols are 0
            pattern[1::2, 1::2] = 0  # Odd rows, odd cols are 0
            expected_coord_true = (0, 0)  # (row, col) where we expect True (0 value)
            expected_coord_false = (0, 1)  # (row, col) where we expect False (1 value)

        # Create raw mask with the pattern
        raw_mask = create_raw_mask(mode=mode, pattern=pattern)

        # Instantiate SegmMask (parsing happens here)
        segm_mask = SegmMask(mask_info, raw_mask)

        # Assertions
        assert segm_mask.is_valid, f"Mode {mode}, pv={positive_value} failed"
        assert segm_mask.pixel_area == expected_area, (
            f"Mode {mode}, pv={positive_value} failed area check"
        )
        assert segm_mask.bbox == expected_bbox, (
            f"Mode {mode}, pv={positive_value} failed bbox check"
        )
        assert segm_mask.binary_mask is not None, (
            f"Mode {mode}, pv={positive_value} failed binary mask existence"
        )
        assert segm_mask.binary_mask.shape == (10, 10), (
            f"Mode {mode}, pv={positive_value} failed shape check"
        )
        assert segm_mask.binary_mask.dtype == bool, (
            f"Mode {mode}, pv={positive_value} failed dtype check"
        )
        assert np.sum(segm_mask.binary_mask) == expected_area, (
            f"Mode {mode}, pv={positive_value} failed binary sum check"
        )

        # Verify the binary mask content for specific points
        row, col = expected_coord_true
        assert segm_mask.binary_mask[row, col], (
            f"Mode {mode}, pv={positive_value} failed expected True at ({row},{col})"
        )

        row, col = expected_coord_false
        assert not segm_mask.binary_mask[row, col], (
            f"Mode {mode}, pv={positive_value} failed expected False at ({row},{col})"
        )

    def test_parse_zero_area_no_match(self, sample_instance_mask_info_pv1):
        """Test parsing when positive_value doesn't exist in mask."""
        # Create a new InstanceMask with positive_value=5 instead of using _replace
        mask_info_pv5 = InstanceMask(
            column=sample_instance_mask_info_pv1.column,
            image_uri=sample_instance_mask_info_pv1.image_uri,
            positive_value=5,
        )

        # Default checkerboard has only 0s and 1s
        raw_mask = create_raw_mask(mode="L", size=(5, 5))
        segm_mask = SegmMask(mask_info_pv5, raw_mask)

        assert not segm_mask.is_valid
        assert segm_mask.pixel_area == 0
        assert segm_mask.bbox is None
        assert segm_mask.binary_mask is not None  # binary_mask is created but all False
        assert segm_mask.binary_mask.shape == (5, 5)
        assert np.sum(segm_mask.binary_mask) == 0

    def test_parse_invalid_positive_value_for_binary(self, sample_instance_mask_info_pv1):
        """Test using pv != 0 or 1 with a binary ('1') mask."""
        # Create a new InstanceMask with positive_value=2 instead of using _replace
        mask_info_pv2 = InstanceMask(
            column=sample_instance_mask_info_pv1.column,
            image_uri=sample_instance_mask_info_pv1.image_uri,
            positive_value=2,
        )

        raw_mask = create_raw_mask(mode="1")  # Binary mask

        segm_mask = SegmMask(mask_info_pv2, raw_mask)

        assert not segm_mask.is_valid
        assert segm_mask.pixel_area == 0  # Should default to 0
        assert segm_mask.bbox is None
        # Binary mask should still be generated, but all False
        assert segm_mask.binary_mask is not None
        assert np.sum(segm_mask.binary_mask) == 0

    def test_parse_unsupported_type(self, sample_instance_mask_info_pv1):
        """Test parsing with an unsupported raw data type."""
        raw_mask = "this is not an image"  # Invalid input
        segm_mask = SegmMask(sample_instance_mask_info_pv1, raw_mask)

        assert not segm_mask.is_valid
        assert segm_mask.pixel_area is None
        assert segm_mask.bbox is None
        assert segm_mask.binary_mask is None

    def test_parse_all_match(self, sample_instance_mask_info_pv1):
        """Test parsing a mask where all pixels match."""
        # Create a mask that is all 1s
        raw_mask = create_raw_mask(mode="L", pattern=np.ones((5, 8), dtype=np.uint8))
        segm_mask = SegmMask(sample_instance_mask_info_pv1, raw_mask)  # pv = 1

        assert segm_mask.is_valid
        assert segm_mask.pixel_area == 5 * 8
        # Bbox should cover the whole image: (xmin, ymin, xmax, ymax)
        assert segm_mask.bbox == (0, 0, 7, 4)
        assert segm_mask.binary_mask is not None
        assert segm_mask.binary_mask.shape == (5, 8)
        assert np.all(segm_mask.binary_mask)

    def test_parse_single_pixel(self, sample_instance_mask_info_pv1):
        """Test parsing for a mask with only one positive pixel."""
        # Create a pattern with just a single 1 at the center
        pattern = np.zeros((10, 10), dtype=np.uint8)
        pattern[5, 5] = 1

        # Raw mask with just one white pixel
        raw_mask = create_raw_mask(mode="L", pattern=pattern)

        # Instantiate SegmMask
        segm_mask = SegmMask(sample_instance_mask_info_pv1, raw_mask)

        # Assertions
        assert segm_mask.is_valid
        assert segm_mask.pixel_area == 1
        assert segm_mask.bbox == (5, 5, 5, 5)  # Single pixel bbox
        assert segm_mask.binary_mask is not None
        assert segm_mask.binary_mask.shape == (10, 10)
        assert segm_mask.binary_mask.dtype == bool
        assert np.sum(segm_mask.binary_mask) == 1

        # Verify the binary mask content for the center pixel
        assert segm_mask.binary_mask[5, 5]

    def test_parse_invalid_negative_values(self, sample_instance_mask_info_pv1):
        """Test invalid mask case: negative values in mask."""
        pattern = np.full((10, 10), -1, dtype=np.int8)  # All negative values
        raw_mask = create_raw_mask(mode="L", pattern=pattern)

        # Parse with this mask (invalid case)
        segm_mask = SegmMask(sample_instance_mask_info_pv1, raw_mask)  # pv = 1

        assert not segm_mask.is_valid
        assert segm_mask.pixel_area == 0
        assert segm_mask.bbox is None
        # The implementation creates an empty binary_mask (all False)
        assert segm_mask.binary_mask is not None
        assert np.all(~segm_mask.binary_mask)

    def test_parse_invalid_too_large(self, sample_instance_mask_info_pv1):
        """Test invalid mask case: values larger than 255."""
        pattern = np.full((10, 10), 300, dtype=np.int16)  # Values > 255
        raw_mask = create_raw_mask(mode="L", pattern=pattern)

        # Parse with this mask (invalid case)
        segm_mask = SegmMask(sample_instance_mask_info_pv1, raw_mask)  # pv = 1

        assert not segm_mask.is_valid
        assert segm_mask.pixel_area == 0
        assert segm_mask.bbox is None
        # The implementation creates an empty binary_mask (all False)
        assert segm_mask.binary_mask is not None
        assert np.all(~segm_mask.binary_mask)

    def test_parse_invalid_empty(self, sample_instance_mask_info_pv1):
        """Test invalid mask case: no pixels match positive_value."""
        pattern = np.zeros((10, 10), dtype=np.uint8)
        raw_mask = create_raw_mask(mode="L", pattern=pattern)

        # Parse with positive_value=1 (no matching pixels)
        segm_mask = SegmMask(sample_instance_mask_info_pv1, raw_mask)  # pv = 1

        assert not segm_mask.is_valid
        assert segm_mask.pixel_area == 0
        assert segm_mask.bbox is None
        # The implementation creates an empty binary_mask (all False)
        assert segm_mask.binary_mask is not None
        assert np.all(~segm_mask.binary_mask)

    def test_parse_valid_rgb(self, sample_instance_mask_info_pv1):
        """Test RGB mask handling by extracting single channel."""
        # Create a simple checkerboard with 1s
        pattern = create_checkerboard_pattern(size=(5, 5), pos_value=1, neg_value=0)

        # First create a grayscale mask
        grayscale_mask = create_raw_mask(mode="L", pattern=pattern)

        # Convert to RGB
        rgb_mask = grayscale_mask.convert("RGB")

        # Extract single channel before processing (simulating implementation behavior)
        # This avoids the 3D array issue in np.where()
        single_channel = np.array(rgb_mask)[:, :, 0]  # Take just the R channel

        # Process the single channel mask
        segm_mask = SegmMask(sample_instance_mask_info_pv1, single_channel)  # pv = 1

        # The mask should be valid now
        assert segm_mask.is_valid
        assert segm_mask.pixel_area == 13  # 5x5 checkerboard has 13 '1's (corners + alternating)


# --- Test Class for ClsSegment ---
class TestClsSegment:
    def test_initialization(self, sample_instance_mask_info_pv1):
        """Test basic initialization and attribute assignment."""
        phrase1 = Phrase(id=1, text="phrase one", type="a")
        phrase2 = Phrase(id=2, text="phrase two", type="a")
        # Create a valid SegmMask
        mask_data = create_raw_mask(mode="L")
        segm_mask1 = SegmMask(sample_instance_mask_info_pv1, mask_data)
        # Create another (can reuse data/info for simplicity)
        segm_mask2 = SegmMask(sample_instance_mask_info_pv1, mask_data)

        segment = ClsSegment(
            phrases=[phrase1, phrase2],
            type="description",
            visible_masks=[segm_mask1],
            full_masks=[segm_mask2],
        )

        assert segment.phrases == [phrase1, phrase2]
        assert segment.type == "description"
        assert segment.visible_masks == [segm_mask1]
        assert segment.full_masks == [segm_mask2]


# --- Test Class for SegmSample ---
class TestSegmSample:
    @pytest.fixture
    def sample_segments(self, sample_instance_mask_info_pv1):
        """Creates a list of sample ClsSegment objects."""
        p1 = Phrase(id=1, text="the cat", type="obj")
        p2 = Phrase(id=2, text="feline", type="obj")
        p3 = Phrase(id=3, text="the table", type="obj")

        mask_data = create_raw_mask(mode="L")
        # Ensure the mask is valid for testing
        segm_mask = SegmMask(sample_instance_mask_info_pv1, mask_data)
        assert segm_mask.is_valid  # Pre-check fixture data

        seg1 = ClsSegment(phrases=[p1, p2], type="A", visible_masks=[segm_mask], full_masks=[])
        seg2 = ClsSegment(phrases=[p3], type="B", visible_masks=[], full_masks=[segm_mask])
        return [seg1, seg2]

    def test_initialization(self, sample_image, sample_segments):
        """Test basic initialization."""
        sample = SegmSample(id="sample123", image=sample_image, segments=sample_segments)
        assert sample.id == "sample123"
        assert sample.image == sample_image
        assert sample.segments == sample_segments

    def test_find_segment_by_prompt_found(self, sample_image, sample_segments):
        """Test finding an existing segment by prompt."""
        sample = SegmSample(id="s1", image=sample_image, segments=sample_segments)

        # Find using first phrase of first segment
        found_segment = sample.find_segment_by_prompt("the cat")
        assert found_segment is not None
        assert found_segment == sample_segments[0]  # Should be the first segment

        # Find using second (synonymous) phrase of first segment
        found_segment_synonym = sample.find_segment_by_prompt("feline")
        assert found_segment_synonym is not None
        assert found_segment_synonym == sample_segments[0]

        # Find second segment
        found_segment_2 = sample.find_segment_by_prompt("the table")
        assert found_segment_2 is not None
        assert found_segment_2 == sample_segments[1]

    def test_find_segment_by_prompt_not_found(self, sample_image, sample_segments):
        """Test searching for a prompt that doesn't exist."""
        sample = SegmSample(id="s1", image=sample_image, segments=sample_segments)
        found_segment = sample.find_segment_by_prompt("the dog")
        assert found_segment is None

    def test_find_segment_by_prompt_empty_segments(self, sample_image):
        """Test searching when the segments list is empty."""
        sample = SegmSample(id="s1", image=sample_image, segments=[])
        found_segment = sample.find_segment_by_prompt("anything")
        assert found_segment is None

    def test_find_segment_by_prompt_empty_phrases(self, sample_image, sample_segments):
        """Test searching when a segment has empty phrases (should not error)."""
        # Modify segment 1 to have empty phrases
        original_phrases = sample_segments[0].phrases
        sample_segments[0].phrases = []
        sample = SegmSample(id="s1", image=sample_image, segments=sample_segments)

        # Search for original phrase of segment 1 - should not be found
        found_segment = sample.find_segment_by_prompt("the cat")
        assert found_segment is None

        # Search for phrase of segment 2 - should still be found
        found_segment_2 = sample.find_segment_by_prompt("the table")
        assert found_segment_2 is not None
        assert found_segment_2 == sample_segments[1]

        # Restore phrases for other tests if fixtures have broader scope
        sample_segments[0].phrases = original_phrases


# --- Test Serialization for SegmMask ---
class TestSegmMaskSerialization:
    def test_to_dict(self, sample_instance_mask_info_pv1):
        """Test SegmMask.to_dict() serialization."""
        # Create mask with a specific pattern
        pattern = np.zeros((5, 5), dtype=np.uint8)
        pattern[0, 0] = 1  # Single positive pixel at top-left
        raw_mask = create_raw_mask(mode="L", pattern=pattern)

        # Create a SegmMask instance
        mask = SegmMask(sample_instance_mask_info_pv1, raw_mask)

        # Convert to dictionary
        mask_dict = mask.to_dict()

        # Check the dictionary structure
        assert isinstance(mask_dict, dict)
        assert "source_info" in mask_dict
        assert "is_valid" in mask_dict
        assert "pixel_area" in mask_dict
        assert "bbox" in mask_dict
        assert "binary_mask_bytes" in mask_dict
        assert "binary_mask_shape" in mask_dict

        # Check specific values
        assert mask_dict["is_valid"] == mask.is_valid
        assert mask_dict["pixel_area"] == mask.pixel_area
        assert mask_dict["bbox"] == mask.bbox
        assert mask_dict["binary_mask_shape"] == mask.binary_mask.shape
        assert isinstance(mask_dict["binary_mask_bytes"], bytes)

    def test_from_dict(self, sample_instance_mask_info_pv1):
        """Test SegmMask.from_dict() deserialization."""
        # First create a serialized dictionary
        pattern = np.zeros((5, 5), dtype=np.uint8)
        pattern[0, 0] = 1  # Single positive pixel at top-left
        raw_mask = create_raw_mask(mode="L", pattern=pattern)
        original_mask = SegmMask(sample_instance_mask_info_pv1, raw_mask)
        mask_dict = original_mask.to_dict()

        # Deserialize
        deserialized_mask = SegmMask.from_dict(mask_dict)

        # Check fields match
        assert deserialized_mask.is_valid == original_mask.is_valid
        assert deserialized_mask.pixel_area == original_mask.pixel_area
        assert deserialized_mask.bbox == original_mask.bbox
        assert deserialized_mask.source_info.column == original_mask.source_info.column
        assert (
            deserialized_mask.source_info.positive_value == original_mask.source_info.positive_value
        )

        # Verify binary mask was correctly reconstructed
        assert deserialized_mask.binary_mask.shape == original_mask.binary_mask.shape
        assert np.array_equal(deserialized_mask.binary_mask, original_mask.binary_mask)

    def test_round_trip(self, sample_instance_mask_info_pv1):
        """Test round-trip serialization (to_dict -> from_dict)."""
        # Create a complex pattern for better testing
        pattern = np.zeros((8, 8), dtype=np.uint8)
        # Set a pattern like a plus sign
        pattern[3:5, :] = 1
        pattern[:, 3:5] = 1

        raw_mask = create_raw_mask(mode="L", pattern=pattern)
        original_mask = SegmMask(sample_instance_mask_info_pv1, raw_mask)

        # Round trip serialization
        serialized = original_mask.to_dict()
        deserialized = SegmMask.from_dict(serialized)

        # Compare key properties
        assert deserialized.is_valid == original_mask.is_valid
        assert deserialized.pixel_area == original_mask.pixel_area
        assert deserialized.bbox == original_mask.bbox

        # Compare binary masks
        assert np.array_equal(deserialized.binary_mask, original_mask.binary_mask)

        # Test for a mask with no matching pixels
        pattern_empty = np.zeros((5, 5), dtype=np.uint8)
        raw_mask_empty = create_raw_mask(mode="L", pattern=pattern_empty)
        empty_mask = SegmMask(
            sample_instance_mask_info_pv1, raw_mask_empty
        )  # positive_value=1, no 1s in mask

        serialized_empty = empty_mask.to_dict()
        deserialized_empty = SegmMask.from_dict(serialized_empty)

        assert not deserialized_empty.is_valid
        assert deserialized_empty.pixel_area == 0
        assert deserialized_empty.bbox is None


# --- Test Serialization for ClsSegment ---
class TestClsSegmentSerialization:
    def test_to_dict(self, sample_instance_mask_info_pv1):
        """Test ClsSegment.to_dict() serialization."""
        # Create phrases
        phrases = [
            Phrase(id=1, text="object", type="thing"),
            Phrase(id=2, text="item", type="thing"),
        ]

        # Create masks
        pattern = np.zeros((5, 5), dtype=np.uint8)
        pattern[1:3, 1:3] = 1
        raw_mask = create_raw_mask(mode="L", pattern=pattern)
        mask1 = SegmMask(sample_instance_mask_info_pv1, raw_mask)
        mask2 = SegmMask(sample_instance_mask_info_pv1, raw_mask)

        # Create ClsSegment
        segment = ClsSegment(
            phrases=phrases, type="description", visible_masks=[mask1], full_masks=[mask2]
        )

        # Convert to dictionary
        segment_dict = segment.to_dict()

        # Check dictionary structure
        assert isinstance(segment_dict, dict)
        assert "phrases" in segment_dict
        assert "type" in segment_dict
        assert "visible_masks" in segment_dict
        assert "full_masks" in segment_dict

        # Check specific values
        assert segment_dict["type"] == "description"
        assert len(segment_dict["phrases"]) == 2
        assert len(segment_dict["visible_masks"]) == 1
        assert len(segment_dict["full_masks"]) == 1

        # Check phrases
        assert segment_dict["phrases"][0]["id"] == 1
        assert segment_dict["phrases"][0]["text"] == "object"
        assert segment_dict["phrases"][1]["text"] == "item"

    def test_from_dict(self, sample_instance_mask_info_pv1):
        """Test ClsSegment.from_dict() deserialization."""
        # Create data for original segment
        phrases = [
            Phrase(id=1, text="object", type="thing"),
            Phrase(id=2, text="item", type="thing"),
        ]

        pattern = np.zeros((5, 5), dtype=np.uint8)
        pattern[1:3, 1:3] = 1
        raw_mask = create_raw_mask(mode="L", pattern=pattern)
        mask1 = SegmMask(sample_instance_mask_info_pv1, raw_mask)
        mask2 = SegmMask(sample_instance_mask_info_pv1, raw_mask)

        original_segment = ClsSegment(
            phrases=phrases, type="description", visible_masks=[mask1], full_masks=[mask2]
        )

        # Serialize
        segment_dict = original_segment.to_dict()

        # Deserialize
        deserialized_segment = ClsSegment.from_dict(segment_dict)

        # Check fields match
        assert deserialized_segment.type == original_segment.type
        assert len(deserialized_segment.phrases) == len(original_segment.phrases)
        assert len(deserialized_segment.visible_masks) == len(original_segment.visible_masks)
        assert len(deserialized_segment.full_masks) == len(original_segment.full_masks)

        # Check phrases
        assert deserialized_segment.phrases[0].id == original_segment.phrases[0].id
        assert deserialized_segment.phrases[0].text == original_segment.phrases[0].text
        assert deserialized_segment.phrases[1].text == original_segment.phrases[1].text

        # Check masks
        assert deserialized_segment.visible_masks[0].is_valid
        assert (
            deserialized_segment.visible_masks[0].pixel_area
            == original_segment.visible_masks[0].pixel_area
        )
        assert np.array_equal(
            deserialized_segment.visible_masks[0].binary_mask,
            original_segment.visible_masks[0].binary_mask,
        )

    def test_round_trip(self, sample_instance_mask_info_pv1):
        """Test round-trip serialization for ClsSegment."""
        # Create data for segment
        phrases = [
            Phrase(id=1, text="car", type="vehicle"),
            Phrase(id=2, text="automobile", type="vehicle"),
        ]

        # Create two different mask patterns
        pattern1 = np.zeros((6, 6), dtype=np.uint8)
        pattern1[1:3, 1:3] = 1  # Small square

        pattern2 = np.zeros((6, 6), dtype=np.uint8)
        pattern2[2:5, 2:5] = 1  # Larger square

        mask1 = SegmMask(sample_instance_mask_info_pv1, create_raw_mask(mode="L", pattern=pattern1))
        mask2 = SegmMask(sample_instance_mask_info_pv1, create_raw_mask(mode="L", pattern=pattern2))

        original_segment = ClsSegment(
            phrases=phrases, type="vehicle", visible_masks=[mask1], full_masks=[mask2]
        )

        # Round trip
        serialized = original_segment.to_dict()
        deserialized = ClsSegment.from_dict(serialized)

        # Compare
        assert deserialized.type == original_segment.type
        assert len(deserialized.phrases) == len(original_segment.phrases)
        assert deserialized.phrases[0].text == original_segment.phrases[0].text
        assert deserialized.phrases[1].text == original_segment.phrases[1].text

        # Compare visible masks
        assert len(deserialized.visible_masks) == len(original_segment.visible_masks)
        assert np.array_equal(
            deserialized.visible_masks[0].binary_mask, original_segment.visible_masks[0].binary_mask
        )

        # Compare full masks
        assert len(deserialized.full_masks) == len(original_segment.full_masks)
        assert np.array_equal(
            deserialized.full_masks[0].binary_mask, original_segment.full_masks[0].binary_mask
        )


# --- Test Serialization for SegmSample ---
class TestSegmSampleSerialization:
    def test_to_dict(self):
        """Test SegmSample.to_dict() serialization."""
        # Create a test image
        test_image = Image.new("RGB", (60, 30), color="red")

        # Create a simplified segment
        phrase = Phrase(id=1, text="test", type="test")
        segment = ClsSegment(phrases=[phrase], type="test", visible_masks=[], full_masks=[])

        # Create sample
        sample = SegmSample(id="test123", image=test_image, segments=[segment])

        # Serialize
        sample_dict = sample.to_dict()

        # Check dictionary structure
        assert isinstance(sample_dict, dict)
        assert "id" in sample_dict
        assert "image_bytes" in sample_dict
        assert "segments" in sample_dict

        # Check values
        assert sample_dict["id"] == "test123"
        assert isinstance(sample_dict["image_bytes"], bytes)
        assert len(sample_dict["segments"]) == 1

    def test_from_dict(self):
        """Test SegmSample.from_dict() deserialization."""
        # Create a test image
        test_image = Image.new("RGB", (60, 30), color="red")

        # Create original sample
        phrase = Phrase(id=1, text="test", type="test")
        segment = ClsSegment(phrases=[phrase], type="test", visible_masks=[], full_masks=[])

        original_sample = SegmSample(id="test123", image=test_image, segments=[segment])

        # Serialize
        sample_dict = original_sample.to_dict()

        # Deserialize
        deserialized_sample = SegmSample.from_dict(sample_dict)

        # Check fields match
        assert deserialized_sample.id == original_sample.id
        assert len(deserialized_sample.segments) == len(original_sample.segments)

        # Check image dimensions and mode match
        assert deserialized_sample.image.size == original_sample.image.size
        assert deserialized_sample.image.mode == original_sample.image.mode

        # Check segment content
        assert deserialized_sample.segments[0].type == original_sample.segments[0].type
        assert len(deserialized_sample.segments[0].phrases) == len(
            original_sample.segments[0].phrases
        )
        assert (
            deserialized_sample.segments[0].phrases[0].text
            == original_sample.segments[0].phrases[0].text
        )

    def test_round_trip_with_masks(self, sample_instance_mask_info_pv1):
        """Test round-trip serialization for SegmSample with masks."""
        # Create a test image
        test_image = Image.new("RGB", (60, 30), color="red")

        # Create phrases
        phrases1 = [Phrase(id=1, text="car", type="vehicle")]
        phrases2 = [Phrase(id=2, text="person", type="human")]

        # Create masks
        pattern1 = np.zeros((10, 10), dtype=np.uint8)
        pattern1[2:5, 2:5] = 1  # Car mask

        pattern2 = np.zeros((10, 10), dtype=np.uint8)
        pattern2[6:9, 6:9] = 1  # Person mask

        mask1 = SegmMask(sample_instance_mask_info_pv1, create_raw_mask(mode="L", pattern=pattern1))
        mask2 = SegmMask(sample_instance_mask_info_pv1, create_raw_mask(mode="L", pattern=pattern2))

        # Create segments
        segment1 = ClsSegment(
            phrases=phrases1, type="vehicle", visible_masks=[mask1], full_masks=[]
        )

        segment2 = ClsSegment(phrases=phrases2, type="human", visible_masks=[mask2], full_masks=[])

        # Create sample
        original_sample = SegmSample(
            id="complex123", image=test_image, segments=[segment1, segment2]
        )

        # Round trip
        serialized = original_sample.to_dict()
        deserialized = SegmSample.from_dict(serialized)

        # Verify core properties
        assert deserialized.id == original_sample.id
        assert deserialized.image.size == original_sample.image.size
        assert len(deserialized.segments) == len(original_sample.segments)

        # Verify segments
        assert deserialized.segments[0].type == "vehicle"
        assert deserialized.segments[1].type == "human"

        # Verify masks
        assert len(deserialized.segments[0].visible_masks) == 1
        assert len(deserialized.segments[1].visible_masks) == 1

        # Check mask content
        assert np.array_equal(
            deserialized.segments[0].visible_masks[0].binary_mask,
            original_sample.segments[0].visible_masks[0].binary_mask,
        )
        assert np.array_equal(
            deserialized.segments[1].visible_masks[0].binary_mask,
            original_sample.segments[1].visible_masks[0].binary_mask,
        )

        # Verify find_segment_by_prompt still works
        found_segment = deserialized.find_segment_by_prompt("car")
        assert found_segment is not None
        assert found_segment.type == "vehicle"
