import logging
from unittest.mock import ANY, MagicMock, call, patch

import numpy as np
import pytest
from PIL import Image

# Import datamodels directly from datamodel.py
from src.dataops.cov_segm.datamodel import (
    ProcessedConversationItem,
    ProcessedCovSegmSample,
    ProcessedMask,
)

# Import the functions to test
from src.dataops.cov_segm.visualizer import (
    _apply_color_mask,
    _find_item_by_prompt,
    visualize_prompt_masks,
)

# Configure logging for tests if needed
logger = logging.getLogger("src.dataops.cov_segm.visualizer")
logger.setLevel(logging.DEBUG)

# --- Helper Functions / Fixtures ---


def create_mock_processed_mask(
    mode="L", size=(10, 10), value=1, source="test_mask", area=10, w=5, h=2
) -> ProcessedMask:
    """Creates a mock ProcessedMask dict with a dummy PIL image."""
    # Create a simple mask image (e.g., a single pixel set)
    img = Image.new(mode, size, 0)  # Black background
    if area > 0 and size[0] > 0 and size[1] > 0:
        # Simple way to set some pixels - might not match area exactly but gives non-zero mask
        img.putpixel((0, 0), value if mode != "1" else 1)
        if area > 1 and size[0] > 1 and size[1] > 1:
            img.putpixel((1, 1), value if mode != "1" else 1)

    # Handle palette mode specifically if needed
    if mode == "P":
        # Create a minimal palette (e.g., black, target value color)
        palette = [0, 0, 0] * 256  # Initialize with black
        if 0 <= value < 256:
            palette[value * 3 : (value + 1) * 3] = [
                255,
                0,
                0,
            ]  # Make target value red for visibility
        img.putpalette(palette)

    return {
        "mask": img,
        "positive_value": value,
        "source": source,
        "pixel_area": area,
        "width": w,
        "height": h,
    }


@pytest.fixture
def mock_processed_sample() -> ProcessedCovSegmSample:
    """Creates a mock ProcessedCovSegmSample."""
    mask_vis_1 = create_mock_processed_mask(value=1, source="vis_mask_1", area=5, w=3, h=2)
    mask_vis_2 = create_mock_processed_mask(value=2, source="vis_mask_2", area=8, w=4, h=2)
    mask_full_1 = create_mock_processed_mask(value=1, source="full_mask_1", area=20, w=5, h=4)

    item1: ProcessedConversationItem = {
        "phrases": [{"id": 1, "text": "cat", "type": "obj"}],
        "type": "SEG",
        "processed_instance_masks": [mask_vis_1, mask_vis_2],
        "processed_full_masks": [mask_full_1],
    }
    item2: ProcessedConversationItem = {
        "phrases": [{"id": 2, "text": "dog", "type": "obj"}],
        "type": "SEG",
        "processed_instance_masks": [],
        "processed_full_masks": [],
    }
    sample: ProcessedCovSegmSample = {
        "id": "sample1",
        "image": Image.new("RGB", (50, 50)),  # Dummy main image
        "processed_conversations": [item1, item2],
    }
    return sample


# --- Test _find_item_by_prompt ---


def test_find_item_by_prompt_found(mock_processed_sample):
    found_item = _find_item_by_prompt(mock_processed_sample, "cat")
    assert found_item is not None
    assert found_item["phrases"][0]["text"] == "cat"


def test_find_item_by_prompt_not_found(mock_processed_sample):
    found_item = _find_item_by_prompt(mock_processed_sample, "bird")
    assert found_item is None


# --- Test _apply_color_mask ---


@pytest.mark.parametrize(
    "mask_mode, mask_value, positive_value, should_match",
    [
        ("L", 1, 1, True),  # Grayscale direct match
        ("L", 255, 1, True),  # Grayscale scaled match (non-zero)
        ("L", 1, 2, False),  # Grayscale no match
        ("L", 0, 1, False),  # Grayscale zero value, positive=1
        ("L", 0, 0, True),  # Grayscale zero value, positive=0
        ("P", 5, 5, True),  # Palette direct match
        ("P", 5, 6, False),  # Palette no match
        ("1", 1, 1, True),  # Binary direct match (PIL '1' mode pixels are 0 or 1)
        ("1", 0, 1, False),  # Binary no match
        ("L", 10, 10, True),  # Ensure exact value match works
    ],
)
@patch("matplotlib.pyplot.Axes")  # Mock the Axes object directly
def test_apply_color_mask_matching(
    mock_axes_class, mask_mode, mask_value, positive_value, should_match
):
    """Tests mask matching logic for different modes and values."""
    mock_ax = mock_axes_class.return_value  # Get the instance
    # Create a 2x2 mask with the top-left pixel having the target value
    mask_array = np.zeros((2, 2), dtype=np.uint8)
    mask_array[0, 0] = mask_value
    mask_img = Image.fromarray(mask_array, mode=mask_mode)

    # For binary mode '1', we need special handling since it only accepts 0 or 1
    if mask_mode == "1":
        mask_img = Image.new("1", (2, 2), 0)  # Create a black binary image
        if mask_value > 0:
            mask_img.putpixel((0, 0), 1)  # Set top-left pixel to white (1)

    if mask_mode == "P":
        palette = [0, 0, 0] * 256  # Black palette
        if 0 <= mask_value < 256:
            palette[mask_value * 3 : (mask_value + 1) * 3] = [255, 0, 0]  # Make target red
        mask_img.putpalette(palette)

    color = (1, 0, 0)  # Red
    alpha = 0.6

    _apply_color_mask(
        mock_ax, mask_img, color, positive_value=positive_value, alpha=alpha, debug=True
    )

    assert mock_ax.imshow.call_count == 1
    # Check the applied mask
    call_args, call_kwargs = mock_ax.imshow.call_args  # Get both args and kwargs
    applied_mask = call_args[0]
    assert applied_mask.shape == (2, 2, 4)

    # Check pixel values based on whether they should match
    # Top-left pixel (0,0) has mask_value
    # Bottom-right pixel (1,1) has value 0 (unless mask_value is also 0)
    top_left_matches = (
        (mask_value == positive_value)
        or (mask_mode == "1" and mask_value != 0 and positive_value == 1)
        or (mask_mode == "1" and mask_value == 0 and positive_value == 0)
        or (mask_mode == "L" and mask_value != 0 and positive_value > 0 and mask_value > 1)
    )  # Scaled match case

    bottom_right_matches = 0 == positive_value  # Only matches if positive_value is 0
    if mask_mode == "1" and positive_value == 0:
        bottom_right_matches = True  # In mode '1', pixel value 0 should match positive_value 0

    if top_left_matches:
        assert np.allclose(applied_mask[0, 0, :3], color)
        assert np.isclose(applied_mask[0, 0, 3], alpha)
    else:
        assert np.isclose(applied_mask[0, 0, 3], 0.0)

    if bottom_right_matches:
        assert np.allclose(applied_mask[1, 1, :3], color)
        assert np.isclose(applied_mask[1, 1, 3], alpha)
    else:
        assert np.isclose(applied_mask[1, 1, 3], 0.0)

    # Overall check based on should_match parameter (as a sanity check)
    if should_match:
        # At least one pixel should have alpha > 0
        assert np.any(applied_mask[:, :, 3] > 0)
    else:
        # All pixels should have alpha == 0
        assert np.all(applied_mask[:, :, 3] == 0)


@patch("src.dataops.cov_segm.visualizer._find_item_by_prompt")
@patch("matplotlib.pyplot.subplots")
@patch("src.dataops.cov_segm.visualizer._apply_color_mask")
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.close")
def test_visualize_prompt_masks_visible_show(
    mock_close,
    mock_show,
    mock_savefig,
    mock_apply_mask,
    mock_subplots,
    mock_find_item,
    mock_processed_sample,
):
    """Test visualizing visible masks with interactive display."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)
    # Find the item with masks
    target_item = _find_item_by_prompt(mock_processed_sample, "cat")
    mock_find_item.return_value = target_item

    visualize_prompt_masks(
        mock_processed_sample, "cat", mask_type="visible", show=True, output_path=None
    )

    mock_find_item.assert_called_once_with(mock_processed_sample, "cat")
    mock_subplots.assert_called_once()
    mock_ax.imshow.assert_called_once_with(mock_processed_sample["image"])
    assert mock_apply_mask.call_count == len(target_item["processed_instance_masks"])
    # Check calls to apply_mask
    expected_calls = [
        call(
            mock_ax,
            m["mask"],
            ANY,  # color tuple - hard to predict exactly due to colormap
            positive_value=m["positive_value"],
            alpha=ANY,  # default alpha
            debug=ANY,  # default debug
        )
        for m in target_item["processed_instance_masks"]
    ]
    # Need to import ANY from unittest.mock if using directly
    # For simplicity, just check the count and maybe first call args more loosely
    assert mock_apply_mask.call_count == 2

    # Check keyword arguments for positive_value in the first call
    first_call_args, first_call_kwargs = mock_apply_mask.call_args_list[0]
    assert first_call_args[0] == mock_ax
    assert first_call_args[1] == target_item["processed_instance_masks"][0]["mask"]
    assert (
        first_call_kwargs["positive_value"]
        == target_item["processed_instance_masks"][0]["positive_value"]
    )

    mock_show.assert_called_once()
    mock_savefig.assert_not_called()
    mock_close.assert_called_once_with(mock_fig)


@patch("src.dataops.cov_segm.visualizer._find_item_by_prompt")
@patch("matplotlib.pyplot.subplots")
@patch("src.dataops.cov_segm.visualizer._apply_color_mask")
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.close")
def test_visualize_prompt_masks_full_save(
    mock_close,
    mock_show,
    mock_savefig,
    mock_apply_mask,
    mock_subplots,
    mock_find_item,
    mock_processed_sample,
):
    """Test visualizing full masks and saving to file."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)
    target_item = _find_item_by_prompt(mock_processed_sample, "cat")
    mock_find_item.return_value = target_item
    output_file = "test_output.png"

    visualize_prompt_masks(
        mock_processed_sample, "cat", mask_type="full", show=False, output_path=output_file, dpi=100
    )

    mock_find_item.assert_called_once_with(mock_processed_sample, "cat")
    mock_subplots.assert_called_once()
    mock_ax.imshow.assert_called_once_with(mock_processed_sample["image"])
    assert mock_apply_mask.call_count == len(target_item["processed_full_masks"])
    # Check the single mask applied
    assert mock_apply_mask.call_count == 1

    # Check keyword arguments for positive_value in the call
    call_args, call_kwargs = mock_apply_mask.call_args
    assert call_args[0] == mock_ax
    assert call_args[1] == target_item["processed_full_masks"][0]["mask"]
    assert call_kwargs["positive_value"] == target_item["processed_full_masks"][0]["positive_value"]

    mock_show.assert_not_called()
    mock_savefig.assert_called_once_with(output_file, dpi=100, bbox_inches="tight")
    mock_close.assert_called_once_with(mock_fig)


@patch("src.dataops.cov_segm.visualizer._find_item_by_prompt")
@patch("matplotlib.pyplot.subplots")
@patch("logging.Logger.warning")  # Patch the logger directly
def test_visualize_prompt_masks_prompt_not_found(
    mock_log_warning, mock_subplots, mock_find_item, mock_processed_sample
):
    """Test behavior when the prompt is not found."""
    mock_find_item.return_value = None

    visualize_prompt_masks(mock_processed_sample, "nonexistent_prompt")

    mock_find_item.assert_called_once_with(mock_processed_sample, "nonexistent_prompt")
    mock_log_warning.assert_called_once_with(
        "Prompt 'nonexistent_prompt' not found in the provided sample."
    )
    mock_subplots.assert_not_called()


@patch("src.dataops.cov_segm.visualizer._find_item_by_prompt")
@patch("matplotlib.pyplot.subplots")
@patch("logging.Logger.error")  # Patch the logger directly
def test_visualize_prompt_masks_invalid_mask_type(
    mock_log_error, mock_subplots, mock_find_item, mock_processed_sample
):
    """Test behavior with an invalid mask_type."""
    target_item = _find_item_by_prompt(mock_processed_sample, "cat")
    mock_find_item.return_value = target_item  # Assume item is found

    visualize_prompt_masks(mock_processed_sample, "cat", mask_type="invalid_type")

    mock_find_item.assert_called_once_with(mock_processed_sample, "cat")
    mock_log_error.assert_called_once_with(
        "Invalid mask_type: invalid_type. Choose 'visible' or 'full'."
    )
    mock_subplots.assert_not_called()


@patch("src.dataops.cov_segm.visualizer._find_item_by_prompt")
@patch("matplotlib.pyplot.subplots")
@patch("src.dataops.cov_segm.visualizer._apply_color_mask")
@patch("logging.Logger.warning")  # Patch the logger directly
def test_visualize_prompt_masks_no_masks_found(
    mock_log_warning, mock_apply_mask, mock_subplots, mock_find_item, mock_processed_sample
):
    """Test visualizing when the item is found but has no masks of the requested type."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)
    # Use the 'dog' item which has empty mask lists
    target_item = _find_item_by_prompt(mock_processed_sample, "dog")
    mock_find_item.return_value = target_item

    visualize_prompt_masks(mock_processed_sample, "dog", mask_type="visible", show=False)

    mock_find_item.assert_called_once_with(mock_processed_sample, "dog")
    mock_subplots.assert_called_once()  # Should still show main image
    mock_log_warning.assert_called_once_with(
        "No 'Visible Masks' found for prompt 'dog' in this item."
    )
    mock_apply_mask.assert_not_called()
