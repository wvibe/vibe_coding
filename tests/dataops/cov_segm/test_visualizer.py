import logging
from unittest.mock import ANY, MagicMock, call, patch

import numpy as np
import pytest
from PIL import Image

# Import datamodels directly from datamodel.py
from src.dataops.cov_segm.datamodel import (
    ClsSegment,
    ImageURI,
    InstanceMask,
    Phrase,
    SegmMask,
    SegmSample,
)

# Import the functions to test
from src.dataops.cov_segm.visualizer import (
    _apply_color_mask,
    visualize_prompt_masks,
)

# Configure logging for tests if needed
logger = logging.getLogger("src.dataops.cov_segm.visualizer")
logger.setLevel(logging.DEBUG)

# --- Helper Functions / Fixtures ---


def create_mock_segm_mask(
    column="mock_col",
    positive_value=1,
    shape=(10, 10),
    is_valid=True,
    pixel_area=10,
    bbox=(1, 1, 6, 3),  # (x_min, y_min, x_max, y_max)
) -> SegmMask:
    """Creates a mock SegmMask object with optional binary mask."""
    mock_info = InstanceMask(
        column=column,
        image_uri=ImageURI(jpg="s3://dummy/mask.jpg", format="jpg"),
        positive_value=positive_value,
    )
    # Create a SegmMask instance without raw_mask_data (as _parse is not called directly)
    # We will manually set the post-parsing attributes for testing purposes.
    mask = SegmMask(instance_mask_info=mock_info, raw_mask_data=Image.new("L", shape))

    # Override attributes based on test needs
    mask.is_valid = is_valid
    mask.pixel_area = pixel_area if is_valid else 0
    mask.bbox = bbox if is_valid and pixel_area > 0 else None

    # Create a plausible binary mask if valid
    if is_valid and pixel_area > 0 and shape[0] > 0 and shape[1] > 0:
        binary_mask_arr = np.zeros(shape, dtype=bool)
        # Set some pixels based on bbox to simulate the mask
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            # Ensure indices are within bounds
            y_start = max(0, y_min)
            y_end = min(shape[0], y_max + 1)
            x_start = max(0, x_min)
            x_end = min(shape[1], x_max + 1)
            if y_start < y_end and x_start < x_end:
                binary_mask_arr[y_start:y_end, x_start:x_end] = True
                # Adjust pixel area if the constructed mask doesn't match exactly
                # For simplicity in testing, we might assume the area matches bbox or
                # just use the provided pixel_area and bbox without strict consistency check
                # in the binary mask itself for unit tests.
                pass  # Keep provided pixel_area for testing consistency
        else:  # Simplified setting if no bbox
            binary_mask_arr[0, 0] = True  # Minimal mask
        mask.binary_mask = binary_mask_arr
    else:
        mask.binary_mask = np.zeros(shape, dtype=bool)  # Empty mask

    return mask


@pytest.fixture
def mock_segm_sample() -> SegmSample:
    """Creates a mock SegmSample with ClsSegment and SegmMask objects."""
    vis_mask_1 = create_mock_segm_mask(
        column="vis/0", positive_value=1, pixel_area=5, bbox=(0, 0, 2, 2)
    )
    vis_mask_2 = create_mock_segm_mask(
        column="vis/1", positive_value=2, pixel_area=8, bbox=(5, 5, 8, 8)
    )
    full_mask_1 = create_mock_segm_mask(
        column="full/0", positive_value=1, pixel_area=20, bbox=(0, 0, 4, 4)
    )
    # Add an invalid mask for testing filtering
    invalid_mask = create_mock_segm_mask(column="vis/invalid", is_valid=False)

    seg1 = ClsSegment(
        phrases=[Phrase(id=1, text="cat", type="obj")],
        type="SEG",
        visible_masks=[vis_mask_1, vis_mask_2, invalid_mask],
        full_masks=[full_mask_1],
    )
    seg2 = ClsSegment(
        phrases=[Phrase(id=2, text="dog", type="obj")],
        type="SEG",
        visible_masks=[],  # No masks for dog
        full_masks=[],
    )
    sample = SegmSample(
        id="sample1",
        image=Image.new("RGB", (50, 50)),  # Dummy main image
        segments=[seg1, seg2],
    )
    return sample


# --- Test _apply_color_mask ---


@pytest.mark.parametrize(
    "binary_mask_input, should_have_color",
    [
        (np.array([[False, False], [False, False]], dtype=bool), False),  # All False
        (np.array([[True, False], [False, False]], dtype=bool), True),  # Top-left True
        (np.array([[False, True], [True, False]], dtype=bool), True),  # Some True
        (np.array([[True, True], [True, True]], dtype=bool), True),  # All True
    ],
)
@patch("matplotlib.pyplot.Axes")
def test_apply_color_mask(mock_axes_class, binary_mask_input, should_have_color):
    """Tests applying color based on a boolean binary mask."""
    mock_ax = mock_axes_class.return_value
    color = (0, 1, 0)  # Green
    alpha = 0.7

    _apply_color_mask(mock_ax, binary_mask_input, color, alpha=alpha, debug=True)

    assert mock_ax.imshow.call_count == 1
    call_args, _ = mock_ax.imshow.call_args
    applied_overlay = call_args[0]
    assert applied_overlay.shape == (binary_mask_input.shape[0], binary_mask_input.shape[1], 4)

    # Check alpha channel based on input mask and expected outcome
    expected_alpha_mask = np.where(binary_mask_input, alpha, 0.0)
    assert np.allclose(applied_overlay[:, :, 3], expected_alpha_mask)

    # Check color channel where alpha > 0
    if should_have_color:
        assert np.any(applied_overlay[:, :, 3] > 0)
        colored_pixels = applied_overlay[applied_overlay[:, :, 3] > 0][
            :, :3
        ]  # Get RGB of colored pixels
        assert np.allclose(colored_pixels, color)  # Check if they match the input color
    else:
        assert np.all(applied_overlay[:, :, 3] == 0)


# --- Test visualize_prompt_masks ---


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
    mock_segm_sample,  # Use the refactored fixture
):
    """Test visualizing visible masks with interactive display."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    visualize_prompt_masks(
        mock_segm_sample, "cat", mask_type="visible", show=True, output_path=None
    )

    mock_subplots.assert_called_once()
    mock_ax.imshow.assert_called_once_with(mock_segm_sample.image)

    # Expect apply_mask to be called only for the VALID visible masks (vis_mask_1, vis_mask_2)
    cat_segment = mock_segm_sample.find_segment_by_prompt("cat")
    assert cat_segment is not None
    expected_call_count = sum(
        1 for m in cat_segment.visible_masks if m.is_valid and m.binary_mask is not None
    )
    assert mock_apply_mask.call_count == expected_call_count
    assert expected_call_count == 2  # vis_mask_1 and vis_mask_2 are valid

    # Check calls to apply_mask more carefully
    valid_visible_masks = [
        m for m in cat_segment.visible_masks if m.is_valid and m.binary_mask is not None
    ]
    expected_calls = [
        call(
            mock_ax,
            m.binary_mask,
            ANY,  # color tuple
            alpha=ANY,  # default alpha
            debug=ANY,  # default debug
        )
        for m in valid_visible_masks
    ]
    assert mock_apply_mask.call_args_list == expected_calls

    mock_show.assert_called_once()
    mock_savefig.assert_not_called()
    mock_close.assert_called_once_with(mock_fig)


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
    mock_segm_sample,  # Use fixture
):
    """Test visualizing full masks and saving to file."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)
    output_file = "test_output.png"

    visualize_prompt_masks(
        mock_segm_sample,
        "cat",
        mask_type="full",
        show=False,
        output_path=output_file,
        # Removed dpi argument
    )

    mock_subplots.assert_called_once()
    mock_ax.imshow.assert_called_once_with(mock_segm_sample.image)

    # Expect apply_mask for the single valid full mask
    cat_segment = mock_segm_sample.find_segment_by_prompt("cat")
    assert cat_segment is not None
    expected_call_count = sum(
        1 for m in cat_segment.full_masks if m.is_valid and m.binary_mask is not None
    )
    assert mock_apply_mask.call_count == expected_call_count
    assert expected_call_count == 1  # Only full_mask_1

    # Check the single mask applied
    valid_full_masks = [
        m for m in cat_segment.full_masks if m.is_valid and m.binary_mask is not None
    ]
    assert len(valid_full_masks) == 1
    expected_call = call(
        mock_ax,
        valid_full_masks[0].binary_mask,
        ANY,  # color tuple
        alpha=ANY,
        debug=ANY,
    )
    mock_apply_mask.assert_called_once_with(*expected_call.args, **expected_call.kwargs)

    mock_show.assert_not_called()
    # Removed dpi from assertion
    mock_savefig.assert_called_once_with(output_file, bbox_inches="tight")
    mock_close.assert_called_once_with(mock_fig)


@patch("matplotlib.pyplot.subplots")
@patch("logging.Logger.warning")
def test_visualize_prompt_masks_prompt_not_found(
    mock_log_warning,
    mock_subplots,
    mock_segm_sample,  # Use fixture
):
    """Test behavior when the prompt is not found."""
    visualize_prompt_masks(mock_segm_sample, "nonexistent_prompt")

    # Check log message includes sample ID
    mock_log_warning.assert_called_once_with(
        f"Prompt 'nonexistent_prompt' not found in the provided sample (ID: {mock_segm_sample.id})."
    )
    mock_subplots.assert_not_called()


@patch("matplotlib.pyplot.subplots")
@patch("logging.Logger.error")
def test_visualize_prompt_masks_invalid_mask_type(
    mock_log_error,
    mock_subplots,
    mock_segm_sample,  # Use fixture
):
    """Test behavior with an invalid mask_type."""
    visualize_prompt_masks(mock_segm_sample, "cat", mask_type="invalid_type")

    mock_log_error.assert_called_once_with(
        "Invalid mask_type: invalid_type. Choose 'visible' or 'full'."
    )
    # No need to check find_segment as it happens before mask_type check
    mock_subplots.assert_not_called()


@patch("matplotlib.pyplot.subplots")
@patch("src.dataops.cov_segm.visualizer._apply_color_mask")
@patch("logging.Logger.warning")
def test_visualize_prompt_masks_no_masks_found(
    mock_log_warning,
    mock_apply_mask,
    mock_subplots,
    mock_segm_sample,  # Use fixture
):
    """Test visualizing when the item is found but has no masks of the requested type."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    # Use the 'dog' segment which has empty mask lists
    visualize_prompt_masks(mock_segm_sample, "dog", mask_type="visible", show=False)

    mock_subplots.assert_called_once()  # Should still show main image
    # Check log message includes sample ID
    mock_log_warning.assert_called_once_with(
        f"No valid 'Visible Masks' found for prompt 'dog' in sample {mock_segm_sample.id}."
    )
    mock_apply_mask.assert_not_called()
