# Standard library imports
import logging
from typing import Literal, Optional

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.dataops.cov_segm.datamodel import (
    ProcessedConversationItem,
    ProcessedCovSegmSample,
)

# Local imports

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define a colormap for masks (use 'tab10' for better visual distinction)
DEFAULT_COLORMAP = plt.get_cmap("tab10")


def _find_item_by_prompt(
    processed_sample: ProcessedCovSegmSample, prompt: str
) -> Optional[ProcessedConversationItem]:
    """Finds the first ProcessedConversationItem matching the prompt."""
    for item in processed_sample["processed_conversations"]:
        for phrase in item["phrases"]:
            if phrase["text"] == prompt:
                return item
    return None


def _apply_color_mask(
    ax: plt.Axes,
    mask: Image.Image,
    color: tuple,
    positive_value: int = 1,
    alpha: float = 0.5,
    debug: bool = False,
):
    """Applies a colored mask overlay to the plot axes.

    Args:
        ax: Matplotlib axes to apply the overlay to
        mask: PIL Image of the mask
        color: RGB tuple for the overlay color
        positive_value: The specific pixel value that indicates the target instance
        alpha: Transparency level (0-1) for the overlay
        debug: Whether to log detailed debugging information
    """
    # Convert mask to numpy array for processing
    np_mask = np.array(mask)

    # Only log detailed mask information in debug mode
    if debug:
        logger.info(
            f"Mask info - Mode: {mask.mode}, Min: {np_mask.min()}, "
            f"Max: {np_mask.max()}, Shape: {np_mask.shape}"
        )
        logger.info(f"Looking for positive_value: {positive_value}")

    # Different handling based on mask mode
    if mask.mode == "P":  # Palette-based image
        # Strategy 1: Direct match in palette mode.
        # For palette images, check if pixel values directly match the positive_value.
        # Note: Palette mapping interpretation might be needed for complex cases,
        # but direct matching often works for segmentation masks.
        if debug:
            unique_values = np.unique(np_mask)
            logger.debug(f"Palette Mask: Unique values={unique_values}")

        # Create mask where pixels match positive_value
        mask_indices = np_mask == positive_value

        # Log if palette exists but direct match failed (for debugging)
        if not np.any(mask_indices) and hasattr(mask, "palette"):
            if debug:
                logger.debug(
                    f"Palette Mask: No direct match for positive_value={positive_value}. "
                    f"Palette exists: {mask.palette is not None}. Complex mapping may be needed."
                )

    elif mask.mode == "1":  # Explicit handling for binary masks
        # np.array() on mode '1' image should give a boolean array
        np_mask = np.array(mask)
        # Ensure boolean type just in case
        np_mask_bool = np_mask.astype(bool)
        if debug:
            # Log the boolean array unique values
            logger.debug(f"Binary Mask (as bool): Unique values={np.unique(np_mask_bool)}")

            # Log the actual binary mask values
            logger.debug(f"Binary Mask (as int): {np_mask}")

        if positive_value == 1:
            # For mode '1', pixel values 1 represent True (white)
            mask_indices = np_mask_bool  # Match True (white) pixels
            if debug and np.any(mask_indices):
                logger.debug("Binary: Matched True values for positive_value=1")
            elif debug:
                logger.debug("Binary: No True values found in binary mask")
        elif positive_value == 0:
            mask_indices = ~np_mask_bool  # Match False (black) pixels
            if debug and np.any(mask_indices):
                logger.debug("Binary: Matched False values for positive_value=0")
        else:
            # Handle unexpected positive_value for binary mask
            mask_indices = np.zeros_like(np_mask_bool, dtype=bool)
            if debug:
                logger.warning(
                    f"Binary: Unexpected positive_value {positive_value}, expected 0 or 1."
                )

    else:  # Grayscale ('L') or other non-palette, non-binary modes
        # Convert mask to numpy array for processing
        np_mask = np.array(mask)
        if debug:
            unique_values = np.unique(np_mask)
            logger.debug(f"Non-Palette/Binary Mask: Unique values={unique_values}")

        # Strategy 2: Direct match (exact value) - THIS IS THE PRIMARY METHOD
        mask_indices = np_mask == positive_value
        if debug and np.any(mask_indices):
            logger.debug("Non-Palette/Binary: Matched using direct value comparison.")

        # Strategy 3: Scaled value match (e.g., 0-255 range) - Fallback if direct fails
        if not np.any(mask_indices) and np_mask.max() > 1:
            # Create a temporary boolean mask based on non-zero values
            temp_bool_mask = np_mask > 0
            # Check if the *positive* value itself is non-zero
            if positive_value > 0:
                mask_indices = temp_bool_mask
                if debug and np.any(mask_indices):
                    logger.debug(
                        "Non-Palette/Binary: Matched using non-zero values (fallback assuming scaled binary)."
                    )
            elif debug:  # positive_value is 0, and direct match failed
                logger.debug("Non-Palette/Binary: positive_value is 0, direct match failed.")

    # Create the colored overlay
    h, w = np_mask.shape[:2]  # Handle both 2D and 3D arrays (e.g. 'LA' mode)
    color_mask = np.zeros((h, w, 4))

    # Apply color to the identified mask regions
    if np.any(mask_indices):
        if debug:
            logger.info(f"Found {np.sum(mask_indices)} matching pixels for overlay")
        color_mask[mask_indices, 0] = color[0]  # R
        color_mask[mask_indices, 1] = color[1]  # G
        color_mask[mask_indices, 2] = color[2]  # B
        color_mask[mask_indices, 3] = alpha  # Alpha
    else:
        # Always log warnings about missing matches
        logger.warning(f"No pixels matched positive_value={positive_value} in the mask!")

    # Add the colored overlay to the plot
    ax.imshow(color_mask)


def visualize_prompt_masks(
    sample: ProcessedCovSegmSample,
    prompt: str,
    mask_type: Literal["visible", "full"] = "visible",
    output_path: Optional[str] = None,
    show: bool = True,
    alpha: float = 0.5,
    dpi: int = 150,
    debug: bool = False,
):
    """
    Visualizes the main image and overlays masks corresponding to a specific prompt.

    Args:
        sample: The processed data sample containing images and conversation details.
        prompt: The exact phrase text to search for in the conversation items.
        mask_type: Type of masks to display ('visible' for instance, 'full' for full segment).
        output_path: Optional path to save the visualization. If None, displays interactively.
        show: Whether to display the plot interactively (requires a GUI backend).
        alpha: Transparency level for the mask overlay (0.0 to 1.0).
        dpi: Resolution for the saved image.
        debug: Whether to log detailed debugging information.
    """
    target_item = _find_item_by_prompt(sample, prompt)

    if not target_item:
        logger.warning(f"Prompt '{prompt}' not found in the provided sample.")
        return

    main_image = sample["image"]
    if mask_type == "visible":
        masks_to_show = target_item.get("processed_instance_masks", [])
        mask_label = "Visible Masks"
    elif mask_type == "full":
        masks_to_show = target_item.get("processed_full_masks", [])
        mask_label = "Full Masks"
    else:
        logger.error(f"Invalid mask_type: {mask_type}. Choose 'visible' or 'full'.")
        return

    # Group masks by source for more concise reporting
    mask_sources = {}
    for mask in masks_to_show:
        source = mask.get("source", "unknown")
        if source not in mask_sources:
            mask_sources[source] = 0
        mask_sources[source] += 1

    num_masks = len(masks_to_show)

    # Log a concise summary of the masks
    if num_masks > 0:
        sources_summary = ", ".join([f"{count} from {src}" for src, count in mask_sources.items()])
        logger.info(
            f"Visualizing {num_masks} {mask_label.lower()} for prompt '{prompt}' "
            f"({sources_summary})"
        )
    else:
        logger.warning(f"No '{mask_label}' found for prompt '{prompt}' in this item.")
        # Optionally still show the main image
        # return

    fig, ax = plt.subplots()
    ax.imshow(main_image)
    ax.axis("off")  # Hide axes ticks

    # Use a different color for each mask instance
    # Tab10 has exactly 10 distinct colors - use them directly by index, cycling if needed
    # This ensures we get the distinct categorical colors the colormap was designed for
    colors = [DEFAULT_COLORMAP(i % 10) for i in range(num_masks)] if num_masks > 0 else []

    # Iterate through the list of ProcessedMask dictionaries
    for i, mask_data in enumerate(masks_to_show):
        source = mask_data.get("source", "unknown")
        positive_value = mask_data.get("positive_value", 1)

        # Log details only in debug mode
        if debug:
            logger.debug(
                f"Processing mask {i + 1}/{num_masks} from source '{source}' "
                f"with positive_value={positive_value}"
            )

        # Extract the actual PIL Image from the dictionary
        mask_img = mask_data.get("mask")
        if not isinstance(mask_img, Image.Image):
            logger.warning(
                f"ProcessedMask item {i} does not contain a valid PIL Image "
                f"in its 'mask' key, skipping."
            )
            continue

        # Each mask gets a distinct color from the tab10 colormap
        mask_color = colors[i][:3]  # Get RGB from RGBA

        # Apply the colored mask overlay
        _apply_color_mask(
            ax, mask_img, mask_color, positive_value=positive_value, alpha=alpha, debug=debug
        )

    # Add title with the prompt
    title = f"Prompt: '{prompt}'\n({mask_label})"
    ax.set_title(title, fontsize=10)

    plt.tight_layout()

    if output_path:
        logger.info(f"Saving visualization to: {output_path}")
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        if debug:
            logger.debug("Displaying visualization...")
        plt.show()

    # Close the plot to free memory, especially important in loops/scripts
    plt.close(fig)
