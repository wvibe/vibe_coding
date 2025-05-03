# Standard library imports
import logging
from typing import List, Literal, Optional

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np

from vibelab.dataops.cov_segm.datamodel import (
    ClsSegment,
    SegmMask,
    SegmSample,
)

# Local imports

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define a colormap for masks (use 'tab10' for better visual distinction)
DEFAULT_COLORMAP = plt.get_cmap("tab10")


def _apply_color_mask(
    ax: plt.Axes,
    binary_mask: np.ndarray,
    color: tuple,
    alpha: float = 0.5,
    debug: bool = False,
):
    """Applies a colored mask overlay based on a pre-computed boolean mask.

    Args:
        ax: Matplotlib axes to apply the overlay to
        binary_mask: Boolean NumPy array where True indicates pixels to color.
        color: RGB tuple for the overlay color
        alpha: Transparency level (0-1) for the overlay
        debug: Whether to log detailed debugging information
    """
    # Input is already the boolean mask
    mask_indices = binary_mask

    if debug:
        # Log basic info about the boolean mask
        logger.debug(f"Binary Mask Info - Shape: {binary_mask.shape}, dtype: {binary_mask.dtype}")

    # Create the colored overlay
    # Check mask dimensions, handle potential grayscale boolean mask (shape h, w)
    if binary_mask.ndim == 2:
        h, w = binary_mask.shape
    elif binary_mask.ndim == 3 and binary_mask.shape[2] == 1:  # Handle (h, w, 1) case
        h, w = binary_mask.shape[:2]
        mask_indices = binary_mask.squeeze(axis=2)  # Ensure 2D for indexing
    else:
        logger.warning(
            f"Unexpected binary_mask dimensions: {binary_mask.shape}. Cannot apply overlay."
        )
        return

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
        # Log if the input binary mask was empty
        logger.debug("Input binary_mask contained no True values.")

    # Add the colored overlay to the plot
    ax.imshow(color_mask)


def visualize_prompt_masks(
    sample: SegmSample,
    prompt: str,
    mask_type: Literal["visible", "full"] = "visible",
    output_path: Optional[str] = None,
    show: bool = True,
    alpha: float = 0.5,
    debug: bool = False,
):
    """
    Visualizes the main image and overlays masks corresponding to a specific prompt.

    Args:
        sample: The processed SegmSample object containing the image and segments.
        prompt: The exact phrase text to search for in the conversation items.
        mask_type: Type of masks to display ('visible' or 'full').
        output_path: Optional path to save the visualization. If None, displays interactively.
        show: Whether to display the plot interactively (requires a GUI backend).
        alpha: Transparency level for the mask overlay (0.0 to 1.0).
        debug: Whether to log detailed debugging information.
    """
    # Use the method from SegmSample
    target_segment: Optional[ClsSegment] = sample.find_segment_by_prompt(prompt)

    if not target_segment:
        logger.warning(f"Prompt '{prompt}' not found in the provided sample (ID: {sample.id}).")
        return

    main_image = sample.image  # Access image directly from SegmSample
    if mask_type == "visible":
        masks_to_show: List[SegmMask] = target_segment.visible_masks  # List of SegmMask
        mask_label = "Visible Masks"
    elif mask_type == "full":
        masks_to_show: List[SegmMask] = target_segment.full_masks  # List of SegmMask
        mask_label = "Full Masks"
    else:
        logger.error(f"Invalid mask_type: {mask_type}. Choose 'visible' or 'full'.")
        return

    # Filter for valid masks with non-None binary_mask
    valid_masks_to_show = [m for m in masks_to_show if m.is_valid and m.binary_mask is not None]
    num_masks = len(valid_masks_to_show)

    # Group masks by source column for more concise reporting
    mask_sources = {}
    for segm_mask in valid_masks_to_show:
        # Access source info from SegmMask
        source_col = segm_mask.source_info.column if segm_mask.source_info else "unknown"
        if source_col not in mask_sources:
            mask_sources[source_col] = 0
        mask_sources[source_col] += 1

    # Log a concise summary of the masks
    if num_masks > 0:
        sources_summary = ", ".join(
            [f"{count} from '{src}'" for src, count in mask_sources.items()]
        )
        logger.info(
            f"Visualizing {num_masks} valid {mask_label.lower()} for prompt '{prompt}' "
            f"in sample {sample.id} ({sources_summary})"
        )
    else:
        logger.warning(
            f"No valid '{mask_label}' found for prompt '{prompt}' in sample {sample.id}."
        )
        # Optionally still show the main image if desired, but usually not helpful without masks
        # return

    fig, ax = plt.subplots()
    ax.imshow(main_image)
    ax.axis("off")  # Hide axes ticks

    # Use distinct colors for each valid mask instance
    colors = [DEFAULT_COLORMAP(i % 10) for i in range(num_masks)] if num_masks > 0 else []

    # Iterate through the list of valid SegmMask objects
    for i, segm_mask in enumerate(valid_masks_to_show):
        source_col = segm_mask.source_info.column if segm_mask.source_info else "unknown"

        # Log details only in debug mode
        if debug:
            logger.debug(
                f"Processing valid mask {i + 1}/{num_masks} from source '{source_col}' "
                f" (Area: {segm_mask.pixel_area}, BBox: {segm_mask.bbox})"
            )

        # Each mask gets a distinct color from the tab10 colormap
        mask_color = colors[i][:3]  # Get RGB from RGBA

        # Apply the colored mask overlay using the binary mask directly
        # binary_mask should not be None here due to filtering above
        assert segm_mask.binary_mask is not None
        _apply_color_mask(ax, segm_mask.binary_mask, mask_color, alpha=alpha, debug=debug)

    # Add title with the prompt
    title = f"Sample: {sample.id}\nPrompt: '{prompt}' ({mask_label})"  # Include sample ID
    ax.set_title(title, fontsize=10)

    plt.tight_layout()

    if output_path:
        logger.info(f"Saving visualization to: {output_path}")
        plt.savefig(output_path, bbox_inches="tight")

    if show:
        if debug:
            logger.debug("Displaying visualization...")
        plt.show()

    # Close the plot to free memory, especially important in loops/scripts
    plt.close(fig)
