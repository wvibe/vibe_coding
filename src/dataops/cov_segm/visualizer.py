# Standard library imports
import argparse
import logging
import os
import pprint
from typing import Literal, Optional

# Third-party imports
import datasets
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Local imports
from src.dataops.cov_segm.loader import (
    ProcessedConversationItem,
    ProcessedCovSegmSample,
    get_last_parsed_conversations,
    load_sample,
)

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
        # For palette images, we need to check if the positive_value is in the palette
        # Get unique values to log
        unique_values = np.unique(np_mask)
        if debug:
            logger.info(f"Unique values in mask: {unique_values}")

        # Create mask where pixels match positive_value
        mask_indices = np_mask == positive_value

        # If no matches, try looking at the palette mapping
        if not np.any(mask_indices) and hasattr(mask, "palette"):
            if debug:
                logger.info("No direct matches found, checking palette...")
            # This is a more complex approach if needed
            # We might need to examine the palette and map palette indices to actual values
    else:
        # For non-palette images, handle different value encoding possibilities
        unique_values = np.unique(np_mask)
        if debug:
            logger.info(f"Unique values in mask: {unique_values}")

        # Strategy 1: Direct match (exact value)
        mask_indices = np_mask == positive_value

        # Strategy 2: If no matches and values are scaled (e.g., 0-255 range)
        if not np.any(mask_indices) and np_mask.max() > 1:
            # Try normalized mask (if values might be scaled)
            if debug:
                logger.info("Trying normalized value comparison...")
            normalized_mask = np_mask / np_mask.max()
            normalized_positive = positive_value / max(unique_values.max(), positive_value)
            mask_indices = normalized_mask == normalized_positive

        # Strategy 3: If still no matches, check if the mask has binary-like structure
        if not np.any(mask_indices) and len(unique_values) <= 2:
            if debug:
                logger.info("Trying binary mask approach...")
            # For binary masks, use non-zero values if positive_value > 0
            if positive_value > 0:
                mask_indices = np_mask > 0

    # Create the colored overlay
    h, w = np_mask.shape[:2]  # Handle both 2D and 3D arrays
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


def _create_output_filename(row_id: str, prompt: str, mask_type: str) -> str:
    """Creates a standardized filename for visualization outputs.

    Args:
        row_id: The dataset row ID
        prompt: The prompt text that was visualized
        mask_type: The type of mask ('visible' or 'full')

    Returns:
        A standardized filename
    """
    # Sanitize the prompt text for filename use
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:20])
    return f"{row_id}_{safe_prompt}_{mask_type}.png"


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


def _setup_argparse() -> argparse.ArgumentParser:
    """Creates and configures the argument parser for the CLI.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Visualize masks for a specific prompt in the lab42/cov-segm-v3 dataset."
    )
    parser.add_argument("prompt", type=str, help="The exact text prompt to visualize.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lab42/cov-segm-v3",
        help="Name of the Hugging Face dataset.",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split (e.g., 'train', 'validation')."
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="Starting index of the sample in the dataset."
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=1,
        help="Number of samples to check starting from start_index.",
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        default="visible",
        choices=["visible", "full"],
        help="Type of mask to display ('visible' or 'full').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory path to save visualizations. "
            "Individual filenames include sample ID and prompt."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not attempt to show the plot interactively (e.g., in headless environment).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Force interactive display even when saving to output_dir. "
            "By default, images are not displayed when saving."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Transparency alpha value for mask overlays (0.0 to 1.0).",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Resolution (DPI) for saved images.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debugging output and raw conversation data display.",
    )

    return parser


def _configure_logging(debug_mode: bool):
    """Configures logging based on debug mode.

    Args:
        debug_mode: Whether to enable debug logging
    """
    if debug_mode:
        # Set visualizer module to DEBUG
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled - showing detailed logging information")

        # Also set loader module to DEBUG
        loader_logger = logging.getLogger("src.dataops.cov_segm.loader")
        loader_logger.setLevel(logging.DEBUG)


def _determine_display_behavior(
    output_dir: Optional[str], show_flag: bool, no_show_flag: bool
) -> bool:
    """Determines whether to show interactive display based on arguments.

    Args:
        output_dir: The output directory if specified
        show_flag: Whether --show was specified
        no_show_flag: Whether --no-show was specified

    Returns:
        Boolean indicating whether to show interactive display
    """
    should_show_interactive = True

    # When saving to output_dir, don't show interactively by default
    # Unless explicitly requested with --show
    if output_dir and not show_flag:
        should_show_interactive = False
        logger.info(
            "Saving images to output_dir without interactive display. Use --show to enable display."
        )

    # --no-show always disables interactive display
    if no_show_flag:
        should_show_interactive = False

    return should_show_interactive


def _load_dataset_slice(dataset_name: str, split: str, start_index: int, sample_count: int):
    """Loads a slice of the dataset.

    Args:
        dataset_name: Name of the dataset
        split: Dataset split
        start_index: Starting index
        sample_count: Number of samples

    Returns:
        Dataset slice or None if loading fails
    """
    try:
        # Load only the necessary slice
        dataset_slice = datasets.load_dataset(
            dataset_name,
            split=f"{split}[{start_index}:{start_index + sample_count}]",
            trust_remote_code=True,  # Required for this dataset
        )
        return dataset_slice
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return None


def main():
    """Main entry point for the visualization CLI."""
    parser = _setup_argparse()
    args = parser.parse_args()

    # Configure logging based on debug mode
    _configure_logging(args.debug)

    # Determine interactive display behavior
    should_show_interactive = _determine_display_behavior(args.output_dir, args.show, args.no_show)

    logger.info(f"Loading dataset '{args.dataset_name}', split '{args.split}'...")
    dataset_slice = _load_dataset_slice(
        args.dataset_name, args.split, args.start_index, args.sample_count
    )

    if not dataset_slice:
        exit(1)

    found_prompt = False
    for i, hf_row in enumerate(dataset_slice):
        current_index = args.start_index + i

        # Get the sample ID from the dataset row
        row_id = hf_row.get(
            "id", f"sample_{current_index}"
        )  # Default to sample_index if id not found

        logger.info(f"Processing sample index: {current_index}, ID: {row_id}")

        try:
            processed_sample = load_sample(hf_row)
            if processed_sample:
                # Log available phrases only in debug mode
                if args.debug:
                    available_phrases = []
                    for item in processed_sample["processed_conversations"]:
                        for phrase in item["phrases"]:
                            available_phrases.append(phrase["text"])
                    logger.debug(f"Sample {current_index} Available Phrases: {available_phrases}")

                target_item = _find_item_by_prompt(processed_sample, args.prompt)
                if target_item:
                    logger.info(
                        f"Found prompt '{args.prompt}' in sample index "
                        f"{current_index}. Visualizing..."
                    )

                    # --- Debug: Show raw conversation items when --debug is enabled ---
                    if args.debug:
                        logger.debug(f"-- RAW Conversation Items for Sample {current_index} --")
                        raw_conversations = get_last_parsed_conversations()

                        # Find the raw item that contains our prompt
                        matching_raw_item = None
                        for idx, raw_item in enumerate(raw_conversations):
                            raw_phrases = raw_item.get("phrases", [])
                            if any(phrase.get("text") == args.prompt for phrase in raw_phrases):
                                matching_raw_item = raw_item
                                logger.debug(f"Found prompt in raw conversation item #{idx}")
                                break

                        if matching_raw_item:
                            # Pretty print the full raw item with all its fields
                            logger.debug("Raw Item Structure:")
                            logger.debug(pprint.pformat(matching_raw_item))

                            # Specifically highlight the mask fields
                            instance_masks = matching_raw_item.get("instance_masks", [])
                            instance_full_masks = matching_raw_item.get("instance_full_masks", [])
                            mask_cols = [m.get("column") for m in instance_masks]
                            full_mask_cols = [m.get("column") for m in instance_full_masks]
                            logger.debug(f"instance_masks columns: {mask_cols}")
                            logger.debug(f"instance_full_masks columns: {full_mask_cols}")
                        else:
                            logger.warning("Could not find matching raw item (this is unusual)")
                        logger.debug("-- End RAW Data --")
                    # --- End Debug ---

                    # Determine output path for this specific sample
                    specific_output_path = None

                    # If output_dir is specified, generate a filename based on ID
                    if args.output_dir:
                        # Create the directory if it doesn't exist
                        os.makedirs(args.output_dir, exist_ok=True)

                        # Generate standardized filename
                        filename = _create_output_filename(row_id, args.prompt, args.mask_type)
                        specific_output_path = os.path.join(args.output_dir, filename)
                        logger.info(f"Saving visualization to: {specific_output_path}")

                    visualize_prompt_masks(
                        sample=processed_sample,
                        prompt=args.prompt,
                        mask_type=args.mask_type,
                        output_path=specific_output_path,
                        show=should_show_interactive,
                        alpha=args.alpha,
                        dpi=args.dpi,
                        debug=args.debug,
                    )
                    found_prompt = True
                else:
                    logger.debug(f"Prompt not found in sample index {current_index}.")
            else:
                logger.warning(f"Failed to process sample index {current_index}.")
        except Exception as e:
            logger.error(f"Error processing sample index {current_index}: {e}")

    if not found_prompt:
        logger.warning(
            f"Prompt '{args.prompt}' not found in samples from index {args.start_index} "
            f"to {args.start_index + args.sample_count - 1}."
        )

    logger.info("Visualization script finished.")


if __name__ == "__main__":
    main()
