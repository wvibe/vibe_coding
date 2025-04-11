"""CLI entry point for visualizing cov-segm masks."""

import argparse
import logging
import os
import pprint
from typing import Optional

import datasets

from src.dataops.cov_segm.datamodel import ProcessedCovSegmSample  # For type hinting

# Use full import paths from src
from src.dataops.cov_segm.loader import (
    get_last_parsed_conversations,
    load_sample,
)
from src.dataops.cov_segm.visualizer import visualize_prompt_masks

logger = logging.getLogger(__name__)

# --- CLI Helper Functions (Copied from visualizer.py) ---


def _setup_argparse() -> argparse.ArgumentParser:
    """Creates and configures the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Visualize masks for a specific prompt in the lab42/cov-segm-v3 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show defaults
    )
    # Required argument
    parser.add_argument("prompt", type=str, help="The exact text prompt to visualize.")

    # Dataset selection arguments
    group_dataset = parser.add_argument_group("Dataset Selection")
    group_dataset.add_argument(
        "--dataset_name",
        type=str,
        default="lab42/cov-segm-v3",
        help="Name of the Hugging Face dataset.",
    )
    group_dataset.add_argument(
        "--split", type=str, default="train", help="Dataset split (e.g., 'train', 'validation')."
    )
    group_dataset.add_argument(
        "--start_index", type=int, default=0, help="Starting index of the sample in the dataset."
    )
    group_dataset.add_argument(
        "--sample_count",
        type=int,
        default=1,
        help="Number of samples to check starting from start_index.",
    )

    # Visualization control arguments
    group_viz = parser.add_argument_group("Visualization Control")
    group_viz.add_argument(
        "--mask_type",
        type=str,
        default="visible",
        choices=["visible", "full"],
        help="Type of mask to display ('visible' or 'full').",
    )
    group_viz.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory path to save visualizations. "
            "Individual filenames include sample ID and prompt."
        ),
    )
    group_viz.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Transparency alpha value for mask overlays (0.0 to 1.0).",
    )
    group_viz.add_argument(
        "--dpi", type=int, default=150, help="Resolution (DPI) for saved images."
    )

    # Display/Debug arguments
    group_display = parser.add_argument_group("Display & Debug")
    # Use mutually exclusive group for --show and --no-show
    display_group = group_display.add_mutually_exclusive_group()
    display_group.add_argument(
        "--show",
        action="store_true",
        help=(
            "Force interactive display even when saving to output_dir. "
            "By default, images are not displayed when saving."
        ),
    )
    display_group.add_argument(
        "--no-show",
        action="store_true",
        help="Do not attempt to show the plot interactively (e.g., in headless environment).",
    )
    group_display.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debugging output and raw conversation data display.",
    )

    return parser


def _configure_logging(debug_mode: bool):
    """Configures logging based on debug mode."""
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(log_level)
    if debug_mode:
        logger.debug("Debug mode enabled - showing detailed logging information")
        # Also set loader/visualizer module loggers to DEBUG if needed
        logging.getLogger("src.dataops.cov_segm.loader").setLevel(logging.DEBUG)
        logging.getLogger("src.dataops.cov_segm.visualizer").setLevel(logging.DEBUG)


def _determine_display_behavior(
    output_dir: Optional[str], show_flag: bool, no_show_flag: bool
) -> bool:
    """Determines whether to show interactive display based on arguments."""
    should_show_interactive = True
    if output_dir and not show_flag:
        should_show_interactive = False
        logger.info(
            "Saving images to output_dir without interactive display. Use --show to enable display."
        )
    if no_show_flag:
        should_show_interactive = False
        if output_dir and show_flag:
            logger.warning("--show flag ignored due to --no-show.")
    return should_show_interactive


def _load_dataset_slice(dataset_name: str, split: str, start_index: int, sample_count: int):
    """Loads a slice of the dataset."""
    try:
        dataset_slice = datasets.load_dataset(
            dataset_name,
            split=f"{split}[{start_index}:{start_index + sample_count}]",
            trust_remote_code=True,
        )
        return dataset_slice
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return None


def _create_output_filename(row_id: str, prompt: str, mask_type: str) -> str:
    """Creates a standardized filename for visualization outputs."""
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:20])
    return f"{row_id}_{safe_prompt}_{mask_type}.png"


# --- Main Execution Logic (Copied from visualizer.py) ---


def main():
    """Main entry point for the visualization CLI."""
    parser = _setup_argparse()
    args = parser.parse_args()

    _configure_logging(args.debug)
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
        row_id = hf_row.get("id", f"sample_{current_index}")
        logger.info(f"Processing sample index: {current_index}, ID: {row_id}")

        try:
            processed_sample: Optional[ProcessedCovSegmSample] = load_sample(hf_row)
            if processed_sample:
                if args.debug:
                    available_phrases = []
                    for item in processed_sample["processed_conversations"]:
                        for phrase in item["phrases"]:
                            available_phrases.append(phrase["text"])
                    logger.debug(f"Sample {current_index} Available Phrases: {available_phrases}")

                # Call the core visualization function from the library module
                # Need to check if the prompt exists within the sample first
                prompt_found_in_sample = False
                for item in processed_sample["processed_conversations"]:
                    if any(p["text"] == args.prompt for p in item.get("phrases", [])):
                        prompt_found_in_sample = True
                        break

                if prompt_found_in_sample:
                    logger.info(
                        f"Found prompt '{args.prompt}' in sample index "
                        f"{current_index}. Visualizing..."
                    )
                    found_prompt = True  # Mark that we found the prompt at least once

                    # Debug raw conversation data
                    if args.debug:
                        logger.debug(f"-- RAW Conversation Items for Sample {current_index} --")
                        raw_conversations = get_last_parsed_conversations()
                        matching_raw_item = None
                        for _idx, raw_item in enumerate(raw_conversations):
                            raw_phrases = raw_item.get("phrases", [])
                            if any(phrase.get("text") == args.prompt for phrase in raw_phrases):
                                matching_raw_item = raw_item
                                break
                        if matching_raw_item:
                            logger.debug("Raw Item Structure:")
                            logger.debug(pprint.pformat(matching_raw_item))
                            instance_masks = matching_raw_item.get("instance_masks", [])
                            instance_full_masks = matching_raw_item.get("instance_full_masks", [])
                            mask_cols = [m.get("column") for m in instance_masks]
                            full_mask_cols = [m.get("column") for m in instance_full_masks]
                            logger.debug(f"instance_masks columns: {mask_cols}")
                            logger.debug(f"instance_full_masks columns: {full_mask_cols}")
                        else:
                            logger.warning("Could not find matching raw item (this is unusual)")
                        logger.debug("-- End RAW Data --")

                    # Prepare output path if needed
                    specific_output_path = None
                    if args.output_dir:
                        os.makedirs(args.output_dir, exist_ok=True)
                        filename = _create_output_filename(row_id, args.prompt, args.mask_type)
                        specific_output_path = os.path.join(args.output_dir, filename)
                        # No need to log here, visualize_prompt_masks does it

                    # Call the actual visualization function
                    visualize_prompt_masks(
                        sample=processed_sample,
                        prompt=args.prompt,
                        mask_type=args.mask_type,  # type: ignore
                        output_path=specific_output_path,
                        show=should_show_interactive,
                        alpha=args.alpha,
                        dpi=args.dpi,
                        debug=args.debug,
                    )
                else:
                    logger.debug(
                        f"Prompt '{args.prompt}' not found in sample index {current_index}."
                    )
            else:
                logger.warning(f"Failed to process sample index {current_index}.")
        except Exception as e:
            logger.error(f"Error processing sample index {current_index}: {e}", exc_info=True)

    if not found_prompt:
        logger.warning(
            f"Prompt '{args.prompt}' not found in the processed samples "
            f"(indices {args.start_index} to {args.start_index + args.sample_count - 1})."
        )

    logger.info("Visualization script finished.")


if __name__ == "__main__":
    main()
