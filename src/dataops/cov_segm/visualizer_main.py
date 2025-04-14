"""CLI entry point for visualizing cov-segm masks."""

import argparse
import logging
import os
from typing import Optional

import datasets

# Use full import paths from src
from src.dataops.cov_segm.datamodel import SegmSample  # Updated import
from src.dataops.cov_segm.loader import (
    load_sample,
)
from src.dataops.cov_segm.visualizer import visualize_prompt_masks

logger = logging.getLogger(__name__)

# --- CLI Helper Functions ---


def _setup_argparse() -> argparse.ArgumentParser:
    """Creates and configures the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Visualize masks for a specific prompt in the lab42/cov-segm-v3 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
            "Directory path to save visualizations. If None, displays interactively. "
            "Individual filenames include sample ID and prompt."
        ),
    )
    group_viz.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Transparency alpha value for mask overlays (0.0 to 1.0).",
    )
    # Removed --dpi argument

    # Simplified debug argument (no display group needed)
    parser.add_argument(
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


# Removed _determine_display_behavior function


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


# --- Main Execution Logic ---


def main():
    """Main entry point for the visualization CLI."""
    parser = _setup_argparse()
    args = parser.parse_args()

    _configure_logging(args.debug)

    # Simplified display behavior determination
    should_show_interactive = args.output_dir is None
    if args.output_dir and not should_show_interactive:
        logger.info("Saving images to output_dir without interactive display.")

    logger.info(f"Loading dataset '{args.dataset_name}', split '{args.split}'...")
    dataset_slice = _load_dataset_slice(
        args.dataset_name, args.split, args.start_index, args.sample_count
    )

    if not dataset_slice:
        exit(1)

    found_prompt_globally = False  # Track if prompt found across all samples
    for i, hf_row in enumerate(dataset_slice):
        current_index = args.start_index + i
        # Use sample.id later, but keep hf_row for now to get potential ID early
        hf_row_id = hf_row.get("id", f"sample_{current_index}")
        logger.info(f"Processing sample index: {current_index}, HF ID: {hf_row_id}")

        try:
            # Update type hint and handle None
            processed_sample: Optional[SegmSample] = load_sample(hf_row)

            if processed_sample:
                sample_id = processed_sample.id  # Get ID from processed sample
                if args.debug:
                    available_phrases = []
                    # Iterate through segments and phrases
                    for segment in processed_sample.segments:
                        for phrase in segment.phrases:
                            available_phrases.append(phrase.text)
                    logger.debug(f"Sample {sample_id} Available Phrases: {available_phrases}")

                # Check if the prompt exists using the SegmSample method
                target_segment = processed_sample.find_segment_by_prompt(args.prompt)

                if target_segment:
                    logger.info(
                        f"Found prompt '{args.prompt}' in sample {sample_id}. Visualizing..."
                    )
                    found_prompt_globally = True

                    # Prepare output path if needed
                    specific_output_path = None
                    if args.output_dir:
                        os.makedirs(args.output_dir, exist_ok=True)
                        # Use sample_id from processed_sample
                        filename = _create_output_filename(sample_id, args.prompt, args.mask_type)
                        specific_output_path = os.path.join(args.output_dir, filename)

                    # Call the actual visualization function (Updated arguments)
                    visualize_prompt_masks(
                        sample=processed_sample,
                        prompt=args.prompt,
                        mask_type=args.mask_type,  # type: ignore (Keep Literal hint)
                        output_path=specific_output_path,
                        show=should_show_interactive,
                        alpha=args.alpha,
                        # Removed dpi argument
                        debug=args.debug,
                    )
                else:
                    # Log prompt not found using the definite sample ID
                    logger.debug(f"Prompt '{args.prompt}' not found in sample {sample_id}.")
            else:
                # Log failure using the hf_row_id if sample loading failed
                logger.warning(
                    f"Failed to process sample index {current_index} (HF ID: {hf_row_id})."
                )

        except Exception as e:
            # Log error using the hf_row_id
            logger.error(
                f"Error processing sample index {current_index} (HF ID: {hf_row_id}): {e}",
                exc_info=True,
            )

    if not found_prompt_globally:
        logger.warning(
            f"Prompt '{args.prompt}' was not found in any of the processed samples "
            f"(indices {args.start_index} to {args.start_index + args.sample_count - 1})."
        )

    logger.info("Visualization script finished.")


if __name__ == "__main__":
    main()
