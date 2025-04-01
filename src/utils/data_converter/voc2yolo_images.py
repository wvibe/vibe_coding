#!/usr/bin/env python3
"""
Copy Pascal VOC Images Based on ImageSets.

Copies images from the VOCdevkit/<YEAR>/JPEGImages directory to a structured
output directory (output_root/images/<TAG+YEAR>) based on the image IDs listed
in the corresponding ImageSet file (VOCdevkit/<YEAR>/ImageSets/Main/<TAG>.txt).

Optionally, supports randomly sampling a specified number of images.

Usage:
    python src/utils/data_converter/voc2yolo_images.py \
        --voc-root /path/to/VOC \
        --output-root /path/to/output \
        --years 2007,2012 \
        --tags train,val,test \
        [--sample-count 100] # Optional: Copy only 100 random images per split
"""

import argparse
import logging

# Need os and load_dotenv here now
import os
import random  # Added for sampling
import shutil
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

# Import common utilities from voc2yolo_utils
from src.utils.data_converter.voc2yolo_utils import (
    get_image_path,
    get_image_set_path,
    get_output_image_dir,
    get_voc_dir,
    read_image_ids,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Moved load_dotenv to top level
load_dotenv()  # Load .env variables


def _determine_paths(args: argparse.Namespace) -> Tuple[Optional[Path], Optional[Path]]:
    """Determine and validate VOC root and output root paths."""
    # Determine VOC Root Path
    if args.voc_root:
        voc_root_str = args.voc_root
        logger.info(f"Using specified --voc-root: {voc_root_str}")
    else:
        voc_root_str = os.getenv("VOC_ROOT")
        if not voc_root_str:
            logger.error(
                "VOC root not specified via --voc-root and VOC_ROOT not found in environment."
            )
            return None, None
        logger.info(f"Using VOC_ROOT from environment: {voc_root_str}")

    try:
        voc_root = Path(voc_root_str).expanduser()
        if not voc_root.exists() or not voc_root.is_dir():
            raise ValueError(f"Determined VOC root path is invalid: {voc_root}")
    except ValueError as e:
        logger.error(f"Error validating VOC root path: {e}")
        return None, None

    # Determine Output Root Path
    if args.output_root:
        output_root_str = args.output_root
        logger.info(f"Using specified --output-root: {output_root_str}")
    else:
        output_root_str = voc_root_str  # Default to voc_root if not specified
        logger.info(f"--output-root not specified, defaulting to VOC root: {output_root_str}")

    try:
        output_root = Path(output_root_str).expanduser()
    except Exception as e:
        logger.error(f"Error processing output root path '{output_root_str}': {e}")
        return voc_root, None

    return voc_root, output_root


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Copy VOC images based on ImageSet lists.")
    parser.add_argument(
        "--voc-root",
        type=str,
        default=None,
        help="Path to the VOC dataset root directory (containing VOCdevkit). If not set, uses VOC_ROOT from .env.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Path to the root directory for output images. Defaults to --voc-root if not specified.",
    )
    parser.add_argument(
        "--years",
        type=str,
        required=True,
        help="Comma-separated list of years to process (e.g., 2007,2012).",
    )
    parser.add_argument(
        "--tags",
        type=str,
        required=True,
        help="Comma-separated list of ImageSet tags to process (e.g., train,val,test). Uses ImageSets/Main.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=-1,
        help="Number of images to randomly sample from each split. If <= 0 or >= total images, copy all. (default: -1)",
    )
    return parser.parse_args()


def copy_images_for_split(
    voc_root: Path, output_root: Path, year: str, tag: str, sample_count: int
):
    """Copies images for a specific year, tag, and optional sample count.

    Args:
        voc_root: Path to the VOC dataset root directory.
        output_root: Path to the root directory for output images.
        year: The dataset year (e.g., '2007').
        tag: The ImageSet tag (e.g., 'train', 'val').
        sample_count: Number of images to randomly sample. <= 0 means copy all.

    Returns:
        tuple: (success_count, skipped_count, fail_count)
    """
    logger.info(f"--- Processing Year: {year}, Tag: {tag} ---")
    success_count = 0
    skipped_count = 0
    fail_count = 0

    try:
        # Get paths using utility functions
        voc_year_path = get_voc_dir(voc_root, year)
        # We use the 'detect' (Main) imagesets to define which images to copy
        imageset_file = get_image_set_path(voc_year_path, task_type="detect", tag=tag)
        output_image_dir = get_output_image_dir(output_root, year, tag)

        # Create output directory
        output_image_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output images will be saved to: {output_image_dir}")

        # Get image IDs using utility function
        image_ids = read_image_ids(imageset_file)

    except FileNotFoundError as e:
        logger.error(f"Error setting up paths for {year}/{tag}: {e}. Skipping split.")
        return 0, 0, 0
    except ValueError as e:
        logger.error(f"Configuration error for {year}/{tag}: {e}. Skipping split.")
        return 0, 0, 0
    except IOError as e:
        logger.error(f"Error reading imageset file for {year}/{tag}: {e}. Skipping split.")
        return 0, 0, 0

    if not image_ids:
        logger.warning(f"No image IDs found in {imageset_file}. Skipping copy for {year}/{tag}.")
        return 0, 0, 0

    original_count = len(image_ids)
    logger.info(f"Found {original_count} image IDs in {imageset_file}.")

    # --- Apply Sampling ---
    if 0 < sample_count < original_count:
        logger.info(f"Randomly sampling {sample_count} images out of {original_count}.")
        try:
            image_ids = random.sample(image_ids, sample_count)
        except ValueError as e:
            logger.error(f"Error during sampling: {e}. Check sample count.")
            return 0, 0, 0  # Treat sampling error as failure for the split
        logger.info(f"Proceeding with {len(image_ids)} sampled image IDs.")
    elif sample_count > 0:
        logger.info(
            f"Sample count ({sample_count}) >= total images ({original_count}). Copying all."
        )
    else:
        logger.info("No sampling requested (sample_count <= 0). Copying all.")
    # ----------------------

    logger.info("Processing image copies...")

    for image_id in tqdm(image_ids, desc=f"Copying {year}/{tag}"):
        try:
            src_image_path = get_image_path(voc_year_path, image_id)
            dest_image_path = output_image_dir / f"{image_id}.jpg"

            if dest_image_path.exists():
                skipped_count += 1
                continue

            if not src_image_path.exists():
                logger.warning(f"Source image not found: {src_image_path}. Skipping.")
                fail_count += 1
                continue

            # Copy the image, preserving metadata
            shutil.copy2(src_image_path, dest_image_path)
            success_count += 1

        except IOError as e:
            logger.error(f"Failed to copy {src_image_path} to {dest_image_path}: {e}")
            fail_count += 1
        except Exception as e:
            logger.error(
                f"Unexpected error copying image ID {image_id} for {year}/{tag}: {e}",
                exc_info=True,
            )
            fail_count += 1

    logger.info(
        f"Finished copying for {year}/{tag}: Copied {success_count}, Skipped (exists) {skipped_count}, Failed {fail_count}"
    )
    return success_count, skipped_count, fail_count


def main():
    """Main execution function."""
    args = parse_args()
    # load_dotenv() # Load .env variables (Moved)

    voc_root, output_root = _determine_paths(args)
    if not voc_root or not output_root:
        return  # Error logged in helper

    # Parse comma-separated lists
    try:
        years = [y.strip() for y in args.years.split(",") if y.strip()]
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        if not years or not tags:
            raise ValueError("Years and Tags arguments cannot be empty.")
    except Exception as e:
        logger.error(f"Error parsing --years or --tags argument: {e}")
        return

    sample_count = args.sample_count  # Get sample count from args

    logger.info("Starting VOC image copying process.")
    logger.info(f"Years to process: {years}")
    logger.info(f"Tags to process: {tags}")
    if sample_count > 0:
        logger.info(f"Sampling requested: {sample_count} images per split.")
    else:
        logger.info("No sampling requested.")

    total_success = 0
    total_skipped = 0
    total_fail = 0

    # Loop through each year and tag combination
    for year in years:
        for tag in tags:
            try:
                s, sk, f = copy_images_for_split(
                    voc_root, output_root, year, tag, sample_count
                )  # Pass sample_count
                total_success += s
                total_skipped += sk
                total_fail += f
            except Exception as e:
                logger.error(
                    f"Critical error during processing of {year}/{tag}: {e}", exc_info=True
                )
                # Decide if we should count all potential images as failed or just log
                # For now, just log the critical error and continue if possible.

    logger.info("\nImage copying process completed!")
    logger.info(f"Overall success count: {total_success}")
    logger.info(f"Overall skipped (exists) count: {total_skipped}")
    logger.info(f"Overall failed count: {total_fail}")


if __name__ == "__main__":
    main()
