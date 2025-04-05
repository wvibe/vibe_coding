#!/usr/bin/env python3
"""
Copy Pascal VOC Images Based on ImageSets.

Copies images from the VOCdevkit/<YEAR>/JPEGImages directory to a structured
output directory (output_root/images/<TAG+YEAR>) based on the image IDs listed
in the corresponding ImageSet file (VOCdevkit/<YEAR>/ImageSets/Main/<TAG>.txt).

Optionally, supports randomly sampling a specified number of images.

Usage:
    python -m src.utils.data_converter.voc2yolo_images \\
        --voc-root /path/to/VOC \
        --output-root /path/to/output \
        --years 2007,2012 \
        --tags train,val,test \
        [--sample-count 100] # Optional: Copy only 100 random images per split
"""

import argparse
import logging
import os
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

# from tqdm import tqdm # tqdm removed for simplicity, add back if needed
# Import common utilities from voc2yolo_utils using full path
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

# --- Helper Functions --- #


def apply_sampling_across_splits(
    all_image_ids_map: dict, sample_count: Optional[int], total_ids_found: int, seed: int
) -> dict:
    """Applies random sampling across all collected image IDs if requested."""
    if sample_count is not None and sample_count > 0:
        if sample_count >= total_ids_found:
            logger.info(
                f"Sample count ({sample_count}) >= total images found ({total_ids_found}). "
                f"Using all images."
            )
            return all_image_ids_map
        else:
            logger.info(
                f"Sampling {sample_count} images randomly across all splits (Seed: {seed})."
            )
            # Create a flat list of (year, tag, image_id) tuples for sampling
            flat_list = []
            for (year, tag), ids in all_image_ids_map.items():
                for img_id in ids:
                    flat_list.append(((year, tag), img_id))

            random.seed(seed)
            sampled_list = random.sample(flat_list, sample_count)

            # Reconstruct the map with only sampled IDs
            final_image_ids_map = {}
            for (year, tag), img_id in sampled_list:
                split_key = (year, tag)
                if split_key not in final_image_ids_map:
                    final_image_ids_map[split_key] = []
                final_image_ids_map[split_key].append(img_id)
            return final_image_ids_map
    else:
        # No sampling requested or invalid sample count, use all found IDs
        logger.info("No sampling requested or sample count is invalid/zero. Using all images.")
        return all_image_ids_map


# --- Core Logic --- #


def copy_images_for_split(
    voc_root: Path,
    output_root: Path,
    year: str,
    tag: str,
    task_type: str,
    image_ids: List[str],
) -> Tuple[int, int, int]:
    """Copies images specified by IDs for a single year/tag/task split.

    Args:
        voc_root: Path to the root of the VOC dataset (containing VOCdevkit).
        output_root: Path to the root directory where processed data will be saved.
        year: The dataset year (e.g., '2007').
        tag: The dataset tag (e.g., 'train', 'val').
        task_type: The task type ('detect' or 'segment').
        image_ids: List of image IDs (without extension) to copy.

    Returns:
        Tuple: (success_count, skipped_count, error_count)
    """
    success_count = 0
    skipped_count = 0
    error_count = 0

    voc_year_dir = get_voc_dir(voc_root, year)
    # Use the updated function with task_type
    output_img_dir = get_output_image_dir(output_root, task_type, year, tag)

    try:
        output_img_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory {output_img_dir}: {e}")
        return 0, 0, len(image_ids)  # All failed if directory cannot be created

    # Use tqdm if installed and desired, otherwise just iterate
    # for image_id in tqdm(image_ids, desc=f"Copying {year}/{tag} ({task_type})"):
    for image_id in image_ids:
        src_img_path = get_image_path(voc_year_dir, image_id)
        dest_img_path = output_img_dir / f"{image_id}.jpg"

        if not src_img_path.is_file():
            logger.warning(f"Source image not found, skipping: {src_img_path}")
            error_count += 1
            continue

        if dest_img_path.exists():
            # logger.debug(f"Destination image already exists, skipping: {dest_img_path}")
            skipped_count += 1
            continue

        try:
            shutil.copy2(src_img_path, dest_img_path)  # copy2 preserves metadata
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to copy {src_img_path} to {dest_img_path}: {e}")
            error_count += 1

    return success_count, skipped_count, error_count


# --- Main Function --- #
def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Copy Pascal VOC images to a structured output directory based on year, "
        "tag, and task type.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Example Usage:\n"
            "  # Copy train/val images for VOC2007+2012 for detection task\n"
            "  python -m src.utils.data_converter.voc2yolo_images \\\n"
            "    --years 2007,2012 \\\n"
            "    --tags train,val \\\n"
            "    --task-type detect \\\n"
            "    --voc-root /path/to/datasets/VOC \\\n"
            "    --output-root /path/to/datasets/VOC \n\n"
            "  # Copy 100 random images from VOC2012 trainval for segmentation task\n"
            "  python -m src.utils.data_converter.voc2yolo_images \\\n"
            "    --years 2012 \\\n"
            "    --tags trainval \\\n"
            "    --task-type segment \\\n"
            "    --sample-count 100 \\\n"
            "    --voc-root /path/to/datasets/VOC"
        ),
    )
    # Remove default=None for voc-root to use Path with getenv correctly
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=Path(
            os.getenv("VOC_ROOT", "./datasets/VOC")
        ),  # Default needs adjustment if .env isn't standard
        help="Path to the root VOC dataset directory (containing VOCdevkit). "
        "Defaults to $VOC_ROOT or ./datasets/VOC.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,  # Default to voc_root if not specified
        help="Path to the root directory where processed data will be saved. "
        "Defaults to the value of --voc-root.",
    )
    parser.add_argument(
        "--years",
        type=str,
        required=True,
        help="Comma-separated list of dataset years (e.g., '2007,2012').",
    )
    parser.add_argument(
        "--tags",
        type=str,
        required=True,
        help="Comma-separated list of dataset tags (e.g., 'train,val,test').",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        required=True,
        choices=["detect", "segment"],
        help="Specify the task type ('detect' or 'segment').",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="(Optional) Randomly sample N images TOTAL across all specified splits. "
        "If not set, copies all images.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for random sampling.")

    args = parser.parse_args()

    # Determine and validate paths
    voc_root = args.voc_root.expanduser()
    output_root = args.output_root.expanduser() if args.output_root else voc_root

    if not voc_root.is_dir():
        logger.error(f"VOC Root directory not found or invalid: {voc_root}")
        return
    # Output root will be created by copy_images_for_split if needed

    # Parse comma-separated lists and task type
    try:
        years = [y.strip() for y in args.years.split(",") if y.strip()]
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        task_type = args.task_type  # Get task_type from args
        if not years or not tags:
            raise ValueError("Years and Tags arguments cannot be empty.")
    except Exception as e:
        logger.error(f"Error parsing arguments: {e}")
        return

    logger.info("--- Starting VOC Image Copy --- ")
    logger.info(f"Years: {years}")
    logger.info(f"Tags: {tags}")
    logger.info(f"Task Type: {task_type}")
    logger.info(f"VOC Root: {voc_root}")
    logger.info(f"Output Root: {output_root}")
    logger.info(f"Sampling: {args.sample_count or 'All'}")

    # --- Collect all image IDs first --- #
    all_image_ids_map = {}
    total_ids_found = 0
    for year in years:
        voc_year_dir = get_voc_dir(voc_root, year)
        for tag in tags:
            split_key = (year, tag)
            try:
                # Pass task_type to get_image_set_path
                imageset_path = get_image_set_path(voc_year_dir, task_type, tag)
                image_ids = read_image_ids(imageset_path)
                if not image_ids:
                    logger.warning(
                        f"No image IDs found for {year}/{tag} ({task_type}) in {imageset_path}"
                    )
                    all_image_ids_map[split_key] = []
                else:
                    all_image_ids_map[split_key] = image_ids
                    total_ids_found += len(image_ids)
                    logger.debug(f"Found {len(image_ids)} IDs for {year}/{tag} ({task_type})")
            except FileNotFoundError:
                logger.warning(f"ImageSet file not found, skipping: {imageset_path}")
                all_image_ids_map[split_key] = []
            except Exception as e:
                logger.error(f"Error reading ImageSet {imageset_path}: {e}")
                all_image_ids_map[split_key] = []

    if total_ids_found == 0:
        logger.error("No image IDs found for any specified year/tag combination. Exiting.")
        return

    # --- Apply Sampling (if requested) --- #
    final_image_ids_map = apply_sampling_across_splits(
        all_image_ids_map, args.sample_count, total_ids_found, args.seed
    )

    # --- Copy Images --- #
    total_copied = 0
    total_skipped = 0
    total_errors = 0

    # Iterate through the final map (potentially sampled)
    for (year, tag), image_ids in final_image_ids_map.items():
        if not image_ids:
            continue
        logger.info(f"Processing {len(image_ids)} images for {year} / {tag} ({task_type})...")
        # Pass task_type and correct image_ids list to copy function
        copied, skipped, errors = copy_images_for_split(
            voc_root,
            output_root,
            year,
            tag,
            task_type,  # Pass the parsed task_type
            image_ids,  # Pass the list of IDs for this specific split
        )
        total_copied += copied
        total_skipped += skipped
        total_errors += errors
        logger.info(
            f"Finished {year}/{tag} ({task_type}): "
            f"Copied={copied}, Skipped={skipped}, Errors={errors}"
        )

    logger.info("--- VOC Image Copy Summary --- ")
    logger.info(f"Total images copied: {total_copied}")
    logger.info(f"Total images skipped (already exist): {total_skipped}")
    logger.info(f"Total errors: {total_errors}")
    logger.info(f"Output structure generated under: {output_root}")


if __name__ == "__main__":
    main()
