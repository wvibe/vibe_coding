#!/usr/bin/env python3
"""
Visualize Pascal VOC Ground Truth Detection Annotations.

Reads VOC XML annotation files and draws the bounding boxes
and class labels onto the corresponding images.
Supports processing single images (with display) or batches (saving images).
Optionally calculates and reports statistics on annotations per image.
"""

import argparse
import logging
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

# Project utilities
from src.utils.common.image_annotate import draw_box, get_color
from src.utils.data_converter.voc2yolo_utils import (
    VOC_CLASSES,  # Use this for consistent class mapping if needed?
    get_annotation_path,
    get_image_path,
    get_image_set_path,
    get_voc_dir,
    parse_voc_xml,  # Already parses required info
    read_image_ids,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize Pascal VOC ground truth detection annotations."
    )

    parser.add_argument(
        "--years",
        type=str,
        required=True,
        help="Comma-separated list of dataset years (e.g., '2007', '2007,2012').",
    )
    parser.add_argument(
        "--tags",
        type=str,
        required=True,
        help="Comma-separated list of dataset tags (e.g., 'train', 'val', 'train,val').",
    )

    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--image-id",
        type=str,
        default=None,
        help="Visualize a single specific image ID. Enables display mode.",
    )
    mode_group.add_argument(
        "--sample-count",
        type=int,
        default=-1,
        help=(
            "Randomly sample this many images from the specified splits. "
            "Enables batch mode. Process all if <= 0."
        ),
    )

    parser.add_argument(
        "--voc-root",
        type=str,
        default=None,
        help=(
            "Path to the VOC dataset root directory (containing VOCdevkit). "
            "Uses $VOC_ROOT if not set."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root directory for saving visualizations. Defaults to voc-root.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="visual_detect",  # Changed default name slightly
        help="Subdirectory within output-root to save visualizations.",
    )
    parser.add_argument(
        "--percentiles",
        type=str,
        default=None,
        help=(
            "Comma-separated list of percentiles (0-1) for stats "
            "(e.g., '0.25,0.5,0.75'). Reports average if not set."
        ),
    )
    parser.add_argument(
        "--show-difficult",
        action="store_true",
        help="Add a '*' marker to labels for objects marked as difficult.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")

    return parser.parse_args()


def _setup_paths(
    args: argparse.Namespace, voc_root_env: Optional[str]
) -> Tuple[Path, Path, Path, Path]:
    """Determine base VOC root, VOCdevkit root, output root, and final output directory.

    Handles logic for using --voc-root, $VOC_ROOT, and --output-root arguments,
    including defaults and validation.

    Returns:
        Tuple[Path, Path, Path, Path]: base_voc_root, voc_devkit_root, output_root, output_dir

    Raises:
        ValueError: If VOC root cannot be determined or VOCdevkit is not found.
    """
    base_voc_root_str = args.voc_root or voc_root_env
    if not base_voc_root_str:
        logger.error("VOC root directory not specified via --voc-root or $VOC_ROOT.")
        raise ValueError("VOC root not specified")

    base_voc_root = Path(base_voc_root_str).expanduser()
    voc_devkit_root = base_voc_root / "VOCdevkit"

    if not voc_devkit_root.exists():
        logger.error(f"VOCdevkit directory not found at: {voc_devkit_root}")
        raise FileNotFoundError(f"VOCdevkit not found at {voc_devkit_root}")

    logger.info(f"Using Base VOC Root: {base_voc_root}")
    logger.info(f"Derived VOCdevkit Root: {voc_devkit_root}")

    # Determine Output Root
    if args.output_root:
        output_root = Path(args.output_root).expanduser()
        logger.info(f"Using specified output root: {output_root}")
    else:
        output_root = voc_devkit_root
        logger.info(f"Using default output root (VOCdevkit): {output_root}")

    output_dir = output_root / args.output_subdir
    # Create the directory here for convenience, though it might be mocked in tests
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output visualization directory: {output_dir}")

    return base_voc_root, voc_devkit_root, output_root, output_dir


def _get_batch_image_ids(
    years: List[str], tags: List[str], base_voc_root: Path
) -> List[Tuple[str, str, str]]:
    """Helper to collect all image IDs for specified years/tags in batch mode."""
    all_ids_found = []
    for year in years:
        for tag in tags:
            try:
                # Construct the year-specific VOC directory path
                voc_dir = get_voc_dir(base_voc_root, year)
                # Pass the year-specific voc_dir to the utility function
                imageset_path = get_image_set_path(voc_dir, set_type="detect", tag=tag)
                # Use read_image_ids from the imported utils module
                image_ids = read_image_ids(imageset_path)
                if image_ids:
                    all_ids_found.extend([(img_id, year, tag) for img_id in image_ids])
                else:
                    logger.warning(
                        f"No image IDs found for Detection {year}/{tag} in {imageset_path}"
                    )
            except FileNotFoundError:
                logger.error(
                    f"Could not find ImageSet file for Detection {year}/{tag}: {imageset_path}"
                )
            except Exception as e:
                logger.error(f"Error reading ImageSet for Detection {year}/{tag}: {e}")
    return all_ids_found


def get_target_image_list(
    args: argparse.Namespace, base_voc_root: Path, voc_devkit_dir: Path
) -> List[Tuple[str, str, str]]:
    """Determines the list of (image_id, year, tag) tuples to process."""
    ids_to_process = []
    years = [y.strip() for y in args.years.split(",") if y.strip()]
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    if not years or not tags:
        logger.error("No valid years or tags provided.")
        return []

    if args.image_id:
        # Single image mode - path checking uses base_voc_root + year
        first_year = years[0]
        first_tag = tags[0]
        logger.info(f"Single image mode: Processing {args.image_id} from {first_year}/{first_tag}")
        try:
            year_voc_dir = get_voc_dir(base_voc_root, first_year)
            img_path = get_image_path(year_voc_dir, args.image_id)
            xml_path = get_annotation_path(year_voc_dir, args.image_id)
            if not img_path.exists():
                logger.error(f"Image file not found: {img_path}")
                return []
            if not xml_path.exists():
                logger.error(f"Annotation file not found: {xml_path}")
                return []
        except Exception as e:
            logger.error(f"Error checking paths for {args.image_id}: {e}")
            return []
        ids_to_process = [(args.image_id, first_year, first_tag)]
    else:
        # Batch mode (all or sampled)
        # Pass base_voc_root to the helper, it will construct year-specific paths
        all_ids_found = _get_batch_image_ids(years, tags, base_voc_root)

        if not all_ids_found:
            logger.error(
                "No image IDs found for any specified year/tag combination using Detection sets."
            )
            return []

        logger.info(
            f"Found {len(all_ids_found)} total image IDs across specified Detection splits."
        )

        # Apply sampling if requested
        if args.sample_count > 0 and args.sample_count < len(all_ids_found):
            logger.info(f"Randomly sampling {args.sample_count} images (seed={args.seed}).")
            random.seed(args.seed)
            ids_to_process = random.sample(all_ids_found, args.sample_count)
        else:
            if args.sample_count > 0:
                logger.info(f"Sample count ({args.sample_count}) >= total IDs. Processing all.")
            ids_to_process = all_ids_found

    logger.info(f"Will process {len(ids_to_process)} image(s).")
    return ids_to_process


def process_and_visualize_image(
    image_id: str,
    year: str,
    tag: str,
    voc_root: Path,
    output_dir: Path,
    class_name_to_id_map: dict,
    do_save: bool,
    do_display: bool,
    show_difficult: bool,
) -> Tuple[bool, Optional[int], Optional[int], Optional[int], bool, bool]:
    """Loads, parses, draws, and saves/displays a single image.

    Returns:
        Tuple: (xml_success, num_boxes, num_classes, num_difficult, save_success, display_success)
               Returns counts as None if XML processing failed.
    """
    save_success = False
    display_success = False
    try:
        voc_dir = get_voc_dir(voc_root, year)
        image_path = get_image_path(voc_dir, image_id)
        xml_path = get_annotation_path(voc_dir, image_id)

        # Load Image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Failed to load image: {image_path}. Skipping.")
            return False, None, None, None, save_success, display_success

        # Parse XML
        objects_list, img_dims = parse_voc_xml(xml_path)
        if objects_list is None:
            logger.warning(
                f"Failed to parse or no valid objects in XML: {xml_path}. Skipping drawing."
            )
            return False, 0, 0, 0, save_success, display_success

        # Collect stats
        num_boxes = len(objects_list)
        num_classes = len(set(o["name"] for o in objects_list))
        num_difficult = sum(1 for o in objects_list if o.get("difficult", 0) == 1)

        # Draw Annotations
        image_to_draw = image.copy()
        for obj in objects_list:
            class_name = obj["name"]
            box = obj["bbox"]  # Already in [xmin, ymin, xmax, ymax] pixel format
            is_difficult = obj.get("difficult", 0) == 1

            # Get color based on consistent class ID map
            class_id = class_name_to_id_map.get(class_name, -1)
            color = get_color(class_id if class_id != -1 else None)

            # Format label
            label_text = f"{class_name}*" if show_difficult and is_difficult else class_name

            draw_box(image_to_draw, box, label_text, color=color)

        # Save or Display
        if do_save:
            save_subdir = output_dir / f"{tag}{year}"
            save_subdir.mkdir(parents=True, exist_ok=True)
            save_path = save_subdir / f"{image_id}.png"
            try:
                cv2.imwrite(str(save_path), image_to_draw)
                save_success = True
            except Exception as e:
                logger.error(f"Failed to save image {save_path}: {e}")

        if do_display:
            try:
                cv2.imshow(f"VOC Detect Viz - {image_id}", image_to_draw)
                logger.info(f"Displaying image {image_id}. Press any key to continue...")
                cv2.waitKey(0)
                display_success = True
            except Exception as e:
                logger.error(f"Failed to display image {image_id}: {e}")
            finally:
                cv2.destroyAllWindows()

        # Return status and stats
        return True, num_boxes, num_classes, num_difficult, save_success, display_success

    except FileNotFoundError as e:
        logger.error(f"File not found error for {image_id}: {e}. Skipping.")
        return False, None, None, None, save_success, display_success
    except Exception as e:
        logger.error(f"Unexpected error processing {image_id}: {e}", exc_info=True)
        return False, None, None, None, save_success, display_success


def _initialize_processing_stats():
    """Initialize statistics tracking for image processing."""
    return {
        "boxes_per_image": [],
        "classes_per_image": [],
        "difficult_per_image": [],
        "images_processed": 0,
        "xml_read_success": 0,
        "images_saved": 0,
        "images_displayed": 0,
        "total_difficult_objs": 0,
    }


def _process_images(
    ids_to_process,
    base_voc_root,
    output_dir,
    class_name_to_id_map,
    do_save,
    do_display,
    show_difficult,
):
    """Process a list of images and collect statistics."""
    stats = _initialize_processing_stats()

    logger.info("Starting visualization processing...")
    for image_id, year, tag in tqdm(ids_to_process, desc="Processing Images"):
        stats["images_processed"] += 1

        xml_success, num_boxes, num_classes, num_difficult, saved, displayed = (
            process_and_visualize_image(
                image_id,
                year,
                tag,
                base_voc_root,
                output_dir,
                class_name_to_id_map,
                do_save,
                do_display,
                show_difficult,
            )
        )

        if saved:
            stats["images_saved"] += 1
        if displayed:
            stats["images_displayed"] += 1

        if xml_success:
            stats["xml_read_success"] += 1
            # Append stats only if XML was successfully processed
            if num_boxes is not None:
                stats["boxes_per_image"].append(num_boxes)
            if num_classes is not None:
                stats["classes_per_image"].append(num_classes)
            if num_difficult is not None:
                stats["difficult_per_image"].append(num_difficult)
                stats["total_difficult_objs"] += num_difficult

    return stats


def main():
    """Main execution function."""
    load_dotenv()
    args = parse_arguments()

    try:
        # --- Path Setup ---
        voc_root_env = os.getenv("VOC_ROOT")
        base_voc_root, voc_devkit_root, _, output_dir = _setup_paths(args, voc_root_env)

        # --- Identify Images ---
        ids_to_process = get_target_image_list(args, base_voc_root, voc_devkit_root)
        if not ids_to_process:
            return  # Exit cleanly if no images found

        # --- Determine Output Actions ---
        is_single_image_mode = args.image_id is not None
        do_display = is_single_image_mode
        do_save = not is_single_image_mode

        # --- Initialize Class Mapping ---
        class_name_to_id_map = {name: i for i, name in enumerate(VOC_CLASSES)}

        # --- Process Images ---
        stats = _process_images(
            ids_to_process,
            base_voc_root,
            output_dir,
            class_name_to_id_map,
            do_save,
            do_display,
            args.show_difficult,
        )

        # --- Statistics Reporting ---
        report_statistics(
            args,
            stats["boxes_per_image"],
            stats["classes_per_image"],
            stats["difficult_per_image"],
            len(ids_to_process),
            stats["images_processed"],
            stats["xml_read_success"],
            stats["images_saved"],
            stats["images_displayed"],
            stats["total_difficult_objs"],
        )

    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Path setup failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


def report_statistics(
    args: argparse.Namespace,
    boxes_per_image: List[int],
    classes_per_image: List[int],
    difficult_per_image: List[int],
    num_targeted: int,
    num_processed: int,
    num_xml_success: int,
    num_saved: int,
    num_displayed: int,
    total_difficult: int,
) -> None:
    """Calculates and logs processing and annotation statistics."""
    logger.info("--- Processing Complete ---")
    logger.info(f"Total images targeted: {num_targeted}")
    logger.info(f"Images processed attempt: {num_processed}")
    logger.info(f"XML annotations successfully read: {num_xml_success}")
    logger.info(f"Images saved: {num_saved}")
    logger.info(f"Images displayed: {num_displayed}")

    if boxes_per_image:
        boxes_arr = np.array(boxes_per_image)
        classes_arr = np.array(classes_per_image)
        difficult_arr = np.array(difficult_per_image)
        logger.info("--- Annotation Statistics (per successfully processed XML) ---")
        if args.percentiles:
            try:
                percentiles_str = args.percentiles.split(",")
                percentiles = [float(p.strip()) * 100 for p in percentiles_str]
                if not all(0 <= p <= 100 for p in percentiles):
                    raise ValueError("Percentiles must be between 0 and 1.")

                box_percentiles = np.percentile(boxes_arr, percentiles)
                class_percentiles = np.percentile(classes_arr, percentiles)
                difficult_percentiles = np.percentile(difficult_arr, percentiles)
                logger.info(f"Percentiles requested: {[f'{p / 100:.2f}' for p in percentiles]}")
                logger.info(f"Box count percentiles: {np.round(box_percentiles, 2).tolist()}")
                logger.info(
                    f"Unique class count percentiles: {np.round(class_percentiles, 2).tolist()}"
                )
                logger.info(
                    "Difficult obj count percentiles: "
                    f"{np.round(difficult_percentiles, 2).tolist()}"
                )
            except ValueError as e:
                logger.error(
                    f"Invalid format or value for --percentiles: {e}. "
                    "Use comma-separated floats between 0 and 1. Reporting averages instead."
                )
                # Fallback to average if percentiles are invalid
                logger.info(f"Average boxes per image: {boxes_arr.mean():.2f}")
                logger.info(f"Average unique classes per image: {classes_arr.mean():.2f}")
                logger.info(f"Average difficult objects per image: {difficult_arr.mean():.2f}")
        else:
            # Report averages if percentiles not requested
            logger.info(f"Average boxes per image: {boxes_arr.mean():.2f}")
            logger.info(f"Average unique classes per image: {classes_arr.mean():.2f}")
            logger.info(f"Average difficult objects per image: {difficult_arr.mean():.2f}")
    else:
        logger.info("No annotation statistics generated (no XMLs successfully processed).")
    logger.info(f"Total difficult objects found across all processed XMLs: {total_difficult}")


if __name__ == "__main__":
    main()
