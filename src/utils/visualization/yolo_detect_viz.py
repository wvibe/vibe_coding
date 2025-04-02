#!/usr/bin/env python3
"""
Visualize YOLO Format Detection Labels.

Reads YOLO detection label files (.txt) and the corresponding images,
then draws the bounding boxes and class labels onto the images.
Supports processing single images (with display) or batches (saving images).
Optionally calculates and reports statistics on annotations per image.

Expects dataset structure:
    <voc_root>/images/<tag><year>/<image_id>.jpg
    <voc_root>/labels_detect/<tag><year>/<image_id>.txt
Output structure:
    <output_root>/visual_detect/<tag><year>/<image_id>.png
"""

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

# Project utilities
from src.utils.common.image_annotate import draw_box, get_color
from src.utils.data_converter.voc2yolo_utils import VOC_CLASSES  # Reverted to correct import

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize YOLO format detection labels.")

    parser.add_argument(
        "--year",
        type=str,
        required=True,
        help=(
            "Comma-separated list of dataset years (e.g., '2007', '2007,2012'). Used to find input"
            " folders."
        ),
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help=(
            "Comma-separated list of dataset tags (e.g., 'train', 'val'). Used to find input"
            " folders."
        ),
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
            "Path to the VOC dataset root directory (containing images/, labels_detect/). "
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
        default="visual_detect",  # Default output subfolder name
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
    # Removed --show-difficult argument
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")

    return parser.parse_args()


def _setup_paths(args: argparse.Namespace, voc_root_env: Optional[str]) -> Tuple[Path, Path, Path]:
    """Determine base VOC root, output root, and final output directory.

    Handles logic for using --voc-root, $VOC_ROOT, and --output-root arguments,
    including defaults and validation.

    Returns:
        Tuple[Path, Path, Path]: base_voc_root, output_root, output_dir

    Raises:
        ValueError: If VOC root cannot be determined or doesn't seem valid.
    """
    base_voc_root_str = args.voc_root or voc_root_env
    if not base_voc_root_str:
        logger.error("VOC root directory not specified via --voc-root or $VOC_ROOT.")
        raise ValueError("VOC root not specified")

    base_voc_root = Path(base_voc_root_str).expanduser().resolve()

    # Basic check: Does it contain 'images' and 'labels_detect'?
    if not (base_voc_root / "images").is_dir() or not (base_voc_root / "labels_detect").is_dir():
        logger.warning(
            f"Base VOC Root {base_voc_root} does not contain expected 'images' and 'labels_detect'"
            f" subdirectories."
        )
        # Continue anyway, maybe structure is different but paths will fail later if invalid

    logger.info(f"Using Base VOC Root: {base_voc_root}")

    # Determine Output Root
    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
        logger.info(f"Using specified output root: {output_root}")
    else:
        output_root = base_voc_root  # Default output root to base_voc_root
        logger.info(f"Using default output root (same as voc-root): {output_root}")

    output_dir = output_root / args.output_subdir
    # Create the directory here for convenience
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output visualization directory: {output_dir}")

    return base_voc_root, output_root, output_dir


def _get_image_ids_for_split(
    year: str, tag: str, base_voc_root: Path
) -> List[Tuple[str, str, str]]:
    """Helper function to find image IDs for a specific year/tag split.

    Scans the label directory and checks for corresponding images.

    Returns:
        List of (image_id, year, tag) tuples found for this split.
    """
    tag_year = f"{tag}{year}"
    label_dir = base_voc_root / "labels_detect" / tag_year
    image_dir = base_voc_root / "images" / tag_year
    ids_found = []

    if not label_dir.is_dir():
        logger.warning(f"Label directory not found, skipping split: {label_dir}")
        return []

    logger.info(f"Scanning for labels in: {label_dir}")
    found_in_split = 0
    missing_images = 0
    for label_file in label_dir.glob("*.txt"):
        image_id = label_file.stem
        expected_image_path = image_dir / f"{image_id}.jpg"
        if expected_image_path.is_file():
            ids_found.append((image_id, year, tag))
            found_in_split += 1
        else:
            # logger.warning(f"Label found ({label_file}) but image missing: {expected_image_path}")
            missing_images += 1

    logger.info(f"Found {found_in_split} label/image pairs in {tag_year}.")
    if missing_images > 0:
        logger.warning(
            f"Could not find corresponding images for {missing_images} labels in {tag_year}."
        )
    return ids_found


def get_target_image_list(
    args: argparse.Namespace, base_voc_root: Path
) -> List[Tuple[str, str, str]]:
    """Determines the list of (image_id, year, tag) tuples to process
    by scanning label directories and checking for corresponding images."""
    ids_to_process = []
    years = [y.strip() for y in args.year.split(",") if y.strip()]
    tags = [t.strip() for t in args.tag.split(",") if t.strip()]

    if not years or not tags:
        logger.error("No valid years or tags provided.")
        return []

    if args.image_id:
        # Single image mode - Keep this part simple
        first_year = years[0]
        first_tag = tags[0]
        logger.info(f"Single image mode: Checking {args.image_id} from {first_tag}{first_year}")
        image_path = base_voc_root / "images" / f"{first_tag}{first_year}" / f"{args.image_id}.jpg"
        label_path = (
            base_voc_root / "labels_detect" / f"{first_tag}{first_year}" / f"{args.image_id}.txt"
        )

        if not image_path.is_file():
            logger.error(f"Image file not found: {image_path}")
            return []
        if not label_path.is_file():
            logger.error(f"Label file not found: {label_path}")
            return []

        ids_to_process = [(args.image_id, first_year, first_tag)]
    else:
        # Batch mode (all or sampled) - Use helper function
        all_ids_found = []
        for year in years:
            for tag in tags:
                split_ids = _get_image_ids_for_split(year, tag, base_voc_root)
                all_ids_found.extend(split_ids)

        if not all_ids_found:
            logger.error("No valid label/image pairs found for any specified year/tag combination.")
            return []

        logger.info(
            f"Found {len(all_ids_found)} total image IDs with labels across specified splits."
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


def parse_yolo_detection_label(
    label_path: Path, img_width: int, img_height: int, class_names: List[str]
) -> Optional[List[Dict]]:
    """Parses a YOLO detection label file.

    Args:
        label_path: Path to the YOLO .txt label file.
        img_width: Width of the corresponding image.
        img_height: Height of the corresponding image.
        class_names: List of class names, indexed by class_id.

    Returns:
        List of dictionaries, each containing 'name' and 'bbox' [xmin, ymin, xmax, ymax],
        or None if the file cannot be read or parsed.
    """
    objects_list = []
    try:
        with open(label_path, "r") as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        logger.warning(f"Skipping invalid line in {label_path}: {line.strip()}")
                        continue

                    class_id = int(parts[0])
                    cx_norm = float(parts[1])
                    cy_norm = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])

                    # Denormalize
                    cx = cx_norm * img_width
                    cy = cy_norm * img_height
                    w = w_norm * img_width
                    h = h_norm * img_height

                    # Convert to xmin, ymin, xmax, ymax
                    xmin = int(round(cx - w / 2))
                    ymin = int(round(cy - h / 2))
                    xmax = int(round(cx + w / 2))
                    ymax = int(round(cy + h / 2))

                    # Clamp coordinates to image bounds
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(img_width - 1, xmax)
                    ymax = min(img_height - 1, ymax)

                    if xmin >= xmax or ymin >= ymax:
                        logger.warning(
                            f"Skipping invalid box dimensions in {label_path} after clamping:"
                            f" class={class_id}, box=[{xmin},{ymin},{xmax},{ymax}]"
                        )
                        continue

                    # Get class name
                    if 0 <= class_id < len(class_names):
                        class_name = class_names[class_id]
                    else:
                        logger.warning(
                            f"Invalid class ID {class_id} in {label_path}. Using 'Unknown'."
                        )
                        class_name = "Unknown"

                    objects_list.append({"name": class_name, "bbox": [xmin, ymin, xmax, ymax]})
                except ValueError as e:
                    logger.warning(f"Skipping malformed line in {label_path}: {line.strip()} - {e}")
                    continue
    except FileNotFoundError:
        logger.error(f"Label file not found: {label_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading or parsing label file {label_path}: {e}", exc_info=True)
        return None

    return objects_list


def process_and_visualize_image(
    image_id: str,
    year: str,
    tag: str,
    voc_root: Path,
    output_dir: Path,  # This is <output_root>/visual_detect
    class_names: List[str],
    do_save: bool,
    do_display: bool,
) -> Tuple[bool, Optional[int], Optional[int], bool, bool]:
    """Loads image, parses YOLO label, draws boxes, and saves/displays.

    Returns:
        Tuple: (label_parse_success, num_boxes, num_classes, save_success, display_success)
               Returns counts as None if label processing failed.
    """
    save_success = False
    display_success = False
    tag_year = f"{tag}{year}"
    image_path = voc_root / "images" / tag_year / f"{image_id}.jpg"
    label_path = voc_root / "labels_detect" / tag_year / f"{image_id}.txt"

    try:
        # Load Image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Failed to load image: {image_path}. Skipping.")
            return False, None, None, save_success, display_success
        img_height, img_width = image.shape[:2]

        # Parse YOLO Label
        objects_list = parse_yolo_detection_label(label_path, img_width, img_height, class_names)
        if objects_list is None:
            logger.warning(
                f"Failed to parse or no valid objects in label: {label_path}. Skipping drawing."
            )
            # Still return False for label_parse_success, but 0 counts for stats
            return False, 0, 0, save_success, display_success

        # Collect stats
        num_boxes = len(objects_list)
        num_classes = len(set(o["name"] for o in objects_list))

        # Draw Annotations
        image_to_draw = image.copy()
        class_name_to_id_map = {name: i for i, name in enumerate(class_names)}

        for obj in objects_list:
            class_name = obj["name"]
            box = obj["bbox"]  # Already in [xmin, ymin, xmax, ymax] pixel format

            # Get color based on consistent class ID map
            class_id = class_name_to_id_map.get(class_name, -1)
            color = get_color(class_id if class_id != -1 else None)

            label_text = class_name  # No difficult flag

            draw_box(image_to_draw, box, label_text, color=color)

        # Save or Display
        if do_save:
            save_subdir = (
                output_dir / tag_year
            )  # Output is <output_root>/visual_detect/<tag><year>/
            save_subdir.mkdir(parents=True, exist_ok=True)
            save_path = save_subdir / f"{image_id}.png"  # Save as <id>.png
            try:
                cv2.imwrite(str(save_path), image_to_draw)
                save_success = True
            except Exception as e:
                logger.error(f"Failed to save image {save_path}: {e}")

        if do_display:
            try:
                cv2.imshow(f"YOLO Detect Viz - {image_id}", image_to_draw)
                logger.info(f"Displaying image {image_id}. Press any key to continue...")
                cv2.waitKey(0)
                display_success = True
            except Exception as e:
                logger.error(f"Failed to display image {image_id}: {e}")
            finally:
                cv2.destroyAllWindows()

        # Return status and stats
        return True, num_boxes, num_classes, save_success, display_success

    except FileNotFoundError as e:
        # This might catch the image file not found if the check in get_target_image_list somehow
        # missed it
        logger.error(f"File not found error for {image_id}: {e}. Skipping.")
        return False, None, None, save_success, display_success
    except Exception as e:
        logger.error(f"Unexpected error processing {image_id}: {e}", exc_info=True)
        return False, None, None, save_success, display_success


def main():
    """Main execution function."""
    load_dotenv()
    args = parse_arguments()

    try:
        # --- Path Setup ---
        voc_root_env = os.getenv("VOC_ROOT")
        base_voc_root, _, output_dir = _setup_paths(
            args, voc_root_env
        )  # output_root not needed here

        # --- Identify Images ---
        ids_to_process = get_target_image_list(args, base_voc_root)
        if not ids_to_process:
            logger.info("No images found to process based on arguments.")
            return  # Exit cleanly

        # --- Determine Output Actions ---
        is_single_image_mode = args.image_id is not None
        do_display = is_single_image_mode  # Display only in single image mode
        do_save = not is_single_image_mode  # Save only in batch mode

        # --- Initialize Stats & Resources ---
        boxes_per_image: List[int] = []
        classes_per_image: List[int] = []
        images_processed = 0
        label_read_success = 0
        images_saved = 0
        images_displayed = 0
        class_names = VOC_CLASSES  # Load class names

        # --- Processing Loop ---
        logger.info("Starting visualization processing...")
        for image_id, year, tag in tqdm(ids_to_process, desc="Processing Images"):
            images_processed += 1

            label_success, num_boxes, num_classes, saved, displayed = process_and_visualize_image(
                image_id,
                year,
                tag,
                base_voc_root,
                output_dir,  # Pass the final output dir: <output_root>/visual_detect
                class_names,
                do_save,
                do_display,
            )

            if saved:
                images_saved += 1
            if displayed:
                images_displayed += 1

            if label_success:
                label_read_success += 1
                # Append stats only if label was successfully processed
                if num_boxes is not None:
                    boxes_per_image.append(num_boxes)
                if num_classes is not None:
                    classes_per_image.append(num_classes)

        # --- Statistics Reporting ---
        report_statistics(
            args,
            boxes_per_image,
            classes_per_image,
            len(ids_to_process),
            images_processed,
            label_read_success,
            images_saved,
            images_displayed,
        )

    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Setup or file finding failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during main execution: {e}", exc_info=True)


def report_statistics(
    args: argparse.Namespace,
    boxes_per_image: List[int],
    classes_per_image: List[int],
    num_targeted: int,
    num_processed: int,
    num_label_success: int,
    num_saved: int,
    num_displayed: int,
) -> None:
    """Calculates and logs processing and annotation statistics."""
    logger.info("--- Processing Complete ---")
    logger.info(f"Total image/label pairs targeted: {num_targeted}")
    logger.info(f"Images processed attempt: {num_processed}")
    logger.info(f"YOLO labels successfully read and parsed: {num_label_success}")
    logger.info(f"Images saved: {num_saved}")
    logger.info(f"Images displayed: {num_displayed}")

    if boxes_per_image:
        boxes_arr = np.array(boxes_per_image)
        classes_arr = np.array(classes_per_image)
        logger.info("--- Annotation Statistics (per successfully processed label) ---")
        if args.percentiles:
            try:
                percentiles_str = args.percentiles.split(",")
                percentiles = [float(p.strip()) * 100 for p in percentiles_str]
                if not all(0 <= p <= 100 for p in percentiles):
                    raise ValueError("Percentiles must be between 0 and 1.")

                box_percentiles = np.percentile(boxes_arr, percentiles)
                class_percentiles = np.percentile(classes_arr, percentiles)

                logger.info(f"Percentiles requested: {[f'{p / 100:.2f}' for p in percentiles]}")
                logger.info(f"Box count percentiles: {np.round(box_percentiles, 2).tolist()}")
                logger.info(
                    f"Unique class count percentiles: {np.round(class_percentiles, 2).tolist()}"
                )
            except ValueError as e:
                logger.error(
                    f"Invalid format or value for --percentiles: {e}. "
                    "Use comma-separated floats between 0 and 1. Reporting averages instead."
                )
                # Fallback to average if percentiles are invalid
                logger.info(f"Average boxes per image: {boxes_arr.mean():.2f}")
                logger.info(f"Average unique classes per image: {classes_arr.mean():.2f}")
        else:
            # Report averages if percentiles not requested
            logger.info(f"Average boxes per image: {boxes_arr.mean():.2f}")
            logger.info(f"Average unique classes per image: {classes_arr.mean():.2f}")
    else:
        logger.info("No annotation statistics generated (no labels successfully processed).")


if __name__ == "__main__":
    main()
