#!/usr/bin/env python3
"""
Visualize YOLO Format Segmentation Labels.

Reads YOLO segmentation label files (.txt) and the corresponding images,
then draws the polygons and class labels onto the images.
Supports processing single images (with display) or batches (saving images).
Optionally calculates and reports statistics on annotations per image.

Expected dataset structure:
    <dataset_root>/<dataset_name>/images/<tag>[<year>]/<image_id>.jpg
    <dataset_root>/<dataset_name>/labels/<tag>[<year>]/<image_id>.txt
Where [<year>] is optional. If --years is not provided, it expects:
    <dataset_root>/<dataset_name>/images/<tag>/<image_id>.jpg
    <dataset_root>/<dataset_name>/labels/<tag>/<image_id>.txt

Output structure:
    <output_root>/<output_subdir>/<tag>[<year>]/<image_id>.png
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
from src.utils.common.image_annotate import (
    draw_polygon,
    get_color,
    overlay_mask,
)
from src.utils.data_converter.voc2yolo_utils import VOC_CLASSES  # Reverted to correct import

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize YOLO format segmentation labels.")

    parser.add_argument(
        "--ds-root-path",
        type=str,
        default=None,
        help=("Direct path to the dataset root directory. Overrides environment variable if set."),
    )
    parser.add_argument(
        "--ds-root-env",
        type=str,
        default="VOC_ROOT",  # Keep VOC_ROOT as default for backward compatibility
        help=(
            "Name of the environment variable containing the dataset root path "
            "(e.g., 'VOC_ROOT', 'COV_SEGM_ROOT'). Used if --ds-root-path is not set."
        ),
    )
    parser.add_argument(
        "--ds-subname",
        type=str,
        required=True,
        help="Subdirectory name for the dataset within the root (e.g., 'segment', 'coco').",
    )

    parser.add_argument(
        "--years",
        type=str,
        default=None,  # Make year optional
        help=(
            "Optional: Comma-separated list of dataset years (e.g., '2007', '2007,2012'). "
            "If provided, expects folder structure '<tag><year>'. "
            "If not provided, expects folder structure '<tag>'."
        ),
    )
    parser.add_argument(
        "--tags",
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
        "--output-root",
        type=str,
        default=None,
        help="Root directory for saving visualizations. Defaults to dataset-root.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="visual",  # Default output subfolder name, relative to output-root
        help=(
            "Subdirectory within output-root to save visualizations "
            "(e.g., 'segment/visual', 'coco/visual')."
        ),
    )
    parser.add_argument(
        "--fill-polygons",
        action="store_true",
        help="If set, fills polygons with semi-transparent color. Otherwise, only draws outlines.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Alpha (transparency) value for filled polygons (0-1). Default: 0.3",
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")

    return parser.parse_args()


def _setup_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    """Determine base dataset root and final output directory.

    Handles logic for using --ds-root-path, --ds-root-env, and --output-root arguments,
    including defaults and validation.

    Returns:
        Tuple[Path, Path]: base_dataset_root, output_dir

    Raises:
        ValueError: If dataset root cannot be determined or doesn't seem valid.
    """
    # Determine Dataset Root
    base_dataset_root_str = args.ds_root_path or os.getenv(args.ds_root_env)
    if not base_dataset_root_str:
        logger.error(
            f"Dataset root directory not specified via --ds-root-path or ${args.ds_root_env}."
        )
        raise ValueError("Dataset root not specified")

    base_dataset_root = Path(base_dataset_root_str).expanduser().resolve()

    # Basic check: Does it contain '<ds_subname>/images' and '<ds_subname>/labels'?
    expected_images_dir = base_dataset_root / args.ds_subname / "images"
    expected_labels_dir = base_dataset_root / args.ds_subname / "labels"
    if not expected_images_dir.is_dir() or not expected_labels_dir.is_dir():
        logger.warning(
            f"Base Dataset Root {base_dataset_root} does not contain expected "
            f"'{args.ds_subname}/images' and '{args.ds_subname}/labels' subdirectories."
        )
        # Continue anyway, maybe structure is different but paths will fail later if invalid

    logger.info(f"Using Base Dataset Root: {base_dataset_root}")

    # Determine Output Root
    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
        logger.info(f"Using specified output root: {output_root}")
    else:
        output_root = base_dataset_root  # Default output root to base_dataset_root
        logger.info(f"Using default output root (same as dataset-root): {output_root}")

    output_dir = output_root / args.output_subdir
    # Create the directory here for convenience
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Base output visualization directory: {output_dir}")

    return base_dataset_root, output_dir


def _get_image_ids_for_split(
    year: Optional[str], tag: str, base_dataset_root: Path, ds_subname: str
) -> List[Tuple[str, Optional[str], str]]:
    """Helper function to find image IDs for a specific tag (and optional year) split.

    Scans the label directory and checks for corresponding images.

    Returns:
        List of (image_id, year, tag) tuples found for this split. Year is None if not applicable.
    """
    split_name = f"{tag}{year}" if year else tag
    label_dir = base_dataset_root / ds_subname / "labels" / split_name
    image_dir = base_dataset_root / ds_subname / "images" / split_name
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
            missing_images += 1

    logger.info(f"Found {found_in_split} label/image pairs in {split_name}.")
    if missing_images > 0:
        logger.warning(
            f"Could not find corresponding images for {missing_images} labels in {split_name}."
        )
    return ids_found


def get_target_image_list(
    args: argparse.Namespace, base_dataset_root: Path
) -> List[Tuple[str, Optional[str], str]]:
    """Determines the list of (image_id, year, tag) tuples to process
    by scanning label directories and checking for corresponding images."""
    ids_to_process = []
    years = (
        [y.strip() for y in args.years.split(",") if y.strip()] if args.years else [None]
    )  # Use [None] if years not specified
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    if not tags:
        logger.error("No valid tags provided.")
        return []

    if args.image_id:
        # Single image mode
        first_tag = tags[0]
        first_year = years[0]  # This will be None if args.years is None
        split_name = f"{first_tag}{first_year}" if first_year else first_tag

        logger.info(f"Single image mode: Checking {args.image_id} from split '{split_name}'")
        image_path = (
            base_dataset_root / args.ds_subname / "images" / split_name / f"{args.image_id}.jpg"
        )
        label_path = (
            base_dataset_root / args.ds_subname / "labels" / split_name / f"{args.image_id}.txt"
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
                split_ids = _get_image_ids_for_split(year, tag, base_dataset_root, args.ds_subname)
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


def parse_yolo_segmentation_label(
    label_path: Path, img_width: int, img_height: int, class_names: List[str]
) -> Optional[List[Dict]]:
    """Parses a YOLO segmentation label file.

    Args:
        label_path: Path to the YOLO .txt label file.
        img_width: Width of the corresponding image.
        img_height: Height of the corresponding image.
        class_names: List of class names, indexed by class_id.

    Returns:
        List of dictionaries, each containing 'name', 'points' (list of [x,y] coords)
        and 'points_flat' (for drawing), or None if file cannot be read or parsed.
    """
    objects_list = []
    try:
        with open(label_path, "r") as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    if len(parts) < 7:  # Minimum: class_id + at least 3 points (3 x,y pairs)
                        logger.warning(f"Skipping invalid line in {label_path}: {line.strip()}")
                        continue

                    class_id = int(parts[0])
                    coordinates = [float(p) for p in parts[1:]]

                    # Must have even number of coordinates (x,y pairs)
                    if len(coordinates) % 2 != 0:
                        logger.warning(
                            f"Skipping line with odd number of coordinates in {label_path}: {line.strip()}"
                        )
                        continue

                    # Must have at least 3 points
                    if len(coordinates) < 6:  # 3 x,y pairs
                        logger.warning(
                            f"Skipping polygon with < 3 points in {label_path}: {line.strip()}"
                        )
                        continue

                    # Prepare points for drawing - convert to pixel coordinates
                    points_pixel = []
                    for i in range(0, len(coordinates), 2):
                        x_norm, y_norm = coordinates[i], coordinates[i + 1]
                        # Validate coordinates are within [0-1] range
                        if not (0 <= x_norm <= 1 and 0 <= y_norm <= 1):
                            logger.warning(
                                f"Invalid normalized coordinates ({x_norm}, {y_norm}) in {label_path}"
                            )
                            continue
                        # Convert to pixel coordinates
                        x_px = int(x_norm * img_width)
                        y_px = int(y_norm * img_height)
                        points_pixel.append((x_px, y_px))

                    # Skip if we don't have at least 3 valid points after validation
                    if len(points_pixel) < 3:
                        logger.warning(
                            f"Skipping polygon with < 3 valid points after validation in {label_path}"
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

                    # Create mask for filled polygon if needed
                    mask = None
                    points_array = np.array(points_pixel)

                    objects_list.append(
                        {
                            "name": class_name,
                            "points": points_pixel,
                            "points_array": points_array,
                            "original_coords": coordinates,
                        }
                    )

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
    year: Optional[str],  # Year can be None
    tag: str,
    base_dataset_root: Path,  # Renamed from voc_root
    output_dir: Path,  # This is <output_root>/<output_subdir>
    ds_subname: str,  # Added dataset subname
    class_names: List[str],
    do_save: bool,
    do_display: bool,
    fill_polygons: bool,
    alpha: float = 0.3,
) -> Tuple[bool, Optional[int], Optional[int], Optional[List[int]], bool, bool]:
    """Loads image, parses YOLO label, draws polygons, and saves/displays.

    Returns:
        Tuple: (label_parse_success, num_polygons, num_classes, points_per_polygon,
                save_success, display_success)
               Returns counts as None if label processing failed.
    """
    save_success = False
    display_success = False
    split_name = f"{tag}{year}" if year else tag
    image_path = base_dataset_root / ds_subname / "images" / split_name / f"{image_id}.jpg"
    label_path = base_dataset_root / ds_subname / "labels" / split_name / f"{image_id}.txt"

    try:
        # Load Image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Failed to load image: {image_path}. Skipping.")
            return False, None, None, None, save_success, display_success
        img_height, img_width = image.shape[:2]

        # Parse YOLO Label
        objects_list = parse_yolo_segmentation_label(label_path, img_width, img_height, class_names)
        if objects_list is None:
            logger.warning(
                f"Failed to parse or no valid objects in label: {label_path}. Skipping drawing."
            )
            # Still return False for label_parse_success, but 0 counts for stats
            return False, 0, 0, [], save_success, display_success

        # Collect stats
        num_polygons = len(objects_list)
        num_classes = len(set(o["name"] for o in objects_list))
        points_per_polygon = [len(o["points"]) for o in objects_list]

        # Draw Annotations
        image_to_draw = image.copy()
        class_name_to_id_map = {name: i for i, name in enumerate(class_names)}

        for obj in objects_list:
            class_name = obj["name"]
            points = obj["points_array"]  # Already in pixel format

            # Get color based on consistent class ID map
            class_id = class_name_to_id_map.get(class_name, -1)
            color = get_color(class_id if class_id != -1 else None)

            label_text = class_name

            if fill_polygons:
                # Create a binary mask for this polygon
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                # Convert points to the format expected by fillPoly (list of points arrays)
                cv2.fillPoly(mask, [points], 1)
                # Overlay the mask with transparency
                overlay_mask(image_to_draw, mask, label_text, color, alpha=alpha)
            else:
                # Just draw the polygon outline with label
                draw_polygon(image_to_draw, points, label_text, color=color)

        # Save or Display
        if do_save:
            save_subdir = (
                output_dir / split_name
            )  # Output is <output_root>/<output_subdir>/<split_name>/
            save_subdir.mkdir(parents=True, exist_ok=True)
            save_path = save_subdir / f"{image_id}.png"  # Save as <id>.png
            try:
                cv2.imwrite(str(save_path), image_to_draw)
                save_success = True
            except Exception as e:
                logger.error(f"Failed to save image {save_path}: {e}")

        if do_display:
            try:
                cv2.imshow(f"YOLO Segment Viz - {image_id}", image_to_draw)
                logger.info(f"Displaying image {image_id}. Press any key to continue...")
                cv2.waitKey(0)
                display_success = True
            except Exception as e:
                logger.error(f"Failed to display image {image_id}: {e}")
            finally:
                cv2.destroyAllWindows()

        # Return status and stats
        return True, num_polygons, num_classes, points_per_polygon, save_success, display_success

    except FileNotFoundError as e:
        # This might catch the image file not found if the check in get_target_image_list somehow
        # missed it
        logger.error(f"File not found error for {image_id} in split {split_name}: {e}. Skipping.")
        return False, None, None, None, save_success, display_success
    except Exception as e:
        logger.error(
            f"Unexpected error processing {image_id} in split {split_name}: {e}", exc_info=True
        )
        return False, None, None, None, save_success, display_success


def _initialize_stats():
    """Create empty statistics tracking structures."""
    return {
        "polygons_per_image": [],
        "classes_per_image": [],
        "points_per_polygon": [],
        "images_processed": 0,
        "label_read_success": 0,
        "images_saved": 0,
        "images_displayed": 0,
    }


def _report_statistics(
    args: argparse.Namespace,
    stats: Dict,
    num_targeted: int,
) -> None:
    """Calculates and logs processing and annotation statistics."""
    logger.info("--- Processing Complete ---")
    logger.info(f"Total image/label pairs targeted: {num_targeted}")
    logger.info(f"Images processed attempt: {stats['images_processed']}")
    logger.info(f"YOLO labels successfully read and parsed: {stats['label_read_success']}")
    logger.info(f"Images saved: {stats['images_saved']}")
    logger.info(f"Images displayed: {stats['images_displayed']}")

    if stats["polygons_per_image"]:
        polygons_arr = np.array(stats["polygons_per_image"])
        classes_arr = np.array(stats["classes_per_image"])
        points_arr = np.array(stats["points_per_polygon"])

        logger.info("--- Annotation Statistics (per successfully processed label) ---")
        if args.percentiles:
            try:
                percentiles_str = args.percentiles.split(",")
                percentiles = [float(p.strip()) * 100 for p in percentiles_str]
                if not all(0 <= p <= 100 for p in percentiles):
                    raise ValueError("Percentiles must be between 0 and 1.")

                polygon_percentiles = np.percentile(polygons_arr, percentiles)
                class_percentiles = np.percentile(classes_arr, percentiles)
                points_percentiles = np.percentile(points_arr, percentiles)

                logger.info(f"Percentiles requested: {[f'{p / 100:.2f}' for p in percentiles]}")
                logger.info(
                    f"Polygon count percentiles: {np.round(polygon_percentiles, 2).tolist()}"
                )
                logger.info(
                    f"Unique class count percentiles: {np.round(class_percentiles, 2).tolist()}"
                )
                logger.info(
                    f"Points per polygon percentiles: {np.round(points_percentiles, 2).tolist()}"
                )
            except ValueError as e:
                logger.error(
                    f"Invalid format or value for --percentiles: {e}. "
                    "Use comma-separated floats between 0 and 1. Reporting averages instead."
                )
                # Fallback to average if percentiles are invalid
                logger.info(f"Average polygons per image: {polygons_arr.mean():.2f}")
                logger.info(f"Average unique classes per image: {classes_arr.mean():.2f}")
                logger.info(f"Average points per polygon: {points_arr.mean():.2f}")
        else:
            # Report averages if percentiles not requested
            logger.info(f"Average polygons per image: {polygons_arr.mean():.2f}")
            logger.info(f"Average unique classes per image: {classes_arr.mean():.2f}")
            logger.info(f"Average points per polygon: {points_arr.mean():.2f}")
    else:
        logger.info("No annotation statistics generated (no labels successfully processed).")


def _process_images(
    ids_to_process: List[Tuple[str, Optional[str], str]],  # Year is optional
    base_dataset_root: Path,  # Renamed
    output_dir: Path,  # Base output dir
    ds_subname: str,  # Added
    class_names: List[str],
    do_display: bool,
    do_save: bool,
    fill_polygons: bool,
    alpha: float,
) -> Dict:
    """Process and visualize images based on the target list."""
    stats = _initialize_stats()

    for image_id, year, tag in tqdm(ids_to_process, desc="Processing Images"):
        stats["images_processed"] += 1

        label_success, num_polygons, num_classes, points_per_polygon, saved, displayed = (
            process_and_visualize_image(
                image_id,
                year,  # Pass year (can be None)
                tag,
                base_dataset_root,  # Pass renamed root
                output_dir,  # Pass base output dir
                ds_subname,  # Pass dataset subname
                class_names,
                do_save,
                do_display,
                fill_polygons,
                alpha,
            )
        )

        if saved:
            stats["images_saved"] += 1
        if displayed:
            stats["images_displayed"] += 1

        if label_success:
            stats["label_read_success"] += 1
            # Append stats only if label was successfully processed
            if num_polygons is not None:
                stats["polygons_per_image"].append(num_polygons)
            if num_classes is not None:
                stats["classes_per_image"].append(num_classes)
            if points_per_polygon:
                stats["points_per_polygon"].extend(points_per_polygon)

    return stats


def main():
    """Main execution function."""
    load_dotenv()
    args = parse_arguments()

    try:
        # --- Path Setup ---
        base_dataset_root, output_dir = _setup_paths(args)  # Get base paths

        # --- Identify Images ---
        ids_to_process = get_target_image_list(args, base_dataset_root)
        if not ids_to_process:
            logger.info("No images found to process based on arguments.")
            return  # Exit cleanly

        # --- Determine Output Actions ---
        is_single_image_mode = args.image_id is not None
        do_display = is_single_image_mode  # Display only in single image mode
        do_save = not is_single_image_mode  # Save only in batch mode

        # --- Processing Loop ---
        logger.info("Starting visualization processing...")
        class_names = VOC_CLASSES  # Load class names

        stats = _process_images(
            ids_to_process,
            base_dataset_root,  # Pass renamed root
            output_dir,  # Pass base output dir
            args.ds_subname,  # Pass dataset subname
            class_names,
            do_display,
            do_save,
            args.fill_polygons,
            args.alpha,
        )

        # --- Statistics Reporting ---
        _report_statistics(
            args,
            stats,
            len(ids_to_process),
        )

    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Setup or file finding failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during main execution: {e}", exc_info=True)


if __name__ == "__main__":
    main()
