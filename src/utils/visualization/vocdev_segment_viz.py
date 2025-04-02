#!/usr/bin/env python3
"""
Visualize Pascal VOC Ground Truth Segmentation Masks.

Reads VOC segmentation mask files (`SegmentationObject`) and overlays them
onto the corresponding images from `VOCdevkit`.
Supports processing single images (with display) or batches (saving images).
Optionally calculates and reports statistics on instances per image.
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
from src.utils.common.image_annotate import (
    get_color,
    overlay_mask,  # Use this to draw masks
)
from src.utils.common.iou import calculate_iou  # Added for matching
from src.utils.data_converter.voc2yolo_utils import (
    get_annotation_path,  # Added for XML loading
    get_image_path,
    get_image_set_path,
    get_segmentation_mask_path,  # Use this for masks
    get_voc_dir,
    parse_voc_xml,  # Added for XML parsing
    read_image_ids,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize Pascal VOC ground truth segmentation masks."  # Updated desc
    )

    parser.add_argument(
        "--year",
        type=str,
        required=True,
        help="Comma-separated list of dataset years (e.g., '2007', '2007,2012').",
    )
    parser.add_argument(
        "--tag",
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
        help="Root directory for saving visualizations. Defaults to the VOCdevkit directory.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="visual_segment",  # Changed default subdir
        help="Subdirectory within output-root to save visualizations.",
    )
    parser.add_argument(
        "--percentiles",
        type=str,
        default=None,
        help=(
            "Comma-separated list of percentiles (0-1) for instance count stats "
            "(e.g., '0.25,0.5,0.75'). Reports average if not set."
        ),
    )
    # Removed --show-difficult argument
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
                imageset_path = get_image_set_path(voc_dir, set_type="segment", tag=tag)
                # Use read_image_ids from the imported utils module
                image_ids = read_image_ids(imageset_path)
                if image_ids:
                    all_ids_found.extend([(img_id, year, tag) for img_id in image_ids])
                else:
                    logger.warning(
                        f"No image IDs found for Segmentation {year}/{tag} in {imageset_path}"
                    )
            except FileNotFoundError:
                logger.error(
                    f"Could not find ImageSet file for Segmentation {year}/{tag}: {imageset_path}"
                )
            except Exception as e:
                logger.error(f"Error reading ImageSet for Segmentation {year}/{tag}: {e}")
    return all_ids_found


def get_target_image_list(
    args: argparse.Namespace, base_voc_root: Path, voc_devkit_dir: Path
) -> List[Tuple[str, str, str]]:
    """Determines the list of (image_id, year, tag) tuples to process."""
    ids_to_process = []
    years = [y.strip() for y in args.year.split(",") if y.strip()]
    tags = [t.strip() for t in args.tag.split(",") if t.strip()]

    if not years or not tags:
        logger.error("No valid years or tags provided.")
        return []

    if args.image_id:
        # Single image mode
        first_year = years[0]
        first_tag = tags[0]
        logger.info(f"Single image mode: Processing {args.image_id} from {first_year}/{first_tag}")
        # Basic path check for image and mask
        try:
            year_voc_dir = get_voc_dir(base_voc_root, first_year)
            img_path = get_image_path(year_voc_dir, args.image_id)
            mask_path = get_segmentation_mask_path(year_voc_dir, args.image_id)
            if not img_path.exists():
                logger.error(f"Image file not found: {img_path}")
                return []
            if not mask_path.exists():
                logger.error(f"Segmentation mask file not found: {mask_path}")
                return []
        except Exception as e:
            logger.error(f"Error checking paths for {args.image_id}: {e}")
            return []
        ids_to_process = [(args.image_id, first_year, first_tag)]
    else:
        # Batch mode (all or sampled)
        all_ids_found = _get_batch_image_ids(years, tags, base_voc_root)

        if not all_ids_found:
            logger.error(
                "No image IDs found for any specified year/tag combination using Segmentation sets."
            )
            return []

        logger.info(
            f"Found {len(all_ids_found)} total image IDs across specified Segmentation splits."
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
    do_save: bool,
    do_display: bool,
) -> Tuple[bool, Optional[int], bool, bool]:
    """Loads image and mask, draws masks, and saves/displays the image.

    Returns:
        Tuple: (mask_load_success, num_instances_found, save_success, display_success)
               Returns counts as None if mask loading failed.
    """
    save_success = False
    display_success = False
    num_instances_found: Optional[int] = None
    mask_load_success = False
    xml_load_success = False  # Track XML loading
    objects_from_xml = None  # Store parsed XML objects

    try:
        voc_dir = get_voc_dir(voc_root, year)
        image_path = get_image_path(voc_dir, image_id)
        mask_path = get_segmentation_mask_path(voc_dir, image_id)
        xml_path = get_annotation_path(voc_dir, image_id)  # Get XML path

        # Load Image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Failed to load image: {image_path}. Skipping.")
            return mask_load_success, num_instances_found, save_success, display_success

        # Load Mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning(f"Failed to load segmentation mask: {mask_path}. Skipping drawing.")
            # Still return False for mask_load_success, but 0 instances
            return False, 0, save_success, display_success

        mask_load_success = True  # Mark mask as successfully loaded

        # Load and Parse XML Annotation
        try:
            if xml_path.exists():
                objects_from_xml, _ = parse_voc_xml(xml_path)
                if objects_from_xml is not None:
                    xml_load_success = True
                    logger.debug(f"Successfully loaded and parsed XML: {xml_path}")
                else:
                    logger.warning(f"Failed to parse objects from XML: {xml_path}")
            else:
                logger.warning(
                    f"XML annotation not found: {xml_path}. Cannot determine class names."
                )
        except Exception as e:
            logger.error(f"Error loading/parsing XML {xml_path}: {e}")

        # Find unique instances (non-zero pixel values)
        instance_ids = np.unique(mask[mask > 0])
        num_instances_found = len(instance_ids)

        # Draw Annotations (Masks)
        image_to_draw = image.copy()
        if num_instances_found > 0:
            logger.debug(f"Found instances {instance_ids} in {mask_path}")
            for instance_id in instance_ids:
                # Create binary mask for the current instance (0/255 for overlay_mask)
                binary_mask = (mask == instance_id).astype(np.uint8) * 255

                # Get color based on instance ID
                color = get_color(instance_id)

                # --- Determine Class Name via IoU Matching ---
                class_name = "Unknown"
                best_iou = 0.0

                # Calculate bounding box for the current instance mask
                rows, cols = np.where(binary_mask > 0)
                mask_bbox = None
                if len(rows) > 0:
                    xmin, xmax = np.min(cols), np.max(cols)
                    ymin, ymax = np.min(rows), np.max(rows)
                    mask_bbox = np.array([xmin, ymin, xmax, ymax])

                if xml_load_success and objects_from_xml is not None and mask_bbox is not None:
                    for obj in objects_from_xml:
                        xml_bbox = np.array(obj["bbox"])  # Already [xmin, ymin, xmax, ymax]
                        iou = calculate_iou(mask_bbox, xml_bbox)
                        # Match if IoU is highest so far and above threshold
                        if iou > best_iou and iou >= 0.5:
                            best_iou = iou
                            class_name = obj["name"]

                    if class_name != "Unknown":
                        logger.debug(
                            f"Matched Inst {instance_id} to class '{class_name}' (IoU: {best_iou:.2f})"
                        )
                    else:
                        logger.debug(
                            f"Could not match Inst {instance_id} to XML object (best IoU: {best_iou:.2f})"
                        )

                # Format label including class name if found
                label_text = f"{class_name}.{instance_id}"

                # Overlay the mask with the updated label
                overlay_mask(
                    image_to_draw, binary_mask, label=label_text, color=color, alpha=0.3
                )  # Use new label
        else:
            logger.debug(f"No instances found in mask {mask_path}")

        # Save or Display
        if do_save:
            save_subdir = output_dir / f"{tag}{year}"
            save_subdir.mkdir(parents=True, exist_ok=True)
            # Updated filename suffix
            save_path = save_subdir / f"{image_id}_voc_segment.png"
            try:
                cv2.imwrite(str(save_path), image_to_draw)
                save_success = True
            except Exception as e:
                logger.error(f"Failed to save image {save_path}: {e}")

        if do_display:
            try:
                cv2.imshow(f"VOC Segment Viz - {image_id}", image_to_draw)
                logger.info(f"Displaying image {image_id}. Press any key to continue...")
                cv2.waitKey(0)
                display_success = True
            except Exception as e:
                logger.error(f"Failed to display image {image_id}: {e}")
            finally:
                cv2.destroyAllWindows()

        # Return status and stats
        return mask_load_success, num_instances_found, save_success, display_success

    except FileNotFoundError as e:
        logger.error(f"File not found error for {image_id}: {e}. Skipping.")
        return False, None, save_success, display_success
    except Exception as e:
        logger.error(f"Unexpected error processing {image_id}: {e}", exc_info=True)
        return False, None, save_success, display_success


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

        # --- Initialize Stats & Resources ---
        instances_per_image: List[int] = []
        images_processed = 0
        mask_read_success = 0
        images_saved = 0
        images_displayed = 0
        total_instances_found = 0

        # --- Processing Loop ---
        logger.info("Starting visualization processing...")
        for image_id, year, tag in tqdm(ids_to_process, desc="Processing Images"):
            images_processed += 1

            mask_success, num_instances, saved, displayed = process_and_visualize_image(
                image_id,
                year,
                tag,
                base_voc_root,  # Pass base root for finding year-specific VOC dir
                output_dir,
                do_save,
                do_display,
            )

            if saved:
                images_saved += 1
            if displayed:
                images_displayed += 1

            if mask_success:
                mask_read_success += 1
                if num_instances is not None:
                    instances_per_image.append(num_instances)
                    total_instances_found += num_instances

        # --- Statistics Reporting ---
        report_statistics(
            args,
            instances_per_image,
            len(ids_to_process),
            images_processed,
            mask_read_success,
            images_saved,
            images_displayed,
            total_instances_found,
        )

    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Path setup failed: {e}")
        # Optionally exit with non-zero status
        # sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        # sys.exit(1)


def report_statistics(
    args: argparse.Namespace,
    instances_per_image: List[int],  # Updated stats list name
    num_targeted: int,
    num_processed: int,
    num_mask_success: int,  # Updated stat name
    num_saved: int,
    num_displayed: int,
    total_instances: int,  # Updated stat name
) -> None:
    """Calculates and logs processing and annotation statistics."""
    logger.info("--- Processing Complete ---")
    logger.info(f"Total images targeted: {num_targeted}")
    logger.info(f"Images processed attempt: {num_processed}")
    logger.info(f"Segmentation masks successfully read: {num_mask_success}")  # Updated message
    logger.info(f"Images saved: {num_saved}")
    logger.info(f"Images displayed: {num_displayed}")

    if instances_per_image:
        instances_arr = np.array(instances_per_image)
        logger.info(
            "--- Instance Statistics (per successfully processed mask) ---"
        )  # Updated header
        if args.percentiles:
            try:
                percentiles_str = args.percentiles.split(",")
                percentiles = [float(p.strip()) * 100 for p in percentiles_str]
                if not all(0 <= p <= 100 for p in percentiles):
                    raise ValueError("Percentiles must be between 0 and 1.")

                instance_percentiles = np.percentile(
                    instances_arr, percentiles
                )  # Calc instance percentiles
                logger.info(f"Percentiles requested: {[f'{p / 100:.2f}' for p in percentiles]}")
                logger.info(
                    f"Instance count percentiles: {np.round(instance_percentiles, 2).tolist()}"
                )  # Log instance percentiles

            except ValueError as e:
                logger.error(
                    f"Invalid format or value for --percentiles: {e}. "
                    "Use comma-separated floats between 0 and 1. Reporting averages instead."
                )
                # Fallback to average
                logger.info(
                    f"Average instances per image: {instances_arr.mean():.2f}"
                )  # Log instance average
        else:
            # Report averages if percentiles not requested
            logger.info(
                f"Average instances per image: {instances_arr.mean():.2f}"
            )  # Log instance average
    else:
        logger.info(
            "No instance statistics generated (no masks successfully processed)."
        )  # Updated message
    logger.info(
        f"Total object instances found across all processed masks: {total_instances}"
    )  # Updated message


if __name__ == "__main__":
    main()
