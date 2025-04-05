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
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from scipy.stats import mode  # Add mode import
from tqdm import tqdm

# Project utilities
from src.utils.common.image_annotate import (
    get_color,
    overlay_mask,  # Use this to draw masks
)
from src.utils.data_converter.voc2yolo_utils import (
    VOC_CLASSES,  # Add this import
    get_image_path,
    get_image_set_path,
    get_segm_cls_mask_path,  # Updated name
    get_segm_inst_mask_path,  # Updated name
    get_voc_dir,
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
    # Add back the --show-difficult argument
    parser.add_argument(
        "--show-difficult",
        action="store_true",
        help=(
            "Visualize boundaries/difficult regions (value 255) with 'Unk' label. "
            "In VOC segmentation, 255 typically marks object boundaries or "
            "difficult-to-segment pixels."
        ),
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
    years = [y.strip() for y in args.years.split(",") if y.strip()]
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

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
            mask_path = get_segm_inst_mask_path(year_voc_dir, args.image_id)
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


def generate_instance_label(
    instance_id: int,
    instance_mask: np.ndarray,
    class_mask: Optional[np.ndarray],
    image_id: str = "",
) -> str:
    """Generate label text combining class name and instance ID from masks.

    Args:
        instance_id: Numeric identifier from instance mask
        instance_mask: 2D array from SegmentationObject mask
        class_mask: 2D array from SegmentationClass mask (optional)
        image_id: Image ID for error context

    Returns:
        Formatted label string "ClassName.InstanceID"
    """
    class_name = "Unknown"
    if class_mask is not None:
        try:
            # Get pixels belonging to this instance
            instance_pixels = instance_mask == instance_id
            class_pixels = class_mask[instance_pixels]

            # Filter to valid class pixels (0=background, 255=void/ignore)
            valid_class_pixels = class_pixels[(class_pixels > 0) & (class_pixels < 255)]

            if valid_class_pixels.size > 0:
                # Get the most common class ID for this instance
                mode_result = mode(valid_class_pixels, keepdims=False)
                if mode_result.count > 0:
                    class_id = mode_result.mode
                    logger.debug(
                        f"Instance {instance_id} in {image_id}: Most common class ID {class_id}"
                    )

                    # In PASCAL VOC, class IDs are 1-20, corresponding to the 20 classes
                    if 1 <= class_id <= len(VOC_CLASSES):
                        class_name = VOC_CLASSES[class_id - 1]
                        logger.debug(
                            f"Instance {instance_id} in {image_id}: Mapped to class '{class_name}'"
                        )
                    else:
                        logger.debug(
                            f"Instance {instance_id} in {image_id}: Invalid class ID {class_id}, "
                            f"not in range 1-{len(VOC_CLASSES)}"
                        )
        except Exception as e:
            logger.debug(f"Error processing instance {instance_id} in {image_id}: {e}")

    return f"{class_name}.{instance_id}"


def _load_image_and_masks(
    voc_dir: Path, image_id: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool, bool]:
    """Load image and segmentation masks from files.

    Args:
        voc_dir: VOC directory path
        image_id: Image identifier

    Returns:
        Tuple containing:
        - image: Loaded image or None if failed
        - mask_instance: Loaded instance mask or None if failed
        - mask_class: Loaded class mask or None if failed
        - mask_instance_load_success: Whether instance mask was loaded successfully
        - mask_class_load_success: Whether class mask was loaded successfully
    """
    # Get file paths
    image_path = get_image_path(voc_dir, image_id)
    mask_instance_path = get_segm_inst_mask_path(voc_dir, image_id)
    mask_class_path = get_segm_cls_mask_path(voc_dir, image_id)

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning(f"Failed to load image: {image_path}. Skipping.")
        return None, None, None, False, False

    # Import PIL for handling palette-mode PNG masks
    from PIL import Image

    # Load instance mask using PIL to properly handle palette-mode PNG
    mask_instance = None
    mask_instance_load_success = False
    try:
        mask_instance = np.array(Image.open(str(mask_instance_path)))
        mask_instance_load_success = True
        logger.debug(f"Successfully loaded instance mask with PIL: {mask_instance_path}")
        logger.debug(f"Instance mask unique values: {np.unique(mask_instance)}")
    except Exception as e:
        logger.warning(
            f"Failed to load instance segmentation mask with PIL: {mask_instance_path}. Error: {e}."
        )
        # Removed OpenCV fallback

    # Load class mask using PIL to properly handle palette-mode PNG
    mask_class = None
    mask_class_load_success = False
    try:
        mask_class = np.array(Image.open(str(mask_class_path)))
        mask_class_load_success = True
        logger.debug(f"Successfully loaded class mask with PIL: {mask_class_path}")
        logger.debug(f"Class mask unique values: {np.unique(mask_class)}")
    except Exception as e:
        logger.warning(
            f"Failed to load class segmentation mask with PIL: {mask_class_path}. Error: {e}. "
            "Class names may be 'Unknown'."
        )
        # Removed OpenCV fallback

    return image, mask_instance, mask_class, mask_instance_load_success, mask_class_load_success


def _draw_instance_masks(
    image: np.ndarray,
    mask_instance: np.ndarray,
    mask_class: Optional[np.ndarray],
    image_id: str,
    show_difficult: bool = False,
) -> Tuple[np.ndarray, int]:
    """Draw instance masks on the image.

    Args:
        image: Original image to draw on
        mask_instance: Instance segmentation mask
        mask_class: Class segmentation mask (can be None)
        image_id: Image identifier for logging
        show_difficult: Whether to show difficult/void regions (value 255)

    Returns:
        Tuple of (drawn_image, number_of_instances_found)
    """
    # Find unique instances in the mask
    if show_difficult:
        # Include 255 (difficult/void) if requested
        instance_ids = np.unique(mask_instance[mask_instance > 0])
    else:
        # Exclude 0 (background) and 255 (void/difficult)
        instance_ids = np.unique(mask_instance[(mask_instance > 0) & (mask_instance != 255)])

    num_instances = len(instance_ids)

    # Create a copy of the image to draw on
    image_to_draw = image.copy()

    if num_instances > 0:
        logger.debug(f"Found instances {instance_ids} in image {image_id}")
        for instance_id in sorted(instance_ids):
            # Create binary mask for the current instance
            instance_pixel_mask = mask_instance == instance_id
            binary_mask = instance_pixel_mask.astype(np.uint8) * 255

            # Get color based on instance ID
            color = get_color(instance_id)

            # For difficult/void regions (255), use a special label
            if instance_id == 255 and show_difficult:
                label_text = "Unk.255"
            else:
                # Determine class name via mask lookup
                label_text = generate_instance_label(
                    instance_id, mask_instance, mask_class, image_id
                )

            # Overlay the mask with the label
            overlay_mask(image_to_draw, binary_mask, label=label_text, color=color, alpha=0.3)
    else:
        logger.debug(f"No instances found in mask for {image_id}")

    return image_to_draw, num_instances


def process_and_visualize_image(
    image_id: str,
    year: str,
    tag: str,
    voc_root: Path,
    output_dir: Path,
    do_save: bool,
    do_display: bool,
    show_difficult: bool = False,
) -> Tuple[bool, bool, Optional[int], bool, bool]:
    """Loads image and mask, draws masks, and saves/displays the image.

    Returns:
        Tuple: (mask_instance_load_success, mask_class_load_success,
                num_instances_found, save_success, display_success)
               Returns counts as None if mask loading failed.
    """
    save_success = False
    display_success = False
    num_instances_found: Optional[int] = None

    try:
        voc_dir = get_voc_dir(voc_root, year)

        # Load image and masks
        image, mask_instance, mask_class, mask_instance_load_success, mask_class_load_success = (
            _load_image_and_masks(voc_dir, image_id)
        )

        if image is None or not mask_instance_load_success:
            return mask_instance_load_success, mask_class_load_success, 0, False, False

        # Draw instance masks
        image_to_draw, num_instances_found = _draw_instance_masks(
            image, mask_instance, mask_class, image_id, show_difficult
        )

        # Save or display the image
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
                cv2.imshow(f"VOC Segment Viz - {image_id}", image_to_draw)
                logger.info(f"Displaying image {image_id}. Press any key to continue...")
                cv2.waitKey(0)
                display_success = True
            except Exception as e:
                logger.error(f"Failed to display image {image_id}: {e}")
            finally:
                cv2.destroyAllWindows()

        return (
            mask_instance_load_success,
            mask_class_load_success,
            num_instances_found,
            save_success,
            display_success,
        )

    except FileNotFoundError as e:
        logger.error(f"File not found error for {image_id}: {e}. Skipping.")
        return False, False, None, False, False
    except Exception as e:
        logger.error(f"Unexpected error processing {image_id}: {e}", exc_info=True)
        return False, False, None, False, False


def track_class_mask_distribution(
    mask: np.ndarray, value_distribution: Dict[int, int], image_id: str
) -> None:
    """Track the distribution of class values in a mask array.

    For each unique value in the mask, increment the count in value_distribution.

    Args:
        mask: The mask array to analyze.
        value_distribution: Dictionary mapping pixel values to count of images containing the value.
        image_id: Image identifier for logging.
    """
    if mask is None:
        return

    unique_values = set(np.unique(mask))
    logger.debug(f"Class mask for {image_id} has values: {unique_values}")

    for value in unique_values:
        value_distribution[value] = value_distribution.get(value, 0) + 1


def _initialize_stats():
    """Initialize statistics tracking for image processing."""
    return {
        "instances_per_image": [],
        "images_processed": 0,
        "mask_instance_read_success": 0,
        "mask_class_read_success": 0,
        "images_saved": 0,
        "images_displayed": 0,
        "total_instances_found": 0,
        "class_mask_value_distribution": {},
    }


def _process_images(
    ids_to_process,
    base_voc_root,
    output_dir,
    do_save,
    do_display,
    show_difficult,
):
    """Process a batch of images and collect statistics.

    Args:
        ids_to_process: List of (image_id, year, tag) tuples to process
        base_voc_root: Base VOC directory path
        output_dir: Output directory for visualizations
        do_save: Whether to save processed images
        do_display: Whether to display processed images
        show_difficult: Whether to show difficult/boundary regions

    Returns:
        Dictionary containing processing statistics
    """
    # Initialize statistics
    stats = _initialize_stats()

    # Processing loop
    logger.info("Starting visualization processing...")
    for image_id, year, tag in tqdm(ids_to_process, desc="Processing Images"):
        stats["images_processed"] += 1

        # Analyze class mask distribution
        try:
            voc_dir = get_voc_dir(base_voc_root, year)
            mask_class_path = get_segm_cls_mask_path(voc_dir, image_id)

            from PIL import Image  # Keep import local for potential lazy loading

            try:
                mask_class = np.array(Image.open(str(mask_class_path)))
                track_class_mask_distribution(
                    mask_class, stats["class_mask_value_distribution"], image_id
                )
            except Exception as e:
                logger.debug(f"Failed to analyze class mask for {image_id}: {e}")

        except Exception as e:
            logger.debug(f"Failed to get paths for mask analysis for {image_id}: {e}")

        # Process and visualize the image
        instance_success, class_success, num_instances, saved, displayed = (
            process_and_visualize_image(
                image_id,
                year,
                tag,
                base_voc_root,
                output_dir,
                do_save,
                do_display,
                show_difficult,
            )
        )

        if saved:
            stats["images_saved"] += 1
        if displayed:
            stats["images_displayed"] += 1

        if instance_success:
            stats["mask_instance_read_success"] += 1
            if num_instances is not None:
                stats["instances_per_image"].append(num_instances)
                stats["total_instances_found"] += num_instances
        if class_success:
            stats["mask_class_read_success"] += 1

    return stats


def _report_instance_statistics(args: argparse.Namespace, instances_per_image: List[int]) -> None:
    """Report statistics about instance counts.

    Args:
        args: Command-line arguments including percentiles
        instances_per_image: List of instance counts per image
    """
    if not instances_per_image:
        logger.info("No instance statistics generated (no instance masks successfully processed).")
        return

    instances_arr = np.array(instances_per_image)
    logger.info("--- Instance Statistics (per successfully processed mask) ---")

    if args.percentiles:
        try:
            percentiles_str = args.percentiles.split(",")
            percentiles = [float(p.strip()) * 100 for p in percentiles_str]
            if not all(0 <= p <= 100 for p in percentiles):
                raise ValueError("Percentiles must be between 0 and 1.")

            instance_percentiles = np.percentile(instances_arr, percentiles)
            logger.info(f"Percentiles requested: {[f'{p / 100:.2f}' for p in percentiles]}")
            logger.info(f"Instance count percentiles: {np.round(instance_percentiles, 2).tolist()}")

        except ValueError as e:
            logger.error(
                f"Invalid format or value for --percentiles: {e}. "
                "Use comma-separated floats between 0 and 1. Reporting averages instead."
            )
            logger.info(f"Average instances per image: {instances_arr.mean():.2f}")
    else:
        logger.info(f"Average instances per image: {instances_arr.mean():.2f}")


def _report_class_distributions(class_mask_value_distribution: Dict[int, int]) -> None:
    """Report statistics about class distribution in masks.

    Args:
        class_mask_value_distribution: Dictionary mapping class values to occurrence counts
    """
    if not class_mask_value_distribution:
        return

    logger.info("--- Class Mask Value Distribution ---")
    logger.info("Showing count of images containing each class value:")

    # Check for unexpected class values (21-254)
    unexpected_values_found = {}
    for value, count in class_mask_value_distribution.items():
        if 21 <= value <= 254:
            unexpected_values_found[value] = count

    if unexpected_values_found:
        logger.warning("Found unexpected class values (21-254) in some images:")
        for value, count in sorted(unexpected_values_found.items()):
            logger.warning(f"  Unexpected class value {value}: {count} images")

    # Report counts for each valid class (1-20)
    for value in range(1, 21):
        if value in class_mask_value_distribution and value <= len(VOC_CLASSES):
            count = class_mask_value_distribution[value]
            class_name = VOC_CLASSES[value - 1]
            logger.info(f"  Class value {value} ({class_name}): {count} images")


def report_statistics(
    args: argparse.Namespace,
    instances_per_image: List[int],
    num_targeted: int,
    num_processed: int,
    num_mask_instance_success: int,
    num_mask_class_success: int,
    num_saved: int,
    num_displayed: int,
    total_instances: int,
    class_mask_value_distribution: Dict[int, int],
) -> None:
    """Calculates and logs processing and annotation statistics."""
    # Report processing summary
    logger.info("--- Processing Complete ---")
    logger.info(f"Total images targeted: {num_targeted}")
    logger.info(f"Images processed attempt: {num_processed}")
    logger.info(f"Instance segmentation masks successfully read: {num_mask_instance_success}")
    logger.info(f"Class segmentation masks successfully read: {num_mask_class_success}")
    logger.info(f"Images saved: {num_saved}")
    logger.info(f"Images displayed: {num_displayed}")

    # Report instance statistics
    _report_instance_statistics(args, instances_per_image)

    # Report total instances count
    logger.info(
        f"Total object instances found across all processed instance masks: {total_instances}"
    )

    # Report class mask value distributions
    _report_class_distributions(class_mask_value_distribution)


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
            return

        # --- Determine Output Actions ---
        is_single_image_mode = args.image_id is not None
        do_display = is_single_image_mode
        do_save = not is_single_image_mode
        show_difficult = args.show_difficult

        # --- Process Images ---
        stats = _process_images(
            ids_to_process,
            base_voc_root,
            output_dir,
            do_save,
            do_display,
            show_difficult,
        )

        # --- Statistics Reporting ---
        report_statistics(
            args,
            stats["instances_per_image"],
            len(ids_to_process),
            stats["images_processed"],
            stats["mask_instance_read_success"],
            stats["mask_class_read_success"],
            stats["images_saved"],
            stats["images_displayed"],
            stats["total_instances_found"],
            stats["class_mask_value_distribution"],
        )

    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Path setup failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
