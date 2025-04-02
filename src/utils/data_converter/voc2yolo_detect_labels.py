#!/usr/bin/env python3
"""
Convert VOC Dataset XML Annotations to YOLO Detection Format for a Specific ImageSet.

Reads XML annotations from the specified VOCdevkit structure based on a given
year and ImageSet tag, and outputs YOLO format bounding box labels (.txt)
directly into the specified output directory under labels_detect/<tag+year>.

Usage:
    python src/utils/data_converter/voc2yolo_detect_labels.py \
        --voc-root /path/to/VOC \
        --output-root /path/to/output \
        --year 2007 \
        --tag trainval
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from src.utils.data_converter.voc2yolo_utils import (
    VOC_CLASS_TO_ID,
    get_annotation_path,
    get_image_set_path,
    get_output_detect_label_dir,
    get_voc_dir,
    parse_voc_xml,
    read_image_ids,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
    except Exception as e:  # Catch potential path errors
        logger.error(f"Error processing output root path '{output_root_str}': {e}")
        return voc_root, None

    return voc_root, output_root


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert VOC XML annotations to YOLO detection format."
    )
    parser.add_argument(
        "--voc-root",
        type=str,
        default=None,
        help=(
            "Path to the VOC dataset root directory (containing VOCdevkit). "
            "If not set, uses VOC_ROOT from .env."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Path to the root directory for output labels. Defaults to --voc-root if not specified."
        ),
    )
    parser.add_argument(
        "--years",
        type=str,
        required=True,
        help=("Comma-separated list of years to process (e.g., 2007,2012)."),
    )
    parser.add_argument(
        "--tags",
        type=str,
        required=True,
        help=(
            "Comma-separated list of ImageSet tags to process "
            "(e.g., train,val,test). Uses ImageSets/Main."
        ),
    )
    return parser.parse_args()


def convert_box(size, box):
    """Convert VOC bounding box (xmin, xmax, ymin, ymax) to YOLO format.

    Args:
        size: Tuple of (width, height) of the image
        box: Tuple of (xmin, ymin, xmax, ymax) in absolute coordinates

    Returns:
        Tuple of (x_center, y_center, width, height) in normalized coordinates
    """
    dw, dh = 1.0 / size[0], 1.0 / size[1]  # width, height
    # XML box is xmin, ymin, xmax, ymax - careful with indices from parse_voc_xml
    # parse_voc_xml returns bbox=[xmin, ymin, xmax, ymax]
    xml_xmin, xml_ymin, xml_xmax, xml_ymax = box
    x_center = (xml_xmin + xml_xmax) / 2.0
    y_center = (xml_ymin + xml_ymax) / 2.0
    width = xml_xmax - xml_xmin
    height = xml_ymax - xml_ymin
    return x_center * dw, y_center * dh, width * dw, height * dh


def convert_annotation(xml_path: Path, output_label_dir: Path) -> bool:
    """Convert a single VOC XML annotation to a YOLO txt file in the specified directory."""
    # Construct output path within the target directory
    out_path = output_label_dir / f"{xml_path.stem}.txt"

    # Use the common XML parser
    objects, img_dims = parse_voc_xml(xml_path)

    # Handle parsing errors or no objects found
    if objects is None or img_dims is None:
        # Error logged in parse_voc_xml
        return False
    if not objects:
        logger.debug(f"No valid objects found in {xml_path.name}, creating empty label file.")
        # Create empty file
        try:
            # output_label_dir should already exist (created in main)
            out_path.touch()
            return True  # Considered success (empty annotation)
        except OSError as e:
            logger.error(f"Failed to create empty label file {out_path}: {e}")
            return False

    img_width, img_height = img_dims

    try:
        # output_label_dir should already exist
        with open(out_path, "w") as out_file:
            for obj in objects:
                cls_name = obj["name"]
                # Class name already validated in parse_voc_xml
                cls_id = VOC_CLASS_TO_ID[cls_name]
                # Bbox is [xmin, ymin, xmax, ymax] absolute
                bbox_abs = obj["bbox"]

                # Convert to YOLO format
                bb_yolo = convert_box((img_width, img_height), bbox_abs)

                # Final validation for normalized coords before writing
                if not (
                    0 <= bb_yolo[0] <= 1
                    and 0 <= bb_yolo[1] <= 1
                    and 0 < bb_yolo[2] <= 1
                    and 0 < bb_yolo[3] <= 1
                ):
                    logger.warning(
                        f"Skipping object '{cls_name}' in {xml_path.name}: "
                        f"Invalid normalized coordinates {bb_yolo}."
                    )
                    continue

                out_file.write(f"{cls_id} {' '.join(map(str, bb_yolo))}\n")

        return True
    except Exception as e:
        logger.error(f"Error writing annotation file {out_path} for {xml_path.name}: {e}")
        return False


def convert_split(
    voc_root: Path,
    output_root: Path,
    year: str,
    tag: str,
) -> tuple[int, int]:
    """Convert annotations for a specific year and tag.

    Args:
        voc_root: Path to VOC dataset root directory
        output_root: Path to output root directory
        year: Dataset year (e.g., '2007')
        tag: ImageSet tag (e.g., 'train', 'val')

    Returns:
        Tuple of (success_count, fail_count)
    """
    logger.info(f"--- Processing Year: {year}, Tag: {tag} ---")
    success_count = fail_count = 0

    try:
        # Use utility functions to get paths
        voc_year_path = get_voc_dir(voc_root, year)
        imageset_file = get_image_set_path(voc_year_path, set_type="detect", tag=tag)
        output_label_dir = get_output_detect_label_dir(output_root, year, tag)

        # Create output directory
        output_label_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output labels will be saved to: {output_label_dir}")

        # Get image IDs using utility function
        image_ids = read_image_ids(imageset_file)

    except FileNotFoundError as e:
        logger.error(f"Error setting up paths: {e}")
        return 0, 0
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 0, 0
    except IOError as e:
        logger.error(f"Error reading imageset file: {e}")
        return 0, 0

    if not image_ids:
        logger.warning(f"No image IDs found or read from {imageset_file}. Skipping.")
        return 0, 0

    logger.info(f"Found {len(image_ids)} image IDs. Converting annotations...")

    # Process images defined in the specific tag file
    for image_id in tqdm(image_ids, desc=f"Processing {year}/{tag}"):
        try:
            # Get XML path using utility
            src_xml = get_annotation_path(voc_year_path, image_id)

            if not src_xml.exists():
                logger.warning(f"Annotation file not found for ID {image_id}: {src_xml}, skipping.")
                fail_count += 1
                continue

            # Pass the specific output directory to the conversion function
            if convert_annotation(src_xml, output_label_dir):
                success_count += 1
            else:
                fail_count += 1
                # Warning/error logged within convert_annotation
        except Exception as e:
            logger.error(
                f"Unexpected error processing image ID {image_id} for tag {tag}: {e}", exc_info=True
            )
            fail_count += 1

    logger.info(
        f"Finished converting for {year}/{tag}: Success {success_count}, Failed {fail_count}"
    )
    return success_count, fail_count


def main() -> None:
    """Main execution function."""
    args = parse_args()

    voc_root, output_root = _determine_paths(args)
    if not voc_root or not output_root:
        return  # Error logged in helper function

    # Parse comma-separated lists
    try:
        years = [y.strip() for y in args.years.split(",") if y.strip()]
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        if not years or not tags:
            raise ValueError("Years and Tags arguments cannot be empty.")
    except Exception as e:
        logger.error(f"Error parsing --years or --tags argument: {e}")
        return

    logger.info("Starting VOC to YOLO detection conversion.")
    logger.info(f"Years to process: {years}")
    logger.info(f"Tags to process: {tags}")

    total_success = 0
    total_fail = 0

    # Process each year and tag combination
    for year in years:
        for tag in tags:
            try:
                s, f = convert_split(voc_root, output_root, year, tag)
                total_success += s
                total_fail += f
            except Exception as e:
                logger.error(
                    f"Critical error during processing of {year}/{tag}: {e}", exc_info=True
                )

    logger.info("\nConversion completed!")
    logger.info(f"Overall success count: {total_success}")
    logger.info(f"Overall failed count: {total_fail}")


if __name__ == "__main__":
    main()
