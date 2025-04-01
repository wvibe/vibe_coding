#!/usr/bin/env python3
"""
Convert VOC Dataset XML Annotations to YOLO Detection Format for a Specific ImageSet.

Reads XML annotations from the specified VOCdevkit structure based on a given
year and ImageSet tag, and outputs YOLO format bounding box labels (.txt)
directly into the specified output directory.

Usage:
    python src/utils/data_converter/detect_voc2yolo.py \
        --devkit-path /path/to/VOCdevkit \
        --year 2007 \
        --tag trainval \
        --output-dir /path/to/output/labels_detect
"""

import argparse
import logging
import os
from pathlib import Path

from tqdm import tqdm

# Import common utilities
from src.utils.data_converter.converter_utils import (
    VOC_CLASS_TO_ID,
    parse_voc_xml,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert VOC XML annotations for a specific ImageSet tag to YOLO detection format."
    )
    parser.add_argument(
        "--devkit-path",
        type=str,
        required=True,
        help="Path to the VOCdevkit directory (e.g., /path/to/VOC/VOCdevkit)",
    )
    parser.add_argument(
        "--year",
        type=str,
        required=True,
        choices=["2007", "2012"],  # Add more years if needed
        help="Year of the VOC dataset (e.g., 2007, 2012)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="ImageSet tag to process (e.g., train, val, test, trainval, person_trainval)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the flat output directory for label files",
    )
    return parser.parse_args()


def convert_box(size, box):
    """Convert VOC bounding box (xmin, xmax, ymin, ymax) to YOLO format (x_center, y_center, width, height, normalized)."""
    dw, dh = 1.0 / size[0], 1.0 / size[1]  # width, height
    # XML box is xmin, ymin, xmax, ymax - careful with indices from parse_voc_xml
    # parse_voc_xml returns bbox=[xmin, ymin, xmax, ymax]
    xml_xmin, xml_ymin, xml_xmax, xml_ymax = box
    x_center = (xml_xmin + xml_xmax) / 2.0
    y_center = (xml_ymin + xml_ymax) / 2.0
    width = xml_xmax - xml_xmin
    height = xml_ymax - xml_ymin
    return x_center * dw, y_center * dh, width * dw, height * dh


def convert_annotation(xml_path: Path, out_path: Path):
    """Convert a single VOC XML annotation to a YOLO txt file."""
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
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.touch()
            return True  # Considered success (empty annotation)
        except OSError as e:
            logger.error(f"Failed to create empty label file {out_path}: {e}")
            return False

    img_width, img_height = img_dims

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
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
                        f"Skipping object '{cls_name}' in {xml_path.name}: Invalid normalized coordinates {bb_yolo}."
                    )
                    continue

                out_file.write(f"{cls_id} {' '.join(map(str, bb_yolo))}\n")

        return True
    except Exception as e:
        logger.error(f"Error writing annotation file {out_path} for {xml_path.name}: {e}")
        # Clean up potentially partially written file
        if out_path.exists():
            try:
                os.remove(out_path)
            except OSError:
                logger.error(f"Failed to remove partial output file: {out_path}")
        return False


def main():
    args = parse_args()
    devkit_path = Path(os.path.expanduser(args.devkit_path))
    year = args.year
    tag = args.tag
    output_dir = Path(args.output_dir)

    logger.info("Starting VOC to YOLO detection conversion.")
    logger.info(f"Processing VOCdevkit: {devkit_path}")
    logger.info(f"Year: {year}, Tag: {tag}")
    logger.info(f"Output Directory: {output_dir}")

    # Construct paths
    imageset_file = devkit_path / f"VOC{year}/ImageSets/Main/{tag}.txt"
    annotations_dir = devkit_path / f"VOC{year}/Annotations"

    # Validate paths
    if not devkit_path.exists() or not devkit_path.is_dir():
        logger.error(f"VOCdevkit path does not exist or is not a directory: {devkit_path}")
        return
    if not imageset_file.exists():
        logger.error(f"ImageSet file not found: {imageset_file}")
        return
    if not annotations_dir.exists():
        logger.error(f"Annotations directory not found: {annotations_dir}")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get image IDs
    try:
        with open(imageset_file) as f:
            image_ids = [line.strip().split()[0] for line in f if line.strip()]
    except IOError as e:
        logger.error(f"Could not read ImageSet file {imageset_file}: {e}")
        return

    if not image_ids:
        logger.warning(f"No image IDs found in {imageset_file}. Exiting.")
        return

    logger.info(f"Found {len(image_ids)} image IDs in {imageset_file}. Converting annotations...")

    success_count = 0
    fail_count = 0
    # Process images defined in the specific tag file
    for image_id in tqdm(image_ids, desc=f"Processing {year}/{tag}"):
        src_xml = annotations_dir / f"{image_id}.xml"
        target_label = output_dir / f"{image_id}.txt"  # Write directly to output dir

        if not src_xml.exists():
            logger.warning(f"Annotation file not found for ID {image_id}: {src_xml}, skipping.")
            fail_count += 1
            continue

        if convert_annotation(src_xml, target_label):
            success_count += 1
        else:
            fail_count += 1
            # Warning/error logged within convert_annotation

    logger.info("\nConversion completed!")
    logger.info(f"Successfully converted: {success_count}")
    logger.info(f"Failed/Skipped: {fail_count}")
    logger.info(f"Output labels saved to: {output_dir}")


if __name__ == "__main__":
    main()
