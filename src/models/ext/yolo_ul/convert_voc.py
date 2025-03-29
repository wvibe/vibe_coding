#!/usr/bin/env python3
"""
Script to convert VOC XML annotations to YOLO format.
This creates the necessary label files in YOLO format for training.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
from tqdm import tqdm

# VOC class names
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def convert_voc_annotation(voc_dir, year, image_set, output_dir, classes=VOC_CLASSES):
    """
    Convert VOC XML annotations to YOLO txt format

    Args:
        voc_dir: Path to VOCdevkit directory
        year: Dataset year (e.g., '2007')
        image_set: Dataset split ('train', 'val', 'trainval', 'test')
        output_dir: Directory to save label files
        classes: List of class names
    """
    voc_path = Path(voc_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Get image IDs from the image set file
    with open(voc_path / f'VOC{year}/ImageSets/Main/{image_set}.txt') as f:
        image_ids = f.read().strip().split()

    print(f'Converting {len(image_ids)} images from VOC{year} {image_set}')

    for image_id in tqdm(image_ids):
        # Convert the annotation
        convert_single_annotation(voc_path, year, image_id, output_path, classes)

    print(f'Conversion complete. Labels saved to {output_path}')


def convert_single_annotation(voc_path, year, image_id, output_path, classes):
    """Convert a single XML annotation to YOLO format"""
    xml_file = voc_path / f'VOC{year}/Annotations/{image_id}.xml'

    # Skip if file doesn't exist
    if not xml_file.exists():
        print(f"Warning: {xml_file} does not exist")
        return

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image size
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # Create output file
    out_file = output_path / f'{image_id}.txt'

    with open(out_file, 'w') as f:
        # Process each object
        for obj in root.findall('object'):
            # Get class name
            class_name = obj.find('name').text
            if class_name not in classes:
                continue

            # Get class index
            class_idx = classes.index(class_name)

            # Get bounding box coordinates
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Convert to YOLO format (center x, center y, width, height) - normalized
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            # Write to file
            f.write(f"{class_idx} {x_center} {y_center} {bbox_width} {bbox_height}\n")


def main():
    parser = argparse.ArgumentParser(description='Convert VOC XML annotations to YOLO format')
    parser.add_argument('--voc-dir', type=str, default='/Users/wmu/vibe/hub/data/VOCdevkit',
                        help='Path to VOCdevkit directory')
    parser.add_argument('--year', type=str, default='2007',
                        help='Dataset year (e.g., 2007, 2012)')
    parser.add_argument('--output-dir', type=str, default='labels',
                        help='Directory to save label files')
    parser.add_argument('--trainval', action='store_true',
                        help='Convert trainval split')
    parser.add_argument('--test', action='store_true',
                        help='Convert test split')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert trainval split
    if args.trainval:
        convert_voc_annotation(
            args.voc_dir,
            args.year,
            'trainval',
            os.path.join(args.output_dir, 'trainval')
        )

    # Convert test split
    if args.test:
        convert_voc_annotation(
            args.voc_dir,
            args.year,
            'test',
            os.path.join(args.output_dir, 'test')
        )

    # If no splits specified, convert both
    if not args.trainval and not args.test:
        convert_voc_annotation(
            args.voc_dir,
            args.year,
            'trainval',
            os.path.join(args.output_dir, 'trainval')
        )
        convert_voc_annotation(
            args.voc_dir,
            args.year,
            'test',
            os.path.join(args.output_dir, 'test')
        )


if __name__ == '__main__':
    main()