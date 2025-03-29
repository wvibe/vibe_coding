#!/usr/bin/env python3
"""
Convert VOC Dataset to YOLO Format

This script converts the Pascal VOC dataset to the format expected by Ultralytics YOLOv5.
It reorganizes the directory structure and converts XML annotations to YOLO format.

Usage:
    python scripts/convert_voc_to_yolo.py
"""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import argparse

# VOC class names (from VOC.yaml)
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def parse_args():
    parser = argparse.ArgumentParser(description='Convert VOC dataset to YOLO format')
    parser.add_argument('--voc-path', type=str, default='~/vibe/hub/datasets/VOC',
                        help='Path to the VOC dataset root (default: ~/vibe/hub/datasets/VOC)')
    return parser.parse_args()

def convert_box(size, box):
    """Convert VOC bounding box to YOLO format (normalized)."""
    dw, dh = 1. / size[0], 1. / size[1]
    x, y, w, h = (box[0] + box[1]) / 2.0, (box[2] + box[3]) / 2.0, box[1] - box[0], box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

def convert_annotation(voc_path, year, image_id, out_path):
    """Convert VOC XML annotation to YOLO txt format."""
    try:
        in_file = open(f'{voc_path}/VOCdevkit/VOC{year}/Annotations/{image_id}.xml')
        out_file = open(out_path, 'w')

        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in VOC_CLASSES:
                continue

            # Skip difficult objects if desired
            # difficult = int(obj.find('difficult').text)
            # if difficult:
            #     continue

            cls_id = VOC_CLASSES.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text),
                 float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert_box((w, h), b)
            out_file.write(f"{cls_id} {' '.join(map(str, bb))}\n")

        in_file.close()
        out_file.close()
        return True
    except Exception as e:
        print(f"Error converting {image_id}: {e}")
        return False

def process_split(voc_path, split_data):
    """Process a specific train/val/test split."""
    year, image_set = split_data

    # Source paths
    voc_src_path = f"{voc_path}/VOCdevkit/VOC{year}"

    # Target paths
    img_target_dir = f"{voc_path}/images/{image_set}{year}"
    label_target_dir = f"{voc_path}/labels/{image_set}{year}"

    # Create target directories
    os.makedirs(img_target_dir, exist_ok=True)
    os.makedirs(label_target_dir, exist_ok=True)

    # Get image IDs
    with open(f"{voc_src_path}/ImageSets/Main/{image_set}.txt") as f:
        image_ids = f.read().strip().split()

    print(f"Processing {image_set}{year} with {len(image_ids)} images")

    for image_id in tqdm(image_ids, desc=f"{image_set}{year}"):
        # Source image path
        src_img = f"{voc_src_path}/JPEGImages/{image_id}.jpg"

        # Target paths
        target_img = f"{img_target_dir}/{image_id}.jpg"
        target_label = f"{label_target_dir}/{image_id}.txt"

        # Copy image
        if not os.path.exists(target_img):
            shutil.copy(src_img, target_img)

        # Convert annotation
        convert_annotation(voc_path, year, image_id, target_label)

    return len(image_ids)

def main():
    args = parse_args()
    voc_path = os.path.expanduser(args.voc_path)

    # Create main directories
    os.makedirs(f"{voc_path}/images", exist_ok=True)
    os.makedirs(f"{voc_path}/labels", exist_ok=True)

    # Define splits to process (year, image_set)
    splits = [
        ('2007', 'train'),
        ('2007', 'val'),
        ('2007', 'test'),
        ('2012', 'train'),
        ('2012', 'val')
    ]

    # Process all splits
    total_processed = 0
    for split_data in splits:
        count = process_split(voc_path, split_data)
        total_processed += count

    print(f"\nDataset conversion completed!")
    print(f"Total processed images: {total_processed}")
    print(f"Dataset structure:")
    print(f"  - {voc_path}/images/ - Contains images organized by split")
    print(f"  - {voc_path}/labels/ - Contains YOLO format labels organized by split")
    print(f"\nYou can now use this dataset with Ultralytics YOLOv5/YOLOv8 by setting the path in your YAML config.")

if __name__ == "__main__":
    main()