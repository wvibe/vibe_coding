"""
Utility functions for Ultralytics YOLO integration
"""

import os

import dotenv
import yaml

# Load environment variables from .env file (expecting to run from project root)
dotenv.load_dotenv(".env")

# Environment variables - will raise KeyError if not found to fail fast
VIBE_ROOT = os.environ["VIBE_ROOT"]
VHUB_ROOT = os.environ["VHUB_ROOT"]
DATA_ROOT = os.environ["DATA_ROOT"]
CHECKPOINTS_ROOT = os.environ["CHECKPOINTS_ROOT"]

# Dataset paths
VOC_ROOT = os.environ["VOC_ROOT"]
VOC2007_DIR = os.environ["VOC2007_DIR"]
VOC2012_DIR = os.environ["VOC2012_DIR"]
COCO_ROOT = os.environ["COCO_ROOT"]

# Weights & Biases API key (kept for backward compatibility)
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)


def resolve_source_path(source):
    """
    Resolve source path using environment variables if applicable

    Examples:
        VOC2007/JPEGImages/000001.jpg -> os.environ["VOC2007_DIR"] + "/JPEGImages/000001.jpg"
        VOCdevkit/VOC2007/JPEGImages/000001.jpg -> os.environ["VOC_ROOT"] + "/VOC2007/JPEGImages/000001.jpg"
        COCO/train2017/000000000001.jpg -> os.environ["COCO_ROOT"] + "/train2017/000000000001.jpg"
        /absolute/path/to/image.jpg -> /absolute/path/to/image.jpg (unchanged)

    Args:
        source: Source path string

    Returns:
        Resolved absolute path
    """
    # If it's an absolute path or URL, return as is
    if os.path.isabs(source) or source.startswith("http"):
        return source

    # Check for dataset prefixes
    if source.startswith("VOC2007/"):
        return os.path.join(VOC2007_DIR, source[len("VOC2007/") :])
    elif source.startswith("VOC2012/"):
        return os.path.join(VOC2012_DIR, source[len("VOC2012/") :])
    elif source.startswith("COCO/"):
        return os.path.join(COCO_ROOT, source[len("COCO/") :])
    elif source.startswith("VOCdevkit/"):
        return os.path.join(VOC_ROOT, source[len("VOCdevkit/") :])

    # For other relative paths, assume they're relative to DATA_ROOT
    return os.path.join(DATA_ROOT, source)


def load_yaml_config(file_path):
    """
    Load a YAML configuration file

    Args:
        file_path: Path to YAML file

    Returns:
        Dictionary with configuration
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
