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
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(os.getcwd(), "data"))
CHECKPOINTS_ROOT = os.environ.get("CHECKPOINTS_ROOT", None)

# Dataset paths
VOC_ROOT = os.environ.get("VOC_ROOT", None)
VOC2007_DIR = os.environ.get("VOC2007_DIR", None)
VOC2012_DIR = os.environ.get("VOC2012_DIR", None)
COCO_ROOT = os.environ.get("COCO_ROOT", None)

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
        if VOC2007_DIR is None:
            raise ValueError("VOC2007_DIR environment variable not set, but required by source path.")
        return os.path.join(VOC2007_DIR, source[len("VOC2007/") :])
    elif source.startswith("VOC2012/"):
        if VOC2012_DIR is None:
            raise ValueError("VOC2012_DIR environment variable not set, but required by source path.")
        return os.path.join(VOC2012_DIR, source[len("VOC2012/") :])
    elif source.startswith("COCO/"):
        if COCO_ROOT is None:
            raise ValueError("COCO_ROOT environment variable not set, but required by source path.")
        return os.path.join(COCO_ROOT, source[len("COCO/") :])
    elif source.startswith("VOCdevkit/"):
        if VOC_ROOT is None:
            raise ValueError("VOC_ROOT environment variable not set, but required by source path.")
        return os.path.join(VOC_ROOT, source[len("VOCdevkit/") :])

    # For other relative paths, assume they're relative to DATA_ROOT
    if DATA_ROOT is None:
        raise ValueError("DATA_ROOT environment variable not set and no default available.")
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
