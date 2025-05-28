"""
Dataset utilities for VLM inference.

Simple utilities for loading and handling HuggingFace datasets.
"""

import re
from typing import Dict

from datasets import Dataset, load_dataset

# Dataset column mappings
DATASET_COLUMN_MAPPINGS = {
    "HuggingFaceM4/ChartQA": {
        "test": {
            "image_column": "image",
            "question_column": "query",
            "id_column": None,  # ChartQA doesn't have an ID column, will use index
            "answer_column": "label",  # Ground truth answers
        },
        "train": {
            "image_column": "image",
            "question_column": "query",
            "id_column": None,  # ChartQA doesn't have an ID column, will use index
            "answer_column": "label",  # Ground truth answers
        },
        "val": {
            "image_column": "image",
            "question_column": "query",
            "id_column": None,  # ChartQA doesn't have an ID column, will use index
            "answer_column": "label",  # Ground truth answers
        },
    }
}

# Default column mapping for unknown datasets
DEFAULT_COLUMNS = {"image_column": "image", "question_column": "question", "id_column": "id", "answer_column": "answer"}


def get_dataset_columns(dataset_name: str, dataset_split: str) -> Dict[str, str]:
    """
    Get the appropriate column names for a dataset.

    Args:
        dataset_name: Name of the dataset.
        dataset_split: Split of the dataset.

    Returns:
        Dictionary with column names.
    """
    if dataset_name in DATASET_COLUMN_MAPPINGS:
        dataset_config = DATASET_COLUMN_MAPPINGS[dataset_name]
        if dataset_split in dataset_config:
            return dataset_config[dataset_split]
        else:
            # Use first available split as fallback
            return next(iter(dataset_config.values()))

    return DEFAULT_COLUMNS


def parse_sample_slices(slice_str: str) -> slice:
    """
    Parse a slice string like "[150:160]" into a Python slice object.
    Based on the implementation in src/vibelab/dataops/cov_segm/converter.py

    Args:
        slice_str: String representation of a slice, e.g., "[150:160]", "[10:]", "[:50]"

    Returns:
        Python slice object.
    """
    pattern = r"^\[(.*?)(:(.*?))?\]$"
    match = re.match(pattern, slice_str)
    if not match:
        raise ValueError(
            f"Invalid slice format: '{slice_str}'. Expected format like '[start:stop]'."
        )

    start_str, _, stop_str = match.groups(default="")
    try:
        start = int(start_str) if start_str else None
        stop = int(stop_str) if stop_str else None
    except ValueError as e:
        raise ValueError(
            f"Invalid slice components in '{slice_str}'. Start and stop must be integers."
        ) from e

    if start is not None and start < 0:
        raise ValueError(f"Slice start cannot be negative: {start}")
    if stop is not None and stop < 0:
        raise ValueError(f"Slice stop cannot be negative: {stop}")

    return slice(start, stop)


def load_hf_dataset(dataset_name: str, dataset_split: str) -> Dataset:
    """
    Load a dataset from HuggingFace.

    Args:
        dataset_name: Name of the HuggingFace dataset.
        dataset_split: Split to load.

    Returns:
        Loaded dataset.
    """
    return load_dataset(dataset_name, split=dataset_split)
