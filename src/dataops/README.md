# Data Operations (`src/dataops`)

This directory contains all modules and utilities related to dataset loading, parsing, processing, conversion, and visualization for the project. The goal is to centralize data handling logic, making it modular, reusable, and testable.

## Organization

Modules are primarily organized **by dataset**. Each dataset has its own subdirectory (e.g., `cov_segm/`) containing the specific logic needed to handle it:

*   **`loader.py`**: Handles loading the raw dataset (often using Hugging Face `datasets`) and orchestrates parsing and data fetching to return processed samples.
*   **`parser.py`**: Contains logic to parse complex or nested fields specific to the dataset format (e.g., JSON strings within columns).
*   **`datamodel.py`**: (Recommended) Defines data classes or Pydantic models representing the structure of the parsed data for clarity and validation.
*   **`visualizer.py`**: Provides functions to visualize samples or specific aspects of the dataset.
*   **`converter.py`**: Includes logic to convert the dataset format for specific downstream tasks (e.g., to YOLO or COCO format).

Shared utilities that are applicable across multiple datasets reside in the `common/` directory:

*   **`common/s3_fetcher.py`**: A utility specifically designed to download content from S3 URIs that might be embedded within data fields (e.g., inside JSON strings), which standard dataset loaders might not handle automatically.

## Workflow

Typically, a notebook or training script would:
1. Import the specific loader from the relevant dataset module (e.g., `from src.dataops.cov_segm.loader import load_processed_sample`).
2. Use the loader function to get processed data samples.
3. (Optionally) Import and use visualization functions from the same dataset module (e.g., `from src.dataops.cov_segm.visualizer import display_overlay`).
4. (Optionally) Use a converter from the dataset module if format transformation is needed.

## Contribution

When adding support for a new dataset, create a new subdirectory named after the dataset and populate it with the necessary `loader.py`, `parser.py`, etc. Adapt existing logic where possible, especially from the `ref/` directory (copying and commenting the origin is preferred over direct modification of `ref/`). Place genuinely reusable, dataset-agnostic utilities in `common/`.