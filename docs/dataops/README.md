# Data Operations (`vibelab/dataops`)

This directory contains all modules and utilities related to dataset loading, parsing, processing, conversion, and visualization for the project. The goal is to centralize data handling logic for all machine learning datasets from any source, making it modular, reusable, and testable.

## Scope

The `dataops` module is designed to handle all dataset-related operations needed for machine learning workflows:

* **Loading and parsing** datasets from various sources (Hugging Face, local files, cloud storage, etc.)
* **Analyzing** dataset statistics and distributions
* **Processing** raw data into formats suitable for model training
* **Visualizing** dataset samples and annotations
* **Converting** between different dataset formats (YOLO, COCO, VOC, etc.)
* **Transforming** data as needed for specific model requirements
* **Filtering and sampling** data based on various criteria

## Organization

Modules are primarily organized **by dataset**. Each dataset has its own subdirectory (e.g., `cov_segm/`) containing the specific logic needed to handle it:

*   **`loader.py`**: Handles loading the raw dataset (e.g., from Hugging Face `datasets`, local files, S3, etc.) and orchestrates parsing and data fetching to return processed samples.
*   **`datamodel.py`**: Defines data classes or Pydantic models representing the structure of the parsed data for clarity and validation.
*   **`visualizer.py`**: Provides functions to visualize samples or specific aspects of the dataset.
*   **`converter.py`**: Includes logic to convert the dataset format for specific downstream tasks (e.g., to YOLO or COCO format).
*   **`analyzer.py`**: Provides tools for analyzing dataset statistics and characteristics.

Shared utilities that are applicable across multiple datasets reside in the `common/` directory:

*   **`common/s3_fetcher.py`**: A utility specifically designed to download content from S3 URIs that might be embedded within data fields (e.g., inside JSON strings), which standard dataset loaders might not handle automatically.

## Current and Future Datasets

Currently, the following datasets are implemented:

* **`cov_segm`**: A comprehensive implementation for the `lab42/cov-segm-v3` dataset (conversation-based segmentation) from Hugging Face.

Future plans include adding support for:

* Standard object detection datasets (COCO, VOC, etc.)
* Custom internal datasets
* More specialized datasets as needed for research
* Image classification datasets
* Natural language datasets

Each dataset implementation will follow the same modular structure while addressing dataset-specific requirements.

## Workflow

Typically, a notebook or training script would:
1. Import the specific loader from the relevant dataset module (e.g., `from vibelab.dataops.cov_segm.loader import load_sample`).
2. Use the loader function to get processed data samples.
3. (Optionally) Import and use visualization functions from the same dataset module (e.g., `from vibelab.dataops.cov_segm.visualizer import visualize_prompt_masks`).
4. (Optionally) Use a converter from the dataset module if format transformation is needed.

## Contribution

When adding support for a new dataset, create a new subdirectory named after the dataset and populate it with the necessary `loader.py`, `datamodel.py`, etc. Adapt existing logic where possible, especially from the `ref/` directory (copying and commenting the origin is preferred over direct modification of `ref/`). Place genuinely reusable, dataset-agnostic utilities in `common/`.

## Testing Philosophy

The testing approach for this module focuses on validating core algorithmic functionality rather than visual outputs or end-to-end results. This ensures:

- **Unit Tests are Deterministic:** Tests should produce the same results regardless of execution environment
- **Headless Execution:** Tests can run in CI/CD pipelines without requiring display capabilities
- **Focus on Logic, not Appearance:** Tests validate that the right pixels are processed, not how they look

For example, in `tests/dataops/cov_segm/test_visualizer.py`:
- Matplotlib components are mocked to avoid actual plotting
- Various mask types (grayscale, binary, palette) are tested with parameterized test cases
- The tests verify mask interpretation logic, not the visual appearance
- Binary mask handling is tested through proper PIL image creation and numpy array manipulation

When adding tests, follow these principles:
- Use appropriate mocking to isolate the functionality being tested
- Add parameterized tests for diverse inputs when applicable
- Use fixtures to create test data that exercises key logic paths
- Focus on assertions that validate behavior, not implementation details