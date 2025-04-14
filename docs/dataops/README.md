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

## Analyzer Module (`src/dataops/cov_segm/analyzer.py`)

The `Analyzer` module is responsible for aggregating and summarizing phrase statistics from the `lab42/cov-segm-v3` dataset metadata. It operates solely on metadata for efficiency.

### Key Features
- Aggregates phrase statistics including appearance counts, visible/full mask counts, and alternative phrase counts.
- Command-line interface for easy usage.

### Command-Line Usage
To run the analyzer, use the following command:
```bash
python -m src.dataops.cov_segm.analyzer --split train --sample_slice '[:100]' --top 10 --skip_zero
```

### Key Arguments
- `--split`: Specify dataset split (default: `train`).
- `--sample_slice`: Specify sample slice (e.g., `[:100]`, `[50:150]`, default: `[:100]`, `''` for all).
- `--top`: Number of top phrases to include in the output (default: `10`).
- `--skip_zero`: Flag to ignore phrases associated with zero masks in a sample during aggregation.

## Converter Module (`src/dataops/cov_segm/converter.py`)

The `Converter` module transforms the `lab42/cov-segm-v3` dataset into YOLO format for segmentation tasks. It utilizes the OOP data models and supports parallel processing for efficiency with large datasets.

### Key Features
- Converts dataset samples to YOLO segmentation format based on configurable class mappings
- Supports parallel processing via Hugging Face's `datasets.map()` for efficient processing
- Provides detailed statistics on processed samples, segments, and masks
- Implements flexible sampling options with both per-class and global ratios

### Command-Line Usage
For a small test run:
```bash
python -m src.dataops.cov_segm.converter --train-split validation --mask-tag visible --sample-count 10 --num-proc 4
```

For a production run with sampling:
```bash
python -m src.dataops.cov_segm.converter --train-split train --mask-tag visible --sample-ratio 0.2 --num-proc 8 --output-name visible_20pct
```

### Key Arguments
- `--train-split`: Dataset split to process (e.g., 'train', 'validation')
- `--mask-tag`: Type of mask to convert ('visible' or 'full')
- `--num-proc`: Number of processor cores for parallel loading
- `--sample-ratio`: Global sampling ratio applied to all classes
- `--sample-count`: Maximum number of samples to process (for testing)
- `--mapping-config`: Path to the CSV mapping file
- `--output-dir`: Root directory for YOLO dataset output
- `--output-name`: Subdirectory name for this dataset version