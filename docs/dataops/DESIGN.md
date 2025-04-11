# Data Operations (`src/dataops`) Design

This document outlines the design choices and architecture for the `dataops` module.

## Core Principles

- **Modularity:** Each dataset gets its own sub-package (e.g., `cov_segm/`). Within each sub-package:
    - `datamodel.py`: Defines Pydantic models mirroring the raw dataset structure (e.g., JSON fields).
    - `loader.py`: Contains the primary function (`load_sample`) responsible for taking a raw dataset row (e.g., from Hugging Face `datasets`), parsing relevant metadata fields, fetching external assets (like S3 images), loading embedded assets (like mask columns), and returning a unified, processed data structure (often using `TypedDict` for clarity).
    - `visualizer.py`: Provides functions to visualize the processed data structures returned by the loader.
    - `converter.py`: Contains functions to convert the processed data structure to other formats (e.g., YOLO).
- **Reusability:** Common, dataset-agnostic utilities reside in `dataops/common/` (e.g., `s3_fetcher.py`).
    - **Reference Code:** Where necessary, small, self-contained utility functions can be copied from the `ref/` directory *with clear attribution comments* indicating the source file. Avoid direct imports from `ref` into `src`.
- **Clarity:** Use Pydantic (`datamodel.py`) for parsing/validating raw structured data (like JSON). Use `typing.TypedDict` to define the structure of *processed* data returned by loaders for better readability and type checking.

## `cov_segm` Dataset Specifics

- **Input:** The `cov_segm.loader.load_sample` function expects a single dictionary row (`hf_cov_segm_row`) obtained from `datasets.load_dataset('lab42/cov-segm-v3', ...)`. This dictionary contains both the `conversations` JSON string and the actual mask image data in columns like `mask_0`, `mask_1`, etc.
- **Processing:**
    - The `conversations` JSON string is parsed using Pydantic models defined in `cov_segm.datamodel`.
    - The main image is fetched from the S3 URI specified in the `conversations` data.
    - Masks are loaded directly from the columns (e.g., `mask_0`) in the input `hf_cov_segm_row` dictionary, guided by the `column` field in the parsed `conversations` metadata.
- **Output Structure (`ProcessedCovSegmSample` TypedDict):** The `load_sample` function returns a dictionary conforming to the `ProcessedCovSegmSample` structure defined in `cov_segm.loader.py`. This structure contains the loaded main `PIL.Image` and a list of `ProcessedConversationItem` dictionaries.
- **Semantic Interpretation:** Each `ProcessedConversationItem` within the output list represents a single semantic concept or grounding task for the main image. Although structurally containing a list of `phrases` and lists of `processed_instance_masks`/`processed_full_masks`, it should be interpreted that **all `phrases` within a single `ProcessedConversationItem` are equivalent descriptions** referring to the concept(s) depicted by the **collective set of masks** within that same item. Downstream code should treat the phrases within an item as synonymous for the purpose of linking to the corresponding masks of that item.

## Key Components (Summary)

- **Loaders (`<dataset>/loader.py` -> `load_sample`):** Orchestrate loading, parsing, fetching, combining data.
- **Data Models (`<dataset>/datamodel.py`):** Define Pydantic models for raw data validation.
- **Visualizers (`<dataset>/visualizer.py`):** Plot processed data.
- **Converters (`<dataset>/converter.py`):** Transform processed data formats.
- **Common Utilities (`common/`):** Shared tools (e.g., `s3_fetcher.py`).

## Future Considerations

*(Add notes on potential refactoring, future dataset support, etc.)*

## Analyzer Module (`src/dataops/cov_segm/analyzer.py`)

This module provides functions for analyzing and aggregating statistics across the `lab42/cov-segm-v3` dataset after it has been processed by the loader.

### `aggregate_phrase_stats`

- **Purpose:** To efficiently iterate through raw dataset samples and aggregate statistics based on unique phrases found by parsing the `conversations` JSON metadata, *without* loading full image/mask data.
- **Input:**
    - `dataset_iterable`: An `Iterable` yielding raw dataset rows (dictionaries).
    - `verbose` (bool): Enables detailed logging.
    - `debug_phrase` (Optional[str]): Logs details when a specific phrase is encountered.
    - `skip_zero_masks` (bool): If True, ignores phrases in a sample if they have 0 visible AND 0 full masks.
- **Processing:**
    - Retrieves the `conversations` JSON string from each raw row.
    - Uses `src.dataops.cov_segm.loader.parse_conversations` to parse the JSON into Pydantic `ConversationItem` models.
    - Extracts the primary phrase (first phrase text) from each `ConversationItem`.
    - Counts items in `instance_masks` (visible masks) and `instance_full_masks` (full masks) for each phrase per sample.
    - Handles potential errors during JSON/Pydantic parsing gracefully.
    - Aggregates statistics per unique phrase text based on its *first appearance* within a sample, applying `skip_zero_masks` logic if enabled.
- **Output Structure:** Returns a tuple `(aggregated_stats_dict, total_processed_count)` where:
    - `aggregated_stats_dict`: Dictionary mapping phrase text to:
        ```python
        {
            "appearance_count": int,           # How many samples contain this phrase (respecting skip_zero_masks)
            "sample_ids": List[str],         # IDs of samples containing this phrase
            "total_visible_mask_count": int, # Sum of visible masks across all appearances
            "visible_mask_counts_per_image": List[int], # List of visible mask counts per appearance
            "total_full_mask_count": int,    # Sum of full masks across all appearances
            "full_mask_counts_per_image": List[int]  # List of full mask counts per appearance
        }
        ```
    - `total_processed_count`: The number of samples successfully parsed and considered.

### `calculate_summary_stats`

- **Purpose:** To process the aggregated statistics and compute summary metrics per phrase.
- **Input:**
    - `aggregated_stats`: The dictionary output from `aggregate_phrase_stats`.
    - `total_processed_samples`: The total count returned by `aggregate_phrase_stats`.
    - `percentiles`: A list of percentiles (0.0-1.0) to calculate.
- **Processing:**
    - Calculates appearance percentage relative to `total_processed_samples`.
    - Calculates the mean number of visible and full masks per image appearance.
    - Calculates the specified percentiles for the distribution of visible and full mask counts per image.
- **Output Structure:** Returns a *list* of dictionaries, one per phrase, sorted by `appearance_count` (descending). Each dictionary contains:
    ```python
    {
        "phrase": str,
        "appearance_count": int,
        "appearance_percentage": float,
        "avg_visible_masks_per_image": float,
        "avg_full_masks_per_image": float,
        "visible_mask_percentiles": Dict[float, float],
        "full_mask_percentiles": Dict[float, float]
    }
    ```

### Command-Line Usage (`python -m src.dataops.cov_segm.analyzer ...`)

The script can be run directly to perform aggregation and summarization.
*   **Key Arguments:**
    *   `--split`: Specify dataset split (default: `validation`).
    *   `--sample_slice`: Specify sample slice (e.g., `[:100]`, `[50:150]`, default: `[:20]`, `''` for all).
    *   `--output_file`: Base path for saving results (`_agg.json` and `_summary.json` are appended). If omitted, prints summary to console.
    *   `--top`: Number of top phrases to print to console (default: `20`).
    *   `--percentiles`: List of percentiles to calculate (default: `[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]`).
    *   `--skip_zero`: Flag to ignore phrases with zero masks in a sample during aggregation.
    *   `--debug_phrase`: Specify a phrase for detailed debug logging (requires `-v`).
    *   `-v` / `--verbose`: Enable verbose logging.
*   **Output:** Either saves two JSON files or prints a formatted summary of the top N phrases to the console.

## Visualizer Usage Examples (`src/dataops/cov_segm/visualizer.py`)

The visualizer script provides a command-line interface to inspect masks associated with specific text prompts within the dataset samples.

**Basic Usage (Show visible masks for a prompt in the first sample):**

```bash
python -m src.dataops.cov_segm.visualizer "the red car"
```

**Search for a prompt in the first 10 samples of the validation split:**

```bash
python -m src.dataops.cov_segm.visualizer "a window reflection" --split validation --start_index 0 --sample_count 10
```

**Visualize 'full' segmentation masks instead of 'visible' instance masks:**

```bash
python -m src.dataops.cov_segm.visualizer "the license plate" --mask_type full
```

**Save the visualization to a file instead of displaying interactively:**

```bash
# Creates ./viz_outputs/sample_0_the_dog_visible.png (filename depends on sample ID/index)
python -m src.dataops.cov_segm.visualizer "the dog" --output_dir ./viz_outputs --no-show
```

**Enable debug logging for detailed information:**

```bash
python -m src.dataops.cov_segm.visualizer "tree branches" --debug
```

**Combine options (Save full masks for a prompt in sample 5, with higher DPI):**

```bash
python -m src.dataops.cov_segm.visualizer "the side mirror" --start_index 5 --sample_count 1 --mask_type full --output_dir ./viz_outputs --no-show --dpi 300
```