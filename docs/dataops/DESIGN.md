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
- **Converters (`<dataset>/converter.py`):** Transform processed data formats (e.g., to YOLO).
- **Common Utilities (`common/`):** Shared tools (e.g., `s3_fetcher.py`).

## Future Considerations

*(Add notes on potential refactoring, future dataset support, etc.)*

## Analyzer Module (`src/dataops/cov_segm/analyzer.py`)

This module provides functions for analyzing and aggregating statistics across the `lab42/cov-segm-v3` dataset. It operates in two modes: a fast `counts_only` mode using metadata, and a slower `deep_stats` mode that loads mask data to calculate geometric properties.

### `aggregate_phrase_stats`

- **Purpose:** To iterate through dataset samples and aggregate statistics based on unique phrases. The level of aggregation depends on the selected `--mode`.
- **Input:**
    - `dataset_iterable`: An `Iterable` yielding raw dataset rows (dictionaries).
    - `mode` (str): Determines the analysis depth ('counts_only' or 'deep_stats').
    - `verbose` (bool): Enables detailed logging.
    - `debug_phrase` (Optional[str]): Logs details when a specific phrase is encountered.
    - `skip_zero_masks` (bool): If True, ignores phrases in a sample if they have 0 visible AND 0 full masks (only relevant in `counts_only` mode).
- **Processing (`counts_only` mode):**
    - Retrieves the `conversations` JSON string from each raw row.
    - Uses `src.dataops.cov_segm.loader.parse_conversations` to parse the JSON into Pydantic `ConversationItem` models.
    - Extracts the primary phrase (first phrase text) from each `ConversationItem`.
    - Counts items in `instance_masks` (visible masks) and `instance_full_masks` (full masks) for each phrase per sample based on list lengths in the metadata.
    - Handles potential errors during JSON/Pydantic parsing gracefully.
    - Aggregates statistics per unique phrase text based on its *first appearance* within a sample, applying `skip_zero_masks` logic if enabled.
- **Processing (`deep_stats` mode):**
    - Uses `src.dataops.cov_segm.loader.load_sample` to load the full sample data (image and processed masks with pre-calculated geometry) for each row.
    - Handles potential errors during `load_sample` gracefully (logs and skips).
    - Extracts the primary phrase (first phrase text) from each `processed_conversation`.
    - Aggregates statistics per unique phrase text based on its *first appearance* within a sample:
        - Counts visible and full masks based on the length of `processed_instance_masks` and `processed_full_masks`.
        - Aggregates pre-calculated `pixel_area`, `width`, and `height` from the `ProcessedMask` dictionaries within `processed_instance_masks` and `processed_full_masks`.
- **Output Structure:** Returns a tuple `(aggregated_stats_dict, total_processed_count)` where:
    - `aggregated_stats_dict`: Dictionary mapping phrase text to:
        ```python
        {
            "appearance_count": int,           # How many samples contain this phrase
            "sample_ids": List[str],         # IDs of samples containing this phrase
            # --- Counts (Always Populated) ---
            "total_visible_mask_count": int, # Sum of visible masks across all appearances
            "visible_mask_counts_per_image": List[int], # List of visible mask counts per appearance
            "total_full_mask_count": int,    # Sum of full masks across all appearances
            "full_mask_counts_per_image": List[int],  # List of full mask counts per appearance
            # --- Deep Stats (Populated only if mode=='deep_stats') ---
            "visible_mask_pixel_areas": List[int], # List of pixel areas for each visible mask instance
            "visible_mask_widths": List[int],      # List of widths for each visible mask instance
            "visible_mask_heights": List[int],     # List of heights for each visible mask instance
            "full_mask_pixel_areas": List[int],    # List of pixel areas for each full mask instance
            "full_mask_widths": List[int],         # List of widths for each full mask instance
            "full_mask_heights": List[int],        # List of heights for each full mask instance
        }
        ```
    - `total_processed_count`: The number of samples successfully processed (parsed or loaded).

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
    - **If deep stats data is present** (i.e., `*_pixel_areas`, `*_widths`, `*_heights` lists exist):
        - Calculates the mean pixel area, width, and height for visible and full masks.
        - Calculates the specified percentiles for the distributions of pixel area, width, and height for visible and full masks.
- **Output Structure:** Returns a *list* of dictionaries, one per phrase, sorted by `appearance_count` (descending). Each dictionary contains:
    ```python
    {
        "phrase": str,
        "appearance_count": int,
        "appearance_percentage": float,
        # --- Count Stats ---
        "avg_visible_masks_per_image": float,
        "avg_full_masks_per_image": float,
        "visible_mask_percentiles": Dict[float, float], # Count percentiles
        "full_mask_percentiles": Dict[float, float],    # Count percentiles
        # --- Deep Stats (Optional) ---
        "avg_visible_mask_pixels": Optional[float],
        "avg_visible_mask_width": Optional[float],
        "avg_visible_mask_height": Optional[float],
        "visible_mask_pixel_percentiles": Optional[Dict[float, float]],
        "visible_mask_width_percentiles": Optional[Dict[float, float]],
        "visible_mask_height_percentiles": Optional[Dict[float, float]],
        "avg_full_mask_pixels": Optional[float],
        "avg_full_mask_width": Optional[float],
        "avg_full_mask_height": Optional[float],
        "full_mask_pixel_percentiles": Optional[Dict[float, float]],
        "full_mask_width_percentiles": Optional[Dict[float, float]],
        "full_mask_height_percentiles": Optional[Dict[float, float]],
    }
    ```
    *(Note: Deep stats fields will be `None` or absent if run in `counts_only` mode)*

### Command-Line Usage (`python -m src.dataops.cov_segm.analyzer ...`)

The script can be run directly to perform aggregation and summarization.
*   **Key Arguments:**
    *   `--mode`: Analysis mode. Choices: `counts_only` (default, fast, uses metadata), `deep_stats` (slow, loads masks, calculates geometry).
    *   `--split`: Specify dataset split (default: `validation`).
    *   `--sample_slice`: Specify sample slice (e.g., `[:100]`, `[50:150]`, default: `[:20]`, `''` for all).
    *   `--output_file`: Base path for saving results (`_agg.json` and `_summary.json` are appended). If omitted, prints summary to console.
    *   `--top`: Number of top phrases to print to console (default: `20`).
    *   `--percentiles`: List of percentiles to calculate (default: `[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]`).
    *   `--skip_zero`: Flag to ignore phrases with zero masks in a sample during aggregation (only relevant in `counts_only` mode).
    *   `--debug_phrase`: Specify a phrase for detailed debug logging (requires `-v`).
    *   `-v` / `--verbose`: Enable verbose logging.
*   **Output:** Either saves two JSON files or prints a formatted summary of the top N phrases to the console, including geometric stats if run in `deep_stats` mode.

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

## Visualizer Implementation Details

The `cov_segm.visualizer` module has been implemented with careful attention to diverse mask formats:

### Mask Format Handling

- **Grayscale Masks (mode "L")**: Handles both direct matching (e.g., pixel value = 1) and non-zero matching for scaled values (e.g., 0-255 range).
- **Binary Masks (mode "1")**: Special handling for PIL's binary format where pixels are either 0 or 1, with proper boolean array conversion.
- **Palette Masks (mode "P")**: Direct matching against the index value, with palette interpretation capabilities.

### Binary Mask Handling

Special processing for binary masks (PIL mode "1") includes:
- Correct conversion between numpy boolean arrays and PIL binary images
- Proper handling of both positive_value=0 (matching black pixels) and positive_value=1 (matching white pixels)
- Detailed logging for troubleshooting mask processing

### Enhanced Debugging

The visualizer includes robust debugging features:
- Detailed logging of mask characteristics (mode, shape, min/max values)
- Step-by-step tracking of mask interpretation
- Clear error reporting for unmatched masks or unexpected values

## YOLO Format Converter (`src/dataops/cov_segm/converter.py`)

This module is responsible for converting the `lab42/cov-segm-v3` dataset, loaded via `src.dataops.cov_segm.loader.load_sample`, into the YOLO segmentation format.

### Core Functionality

- **Input:** Takes samples processed by `loader.load_sample` (`ProcessedCovSegmSample` TypedDict) from a specified HF dataset (`--hf-dataset-path`) and split (`--train-split`).
- **Configuration:** Reads mapping and sampling rules from a user-provided CSV file (`--mapping-config`). The CSV defines:
    - `yolo_class_id`: Unique integer ID for the target YOLO class.
    - `yolo_class_name`: Human-readable name for the target class.
    - `hf_phrase`: Phrase text to match in conversation items.
    - `sampling_ratio`: Probability (0.0-1.0) for including matches of this phrase (ignored if `--sample-count` is specified).
- **Mask Selection:** Processes either 'visible' or 'full' masks based on the `--mask-tag` argument. Optionally skips conversations with zero masks of the selected type (`--skip-zero`).
- **Mask Conversion:** Uses `src.utils.common.geometry.mask_to_yolo_polygons` to convert binary masks (NumPy arrays) into normalized polygon coordinates required by YOLO.
- **Polygon Handling:** For masks that generate multiple polygons, the converter tracks this in statistics but only processes the first polygon to maintain one annotation per mask.
- **Sampling:** Applies the `sampling_ratio` from the configuration CSV probabilistically (`if random.random() < ratio: keep`) to each conversation item before processing its masks. This sampling is bypassed when `--sample-count` is specified for deterministic debugging runs.
- **File Handling:** With the `--no-overwrite` flag, skips writing image and label files that already exist, tracking these skips in statistics.
- **Output Structure:** Generates the standard YOLO dataset structure within a specific subfolder (`--output-name` or derived from `--mask-tag`) under the main output directory (`--output-dir` or `COV_SEGM_ROOT`):
    - `{output_dir}/{dataset_name}/images/{split_name}/{image_id}.jpg`: Copied source images (only if annotations exist).
    - `{output_dir}/{dataset_name}/labels/{split_name}/{image_id}.txt`: Text files containing annotations (`<class_id> <norm_x1> ...`).
- **Dataset YAML:** Generates a `dataset.yaml` file (e.g., `configs/yolov11/cov_segm_segment_{dataset_name}.yaml`) containing the absolute dataset path and class names, suitable for training frameworks.
- **Split Handling:** Processes only one dataset split (`--train-split`) per execution.

### Statistics Tracking

The converter tracks comprehensive statistics during processing:
- **Sample-level stats:** Total samples, processed samples, samples with errors
- **Conversation-level stats:** Total conversations, phrases, conversations skipped due to no mapping or sampling
- **Mask-level stats:** Masks producing no polygons or multiple polygons
- **File-level stats:** Files skipped due to existing (when `--no-overwrite` is used)
- **Annotation-level stats:** Total annotations generated, images with annotations, class distribution

### Statistics Reporting

- **Basic Reporting:** At the end of conversion, logs summary statistics about the processing run
- **Class-based Statistics:** Generates a table with statistics per class:
  - Configurable metrics including count, sum, mean, percentiles, and max/min
  - Header formatting based on chosen statistics
  - Individual class rows with count, average, and percentile data

### Implementation Details

- **Main Script:** Provides a command-line interface using `argparse` to specify configurations (mapping, output, mask tag, split, sampling, etc.). Allows partial processing via `--sample-count`.
- **Dependencies:** Requires `opencv-python-headless`, `datasets`, `pillow`, `numpy`, `python-dotenv`, `tqdm`.
- **Error Handling:** Includes checks for file/directory existence, graceful handling of sample loading errors, mapping errors, and mask conversion issues.
- **Reproducibility:** Uses `random.seed()` to ensure consistent sampling results with the same seed value.