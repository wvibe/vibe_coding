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
- **Clarity & Encapsulation:** Use Pydantic (`datamodel.py`) for parsing/validating raw structured data (like JSON). For processed data returned by loaders, **Object-Oriented classes** (e.g., `SegmSample`, `Segment`, `SegmMask` for `cov_segm`) are preferred over `TypedDict`s to improve encapsulation (bundling data with methods like mask parsing), maintainability, and clarity.

## `cov_segm` Dataset Specifics

- **Input:** The `cov_segm.loader.load_sample` function expects a single dictionary row (`hf_cov_segm_row`) obtained from `datasets.load_dataset('lab42/cov-segm-v3', ...)`. This dictionary contains both the `conversations` JSON string and the actual mask image data in columns like `mask_0`, `mask_1`, etc.
- **Processing:**
    - The `conversations` JSON string is parsed using Pydantic models defined in `cov_segm.datamodel` (`ConversationItem`, etc.).
    - The main image is fetched from the S3 URI specified in the `conversations` data.
    - Masks are loaded directly from the columns (e.g., `mask_0`) in the input `hf_cov_segm_row` dictionary, guided by the `column` and `positive_value` fields in the parsed `conversations` metadata.
- **Output Structure (`SegmSample` Class):** The `load_sample` function returns an instance of the `SegmSample` class (defined in `cov_segm.datamodel`). This object contains the loaded main `PIL.Image` and a list of `ClsSegment` objects (representing class concepts). Each `ClsSegment` contains a list of `Phrase` objects (synonymous descriptions for the concept) and lists of `SegmMask` objects. The `SegmMask` class encapsulates the mask parsing logic, storing the resulting binary mask, pixel area, and bounding box (`bbox` tuple: `(x_min, y_min, x_max, y_max)`), derived from the raw mask data and the `positive_value`.
- **Semantic Interpretation:** Each `ClsSegment` within the output list represents a single semantic *class concept* or grounding task for the main image. It should be interpreted that **all `phrases` within a single `ClsSegment` are equivalent descriptions** referring to the concept depicted by the **collective set of masks** (`visible_masks`/`full_masks`) within that same `ClsSegment`. Downstream code should treat the `phrases` within a `ClsSegment` as synonymous for the purpose of linking to the corresponding masks.

## Key Components (Summary)

- **Loaders (`<dataset>/loader.py` -> `load_sample`):** Orchestrate loading, parsing, fetching, combining data. Returns processed data objects (e.g., `SegmSample`).
- **Data Models (`<dataset>/datamodel.py`):** Define Pydantic models for raw data validation and **classes for processed data representation** (e.g., `SegmSample`, `ClsSegment`, `SegmMask` for `cov_segm`).
- **Visualizers (`<dataset>/visualizer.py`):** Plot processed data.
- **Converters (`<dataset>/converter.py`):** Transform processed data formats (e.g., to YOLO).
- **Common Utilities (`common/`):** Shared tools (e.g., `s3_fetcher.py`).

## Implementation Notes

- **OOP Data Model:** The refactoring from `TypedDict` to OOP classes has been completed for `datamodel.py` and `loader.py`. The `SegmMask`, `ClsSegment`, and `SegmSample` classes are now fully implemented and tested. The `SegmMask` class robustly handles various mask formats and parsing logic. The `loader.load_sample` function now correctly utilizes these classes and helper functions (`_resolve_reference_path`, `_process_mask_list`) to fetch raw data, delegate parsing to `SegmMask`, and construct the `SegmSample` object graph.
- **Binary Mask Operations:** When working with binary masks and boolean values:
  - Use direct boolean assertions rather than equality comparisons (e.g., `assert mask.is_valid` instead of `assert mask.is_valid == True`)
  - For negating boolean assertions, use `not` (e.g., `assert not mask.is_valid` instead of `assert mask.is_valid == False`)
  - For NumPy binary masks, use `~` for logical negation of entire arrays (e.g., `np.all(~binary_mask)` instead of `np.all(binary_mask == False)`)
- **Handling RGB Images:** The `SegmMask._parse` method has limitations with RGB images. When working with RGB masks, convert them to grayscale or extract a single channel before processing to avoid dimension issues.

## Future Considerations - Implemented

All planned features have now been implemented:

*   **Parallel Processing:** Implementation complete for the `converter.py` module. The OOP data model with serialization support allows for efficient parallel processing using Hugging Face `datasets.map(num_proc=N)`.

*   **Serialization:** Added serialization support to all OOP classes (`SegmSample`, `ClsSegment`, `SegmMask`) with efficient binary data handling for masks and images.

*   **Memory Efficiency:** Implemented compression techniques for binary masks using `np.packbits()` to minimize memory usage during parallel processing.

*   **Progress Tracking:** Added tqdm progress bars to clearly display both loading and conversion progress.

*   **Sampling Controls:** Added global sampling ratio parameter for better control over dataset size during conversion.

## Analyzer Module (`src/dataops/cov_segm/analyzer.py`)

This module provides functions for analyzing and aggregating phrase statistics across the `lab42/cov-segm-v3` dataset. It operates *only* on dataset metadata for speed and efficiency.

### `_aggregate_stats_from_metadata` (Internal Function)

- **Purpose:** To iterate through dataset samples and aggregate statistics based on unique phrases, using only the metadata present in the raw dataset rows (specifically the `conversations` JSON).
- **Input:**
    - `dataset_iterable`: An `Iterable` yielding raw dataset rows (dictionaries).
    - `skip_zero_masks` (bool): If True, ignores phrases in a sample if they have 0 visible AND 0 full masks according to the metadata counts.
- **Processing:** Retrieves the `conversations` JSON string from each raw row, parses it, and aggregates statistics.
- **Output Structure:** Returns a tuple `(phrase_agg_stats, total_samples_processed, total_conversations, total_valid_conversations)` where:
    - `phrase_agg_stats`: Dictionary mapping primary phrase text to its aggregated stats.

### `_print_phrase_details` (Internal Function)

- **Purpose:** To format and print the aggregated statistics for the top N phrases.
- **Input:** List of `(phrase, data)` tuples, sorted by appearance count, with pre-calculated percentage.
- **Output:** Prints formatted text to the console.

### Command-Line Usage
To run the analyzer, use the following command:
```bash
python -m src.dataops.cov_segm.analyzer --split train --sample_slice '[:100]' --top 10 --skip_zero
```

## Visualizer Usage Examples (`src/dataops/cov_segm/visualizer.py`)

The visualizer script provides a command-line interface to inspect masks associated with specific text prompts within the dataset samples.

**Basic Usage (Show visible masks for a prompt in the first sample):**

```bash
python -m src.dataops.cov_segm.visualizer_main "the red car"
```

**Search for a prompt in the first 10 samples of the validation split:**

```bash
python -m src.dataops.cov_segm.visualizer_main "a window reflection" --split validation --start_index 0 --sample_count 10
```

**Visualize 'full' segmentation masks instead of 'visible' instance masks:**

```bash
python -m src.dataops.cov_segm.visualizer_main "the license plate" --mask_type full
```

**Save the visualization to a file instead of displaying interactively:**

```bash
# Creates ./viz_outputs/sample_ID_the_dog_visible.png (filename depends on sample ID)
python -m src.dataops.cov_segm.visualizer_main "the dog" --output_dir ./viz_outputs
```

**Enable debug logging for detailed information:**

```bash
python -m src.dataops.cov_segm.visualizer_main "tree branches" --debug
```

**Combine options (Save full masks for a prompt in sample 5):**

```bash
# Removed --dpi flag, as it's no longer supported
python -m src.dataops.cov_segm.visualizer_main "the side mirror" --start_index 5 --sample_count 1 --mask_type full --output_dir ./viz_outputs
```

## Visualizer Implementation Details

The `cov_segm.visualizer` module has been refactored to work with the OOP data models (`SegmSample`, `ClsSegment`, `SegmMask`).

- **Input:** The primary visualization function `visualize_prompt_masks` now accepts a `SegmSample` object.
- **Mask Handling:** It directly uses the pre-computed boolean `binary_mask` attribute from the `SegmMask` objects within the target `ClsSegment`.
- **Logic Simplification:** The internal helper `_apply_color_mask` no longer needs to handle different mask modes (PIL modes like 'L', 'P', '1') or interpret `positive_value`, as this parsing is encapsulated within the `SegmMask` class during the loading phase (`loader.py`). It simply applies color based on the provided boolean NumPy array.
- **Filtering:** The visualization function filters the list of `SegmMask` objects to only include those marked as valid (`is_valid=True`) and having a non-`None` `binary_mask` before attempting to draw them.
- **CLI (`visualizer_main.py`):** The command-line interface has been simplified, removing the `--dpi`, `--show`, and `--no-show` flags. Output is displayed interactively only if `--output_dir` is *not* provided.

### Enhanced Debugging (Updated)

The visualizer still includes debugging features:
- Detailed logging of which *valid* masks are being processed, including source column and geometry (area, bbox) from the `SegmMask` object.
- The `--debug` flag in `visualizer_main.py` enables detailed logging in both the visualizer and loader modules.
- Raw conversation data debugging (using `get_last_parsed_conversations`) has been removed from `visualizer_main.py` as the function was removed from `loader.py`.

## YOLO Format Converter (`src/dataops/cov_segm/converter.py`)

This module is responsible for converting the `lab42/cov-segm-v3` dataset, loaded via `src.dataops.cov_segm.loader.load_sample`, into the YOLO segmentation format.

### Core Functionality

- **Input:** Takes samples processed by `loader.load_sample` (`SegmSample` objects) from a specified HF dataset (`--hf-dataset-path`) and split (`--train-split`).
- **Configuration:** Reads mapping and sampling rules from a user-provided CSV file (`--mapping-config`). The CSV defines:
    - `yolo_class_id`: Unique integer ID for the target YOLO class.
    - `yolo_class_name`: Human-readable name for the target class.
    - `hf_phrase`: Phrase text to match in conversation items.
    - `sampling_ratio`: Probability (0.0-1.0) for including matches of this phrase.
- **Mask Selection:** Processes either 'visible' or 'full' masks based on the `--mask-tag` argument. Optionally skips segments with zero masks of the selected type (`--skip-zero`).
- **Parallel Processing:** Uses Hugging Face's `datasets.map()` with the specified number of processes (`--num-proc`) to load samples in parallel from the dataset.
- **Mask Conversion:** Uses `src.utils.common.geometry.mask_to_yolo_polygons` to convert binary masks (NumPy arrays) from `SegmMask.binary_mask` into normalized polygon coordinates required by YOLO.
- **Polygon Handling:** For masks that generate multiple polygons, the converter tracks this in statistics but only processes the first polygon to maintain one annotation per mask.
- **Sampling:**
    - Applies a global sampling ratio (`--sample-ratio`) as a multiplier to each class's individual ratio.
    - The effective ratio for a class becomes `global_ratio * local_ratio`.
    - For deterministic testing, sampling is bypassed when `--sample-count` is specified.
- **File Handling:** With the `--no-overwrite` flag, skips writing image and label files that already exist, tracking these skips in statistics.
- **Output Structure:** Generates the standard YOLO dataset structure within a specific subfolder (`--output-name` or derived from `--mask-tag`) under the main output directory (`--output-dir` or `COV_SEGM_ROOT`):
    - `{output_dir}/{dataset_name}/images/{split_name}/{image_id}.jpg`: Copied source images (only if annotations exist).
    - `{output_dir}/{dataset_name}/labels/{split_name}/{image_id}.txt`: Text files containing annotations (`<class_id> <norm_x1> ...`).
- **Dataset YAML:** Generates a `dataset.yaml` file (e.g., `configs/yolov11/cov_segm_segment_{dataset_name}.yaml`) containing the absolute dataset path and class names, suitable for training frameworks.
- **Split Handling:** Processes only one dataset split (`--train-split`) per execution.

### Statistics Tracking

The converter tracks comprehensive statistics during processing:
- **Sample-level stats:** Total samples, processed samples, samples with errors
- **Segment-level stats:** Segments skipped due to no mapping, sampling, or zero masks
- **Mask-level stats:** Masks producing no polygons or multiple polygons
- **File-level stats:** Files skipped due to existing (when `--no-overwrite` is used)
- **Annotation-level stats:** Total annotations generated, images with annotations, class distribution

### Statistics Reporting

- **Basic Reporting:** At the end of conversion, logs summary statistics about the processing run
- **Class-based Statistics:** Generates a table with statistics per class using the common `stats.format_statistics_table` utility
- **Metrics:** Reports count, mean, median, and maximum valid masks per segment for each class

### Implementation Details

- **Main Script:** Provides a command-line interface using `argparse` to specify configurations (mapping, output, mask tag, split, sampling, etc.).
- **Parallel Processing:** Leverages the OOP data model to support parallel loading with HF `datasets.map()` while showing progress with tqdm.
- **Dependencies:** Requires `opencv-python-headless`, `datasets`, `pillow`, `numpy`, `python-dotenv`, `tqdm`.
- **Error Handling:** Includes checks for file/directory existence, graceful handling of sample loading errors, mapping errors, and mask conversion issues.
- **Reproducibility:** Uses `random.seed()` to ensure consistent sampling results with the same seed value.