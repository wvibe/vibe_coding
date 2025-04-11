# Data Operations TODO List

This document tracks the development tasks for the `src/dataops` module.

## High Priority: `lab42/cov-segm-v3` Support

*   **[X] Implement Common S3 Fetcher:**
    *   Created `src/dataops/common/s3_fetcher.py`.
    *   **Goal:** Provide `fetch_s3_uri(uri)` to download S3 content bytes.
    *   **Action:** Implemented using `boto3`, `lru_cache`, and helpers from `ref` (with attribution). Added `if __name__ == "__main__"` for basic testing.
*   **[X] Define `cov_segm` Data Models:**
    *   Created `src/dataops/cov_segm/datamodel.py`.
    *   **Decision:** Used **Pydantic** for robust validation of the `conversations` JSON structure.
    *   Defined `Phrase`, `ImageURI`, `InstanceMask`, `ConversationItem` models.
*   **[X] Implement `cov_segm` Loader (including Parser logic):**
    *   Created `src/dataops/cov_segm/loader.py` (renamed from `conversations.py`).
    *   **Merged** `parse_conversations` function (using Pydantic models) directly into `loader.py`.
    *   Implemented `load_sample(hf_cov_segm_row)`:
        *   **Input:** Takes a raw dictionary row from `datasets.load_dataset('lab42/cov-segm-v3')`.
        *   **Action:** Parses `conversations` JSON, fetches main image from S3 URI (using `s3_fetcher`), loads mask images directly from columns (e.g., `mask_0`, `masks_rest/0`) in the input row based on parsed metadata.
        *   **Output:** Returns a `ProcessedCovSegmSample` `TypedDict` containing the loaded main `PIL.Image` and structured conversation data with loaded mask `PIL.Image` objects.
*   **[X] Implement `cov_segm` Visualization:**
    *   Created `src/dataops/cov_segm/visualizer.py`.
    *   **Action:** Implemented `visualize_prompt_masks` function accepting `ProcessedCovSegmSample`.
        *   Overlays masks based on `positive_value` for a specific `prompt`.
        *   Distinguishes instances with different colors (`tab10` colormap).
    *   **CLI:** Added command-line interface with arguments for sample selection (`start_index`, `sample_count`), prompt, mask type, output directory (`output_dir`), and debugging (`--debug`).
    *   **Output Handling:** Saves images to `output_dir` with filenames based on sample ID, prompt, and mask type. Suppresses interactive display when saving, unless `--show` is used.
    *   **[X] Added CLI usage examples to `docs/dataops/DESIGN.md`.**
*   **[X] Update Exploration Notebook:**
    *   Updated `notebooks/dataset/lab42_segm_explorer.ipynb` with:
        *   Proper imports for `datasets`, `PIL`, `matplotlib`.
        *   Imports for `load_sample` from `src.dataops.cov_segm.loader`.
        *   Imports for visualization functions from `src.dataops.cov_segm.visualizer`.
        *   Dataset loading with error handling.
        *   Sample iteration with visualization and logging.
        *   Robust error handling and logging throughout.
*   **[X] Add Unit Tests:**
    *   Created `tests/dataops/cov_segm/test_loader.py`.
    *   Implemented comprehensive tests for:
        *   `parse_conversations` (valid/invalid JSON, schema errors)
        *   `_load_mask` (mocking input row data - PIL, NumPy, missing columns, invalid types)
        *   `load_sample` (mocking input row, mocking S3 fetch, asserting correct output structure)
        *   Added fixtures for PIL images and NumPy arrays
        *   Added error case handling tests
*   **[X] Improve `cov_segm` Visualization & Testing:**
    *   Enhanced binary mask (mode "1") handling in `_apply_color_mask` function
    *   Added detailed logging and debugging for mask interpretation process
    *   Implemented proper test cases focusing on core functionality rather than visual output
    *   Fixed test issues with binary mask handling by correctly creating test mask images
    *   Added parameterized tests for diverse mask formats (grayscale, binary, palette)

## Medium Priority: Dataset Analysis & YOLO Conversion

*   **[X] Implement `cov_segm` Analyzer:**
    *   Created `src/dataops/cov_segm/analyzer.py`.
    *   Implemented `aggregate_phrase_stats`:
        *   Takes raw dataset rows and operates in two modes (`--mode`):
            *   `counts_only` (default): Parses `conversations` JSON using `parse_conversations`. Aggregates phrase occurrences and mask counts based on metadata.
            *   `deep_stats`: Loads full samples using `load_sample`. Aggregates phrase occurrences, mask counts, and pre-calculated mask geometry (area, width, height).
        *   Returns `(aggregated_stats_dict, total_processed_count)`.
        *   Includes options `--mode`, `--verbose`, `--debug_phrase`, `--skip_zero` (counts_only mode).
    *   Implemented `calculate_summary_stats`:
        *   Takes aggregated stats and total count.
        *   Calculates appearance percentage, average mask counts, and percentiles for mask counts.
        *   If `deep_stats` data is present, calculates average and percentile statistics for mask area, width, and height.
        *   Returns a list of summary dicts per phrase, sorted by appearance count.
    *   Added command-line interface (`if __name__ == "__main__"`):
        *   Handles arguments for mode, split, sample slicing, output files, percentiles, top N printing, skip zero, and verbosity.
        *   Prints formatted summary or saves JSON files.
    *   Added unit tests (`test_analyzer.py`) covering both functions and options.
*   **[ ] Implement `cov_segm` to YOLO Format Converter:**
    *   Create `cov_segm/converter.py`.
    *   Add function `convert_to_yolo(processed_sample: ProcessedCovSegmSample, output_dir: str, class_map: Dict[str, int])` (using `ProcessedCovSegmSample` type hint).
    *   Define logic to generate YOLO bounding boxes/segmentation masks from `processed_sample.processed_conversations[*].processed_instance_masks`.
    *   Handle mapping phrase text/types to class IDs using `class_map`.
    *   Write images and labels to `output_dir`.

## Low Priority

*   **[ ] Add Support for Other Datasets:** Create new subdirectories (e.g., `dataops/other_dataset/`) following the established pattern (`datamodel.py`, `loader.py`, etc.).
*   **[ ] Refactor Common Logic:** Identify and move any truly dataset-agnostic logic emerging in dataset modules to `common/`.
*   **[ ] Expand Visualizer Capabilities:**
    *   Add support for additional mask types (RGBA, LA, etc.)
    *   Implement a mask format conversion utility for standardizing diverse input formats
    *   Create an interactive visualization dashboard for exploring multiple samples/prompts
    *   Add options for visual styling (colormap selection, transparency control, etc.)
    *   Implement batch processing mode for generating visualizations across dataset slices