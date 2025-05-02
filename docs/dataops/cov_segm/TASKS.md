# COV-SEGM-V3 Dataset Implementation Tasks

This document tracks the completed development tasks for the `lab42/cov-segm-v3` dataset implementation in the `vibelab/dataops/cov_segm` module.

## High Priority: `lab42/cov-segm-v3` Support

*   **[X] Implement Common S3 Fetcher:**
    *   Created `vibelab/dataops/common/s3_fetcher.py`.
    *   **Goal:** Provide `fetch_s3_uri(uri)` to download S3 content bytes.
    *   **Action:** Implemented using `boto3`, `lru_cache`, and helpers from `ref` (with attribution). Added `if __name__ == "__main__"` for basic testing.
*   **[X] Define `cov_segm` Data Models:**
    *   Created `vibelab/dataops/cov_segm/datamodel.py`.
    *   **Decision:** Used **Pydantic** for robust validation of the `conversations` JSON structure.
    *   Defined `Phrase`, `ImageURI`, `InstanceMask`, `ConversationItem` models.
*   **[X] Implement `cov_segm` Loader (including Parser logic):**
    *   Created `vibelab/dataops/cov_segm/loader.py` (renamed from `conversations.py`).
    *   **Merged** `parse_conversations` function (using Pydantic models) directly into `loader.py`.
    *   Implemented `load_sample(hf_cov_segm_row)`:
        *   **Input:** Takes a raw dictionary row from `datasets.load_dataset('lab42/cov-segm-v3')`.
        *   **Action:** Parses `conversations` JSON, fetches main image from S3 URI (using `s3_fetcher`), loads mask images directly from columns (e.g., `mask_0`, `masks_rest/0`) in the input row based on parsed metadata.
        *   **Output:** Returns a `ProcessedCovSegmSample` `TypedDict` containing the loaded main `PIL.Image` and structured conversation data with loaded mask `PIL.Image` objects.
*   **[X] Implement `cov_segm` Visualization:**
    *   Created `vibelab/dataops/cov_segm/visualizer.py`.
    *   **Action:** Implemented `visualize_prompt_masks` function accepting `ProcessedCovSegmSample`.
        *   Overlays masks based on `positive_value` for a specific `prompt`.
        *   Distinguishes instances with different colors (`tab10` colormap).
    *   **CLI:** Added command-line interface with arguments for sample selection (`start_index`, `sample_count`), prompt, mask type, output directory (`output_dir`), and debugging (`--debug`).
    *   **Output Handling:** Saves images to `output_dir` with filenames based on sample ID, prompt, and mask type. Suppresses interactive display when saving, unless `--show` is used.
    *   **[X] Added CLI usage examples to `docs/dataops/cov_segm/DESIGN.md`.**
*   **[X] Update Exploration Notebook:**
    *   Updated `notebooks/dataset/lab42_segm_explorer.ipynb` with:
        *   Proper imports for `datasets`, `PIL`, `matplotlib`.
        *   Imports for `load_sample` from `vibelab.dataops.cov_segm.loader`.
        *   Imports for visualization functions from `vibelab.dataops.cov_segm.visualizer`.
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
    *   Created `vibelab/dataops/cov_segm/analyzer.py`.
    *   Implemented aggregation of phrase statistics (appearance count, visible/full mask counts, alternative phrase counts) directly from dataset metadata.
    *   Removed the previous `deep_stats` mode and related geometric calculations.
    *   Added command-line interface for metadata-based analysis and unit tests.
*   **[X] Refactor `analyzer.py`:**
    *   Completed refactoring to improve clarity and maintainability.
    *   Updated function names and logic to focus on metadata-only aggregation.
*   **[X] Implement `cov_segm` to YOLO Format Converter:**
    *   **Goal:** Convert `lab42/cov-segm-v3` (using `ProcessedCovSegmSample` from loader) into YOLO segmentation format.
    *   **Location:** `vibelab/dataops/cov_segm/converter.py`
    *   **Configuration:**
        *   `--mapping-config` (str): Path to mapping/sampling CSV. Default: `'configs/dataops/cov_segm_yolo_mapping.csv'`.
        *   `--mask-tag` (str): Mask type to process ('visible' or 'full'). Required.
        *   `--skip-zero` (bool): Skip samples with no masks of the specified tag. Default: True.
        *   `--no-overwrite` (bool): Skip writing files if they already exist. Default: False.
    *   **Core Logic:**
        *   Implement main script with `argparse` (see args below).
        *   Load mapping CSV. Handle errors.
        *   Initialize `random.seed()`.
        *   Process a single `--train-split` (e.g., 'train' or 'test') per run.
        *   Load specified HF dataset (`--hf-dataset-path`) split. Apply `--sample-count` if provided.
        *   Use `loader.load_sample`. Handle errors.
        *   Iterate conversations: Check mapping, check `--skip-zero`, apply sampling ratio (only if `--sample-count` is not specified).
        *   Use `vibelab.utils.common.geometry.mask_to_yolo_polygons` to convert kept binary masks (selected by `--mask-tag`) to normalized polygons.
        *   Handle multiple polygons by taking the first polygon.
        *   Handle dependencies (`opencv-python-headless`, etc.).
    *   **Output:**
        *   Determine output root from `--output-dir` arg or `COV_SEGM_ROOT` env var.
        *   Determine dataset subfolder name from `--output-name` arg or `--mask-tag`.
        *   Save images/labels to `{output_root}/{dataset_name}/{images|labels}/{train_split}/`.
        *   Generate `dataset.yaml` in `configs/yolov11/cov_segm_segment_{dataset_name}.yaml` with absolute path and class names.
    *   **Arguments:** `--mapping-config`, `--output-dir`, `--output-name`, `--mask-tag`, `--train-split`, `--hf-dataset-path`, `--sample-count`, `--seed`, `--skip-zero`, `--no-overwrite`.
    *   **Statistics:** Implemented statistics tracking and reporting, including count, mean, percentiles by class ID.
    *   **Verification:** Perform test run for each required split using `--sample-count`. Check output structure, format, and `dataset.yaml`.
*   **[X] Implement Parallel Processing for Analyzer/Converter:**
    *   **Goal:** Significantly speed up `analyzer.py` and `converter.py` execution on large datasets (like `cov-segm-v3`) by leveraging multiprocessing.
    *   **Approach:** Utilize Hugging Face `datasets.map(load_sample, num_proc=N, ...)` to parallelize the data loading and initial processing (`load_sample` returning `SegmSample`).
    *   **Implementation:** Added parallel processing to `converter.py` using Hugging Face's `datasets.map()` with support for custom `num_proc` parameter.
    *   **Optimizations:**
        *   Added efficient binary mask compression with `np.packbits()` to reduce memory usage
        *   Added serialization methods to all OOP data models (`SegmSample`, `ClsSegment`, `SegmMask`)
        *   Added progress tracking via tqdm to maintain visibility during processing
        *   Implemented global sampling ratio to control dataset size

## Refactoring: `cov_segm` OOP Data Model

**Goal:** Refactor `cov_segm` data handling (`loader`, `datamodel`, `visualizer`, `analyzer`, `converter`) to use a more robust Object-Oriented structure, replacing the previous `TypedDict`-based approach (see `DESIGN.md`). This enhances encapsulation (parsing logic within `SegmMask`), clarifies data flow, and improves maintainability.
**Note:** This refactoring must maintain compatibility with parallel processing via Hugging Face `datasets.map(num_proc=N)` by ensuring `load_sample` and the returned object graph (`SegmSample` -> `ClsSegment` -> `SegmMask`) are pickleable.

*   **[X] Phase 1: Refactor `datamodel.py`**
    *   Define `SegmMask` class:
        *   Attributes: `source_info: InstanceMask`, `binary_mask: Optional[np.ndarray]`, `pixel_area: Optional[int]`, `bbox: Optional[Tuple[int, int, int, int]]` (x_min, y_min, x_max, y_max), `is_valid: bool`.
        *   Methods: `__init__(instance_mask_info, raw_mask_data)` (does *not* store `raw_mask_data`), `_parse(raw_mask_data)` (private, contains parsing logic, calculates bbox).
    *   Define `ClsSegment` class:
        *   Attributes: `phrases: List[Phrase]` (synonymous phrases for the class concept), `type: str`, `visible_masks: List[SegmMask]`, `full_masks: List[SegmMask]`.
        *   Methods: `__init__`.
    *   Define `SegmSample` class:
        *   Attributes: `id: str`, `image: Image.Image`, `segments: List[ClsSegment]`.
        *   Methods: `__init__`, `find_segment_by_prompt(prompt)`.
    *   Remove old `TypedDict`s: `ProcessedMask`, `ProcessedConversationItem`, `ProcessedCovSegmSample`.
    *   Ensure all necessary imports (`typing`, `PIL.Image`, `numpy`, existing Pydantic models) are correct.
*   **[X] Phase 2: Refactor `loader.py`**
    *   Update imports for new classes (`SegmSample`, `ClsSegment`, `SegmMask`, `InstanceMask`). Remove old `TypedDict` imports.
    *   Keep helpers: `_load_image_from_uri`, `parse_conversations`.
    *   Rename and simplify `_resolve_mask_path` to `_resolve_reference_path`.
    *   Remove old functions: `_load_mask`, `_process_mask_metadata`.
    *   Remove helper `_load_raw_mask`.
    *   Add helper `_process_mask_list` to encapsulate mask loading/parsing loop.
    *   Refactor `load_sample` to return `Optional[SegmSample]`, using the new helpers and classes.
    *   Refactored unit tests in `test_loader.py` to match the new implementation.
*   **[X] Phase 3: Refactor `visualizer.py` and `visualizer_main.py`**
    *   Updated imports (`SegmSample`, `ClsSegment`, `SegmMask`).
    *   Removed `_find_item_by_prompt` (use `SegmSample.find_segment_by_prompt`).
    *   Refactored `visualize_prompt_masks` to accept `sample: SegmSample`. Iterated `target_segment.*_masks` (which are `SegmMask` lists), accessed `segm_mask.binary_mask` after filtering for valid masks.
    *   Refactored `_apply_color_mask` signature to accept `binary_mask: np.ndarray` and simplified implementation.
    *   Refactored `visualizer_main.py` CLI: Updated imports, simplified args (removed dpi, display group), adapted logic for `SegmSample`.
    *   Refactored unit tests in `test_visualizer.py` to match the new implementation.
*   **[X] Phase 4: Refactor `analyzer.py`**
    *   Updated imports and function names to reflect the metadata-only approach.
    *   Removed helper `_extract_deep_stats_for_sample` and related geometric calculations.
    *   Refactored `_aggregate_stats_from_metadata` to focus on metadata-only aggregation.
    *   Added `_print_phrase_details` function for improved output formatting.
    *   Updated command-line interface to reflect the simplified approach.
    *   Refactored unit tests to match the new implementation.
*   **[X] Phase 5: Refactor `converter.py`**
    *   Updated imports and function signatures to use the new OOP data models.
    *   Replaced manual mask parsing with `SegmMask` built-in methods.
    *   Implemented parallel processing using HF `datasets.map()` with progress tracking via tqdm.
    *   Added support for a global `--sample-ratio` parameter to apply to all classes.
    *   Replaced the custom statistics table printing with `stats.format_statistics_table` from the common utility.
    *   Updated the argument parser, removing unnecessary flags and simplifying options (added `--sample-slice`, removed `--sample-count`).
    *   Revised counters and logging to focus on the new terminology (segments versus conversations).

# Cov-Segm Dataset Implementation Completed Tasks

## Phase 1: Data Analysis and Strategy
- [X] Implement analyzer module to study class (phrase) distribution
- [X] Document findings on common phrases, frequencies and patterns
- [X] Plan development track for different dataset handling approaches

## Phase 2: Basic Data Handling Models
- [X] Create data models for Phrases, Segments, and Masks
- [X] Implement basic handling of segmentation data
- [X] Create utilities to visualize masks and segments from the dataset

## Phase 3: V1 Data Processing Implementation
- [X] Implement simple single-process OOP-based data loader
- [X] Add tests for the data loading and processing functionality
- [X] Refine masks and segmentation handling

## Phase 4: YOLO Format Conversion
- [X] Implement converter for exporting segments to YOLO polygon format
- [X] Add mapping configuration for translating phrases to standardized classes
- [X] Support sampling to create balanced datasets
- [X] Generate required YOLO folder structure

## Phase 5: Converter Refinement
- [X] Refactor `converter.py` to use OOP data models
  - [X] Use `load_sample` to get complete sample with segments and masks
  - [X] Add serialization to/from dict for all models
  - [X] Support HF dataset.map() for parallel processing

## Phase 6: Production-Ready Features
- [X] Implement parallel processing for Analyzer/Converter
- [X] Add comprehensive statistics gathering
- [X] Improve error handling for robustness
- [X] Optimize performance through caching and efficient processing (e.g., `load_from_cache_file=False`, `writer_batch_size`)

## Low Priority: Conversion Verification

*   **[ ] Implement Conversion Verifier Tool:**
    *   **Goal:** Verify the integrity and accuracy of YOLO datasets generated by `converter.py` against the original `lab42/cov-segm-v3` HF dataset.
    *   **Location:**
        *   Core Logic: `vibelab/dataops/cov_segm/convert_verifier.py`
        *   CLI Entrypoint: `vibelab/dataops/cov_segm/verifier_main.py`
        *   Tests: `tests/dataops/cov_segm/test_verifier.py`
        *   New Utilities:
            *   `vibelab/utils/common/geometry.py` (or `metrics.py`): `polygon_to_mask`, `compute_mask_iou`
    *   **CLI Interface:**
        *   Entry Point: `python -m vibelab.dataops.cov_segm.verifier_main`
        *   Args:
            *   `--train-split` (str, req): `train` or `validation`.
            *   `--mask-type` (str, req): `visible` or `full`.
            *   `--sample-count` (int, req): Number of samples to verify.
            *   `--target-root` (str, opt): Root of YOLO dataset (default: `$COV_SEGM_ROOT`). Expects `{target-root}/{mask-type}/{images|labels}/{train-split}/`.
            *   `--mapping-config` (str, opt): Path to mapping CSV (default: `configs/dataops/cov_segm_yolo_mapping.csv`).
            *   `--hf-dataset-path` (str, opt): HF dataset name (default: `lab42/cov-segm-v3`).
            *   `--bbox-min-iou` (float, opt): BBox IoU threshold (default: `0.95`).
            *   `--mask-min-iou` (float, opt): Mask IoU threshold (default: `0.95`).
            *   `--num-proc` (int, opt): Parallel processes for HF loading (default: `1`).
            *   `--seed` (int, opt): Random seed for sampling.
            *   `--debug` (bool, opt): Enable verbose logging.
            *   `--report-path` (str, opt): Path for detailed JSON mismatch report.
    *   **Core Logic:**
        1.  **(Setup)** Parse args, load mapping (`load_mapping_config`), determine paths.
        2.  **(Sample YOLO)**
            *   List files in `{target-root}/{mask-type}/labels/{train-split}/`.
            *   Limit listing: Use `os.scandir` or similar to efficiently get filenames. If total files > `max(10 * sample_count, 10000)`, list only up to that limit before sampling.
            *   Extract unique `sample_id`s.
            *   Randomly select `sample_count` unique IDs.
        3.  **(Load HF Data)**
            *   Load HF dataset split.
            *   Filter HF dataset for the `sampled_ids` using `dataset.filter(..., num_proc=...)`.
            *   Process filtered rows using `dataset.map(load_sample, num_proc=...)` to get `SegmSample` objects keyed by `sample_id`.
        4.  **(Per-Sample Verification)** For each `sample_id` successfully loaded from both sources:
            *   **Process YOLO:** Read label file, parse `(class_id, norm_poly_coords)`. Read image for `(H, W)`. For each YOLO instance:
                *   Convert norm_poly -> abs_poly.
                *   Generate binary mask using `polygon_to_mask(abs_poly, H, W)` (new util).
                *   Calculate bbox from mask.
                *   Store `(class_id, derived_mask, bbox)`. -> `YoloInstanceRecord`
            *   **Process Cov-Segm (HF):** Get `SegmSample`. Iterate `sample.segments`. Use `_get_sampled_mapping_info` (from `converter.py`) to check if segment should be converted & get `class_id`. If yes, iterate valid `SegmMask` objects (for `args.mask_type`). Store `(class_id, segm_mask.binary_mask, segm_mask.bbox)`. -> `OriginalInstanceRecord`
            *   **Match Instances:** For each `class_id` present in both lists:
                *   Build mask IoU matrix: `iou_matrix[i, j] = compute_mask_iou(original[i].mask, yolo[j].mask)` (new util).
                *   Find optimal pairs using `scipy.optimize.linear_sum_assignment` (or greedy) where mask IoU >= `mask-min-iou`.
            *   **Record Results:** Log matched pairs, lost instances (original w/o match), extra instances (YOLO w/o match). For matched pairs, compute bbox IoU using `calculate_iou` (from `bbox.py`) and check against `bbox-min-iou`.
            *   **Accumulate Stats:** Update global counters.
        5.  **(Report)** Aggregate stats. Print summary table using `format_statistics_table` (from `stats.py`). Save detailed JSON report if `--report-path` given. Exit 0 if all checks pass, 1 otherwise.
    *   **Failure Mode:** Continue processing all samples and report aggregate errors.

## Code Quality and Refactoring

*   **[ ] Refactor Mask Utilities and Matching Logic:**
    *   **Goal:** Improve code organization and reusability of mask utilities and instance matching algorithms.
    *   **Tasks:**
        *   Rename `geometry.py` to `mask.py` to better reflect its focus on mask-related operations.
        *   Rename `compute_mask_iou` to `calculate_mask_iou` for consistency with `bbox.py`.
        *   Create generic `label_match.py` module with a Hungarian algorithm-based matching system.
        *   Update imports in relevant files (converter.py, etc).
        *   Update test files to match the new structure.
        *   Add comprehensive unit tests for the new generic matching algorithm.
    *   **Benefits:**
        *   Better code organization and discoverability by function purpose.
        *   Consistent naming conventions across related utilities.
        *   Reduced code duplication by centralizing the matching algorithm.
        *   Easier maintainability and extensibility of the verification system.
    *   **Implementation Note:** Use `git mv` for renaming files to preserve file history.