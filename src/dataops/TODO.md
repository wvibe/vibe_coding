# Data Operations TODO List

This document tracks the development tasks for the `src/dataops` module.

## High Priority: `lab42/cov-segm-v3` Support

*   **[ ] Implement Common S3 Fetcher:**
    *   Create `common/s3_fetcher.py`.
    *   **Goal:** Provide a function `fetch_s3_uri(uri)` that downloads and returns content bytes.
    *   **Action:** Investigate `ResourceManager` usage in `ref/llm/datasets/conversions/` scripts. Adapt its core S3 fetching logic if possible. Otherwise, implement minimally using `boto3`. Ensure secure AWS credential handling. *This does NOT replace HF datasets' loading; it only fetches URIs found embedded in data fields.*
*   **[ ] Define `cov_segm` Data Models:**
    *   Create `cov_segm/datamodel.py`.
    *   Define Pydantic or dataclasses for the structures within the `conversations` JSON (e.g., `Phrase`, `ImageURI`, `MaskInfo`, `ConversationItem`). Adapt from `ref/llm/datasets/common.py`.
*   **[ ] Implement `cov_segm` Parser:**
    *   Create `cov_segm/parser.py`.
    *   Use models from `datamodel.py`.
    *   Implement logic to parse the `conversations` JSON string, validate structure, and extract key information (image URIs, prompts, mask columns/URIs, positive values). Handle `instance_masks`. Adapt parsing logic from relevant `ref/llm/datasets/conversions/attributes/` scripts.
*   **[ ] Implement `cov_segm` Loader:**
    *   Create `cov_segm/loader.py`.
    *   Implement `load_processed_sample(raw_sample)`:
        *   Uses `datasets.load_dataset` externally to get `raw_sample`.
        *   Uses `cov_segm.parser` to get URIs/info.
        *   Uses `common.s3_fetcher` to download image/mask data from parsed S3 URIs.
        *   Handles cases where mask data might be directly in `raw_sample` fields (e.g., `mask_0`).
        *   Returns a processed sample object/dictionary containing loaded PIL Images / NumPy arrays and structured prompt/mask info (using data models is ideal).
*   **[ ] Implement `cov_segm` Visualization:**
    *   Create `cov_segm/visualizer.py`.
    *   **Action:** Adapt plotting logic (potentially referencing `ref/llm/datasets/visualizers/visualize_segm_json_vqa.py` and the exploration notebook code).
    *   Create functions accepting the processed sample from the loader to plot images, masks per prompt, and overlays.
*   **[ ] Update Exploration Notebook:**
    *   Modify `notebooks/dataset/lab42_segm_explorer.ipynb` to import and use the loader (`cov_segm.loader`) and visualization (`cov_segm.visualizer`) utilities.

## Medium Priority: YOLO Conversion

*   **[ ] Implement `cov_segm` to YOLO Format Converter:**
    *   Create `cov_segm/converter.py`.
    *   Add function `convert_to_yolo(processed_sample, output_dir, class_map)`.
    *   Define logic to generate YOLO bounding boxes/segmentation masks from the processed sample's masks.
    *   Handle mapping text prompts to class IDs.
    *   Write images and labels to the specified `output_dir` in YOLO format.

## Low Priority

*   **[ ] Add Unit Tests:** Implement tests for fetcher, parser, loader, converter.
*   **[ ] Add Support for Other Datasets:** Create new subdirectories (e.g., `dataops/other_dataset/`) following the established pattern.
*   **[ ] Refactor Common Logic:** Identify and move any truly dataset-agnostic logic emerging in dataset modules to `common/`.