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