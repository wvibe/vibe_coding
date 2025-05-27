# VLM Processing System Design (Qwen2.5-VL Focus)

## 1. Overview

This document outlines the design for a modular Python-based system to handle Vision Language Models (VLMs), with an initial implementation targeting the Qwen2.5-VL family of models. The system will support:

*   **Inference:** Generating outputs from a VLM given an image and/or text prompt.
*   **Fine-tuning:** Adapting a pre-trained VLM to a specific dataset or task using techniques like LoRA.
*   **Benchmarking:** Evaluating the performance of base and fine-tuned VLMs on relevant datasets.

The system is designed to be extensible for other VLM architectures in the future by encapsulating model-specific logic.

## 2. Module Structure

All VLM-related Python scripts will reside under the `src/vibelab/models/hf/vlm/` directory.

*   **`qwen_vl_utils.py`:**
    *   **Purpose:** Contains utility functions specifically for Qwen VL models (initially Qwen2.5-VL). This module aims to abstract away Qwen-specific details from the main workflow scripts.
    *   **Key Responsibilities:**
        *   Loading Qwen VL models and processors (including quantization).
        *   Formatting chat messages/prompts according to Qwen's template.
        *   Preparing image inputs.
        *   Wrapping model generation calls.
        *   Providing default configurations (e.g., for BitsAndBytes, LoRA).

*   **`inference_vlm.py`:**
    *   **Purpose:** Command-line script for running inference on a VLM.
    *   **Key Responsibilities:**
        *   Parsing CLI arguments (model ID, dataset details, output directory, sampling options, etc.).
        *   Loading a specified dataset (e.g., ChartQA).
        *   Iterating through dataset samples.
        *   For each sample:
            *   Preparing inputs using `qwen_vl_utils.py`.
            *   Generating a response from the VLM.
            *   Saving the input details, image (if applicable), and VLM output to structured files (`{sample_id}_in.json`, `{sample_id}_img.jpg`, `{sample_id}_out.json`).

*   **`finetune_vlm.py`:**
    *   **Purpose:** Command-line script for fine-tuning a VLM using `trl`'s `SFTTrainer`.
    *   **Key Responsibilities:**
        *   Parsing CLI arguments (base model ID, dataset details, training hyperparameters, LoRA config, output directory).
        *   Loading and preparing the fine-tuning dataset.
        *   Implementing a custom data collator compatible with `SFTTrainer` (v0.17.0+) for VLM data.
        *   Setting up PEFT (LoRA) configuration using utilities from `qwen_vl_utils.py`.
        *   Initializing and running `SFTTrainer`.
        *   Saving the trained adapter (and optionally the merged model).

*   **`benchmark_vlm.py`:**
    *   **Purpose:** Command-line script for evaluating VLM performance.
    *   **Key Responsibilities:**
        *   Parsing CLI arguments (model ID(s), adapter paths, benchmark dataset, metrics, output directory).
        *   Loading base and/or fine-tuned models.
        *   Iterating through the benchmark dataset.
        *   Generating predictions for each sample.
        *   Calculating specified performance metrics (e.g., ROUGE, BLEU, VQA accuracy).
        *   Saving aggregated benchmark results.

## 3. Data Flow Examples

*   **Inference Flow:**
    1.  `inference_vlm.py` receives dataset info, model ID, output path.
    2.  Loads dataset (e.g., ChartQA).
    3.  For each sample:
        a.  Image and text prompt are extracted.
        b.  `qwen_vl_utils.load_qwen_vl_model_and_processor` (called once or as needed).
        c.  `qwen_vl_utils.load_image_pil` loads/processes the image.
        d.  `qwen_vl_utils.prepare_qwen_vl_chat_messages` formats the prompts.
        e.  `qwen_vl_utils.get_model_inputs_for_generation` prepares tensors.
        f.  `qwen_vl_utils.generate_with_qwen_vl` gets the model output.
        g.  Input details, image, and output are saved to files.

*   **Fine-tuning Flow:**
    1.  `finetune_vlm.py` receives model ID, dataset info, training args.
    2.  `qwen_vl_utils.load_qwen_vl_model_and_processor` loads the base model.
    3.  Dataset is loaded and preprocessed (image loading, prompt formatting into a single sequence for SFT).
    4.  `DataCollatorForVLMSFT` prepares batches (tokenizes text, incorporates image information).
    5.  `qwen_vl_utils.get_default_lora_config_for_qwen_vl` provides PEFT setup.
    6.  `SFTTrainer` trains the model.
    7.  Adapter is saved.

## 4. Configuration

*   Primary configuration will be via command-line arguments using Python's `argparse` module for each script (`inference_vlm.py`, `finetune_vlm.py`, `benchmark_vlm.py`).
*   Model-specific defaults (e.g., quantization for Qwen2.5-VL) will be encapsulated within `qwen_vl_utils.py` but can be overridden by CLI arguments where appropriate.

## 5. Key Libraries and Dependencies

*   **Core ML/Transformers:**
    *   `torch`
    *   `transformers` (Hugging Face, potentially from git for latest Qwen2.5-VL support)
    *   `accelerate`
    *   `bitsandbytes` (for quantization)
*   **Fine-tuning:**
    *   `trl` (specifically version 0.17.0 or as specified)
    *   `peft` (for LoRA)
*   **Data Handling:**
    *   `datasets` (Hugging Face, for loading e.g. ChartQA)
    *   `Pillow` (PIL for image manipulation)
    *   `requests` (for downloading images from URLs)
*   **General:**
    *   `python 3.12` (as per project stack)

## 6. Extensibility

*   To support a new VLM architecture (e.g., LLaVA):
    1.  Create a new utility module, e.g., `llava_utils.py`, mirroring the relevant functionalities of `qwen_vl_utils.py` but tailored for LLaVA.
    2.  Modify `inference_vlm.py`, `finetune_vlm.py`, and `benchmark_vlm.py` to accept a `model_type` argument.
    3.  Based on `model_type`, these scripts would dynamically import and use functions from the corresponding utility module (`qwen_vl_utils.py` or `llava_utils.py`).

## 7. Error Handling and Logging

*   Standard Python `logging` module will be used for informative messages, warnings, and errors.
*   Robust error handling (e.g., for file I/O, network requests, model loading issues) will be implemented.

## 8. Implementation Status

### ✅ Completed (Phase 1)

*   **Project Structure:** Created `src/vibelab/models/hf/vlm/` directory with proper `__init__.py` files
*   **Core Utilities (`qwen_vl_utils.py`):** ✅ IMPLEMENTED
    *   `load_qwen_vl_model_and_processor()` - Model and processor loading with quantization support
    *   `load_image_pil()` - Image loading from URLs, paths, or PIL objects
    *   `format_qwen_vl_messages()` - Chat message formatting for Qwen2.5-VL
    *   `format_data()` - Dataset sample formatting for inference
    *   `generate_text_from_sample()` - End-to-end text generation from samples
    *   Default configurations and constants
*   **Dataset Utilities (`dataset_utils.py`):** ✅ IMPLEMENTED
    *   `load_hf_dataset()` - HuggingFace dataset loading
    *   `get_dataset_columns()` - Dataset column mapping (ChartQA support)
    *   `parse_sample_slices()` - Array slice parsing for sample selection
    *   Extensible column mapping system for different datasets
*   **Inference Script (`inference_vlm.py`):** ✅ IMPLEMENTED
    *   Complete CLI argument parsing with simplified arguments
    *   Dataset loading and sample slicing (`[start:end]` format)
    *   Model loading and inference pipeline
    *   Structured output saving (`{sample_id}_in.json`, `{sample_id}_img.jpg`, `{sample_id}_out.json`)
    *   Comprehensive logging and error handling
    *   Default output directory: `outputs/vlm`

### 🚧 Placeholder Files Created

*   **Fine-tuning Script (`finetune_vlm.py`):** Placeholder created for Phase 2
*   **Benchmarking Script (`benchmark_vlm.py`):** Placeholder created for Phase 3

### 📋 Key Design Decisions Made

*   **Simplified CLI Arguments:** Removed over-designed arguments, kept only `--sample_slices` for sample selection
*   **Modular Design:** Separated dataset utilities from model utilities for better organization
*   **Extensible Column Mapping:** Used constant mapping dictionary for dataset-specific column names
*   **Environment Dependencies:** Added `accelerate` dependency, confirmed `vbl` conda environment setup