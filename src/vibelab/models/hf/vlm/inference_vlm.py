#!/usr/bin/env python3
"""
VLM Inference Script

Run inference with Qwen2.5-VL on VQA datasets and save structured outputs.

Usage:
    python -m vibelab.models.hf.vlm.inference_vlm \
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \
        --dataset_name HuggingFaceM4/ChartQA \
        --dataset_split test \
        --sample_slices "[0:5]" \
        --output_dir ./inference_outputs \
        --max_new_tokens 512
"""

import argparse
import json
import logging
import os
import time
from typing import Any, Dict

from vibelab.models.hf.vlm.dataset_utils import (
    get_dataset_columns,
    load_hf_dataset,
    parse_sample_slices,
)
from vibelab.models.hf.vlm.qwen_vl_utils import (
    DEFAULT_QWEN_VL_MODEL_ID,
    DEFAULT_SYSTEM_PROMPT,
    generate_text_from_sample,
    load_image_pil,
    load_qwen_vl_model_and_processor,
)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run VLM inference on VQA datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_QWEN_VL_MODEL_ID,
        help="Hugging Face model identifier",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceM4/ChartQA",
        help="Hugging Face dataset name",
    )

    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to use")

    parser.add_argument(
        "--sample_slices",
        type=str,
        required=True,
        help="Sample slice in format '[start:end]', e.g., '[0:10]', '[150:160]'",
    )

    parser.add_argument(
        "--output_dir", type=str, default="outputs/vlm", help="Directory to save inference outputs"
    )

    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate"
    )

    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for text generation (default: 1.0)"
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def prepare_output_directory(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")


def save_sample_outputs(
    sample_id: str,
    question: str,
    image_reference: str,
    generated_text: str,
    formatted_prompt: str,
    generation_time: float,
    output_dir: str,
    image_object=None,
    model_id: str = "",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ground_truth=None,
) -> None:
    """
    Save the input, image, and output files for a single sample.

    Args:
        sample_id: Unique identifier for the sample
        question: The original question
        image_reference: Reference to the image (URL or path)
        generated_text: Model's generated response
        formatted_prompt: The full formatted prompt sent to the model
        generation_time: Time taken for generation in seconds
        output_dir: Directory to save outputs
        image_object: PIL Image object (if available)
        model_id: Model identifier
        system_prompt: System prompt used
        ground_truth: Ground truth answer (if available)
    """
    base_filename = f"{sample_id}"

    # Save input JSON
    input_data = {
        "model_id": model_id,
        "system_prompt": system_prompt,
        "user_question": question,
        "image_reference": image_reference,
        "full_templated_prompt": formatted_prompt,
    }

    input_path = os.path.join(output_dir, f"{base_filename}_in.json")
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(input_data, f, indent=2, ensure_ascii=False)

    # Save image if available
    if image_object is not None:
        image_path = os.path.join(output_dir, f"{base_filename}_img.jpg")
        image_object.save(image_path, "JPEG", quality=95)
        logging.debug(f"Saved image: {image_path}")

    # Save output JSON
    output_data = {
        "generated_text": generated_text,
        "generation_time_seconds": generation_time,
        "max_new_tokens_used": len(generated_text.split()),  # Rough estimate
    }

    # Add ground truth if available
    if ground_truth is not None:
        output_data["ground_truth"] = ground_truth

    output_path = os.path.join(output_dir, f"{base_filename}_out.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logging.debug(f"Saved outputs for sample {sample_id}")


def process_single_sample(
    sample: Dict[str, Any],
    model,
    processor,
    columns: Dict[str, str],
    args: argparse.Namespace,
    sample_index: int = 0,
) -> None:
    """
    Process a single dataset sample.

    Args:
        sample: Dataset sample
        model: Loaded VLM model
        processor: Loaded VLM processor
        columns: Column name mappings
        args: Command line arguments
        sample_index: Index of the sample
    """
    try:
        # Extract sample data
        sample_id = str(sample[columns["id_column"]]) if columns["id_column"] is not None else str(sample_index)
        question = sample[columns["question_column"]]
        image_ref = sample[columns["image_column"]]

        # Extract ground truth answer if available
        ground_truth = None
        if "answer_column" in columns and columns["answer_column"] is not None:
            if columns["answer_column"] in sample:
                ground_truth = sample[columns["answer_column"]]

        logging.info(f"Processing sample {sample_id}: {question[:50]}...")

        # Load image
        image_object = None
        if image_ref is not None:
            try:
                image_object = load_image_pil(image_ref)
            except Exception as e:
                logging.warning(f"Failed to load image for sample {sample_id}: {e}")
                image_ref = "failed_to_load"

        # Generate response
        start_time = time.time()
        generated_text, formatted_prompt = generate_text_from_sample(
            model=model,
            processor=processor,
            question=question,
            image_input=image_object,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=args.temperature,
        )
        generation_time = time.time() - start_time

        # Save outputs
        save_sample_outputs(
            sample_id=sample_id,
            question=question,
            image_reference=str(image_ref),
            generated_text=generated_text,
            formatted_prompt=formatted_prompt,
            generation_time=generation_time,
            output_dir=args.output_dir,
            image_object=image_object,
            model_id=args.model_id,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            ground_truth=ground_truth,
        )

        logging.info(f"Completed sample {sample_id} in {generation_time:.2f}s")

    except Exception as e:
        sample_id_for_error = sample.get(columns["id_column"], sample_index) if columns["id_column"] is not None else sample_index
        logging.error(f"Error processing sample {sample_id_for_error}: {e}")


def main():
    """Main function."""
    args = parse_arguments()
    setup_logging(args.log_level)

    logging.info("Starting VLM inference script")
    logging.info(f"Model: {args.model_id}")
    logging.info(f"Dataset: {args.dataset_name} ({args.dataset_split})")
    logging.info(f"Sample slices: {args.sample_slices}")

    # Prepare output directory
    prepare_output_directory(args.output_dir)

    # Load model and processor
    logging.info("Loading model and processor...")
    model, processor = load_qwen_vl_model_and_processor(model_id=args.model_id)

    # Load dataset
    logging.info("Loading dataset...")
    dataset = load_hf_dataset(args.dataset_name, args.dataset_split)

    # Get column mappings
    columns = get_dataset_columns(args.dataset_name, args.dataset_split)
    logging.info(f"Using columns: {columns}")

    # Parse sample slices
    sample_slice = parse_sample_slices(args.sample_slices)
    selected_samples = dataset[sample_slice]

    # Convert to list if it's a single sample
    # Use any available column to check if we have multiple samples
    first_column = next(iter(selected_samples.keys()))
    if not isinstance(selected_samples[first_column], list):
        # Single sample case
        selected_samples = {k: [v] for k, v in selected_samples.items()}

    num_samples = len(selected_samples[first_column])
    logging.info(f"Processing {num_samples} samples")

    # Calculate the starting index from the slice
    slice_start = sample_slice.start if sample_slice.start is not None else 0

    # Process each sample
    for i in range(num_samples):
        sample = {k: v[i] for k, v in selected_samples.items()}
        original_index = slice_start + i  # Calculate original dataset index
        process_single_sample(sample, model, processor, columns, args, original_index)

        # Log progress
        if (i + 1) % 5 == 0 or (i + 1) == num_samples:
            logging.info(f"Progress: {i + 1}/{num_samples} samples completed")

    logging.info(f"Inference completed! Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
