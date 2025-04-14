import argparse
import csv
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

# Import the OOP data models
from src.dataops.cov_segm.datamodel import ClsSegment, SegmMask, SegmSample

# Import loader using full path
from src.dataops.cov_segm.loader import load_sample
from src.utils.common.geometry import mask_to_yolo_polygons

# Import statistics formatting utility
from src.utils.common.stats import format_statistics_table

# Global stats counters (initialized in main)
stats_counters: Dict[str, Any] = {
    "total_samples": 0,
    "processed_samples": 0,  # Samples successfully loaded by load_sample
    "segments_loaded": 0,
    "skipped_samples_load_error": 0,
    "skipped_segments_no_mapping": 0,  # Segments skipped
    "skipped_segments_sampling": 0,  # Segments skipped
    "skipped_segments_zero_masks": 0,  # Segments skipped
    "segments_processed": 0,  # Segments processed
    "masks_skipped_invalid": 0,
    "masks_for_annotation": 0,
    "mask_convert_no_polygon": 0,
    "mask_convert_multiple_polygons": 0,
    "skipped_existing_labels": 0,  # Labels skipped due to already existing
    "skipped_existing_images": 0,  # Images skipped due to already existing
    "generated_annotations": 0,
    "images_with_annotations": set(),
    "copied_images": 0,
    "class_masks_processed_per_segment": {},  # Dict mapping class_id -> List[int] (counts per segment)
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _get_sampled_mapping_info(
    segment: ClsSegment,
    phrase_map: Dict[str, Dict[str, Any]],
    global_sample_ratio: float = 1.0,
    random_sampling: bool = True,
) -> Optional[Tuple[Dict[str, Any], str]]:
    """Checks segment phrases against the map and applies sampling.

    Args:
        segment: The segment containing phrases to check against the mapping.
        phrase_map: Mapping of phrases to class information.
        global_sample_ratio: Global sampling ratio (0.0-1.0) to apply to all classes.
        random_sampling: If True, apply random sampling.

    Returns:
        Tuple[Dict, str] (mapping_info, matched_phrase) if a sampled match is found,
        otherwise None.
    """
    global stats_counters
    matched_phrase = None
    mapping_info = None

    # Iterate through phrases in the segment
    for phrase in segment.phrases:
        phrase_text = phrase.text.strip()
        if not phrase_text:  # Skip empty phrases
            logging.warning("Empty phrase in segment")
            continue

        current_mapping = phrase_map.get(phrase_text)
        if current_mapping:  # Found a mapping
            mapping_info = current_mapping
            matched_phrase = phrase_text
            break  # Stop at the first match

    # Check if a mapping was found
    if mapping_info is None or matched_phrase is None:
        stats_counters["skipped_segments_no_mapping"] += 1
        return None

    if random_sampling:
        # Apply sampling based on the found mapping and global ratio
        local_sampling_ratio = mapping_info.get("sampling_ratio", 1.0)
        # Combine global and local ratios (both are 0.0-1.0, so multiply)
        effective_ratio = global_sample_ratio * local_sampling_ratio

        if random.random() > effective_ratio:  # Note: This has > not < as per original code
            stats_counters["skipped_segments_sampling"] += 1
            return None

    return mapping_info, matched_phrase


def load_mapping_config(
    config_path: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, str]]:
    """Loads the phrase-to-class mapping configuration from a CSV file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Mapping config file not found: {config_path}")

    phrase_map: Dict[str, Dict[str, Any]] = {}
    class_names: Dict[int, str] = {}
    required_columns = {
        "yolo_class_id",
        "yolo_class_name",
        "hf_phrase",
        "sampling_ratio",
    }

    try:
        with open(config_path, mode="r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            if not required_columns.issubset(reader.fieldnames or []):
                missing = required_columns - set(reader.fieldnames or [])
                raise ValueError(f"Missing required columns in {config_path}: {missing}")

            for row_idx, row in enumerate(reader):
                try:
                    class_id = int(row["yolo_class_id"])
                    class_name = row["yolo_class_name"].strip()
                    sampling_ratio = float(row["sampling_ratio"])

                    # Get the single phrase (instead of parsing a comma-separated list)
                    phrase = row["hf_phrase"].strip()
                    if not phrase:
                        logging.warning(f"Row {row_idx + 2}: Empty hf_phrase. Skipping row.")
                        continue

                    if not (0.0 <= sampling_ratio <= 1.0):
                        raise ValueError(
                            f"Invalid sampling_ratio {sampling_ratio} for class"
                            f" {class_id} ({class_name}). Must be between 0.0 and 1.0."
                        )

                    if class_id in class_names and class_names[class_id] != class_name:
                        logging.warning(
                            f"Row {row_idx + 2}: Class ID {class_id} has conflicting names:"
                            f" '{class_names[class_id]}' and '{class_name}'. Using"
                            f" '{class_name}'."
                        )
                    class_names[class_id] = class_name

                    mapping_info = {
                        "class_id": class_id,
                        "class_name": class_name,
                        "sampling_ratio": sampling_ratio,
                    }

                    # Only one phrase per row in this format
                    if phrase in phrase_map:
                        logging.warning(
                            f"Row {row_idx + 2}: Phrase '{phrase}' maps to multiple classes."
                            f" Overwriting previous mapping {phrase_map[phrase]} with"
                            f" {mapping_info}."
                        )
                    phrase_map[phrase] = mapping_info

                except ValueError as e:
                    logging.error(
                        f"Error processing CSV row {row_idx + 2}: {row}. Skipping. Error: {e}"
                    )
                    continue
    except Exception:
        logging.exception(f"Failed to load mapping config from {config_path}")
        raise

    if not phrase_map:
        raise ValueError(f"No valid mappings found in {config_path}")

    logging.info(f"Loaded {len(phrase_map)} phrase mappings for {len(class_names)} classes.")
    return phrase_map, class_names


def generate_yolo_yaml(
    output_dir: Path,
    output_name: str,
    class_names: Dict[int, str],
    processed_split: str,
):
    """Generates the dataset.yaml file for YOLO training."""
    dataset_root = output_dir.resolve() / output_name
    yaml_dir = Path("configs/yolov11")
    yaml_dir.mkdir(parents=True, exist_ok=True)  # Use pathlib to ensure dir exists
    yaml_path = yaml_dir / f"cov_segm_segment_{output_name}.yaml"

    # Create a sorted dictionary of class names by ID
    sorted_class_names = dict(sorted(class_names.items()))

    yaml_content = {
        "path": str(dataset_root),
        # Standard names expected by YOLO, even if only one split is processed now.
        # User needs to run converter for each split.
        "train": "images/train",
        "val": "images/validation",
        "test": "images/test",
        # Store the name of the split actually processed by this run for reference
        "_processed_split": processed_split,
        "names": sorted_class_names,
    }

    try:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        logging.info(f"Generated dataset YAML at: {yaml_path}")
    except Exception:
        logging.exception(f"Failed to write YAML file to {yaml_path}")


def _setup_argparse() -> argparse.ArgumentParser:
    """Sets up the argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert lab42/cov-segm-v3 dataset to YOLO segmentation format."
    )
    parser.add_argument(
        "--mapping-config",
        type=str,
        default="configs/dataops/cov_segm_yolo_mapping.csv",
        help="Path to the CSV mapping file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=("Root directory for YOLO dataset output. Uses COV_SEGM_ROOT env var if not set."),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Subdirectory name for this dataset version (default: uses mask-tag).",
    )
    parser.add_argument(
        "--mask-tag",
        type=str,
        choices=["visible", "full"],
        default="visible",
        help="Type of mask to convert ('visible' or 'full').",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        required=True,
        help="Hugging Face dataset split to process (e.g., 'train', 'validation').",
    )
    parser.add_argument(
        "--hf-dataset-path",
        type=str,
        default="lab42/cov-segm-v3",
        help="Path or name of the Hugging Face dataset.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing).",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Global sampling ratio (0.0-1.0) applied to all classes in addition to their individual ratios.",
    )
    parser.add_argument(
        "--skip-zero",
        action="store_true",
        default=True,
        help="Skip segments with zero masks of the specified mask-tag.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Number of processor cores to use for parallel loading (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        default=False,
        help="Skip writing files if they already exist.",
    )
    return parser


def _configure_environment(args: argparse.Namespace) -> Tuple[Path, str, Path, Path]:
    """Configures logging, env vars, paths, and random seed."""
    # Validate sample_ratio
    if not (0.0 <= args.sample_ratio <= 1.0):
        raise ValueError(f"Invalid sample_ratio: {args.sample_ratio}. Must be between 0.0 and 1.0.")

    load_dotenv()  # Load environment variables from .env file

    output_dir_str = args.output_dir or os.getenv("COV_SEGM_ROOT")
    if not output_dir_str:
        raise ValueError(
            "Output directory must be specified via --output-dir or COV_SEGM_ROOT env var."
        )
    output_dir = Path(output_dir_str)

    output_name = args.output_name or args.mask_tag
    logging.info(f"Output dataset name: {output_name}")

    if args.seed is not None:
        random.seed(args.seed)
        logging.info(f"Using random seed: {args.seed}")

    # Define and create output paths
    yolo_root = output_dir / output_name
    image_dir = yolo_root / "images" / args.train_split
    label_dir = yolo_root / "labels" / args.train_split

    image_dir.mkdir(parents=True, exist_ok=True)  # Use pathlib
    label_dir.mkdir(parents=True, exist_ok=True)  # Use pathlib

    logging.info(f"Output images will be saved to: {image_dir}")
    logging.info(f"Output labels will be saved to: {label_dir}")

    return output_dir, output_name, image_dir, label_dir


def _load_data(
    args: argparse.Namespace,
) -> Tuple[Dataset, Dict[str, Dict[str, Any]], Dict[int, str]]:
    """Loads the mapping config and the Hugging Face dataset.

    Returns:
        A tuple of (dataset, phrase_map, class_names)
    """
    try:
        phrase_map, class_names = load_mapping_config(args.mapping_config)
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Failed to load mapping config: {e}")
        raise  # Re-raise after logging

    # Load dataset
    try:
        logging.info(f"Loading dataset '{args.hf_dataset_path}' split '{args.train_split}'...")
        dataset = load_dataset(args.hf_dataset_path, split=args.train_split)
        total_rows = len(dataset)
        logging.info(f"Dataset loaded with {total_rows} samples.")

    except Exception as e:
        logging.exception(f"Failed to load dataset: {e}")
        raise  # Re-raise after logging

    return dataset, phrase_map, class_names


def process_single_sample(sample_row):
    """Process a single sample row and return a serialized dictionary.

    This function is used with datasets.map() for parallel processing.
    It loads a sample using the loader and then serializes it for PyArrow compatibility.

    Args:
        sample_row: A raw dataset row.

    Returns:
        Dictionary containing the serialized sample, or {"error": True} if loading failed.
    """
    try:
        sample = load_sample(sample_row)
        if sample is None:
            return {"error": True, "reason": "load_sample returned None"}

        # Convert the SegmSample object to a serializable dict
        serialized = sample.to_dict()
        # Add a flag to indicate success
        serialized["error"] = False
        return serialized

    except Exception as e:
        # Log the error but keep processing
        logging.warning(f"Sample loading error: {e}", exc_info=True)
        # Return an error flag that can be checked later
        return {"error": True, "reason": str(e)}


def _process_samples(
    dataset: Dataset,
    phrase_map: Dict[str, Dict[str, Any]],
    image_dir: Path,
    label_dir: Path,
    args: argparse.Namespace,
):
    """Process samples in parallel using datasets.map() and the serialization methods.

    Uses the OOP data model's serialization capabilities to work with PyArrow.
    """
    global stats_counters

    # Track total sample count
    total_samples = len(dataset)
    stats_counters["total_samples"] = total_samples

    # Apply sample count limit if specified
    if args.sample_count is not None and args.sample_count < total_samples:
        dataset = dataset.select(range(args.sample_count))
        logging.info(
            f"Processing only the first {args.sample_count} samples (of {total_samples} total)."
        )
        total_samples = len(dataset)

    # Use datasets.map() for parallel processing
    logging.info(f"Loading samples using {args.num_proc} processes...")
    processed_dataset = dataset.map(
        process_single_sample,
        num_proc=args.num_proc if args.num_proc > 1 else None,
        load_from_cache_file=False,  # Don't use cache initially
        desc="Loading samples",  # Description for the progress bar
    )

    # Convert the loaded samples
    logging.info("Converting samples to YOLO format...")
    processed_iterator = tqdm(
        enumerate(processed_dataset), desc="Converting samples", total=len(processed_dataset)
    )

    # Track successful samples
    samples_with_annotations = 0

    # Process each loaded sample
    for i, sample_dict in processed_iterator:
        # Check for loading errors
        if sample_dict.get("error", False):
            stats_counters["skipped_samples_load_error"] += 1
            continue

        try:
            # Deserialize the sample
            sample = SegmSample.from_dict(sample_dict)
            stats_counters["processed_samples"] += 1
        except Exception as e:
            logging.error(f"Error deserializing sample at index {i}: {e}")
            stats_counters["skipped_samples_load_error"] += 1
            continue

        # Get sample properties
        sample_id = sample.id
        main_image = sample.image

        # Initialize tracking for annotation writing
        annotations_for_image = []

        # Process each segment
        for segment in sample.segments:
            stats_counters["segments_loaded"] += 1

            # Apply mapping and sampling
            # Don't apply random sampling if sample_count is specified (deterministic testing)
            random_sampling = args.sample_count is None
            mapping_result = _get_sampled_mapping_info(
                segment, phrase_map, args.sample_ratio, random_sampling
            )

            if mapping_result is None:
                # Mapping or sampling failed
                continue

            mapping_info, matched_phrase = mapping_result
            class_id = mapping_info["class_id"]

            # Get the appropriate masks to process based on mask_tag
            masks_to_process: List[SegmMask] = []
            if args.mask_tag == "visible":
                masks_to_process = segment.visible_masks
            elif args.mask_tag == "full":
                masks_to_process = segment.full_masks

            # Skip if no valid masks and skip_zero is enabled
            valid_masks = [mask for mask in masks_to_process if mask.is_valid]
            if len(valid_masks) != len(masks_to_process):
                stats_counters["masks_skipped_invalid"] += len(masks_to_process) - len(valid_masks)

            if args.skip_zero and len(valid_masks) == 0:
                stats_counters["skipped_segments_zero_masks"] += 1
                continue

            masks_processed = 0
            stats_counters["masks_for_annotation"] += len(valid_masks)
            # Process each valid mask
            for mask in valid_masks:
                try:
                    # Get the binary mask (already properly parsed by SegmMask)
                    binary_mask = mask.binary_mask

                    # Get image dimensions from the mask
                    mask_height, mask_width = binary_mask.shape

                    # Convert mask to YOLO polygons
                    polygons = mask_to_yolo_polygons(
                        binary_mask=binary_mask,
                        img_shape=(mask_height, mask_width),
                        connect_parts=True,
                    )

                    if polygons:
                        if len(polygons) > 1:
                            stats_counters["mask_convert_multiple_polygons"] += 1
                            # For now, just take the first polygon

                        # Process only the first polygon for now
                        polygon = polygons[0]
                        annotation_line = f"{class_id} {' '.join(map(str, polygon))}"
                        annotations_for_image.append(annotation_line)

                        # Update stats for annotations
                        stats_counters["generated_annotations"] += 1
                        masks_processed += 1
                    else:
                        stats_counters["mask_convert_no_polygon"] += 1

                except Exception as e:
                    logging.warning(
                        f"Sample {sample_id}: Failed to convert mask for mapped phrase"
                        f" '{matched_phrase}' to polygon. Error: {e}"
                    )

            # Update per-class stats
            if class_id not in stats_counters["class_masks_processed_per_segment"]:
                stats_counters["class_masks_processed_per_segment"][class_id] = []
            stats_counters["class_masks_processed_per_segment"][class_id].append(masks_processed)
            stats_counters["segments_processed"] += 1

        # If annotations were generated, save them and the image
        if len(annotations_for_image) > 0:
            # Set up file paths
            image_filename = f"{sample_id}.jpg"
            label_filename = f"{sample_id}.txt"
            label_path = label_dir / label_filename
            image_output_path = image_dir / image_filename

            # Write label file if needed
            if args.no_overwrite and label_path.exists():
                stats_counters["skipped_existing_labels"] += 1
            else:
                try:
                    with open(label_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(annotations_for_image) + "\n")
                    stats_counters["images_with_annotations"].add(str(sample_id))
                    samples_with_annotations += 1
                except Exception as e:
                    logging.error(f"Failed to write label file {label_path}: {e}")

            # Save image file if needed
            if args.no_overwrite and image_output_path.exists():
                stats_counters["skipped_existing_images"] += 1
            else:
                try:
                    if main_image.mode != "RGB":
                        main_image = main_image.convert("RGB")
                    main_image.save(image_output_path, "JPEG", quality=95)
                    stats_counters["copied_images"] += 1
                except Exception as e:
                    logging.error(
                        f"Failed to save image {image_filename} to {image_output_path}: {e}"
                    )

    logging.info(
        f"Finished processing. Images with annotations: {samples_with_annotations}/{stats_counters['processed_samples']}"
    )


def _log_summary(counters: Dict[str, Any], class_names: Dict[int, str]):
    """Logs the final conversion statistics from the global counters dict."""
    logging.info("Conversion finished.")
    logging.info(f"Total samples in dataset split: {counters['total_samples']}")
    logging.info(f"Successfully loaded samples: {counters['processed_samples']}")
    logging.info(f"Segments loaded: {counters['segments_loaded']}")
    logging.info(f"Samples skipped (load error): {counters['skipped_samples_load_error']}")
    logging.info(f"Segments skipped (no mapping): {counters['skipped_segments_no_mapping']}")
    logging.info(f"Segments skipped (sampling): {counters['skipped_segments_sampling']}")
    logging.info(f"Segments skipped (zero masks): {counters['skipped_segments_zero_masks']}")
    logging.info(f"Segments processed: {counters['segments_processed']}")
    logging.info(f"Masks skipped (invalid): {counters['masks_skipped_invalid']}")
    logging.info(f"Masks for annotation: {counters['masks_for_annotation']}")
    logging.info(f"Masks skipped (no polygon): {counters['mask_convert_no_polygon']}")
    logging.info(f"Masks skipped (multiple polygons): {counters['mask_convert_multiple_polygons']}")
    logging.info(f"Labels skipped (already exist): {counters['skipped_existing_labels']}")
    logging.info(f"Images skipped (already exist): {counters['skipped_existing_images']}")
    logging.info(f"Total annotations generated: {counters['generated_annotations']}")
    logging.info(f"Unique images with annotations: {len(counters['images_with_annotations'])}")
    logging.info(f"Images copied to output: {counters['copied_images']}")

    # Calculate and log per-class statistics
    if counters["class_masks_processed_per_segment"]:
        logging.info("Annotation statistics:")

        # Prepare data for stats.format_statistics_table
        stats_data = {}
        for class_id, counts in counters["class_masks_processed_per_segment"].items():
            class_name = class_names.get(class_id, f"Unknown-{class_id}")
            stats_data[class_name] = counts

        # Define format string for the table
        # This will create a table with class name, count, mean, median, and max
        format_string = (
            "{key:<20} {count:>6d} {sum:>6d} {mean:>7.1f} {p25:>6d} {p50:>6d} {p75:>6d} {max:>6d}"
        )

        # Generate the formatted table using the utility
        table_lines = format_statistics_table(stats_data, format_string)

        # Log each line of the table
        for line in table_lines:
            logging.info(line)


def main():
    """Main function to orchestrate the dataset conversion."""
    global stats_counters  # Declare intent to modify global dict
    parser = _setup_argparse()
    args = parser.parse_args()

    try:
        # Configure environment
        try:
            output_dir, output_name, image_dir, label_dir = _configure_environment(args)
        except ValueError as e:
            logging.error(f"Configuration error: {e}")
            return 1

        # Load dataset and mapping
        try:
            dataset, phrase_map, class_names = _load_data(args)
        except Exception as e:
            logging.error(f"Failed to load dataset or mapping: {e}", exc_info=True)
            return 1

        # Process samples
        try:
            _process_samples(dataset, phrase_map, image_dir, label_dir, args)
        except Exception as e:
            logging.error(f"Error during sample processing: {e}", exc_info=True)
            return 1

        # Log summary statistics
        _log_summary(stats_counters, class_names)

        # Generate YAML file
        try:
            if class_names:
                generate_yolo_yaml(output_dir, output_name, class_names, args.train_split)
            else:
                logging.warning("No class names found from mapping, skipping YAML generation.")
        except Exception as e:
            logging.error(f"Failed to generate YAML file: {e}", exc_info=True)
            return 1

        logging.info("Conversion completed successfully.")
        return 0

    except Exception as e:
        logging.error(f"Conversion failed with unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
