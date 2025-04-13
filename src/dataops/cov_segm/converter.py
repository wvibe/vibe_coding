import argparse
import csv
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

# Import the necessary types from datamodel using full paths
from src.dataops.cov_segm.datamodel import (
    ProcessedConversationItem,
    ProcessedCovSegmSample,
    ProcessedMask,
)

# Import loader using full path
from src.dataops.cov_segm.loader import load_sample
from src.utils.common.geometry import mask_to_yolo_polygons

# Global stats counters (initialized in main)
stats_counters: Dict[str, Any] = {
    "total_samples": 0,
    "total_conversations": 0,
    "total_phrases": 0,
    "processed_samples": 0,  # Samples successfully loaded by load_sample
    "skipped_samples_load_error": 0,
    "skipped_samples_no_mapping": 0,  # Conversations skipped
    "skipped_samples_sampling": 0,  # Conversations skipped
    "skipped_samples_zero_masks": 0,  # Conversations skipped
    "mask_convert_no_polygon": 0,
    "mask_convert_multiple_polygons": 0,
    "conversations_processed": 0,
    "skipped_existing_labels": 0,  # Labels skipped due to already existing
    "skipped_existing_images": 0,  # Images skipped due to already existing
    "generated_annotations": 0,
    "images_with_annotations": set(),
    "copied_images": 0,
    "class_annotations_per_image": {},  # Dict mapping class_id -> List[int] (counts per image)
}


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _get_sampled_mapping_info(
    conversation: ProcessedConversationItem,
    phrase_map: Dict[str, Dict[str, Any]],
    random_sampling: bool = True,
) -> Optional[Tuple[Dict[str, Any], str]]:
    """Checks conversation phrases against the map and applies sampling.

    Args:
        conversation: The processed conversation item.
        phrase_map: Mapping of phrases to class information.
        random_sampling: If True, apply random sampling.

    Returns:
        Tuple[Dict, str] (mapping_info, matched_phrase) if a sampled match is found,
        otherwise None.
    """
    global stats_counters
    matched_phrase = None
    mapping_info = None

    # Iterate through phrases (now dicts) in the conversation
    for phrase_dict in conversation.get("phrases", []):
        phrase_text = phrase_dict.get("text", "").strip()
        if not phrase_text:  # Skip empty phrases
            raise ValueError(f"Empty phrase in conversation {conversation['id']}")
            continue

        stats_counters["total_phrases"] += 1

        current_mapping = phrase_map.get(phrase_text)
        if current_mapping:  # Found a mapping
            mapping_info = current_mapping
            matched_phrase = phrase_text
            break  # Stop at the first match

    # Check if a mapping was found
    if mapping_info is None or matched_phrase is None:
        stats_counters["skipped_samples_no_mapping"] += 1
        # Reduced logging: No message here
        return None

    if random_sampling:
        # Apply sampling based on the found mapping
        # Skip random sampling in debug mode (when sample_count is specified)
        sampling_ratio = mapping_info.get("sampling_ratio", 1.0)
        if random.random() > sampling_ratio:
            stats_counters["skipped_samples_sampling"] += 1
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
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--skip-zero",
        action="store_true",
        default=True,
        help=(
            "Skip conversation items with zero masks of the specified mask-tag."
            " Use --no-skip-zero to disable."
        ),
    )
    parser.add_argument(
        "--no-skip-zero",
        action="store_false",
        dest="skip_zero",
        help="Process conversation items even if they have zero masks.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        default=False,
        help="Skip writing files if they already exist.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed (DEBUG level) logging.",
    )
    return parser


def _configure_environment(args: argparse.Namespace) -> Tuple[Path, str, Path, Path]:
    """Configures logging, env vars, paths, and random seed."""
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(log_level)

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


def _load_data(args: argparse.Namespace) -> Tuple[Any, Dict[str, Dict[str, Any]], Dict[int, str]]:
    """Loads the mapping config and the Hugging Face dataset."""
    try:
        phrase_map, class_names = load_mapping_config(args.mapping_config)
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Failed to load mapping config: {e}")
        raise  # Re-raise after logging

    # Load dataset
    try:
        logging.info(f"Loading dataset '{args.hf_dataset_path}' split '{args.train_split}'...")
        dataset = load_dataset(args.hf_dataset_path, split=args.train_split)
        if args.sample_count:
            dataset = dataset.select(range(args.sample_count))
            logging.info(f"Processing only the first {args.sample_count} samples.")
    except Exception as e:
        logging.exception(f"Failed to load dataset: {e}")
        raise  # Re-raise after logging

    return dataset, phrase_map, class_names


def _process_samples(
    dataset: Any,
    phrase_map: Dict[str, Dict[str, Any]],
    image_dir: Path,
    label_dir: Path,
    args: argparse.Namespace,
):
    """Iterates through samples, converts masks, writes output. Updates global stats."""
    global stats_counters
    total_samples = len(dataset)
    stats_counters["total_samples"] = total_samples

    logging.info(f"Starting conversion for {total_samples} samples...")

    for i, sample_row in enumerate(tqdm(dataset, desc="Converting samples")):
        # Removed incorrect sample_id assignment here
        annotations_for_image = []
        image_has_annotations = False

        # Placeholder ID for error logging *before* loading succeeds
        temp_id_for_logging = f"sample_at_index_{i}"

        try:
            processed_sample: ProcessedCovSegmSample = load_sample(sample_row)
            stats_counters["processed_samples"] += 1
            # ---- Get the CORRECT ID after loading ----
            correct_id = processed_sample["id"]
            # Validate and clean the ID if necessary (e.g., remove spaces), though expected format is usually safe
            if not isinstance(correct_id, str) or not correct_id:
                logging.warning(
                    f"Invalid or missing ID '{correct_id}' in loaded sample for index {i}. Using fallback."
                )
                correct_id = f"invalid_id_index_{i}"
            # ---- Define filenames using the correct ID ----
            image_filename = f"{correct_id}.jpg"
            label_filename = f"{correct_id}.txt"
            label_path = label_dir / label_filename
            image_output_path = image_dir / image_filename
            # --------------------------------------------

        except Exception as e:
            logging.warning(
                # Use the temporary ID here since loading failed
                f"Skipping sample {temp_id_for_logging} due to loading error: {e}",
                exc_info=args.verbose,
            )
            stats_counters["skipped_samples_load_error"] += 1
            continue

        # Use correct_id for further processing
        main_image: Image.Image = processed_sample["image"]

        for conversation in processed_sample["processed_conversations"]:
            stats_counters["total_conversations"] += 1

            # random sampling only if no sample_count
            random_sampling = args.sample_count is None
            mapping_result = _get_sampled_mapping_info(conversation, phrase_map, random_sampling)

            if mapping_result is None:
                continue

            mapping_info, matched_phrase = mapping_result
            class_id = mapping_info["class_id"]

            masks_to_process: List[ProcessedMask] = []
            if args.mask_tag == "visible":
                masks_to_process = conversation.get("processed_instance_masks", [])
            elif args.mask_tag == "full":
                masks_to_process = conversation.get("processed_full_masks", [])

            if args.skip_zero and not masks_to_process:
                stats_counters["skipped_samples_zero_masks"] += 1
                continue

            for mask_data in masks_to_process:
                # Convert PIL Image to numpy array
                mask_np = np.array(mask_data["mask"])
                if mask_np.dtype != bool and mask_np.dtype.kind != "i":
                    mask_np = mask_np > 0
                elif mask_np.dtype.kind == "i":
                    unique_vals = np.unique(mask_np)
                    if not np.all(np.isin(unique_vals, [0, 1])):
                        mask_np = mask_np > 0

                try:
                    # Get image dimensions from the mask
                    mask_height, mask_width = mask_np.shape
                    polygons = mask_to_yolo_polygons(
                        binary_mask=mask_np,
                        img_shape=(mask_height, mask_width),
                        connect_parts=True,  # Process all contour together
                    )
                    if polygons:
                        if len(polygons) > 1:
                            stats_counters["mask_convert_multiple_polygons"] += 1

                        # only process the first polygon for now
                        polygon = polygons[0]
                        annotation_line = f"{class_id} {' '.join(map(str, polygon))}"
                        annotations_for_image.append(annotation_line)
                        image_has_annotations = True
                        # Update global stats for total annotations and per-class annotations
                        stats_counters["generated_annotations"] += 1
                    else:
                        stats_counters["mask_convert_no_polygon"] += 1

                except Exception as e:
                    logging.warning(
                        # Use correct_id here
                        f"Sample {correct_id}: Failed to convert mask for mapped phrase"
                        f" '{matched_phrase}' to polygon. Error: {e}",
                        exc_info=args.verbose,
                    )

            # Update per-class stats and total conversations processed
            if class_id not in stats_counters["class_annotations_per_image"]:
                stats_counters["class_annotations_per_image"][class_id] = []
            stats_counters["class_annotations_per_image"][class_id].append(len(masks_to_process))
            stats_counters["conversations_processed"] += 1

        if image_has_annotations:
            # Check if label file already exists
            if args.no_overwrite and label_path.exists():
                stats_counters["skipped_existing_labels"] += 1
            else:
                try:
                    with open(label_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(annotations_for_image) + "\n")
                    # Use correct_id here, converting to string just in case
                    stats_counters["images_with_annotations"].add(str(correct_id))
                except Exception as e:
                    logging.error(f"Failed to write label file {label_path}: {e}")

            # Check if image file already exists
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

    # No return value - stats updated globally


def _print_statistics_table(
    class_counts: Dict[int, List[int]],
    class_names: Dict[int, str],
    format_spec: str = "count,mean,p25,p50,p75,p90",
) -> List[str]:
    """Format a complete statistics table with headers and class rows based on format specification.

    Args:
        class_counts: Dictionary mapping class IDs to lists of counts.
        class_names: Dictionary mapping class IDs to class names.
        format_spec: Comma-separated list of statistics to include.
                     Supported: count, mean, min, max, and percentiles as pN (e.g., p25, p90).

    Returns:
        List of formatted lines ready for logging.
    """
    parts = [part.strip().lower() for part in format_spec.split(",")]

    # Create header parts - just capitalize each stat name
    header_parts = ["ID", "Name"]
    for part in parts:
        header_parts.append(part.capitalize())

    # Format header
    header = f"{'ID':<2} {'Name':<15} " + " ".join(f"{part:<6}" for part in header_parts[2:])
    logging.info(header)
    logging.info("-" * (len(header_parts) * 6 + 10))  # Divider line based on number of columns

    # Process each class
    for class_id, counts in sorted(class_counts.items()):
        if not counts:
            continue

        class_name = class_names.get(class_id, "N/A")

        # Format class row
        row = f"{class_id:<2} {class_name[:15]:<15} "

        for part in parts:
            if part == "count":
                value = len(counts)
                value_str = f"{value:<7d}"
            elif part == "sum":
                value = np.sum(counts)
                value_str = f"{value:<7d}"
            elif part == "mean":
                value = np.mean(counts)
                value_str = f"{value:<7.1f}"
            elif part == "min":
                value = np.min(counts)
                value_str = f"{value:<7d}"
            elif part == "max":
                value = np.max(counts)
                value_str = f"{value:<7d}"
            elif part.startswith("p") and part[1:].isdigit():
                value = int(np.percentile(counts, int(part[1:]), method="nearest"))
                value_str = f"{value:<7d}"
            row += value_str

        logging.info(row)


def _log_summary(counters: Dict[str, Any], class_names: Dict[int, str]):
    """Logs the final conversion statistics from the global counters dict."""
    logging.info("Conversion finished.")
    logging.info(f"Total samples in dataset split: {counters['total_samples']}")
    logging.info(f"Successfully loaded samples: {counters['processed_samples']}")
    logging.info(f"Samples skipped (load error): {counters['skipped_samples_load_error']}")
    logging.info(f"Total conversations: {counters['total_conversations']}")
    logging.info(f"Total phrases in conversations: {counters['total_phrases']}")
    logging.info(f"Conversations skipped (no mapping): {counters['skipped_samples_no_mapping']}")
    logging.info(f"Conversations skipped (sampling): {counters['skipped_samples_sampling']}")
    logging.info(f"Conversations skipped (zero masks): {counters['skipped_samples_zero_masks']}")
    logging.info(f"Masks skipped (no polygon): {counters['mask_convert_no_polygon']}")
    logging.info(f"Masks skipped (multiple polygons): {counters['mask_convert_multiple_polygons']}")
    logging.info(f"Conversations processed: {counters['conversations_processed']}")
    logging.info(f"Labels skipped (already exist): {counters['skipped_existing_labels']}")
    logging.info(f"Images skipped (already exist): {counters['skipped_existing_images']}")
    logging.info(f"Total annotations generated: {counters['generated_annotations']}")
    logging.info(f"Unique images with annotations: {len(counters['images_with_annotations'])}")
    logging.info(f"Images copied to output: {counters['copied_images']}")

    # Calculate and log per-class statistics
    if counters["class_annotations_per_image"]:
        logging.info("Annotation statistics:")

        # Define the format specification for statistics
        stats_format = "count,mean,p25,p50,p75,max"

        # Generate and log the entire statistics table
        _print_statistics_table(counters["class_annotations_per_image"], class_names, stats_format)


def main():
    """Main function to orchestrate the dataset conversion."""
    global stats_counters  # Declare intent to modify global dict
    parser = _setup_argparse()
    args = parser.parse_args()

    try:
        output_dir, output_name, image_dir, label_dir = _configure_environment(args)
        dataset, phrase_map, class_names = _load_data(args)

        _process_samples(dataset, phrase_map, image_dir, label_dir, args)
        _log_summary(stats_counters, class_names)  # Pass the global dict to log summary

        # Generate YAML file after successful processing
        if class_names:
            generate_yolo_yaml(output_dir, output_name, class_names, args.train_split)
        else:
            logging.warning("No class names found from mapping, skipping YAML generation.")

    except (FileNotFoundError, ValueError, Exception) as e:
        logging.error(f"Conversion failed: {e}", exc_info=args.verbose)  # Show traceback if verbose
        # Consider sys.exit(1) here for script mode


if __name__ == "__main__":
    main()
