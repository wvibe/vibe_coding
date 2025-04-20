import argparse
import csv
import json
import logging
import os
import random
from collections import defaultdict  # 确保已导入
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Features, Sequence, Value, load_dataset, load_dataset_builder
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

# Import the OOP data models
from src.dataops.cov_segm.datamodel import ClsSegment, SegmMask, SegmSample

# Import loader using full path
from src.dataops.cov_segm.loader import load_sample
from src.utils.common.geometry import mask_to_yolo_polygons

# Import statistics formatting utility
from src.utils.common.stats import format_statistics_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # Use logger instance


# Define stats counter name constants
SC_SAMPLE_LOADED = "sample_loaded"
SC_SEGMENT_LOADED = "segment_loaded"
SC_SEGMENT_SKIPPED_NO_MATCH = "segment_skipped_no_match"
SC_SEGMENT_SKIPPED_SAMPLING = "segment_skipped_sampling"
SC_SEGMENT_SKIPPED_ZERO_MASK = "segment_skipped_zero_mask"
SC_SEGMENT_PROCESSED = "segment_processed"

SC_MASK_FOR_LABELING = "mask_for_labeling"
SC_MASK_SKIPPED_MULTIPLE_POLYGONS = "mask_skipped_multiple_polygons"
SC_MASK_SKIPPED_NO_POLYGON = "mask_skipped_no_polygon"
SC_MASK_LABEL_GENERATED = "mask_label_generated"

SC_IMAGE_WITH_LABELS = "image_with_labels"
SC_CLASS_STATS_DICT = "class_stats_dict"

SC_FILE_LABEL_WRITTEN = "file_label_written"
SC_FILE_LABEL_WRITE_ERROR = "file_label_write_error"
SC_FILE_LABEL_SKIPPED_EXISTING = "file_label_skipped_existing"
SC_FILE_IMAGE_WRITTEN = "file_image_written"
SC_FILE_IMAGE_WRITE_ERROR = "file_image_write_error"
SC_FILE_IMAGE_SKIPPED_EXISTING = "file_image_skipped_existing"

# Mapping config for COV-SEGM dataset to YOLO format
MC_YOLO_CLASS_ID = "yolo_class_id"
MC_YOLO_CLASS_NAME = "yolo_class_name"
MC_HF_PHRASE = "hf_phrase"
MC_SAMPLING_RATIO = "sampling_ratio"

# Mapping info
MI_CLASS_ID = "class_id"
MI_CLASS_NAME = "class_name"
MI_SAMPLING_RATIO = "sampling_ratio"


def _setup_argparse() -> argparse.ArgumentParser:
    """Sets up the argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert lab42/cov-segm-v3 dataset to YOLO segmentation format."
    )
    parser.add_argument(
        "--hf-dataset-path",
        type=str,
        default="lab42/cov-segm-v3",
        help="Path or name of the Hugging Face dataset.",
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
        "--sample-ratio",
        type=float,
        default=1.0,
        help=(
            "Global sampling ratio (0.0-1.0) applied to all classes in addition to their "
            "individual ratios. Ratio 0.0 means no sampling."
        ),
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=None,
        help="Process dataset in slices of this size (e.g., 100000). Processes all at once if None.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Number of processor cores to use for parallel loading (default: 1).",
    )
    parser.add_argument(
        "--skip-zero",
        action="store_true",
        default=True,
        help="Skip segments with zero masks of the specified mask-tag.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        default=False,
        help="Skip writing files if they already exist.",
    )
    return parser


def _configure_environment(args: argparse.Namespace) -> Tuple[Path, Path]:
    """Configures logging, env vars, paths, and random seed."""
    # Validate sample_ratio
    if not (0.0 <= args.sample_ratio <= 1.0):
        raise ValueError("Invalid sample_ratio.")
    if args.slice_size is not None and args.slice_size <= 0:
        raise ValueError("--slice-size must be positive.")
    if args.sample_count is not None and args.sample_count <= 0:
        raise ValueError("--sample-count must be positive.")
    if args.sample_count is not None and args.slice_size is not None:
        logger.warning(
            "--sample-count overrides --slice-size. Processing only first %d samples.",
            args.sample_count,
        )
        args.slice_size = None  # Disable slicing if sample_count is set

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

    return image_dir, label_dir


def _load_mapping_config(
    config_path: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, str]]:
    """Loads the phrase-to-class mapping configuration from a CSV file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Mapping config file not found: {config_path}")

    phrase_map: Dict[str, Dict[str, Any]] = {}
    class_names: Dict[int, str] = {}
    required_columns = {
        MC_YOLO_CLASS_ID,
        MC_YOLO_CLASS_NAME,
        MC_HF_PHRASE,
        MC_SAMPLING_RATIO,
    }

    try:
        with open(config_path, mode="r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            if not required_columns.issubset(reader.fieldnames or []):
                missing = required_columns - set(reader.fieldnames or [])
                raise ValueError(f"Missing required columns in {config_path}: {missing}")

            for row_idx, row in enumerate(reader):
                try:
                    class_id = int(row[MC_YOLO_CLASS_ID])
                    class_name = row[MC_YOLO_CLASS_NAME].strip()
                    sampling_ratio = float(row[MC_SAMPLING_RATIO])

                    # Get the single phrase (instead of parsing a comma-separated list)
                    phrase = row[MC_HF_PHRASE].strip()
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
                            f" '{class_name}'. Skipping the conflicting row."
                        )
                        continue

                    class_names[class_id] = class_name

                    mapping_info = {
                        MI_CLASS_ID: class_id,
                        MI_CLASS_NAME: class_name,
                        MI_SAMPLING_RATIO: sampling_ratio,
                    }

                    # Only one phrase per row in this format
                    if phrase in phrase_map:
                        logging.warning(
                            f"Row {row_idx + 2}: Phrase '{phrase}' maps to multiple classes."
                            f" Overwriting previous mapping {phrase_map[phrase]} with"
                            f" {mapping_info}. Skipping the conflicting row."
                        )
                        continue

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


def _get_sampled_mapping_info(
    segment: ClsSegment,
    phrase_map: Dict[str, Dict[str, Any]],
    stats: Dict[str, Any],
    global_sample_ratio: float = 1.0,
) -> Optional[Tuple[Dict[str, Any], str]]:
    """Checks segment phrases against the map and applies sampling. Updates local stats.

    Args:
        segment: The segment containing phrases to check against the mapping.
        phrase_map: Mapping of phrases to class information.
        stats: Local stats dict to update.
        global_sample_ratio: Global sampling ratio (0.0-1.0] to apply to all classes.
                            ratio 0.0 means no sampling.

    Returns:
        Tuple[Dict, str] (mapping_info, matched_phrase) if a sampled match is found,
        otherwise None.
    """
    matched_phrase = None
    mapping_info = None

    # Iterate through phrases in the segment
    for phrase in segment.phrases:
        phrase_text = phrase.text.strip()
        if not phrase_text:  # Skip empty phrases
            logger.warning("Empty phrase in segment")
            continue

        current_mapping = phrase_map.get(phrase_text)
        if current_mapping:  # Found a mapping
            mapping_info = current_mapping
            matched_phrase = phrase_text
            break  # Stop at the first match

    # Check if a mapping was found
    if mapping_info is None or matched_phrase is None:
        stats[SC_SEGMENT_SKIPPED_NO_MATCH] += 1  # Update local stats
        return None

    if global_sample_ratio > 0.0:
        # Apply sampling based on the found mapping and global ratio
        local_sampling_ratio = mapping_info.get("sampling_ratio", 1.0)
        # Combine global and local ratios (both are 0.0-1.0, so multiply)
        effective_ratio = global_sample_ratio * local_sampling_ratio

        if random.random() > effective_ratio:
            stats[SC_SEGMENT_SKIPPED_SAMPLING] += 1  # Update local stats
            return None

    return mapping_info, matched_phrase


def process_and_convert_sample(
    sample_row: Dict, phrase_map: Dict[str, Dict[str, Any]], args: argparse.Namespace
) -> Dict[str, Any]:
    """
    Processes a single sample row: loads, maps, samples, converts masks to polygons,
    and returns data ready for writing.

    Args:
        sample_row: A raw dataset row.
        phrase_map: Mapping of phrases to class information.
        args: Namespace containing sample_ratio, mask_tag, skip_zero etc.

    Returns:
        Dictionary containing: sample_id, image_bytes (optional), annotation_lines (list),
        stats (dict), error (bool), reason (str, optional).
    """
    stats = defaultdict(int)  # Initialize stats for THIS sample

    try:
        sample: Optional[SegmSample] = load_sample(sample_row)
        if sample is None:
            return {"error": True, "reason": "load_sample returned None", "stats": dict(stats)}
        stats[SC_SAMPLE_LOADED] += 1

        sample_id = sample.id
        main_image = sample.image
        image_bytes = None  # Only serialize if needed
        image_labels = []
        class_inst_cnt = defaultdict(int)  # Track masks processed per class for this sample

        for segment in sample.segments:
            stats[SC_SEGMENT_LOADED] += 1

            # Apply mapping and sampling (pass local stats dict)
            mapping_result = _get_sampled_mapping_info(
                segment, phrase_map, stats, args.sample_ratio
            )

            if mapping_result is None:
                continue  # Stats updated inside _get_sampled_mapping_info

            phrase_info, matched_phrase = mapping_result
            class_id_str = str(phrase_info[MI_CLASS_ID])

            # Get the appropriate masks to process based on mask_tag
            masks_to_process: List[SegmMask] = []
            if args.mask_tag == "visible":
                masks_to_process = segment.visible_masks
            elif args.mask_tag == "full":
                masks_to_process = segment.full_masks

            valid_masks = [mask for mask in masks_to_process if mask.is_valid]
            if len(masks_to_process) - len(valid_masks) > 0:
                logger.warning(f"Skipping {len(masks_to_process) - len(valid_masks)} invalid masks")

            if args.skip_zero and len(valid_masks) == 0:
                stats[SC_SEGMENT_SKIPPED_ZERO_MASK] += 1
                continue

            stats[SC_MASK_FOR_LABELING] += len(valid_masks)
            masks_processed_for_segment = 0
            for mask in valid_masks:
                try:
                    binary_mask = mask.binary_mask
                    mask_height, mask_width = binary_mask.shape

                    polygons = mask_to_yolo_polygons(
                        binary_mask=binary_mask,
                        img_shape=(mask_height, mask_width),
                        connect_parts=True,
                    )

                    if polygons:
                        if len(polygons) > 1:
                            stats[SC_MASK_SKIPPED_MULTIPLE_POLYGONS] += 1

                        # Use first polygon
                        polygon = polygons[0]
                        label = f"{class_id_str} {' '.join(map(str, polygon))}"
                        image_labels.append(label)
                        masks_processed_for_segment += 1
                    else:
                        stats[SC_MASK_SKIPPED_NO_POLYGON] += 1

                except Exception as e:
                    logger.warning(
                        f"Sample {sample_id}: Failed to convert mask for phrase "
                        f"'{matched_phrase}'. Error: {e}"
                    )

            class_inst_cnt[class_id_str] += masks_processed_for_segment
            stats[SC_MASK_LABEL_GENERATED] += masks_processed_for_segment
            stats[SC_SEGMENT_PROCESSED] += 1  # Count segment if masks were processed

        # If annotations were generated, prepare image bytes
        if len(image_labels) > 0:
            try:
                buffer = BytesIO()
                img_to_save = main_image
                if img_to_save.mode != "RGB":
                    img_to_save = img_to_save.convert("RGB")
                img_to_save.save(buffer, format="JPEG", quality=95)
                image_bytes = buffer.getvalue()
                stats[SC_IMAGE_WITH_LABELS] = 1  # Use 1/0 for aggregation
            except Exception as e:
                logger.error(f"Sample {sample_id}: Failed to prepare image bytes. Error: {e}")
                image_bytes = None  # Ensure image is not saved if bytes failed
                image_labels = []  # Do not save labels if image failed preparation

        # Add per-class mask counts to stats dict
        stats[SC_CLASS_STATS_DICT] = dict(class_inst_cnt)

        return {
            "sample_id": sample_id,
            "image_bytes": image_bytes,
            "labels": image_labels,
            "stats_json": json.dumps(dict(stats)),
            "error": False,
            "reason": None,
        }

    except Exception as e:
        logger.error(f"Error processing sample row. Error: {e}", exc_info=True)
        return {
            "sample_id": sample_row.get("id"),  # Try to get ID if possible
            "image_bytes": None,
            "labels": [],
            "stats_json": json.dumps(dict(stats)),
            "error": True,
            "reason": str(e),
        }


# Define Features for the output of process_and_convert_sample (using JSON for stats)
map_output_features = Features(
    {
        "sample_id": Value("string"),
        "image_bytes": Value("large_binary"),  # Image bytes might still be large
        "labels": Sequence(Value("string")),
        "stats_json": Value(
            "large_string"
        ),  # Stats are now a JSON string (use large_string just in case)
        "error": Value("bool"),
        "reason": Value("string"),  # Allowed to be None
    }
)


def process_slice(
    dataset_path: str,
    split: str,
    start: int,
    end: int,
    phrase_map: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
    image_dir: Path,  # Pass specific output dirs for this split
    label_dir: Path,
) -> Dict[str, Any]:
    """Loads, processes, and writes a slice of the dataset."""

    logger.info(f"--- Processing slice: {split}[{start}:{end}] ---")
    aggr_stats = defaultdict(int)
    aggr_cls_stats = defaultdict(list)

    try:
        # 1. Load Dataset Slice
        logger.info("Loading dataset slice...")
        dset_slice = load_dataset(
            dataset_path,
            split=f"{split}[{start}:{end}]",
            streaming=False,  # Process slice in memory/cache
        )
        logger.info(f"Slice loaded with {len(dset_slice)} samples.")

        # 2. Prepare Map Arguments
        map_fn_kwargs = {
            "phrase_map": phrase_map,
            # Pass only necessary args fields to avoid potential pickle issues
            "args": argparse.Namespace(
                sample_ratio=args.sample_ratio,
                mask_tag=args.mask_tag,
                skip_zero=args.skip_zero,
            ),
        }

        # 3. Run Map Operation
        logger.info(f"Applying map function with {args.num_proc} processes...")
        processed_dataset = dset_slice.map(
            process_and_convert_sample,
            num_proc=args.num_proc if args.num_proc > 1 else None,
            load_from_cache_file=False,  # Always re-process
            desc=f"Processing slice {start}-{end}",
            features=map_output_features,  # Use defined features
            fn_kwargs=map_fn_kwargs,
            remove_columns=dset_slice.column_names,  # Keep only map output
        )
        logger.info("Map operation completed for slice.")

        # 4. Process Results and Write Files
        logger.info("Writing outputs and aggregating stats for slice...")
        # Ensure output dirs exist for this slice/split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        write_iterator = tqdm(
            processed_dataset, desc=f"Writing slice {start}-{end}", total=len(processed_dataset)
        )
        for result in write_iterator:
            # Decode stats JSON
            sample_stats = {}
            stats_json_str = result.get("stats_json")
            if stats_json_str:
                try:
                    sample_stats = json.loads(stats_json_str)  # <--- 从 JSON 还原字典
                except json.JSONDecodeError:
                    logger.error(
                        f"Failed to decode stats JSON for sample: {result.get('sample_id')}"
                    )
                    continue

            # Aggregate basic counters
            for key, value in sample_stats.items():
                if key == SC_CLASS_STATS_DICT:
                    if isinstance(value, dict):
                        for class_id_str, count in value.items():
                            # The defaultdict handles list creation automatically
                            aggr_cls_stats[class_id_str].append(count)  # Append directly
                elif isinstance(value, (int, float)):
                    aggr_stats[key] += value

            if result.get("error"):
                logger.error(
                    f"Slice {start}-{end}: Error processing sample {result.get('sample_id')}"
                )
                continue

            labels = result.get("labels", [])
            if not labels:
                continue

            sample_id = result["sample_id"]
            image_bytes = result.get("image_bytes")
            if not image_bytes:
                logger.error(
                    f"Slice {start}-{end}: No image bytes generated for sample "
                    f"{result.get('sample_id')}"
                )
                continue

            # Setup file paths
            image_filename = f"{sample_id}.jpg"
            label_filename = f"{sample_id}.txt"
            image_path = image_dir / image_filename
            label_path = label_dir / label_filename

            # Write label file
            write_label = labels and not (args.no_overwrite and label_path.exists())
            if write_label:
                try:
                    with open(label_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(labels) + "\n")
                    aggr_stats[SC_FILE_LABEL_WRITTEN] += 1
                except Exception as e:
                    aggr_stats[SC_FILE_LABEL_WRITE_ERROR] += 1
                    logger.error(f"Slice {start}-{end}: Failed to write label {label_path}: {e}")
            else:
                aggr_stats[SC_FILE_LABEL_SKIPPED_EXISTING] += 1

            # Save image file (only if label was written or overwrite disabled check passed)
            write_image = image_bytes and not (args.no_overwrite and image_path.exists())
            if write_image:
                try:
                    # Convert bytes back to PIL Image to save
                    img = Image.open(BytesIO(image_bytes))
                    img.save(image_path, "JPEG", quality=95)
                    aggr_stats[SC_FILE_IMAGE_WRITTEN] += 1
                except Exception as e:
                    aggr_stats[SC_FILE_IMAGE_WRITE_ERROR] += 1
                    logger.error(f"Slice {start}-{end}: Failed to save image {image_path}: {e}")
            elif image_bytes:  # Image exists but wasn't written due to no_overwrite
                aggr_stats[SC_FILE_IMAGE_SKIPPED_EXISTING] += 1

        logger.info(f"--- Finished processing slice: {split}[{start}:{end}] ---")

    except Exception as e:
        logger.exception(f"FATAL error processing slice {split}[{start}:{end}]: {e}")

    aggr_stats[SC_CLASS_STATS_DICT] = dict(aggr_cls_stats)
    return dict(aggr_stats)  # Return aggregated stats for this slice


def _log_summary(stats: Dict[str, Any], cls_stats: Dict[str, Any], class_names: Dict[int, str]):
    """Logs the final conversion statistics from the global counters dict."""
    logging.info("Conversion finished.")
    logging.info(f"Total samples loaded: {stats[SC_SAMPLE_LOADED]}")
    logging.info(f"Total segments loaded: {stats[SC_SEGMENT_LOADED]}")
    logging.info(f"Segments skipped (no mapping): {stats[SC_SEGMENT_SKIPPED_NO_MATCH]}")
    logging.info(f"Segments skipped (sampling): {stats[SC_SEGMENT_SKIPPED_SAMPLING]}")
    logging.info(f"Segments skipped (zero mask): {stats[SC_SEGMENT_SKIPPED_ZERO_MASK]}")
    logging.info(f"Segments processed: {stats[SC_SEGMENT_PROCESSED]}")

    logging.info(f"Masks for labeling: {stats[SC_MASK_FOR_LABELING]}")
    logging.info(f"Masks skipped (no polygon): {stats[SC_MASK_SKIPPED_NO_POLYGON]}")
    logging.info(f"Masks skipped (multiple polygons): {stats[SC_MASK_SKIPPED_MULTIPLE_POLYGONS]}")
    logging.info(f"Masks converted to labels: {stats[SC_MASK_LABEL_GENERATED]}")

    logging.info(f"Images generated with labels: {stats[SC_IMAGE_WITH_LABELS]}")
    logging.info(f"Images file written: {stats[SC_FILE_IMAGE_WRITTEN]}")
    logging.info(f"Images file write errored: {stats[SC_FILE_IMAGE_WRITE_ERROR]}")
    logging.info(f"Images file skipped (already exist): {stats[SC_FILE_IMAGE_SKIPPED_EXISTING]}")
    logging.info(f"Labels file written: {stats[SC_FILE_LABEL_WRITTEN]}")
    logging.info(f"Labels file write errored: {stats[SC_FILE_LABEL_WRITE_ERROR]}")
    logging.info(f"Labels file skipped (already exist): {stats[SC_FILE_LABEL_SKIPPED_EXISTING]}")

    # Calculate and log per-class statistics
    if cls_stats:
        logging.info("Class masks statistics:")

        # Prepare data for stats.format_statistics_table
        stats_data = {}
        for class_id_str, counts in cls_stats.items():
            class_id = int(class_id_str)
            class_name = class_names.get(class_id)
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


def _find_dataset_samples(hf_dataset_path: str, train_split: str) -> int:
    # Determine total samples and slicing strategy
    logger.debug(f"Getting total number of samples for split '{train_split}'...")
    try:
        # Load only the necessary info or a small part to get length
        # This still might download index files
        builder = load_dataset_builder(hf_dataset_path)
        return builder.info.splits[train_split].num_examples
    except Exception as e:
        logger.error(f"Failed to get total sample count for split '{train_split}'. Error: {e}")
        raise e


# --- Main Function ---
def main():
    parser = _setup_argparse()
    args = parser.parse_args()
    global_stats = defaultdict(int)  # Use defaultdict for easier aggregation
    global_class_stats = defaultdict(list)

    try:
        try:
            image_dir, label_dir = _configure_environment(args)
        except ValueError as e:
            logging.error(f"Configuration error: {e}")
            return 1

        phrase_map, class_names = _load_mapping_config(args.mapping_config)
        num_total_samples = _find_dataset_samples(args.hf_dataset_path, args.train_split)
        logger.info(f"Found {num_total_samples} total samples in split '{args.train_split}'.")

        # Determine total samples and slicing strategy
        samples_to_process = num_total_samples
        if args.sample_count is not None:
            samples_to_process = min(args.sample_count, num_total_samples)
            logger.info(
                f"Processing a maximum of {samples_to_process} samples due to --sample-count."
            )

        if samples_to_process == 0:
            logger.info("No samples to process, skipping conversion.")
            return 0

        slice_size = args.slice_size
        if slice_size is None or slice_size > samples_to_process:
            slice_size = samples_to_process
            logger.info("Processing all samples in a single pass (no slicing).")

        # Loop through slices
        for i in range(0, samples_to_process, slice_size):
            start = i
            # Adjust end to not exceed samples_to_process
            end = min(i + slice_size, samples_to_process)
            if start >= end:
                continue  # Skip if start >= end

            slice_stats = process_slice(
                dataset_path=args.hf_dataset_path,
                split=args.train_split,
                start=start,
                end=end,
                phrase_map=phrase_map,
                args=args,
                image_dir=image_dir,
                label_dir=label_dir,
            )

            # Simple aggregation for now (modify _aggregate_stats)
            # Aggregate basic counters
            for key, value in slice_stats.items():
                if key == SC_CLASS_STATS_DICT:
                    if isinstance(value, dict):
                        for class_id_str, count in value.items():
                            global_class_stats[class_id_str].extend(count)
                elif isinstance(value, (int, float)):
                    global_stats[key] += value

        # Log final summary from aggregated stats
        _log_summary(global_stats, global_class_stats, class_names)

        logger.info("Conversion finished.")
        return 0

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration or argument error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
