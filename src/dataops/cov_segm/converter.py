import argparse
import csv
import json
import logging
import os
import random
from collections import defaultdict
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
logger = logging.getLogger(__name__)


# Define stats counter name constants
SC_SAMPLE_LOADED = "sample_loaded"
SC_SAMPLE_PREP_ERROR = "sample_prep_error"

SC_SEGMENT_LOADED = "segment_loaded"
SC_SEGMENT_SKIPPED_NO_MATCH = "segment_skipped_no_match"
SC_SEGMENT_SKIPPED_SAMPLING = "segment_skipped_sampling"
SC_SEGMENT_SKIPPED_ZERO_MASK = "segment_skipped_zero_mask"
SC_SEGMENT_PROCESSED_PREP = "segment_processed_prep"  # Processed in preparation phase

SC_MASK_PREPARED_FOR_LABELING = "mask_prepared_for_labeling"  # Masks identified for processing
SC_MASK_SKIPPED_MULTIPLE_POLYGONS = "mask_skipped_multiple_polygons"
SC_MASK_SKIPPED_NO_POLYGON = "mask_skipped_no_polygon"
SC_MASK_LABEL_GENERATED = "mask_label_generated"

SC_IMAGE_WITH_PENDING_LABELS = "image_with_pending_labels"  # Image has masks to process
SC_CLASS_STATS_DICT = "class_stats_dict"  # Key for nested class stats dictionary

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

SC_ERROR_INCIDENT = "error_incident"  # Single counter for all error incidents


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


def prepare_sample_for_conversion(
    sample_row: Dict, phrase_map: Dict[str, Dict[str, Any]], args: argparse.Namespace
) -> Dict[str, Any]:
    """
    Prepares a single sample row for later conversion: loads, maps, samples, and
    serializes valid masks for later polygon generation.

    Args:
        sample_row: A raw dataset row.
        phrase_map: Mapping of phrases to class information.
        args: Namespace containing sample_ratio, mask_tag, skip_zero etc.

    Returns:
        Dictionary containing: sample_id, image_bytes, pending_masks (list of tuples),
        prep_stats_json, error, reason.
    """
    prep_stats = defaultdict(int)  # Initialize stats for THIS sample preparation step

    try:
        sample: Optional[SegmSample] = load_sample(sample_row)
        if sample is None:
            return {
                "sample_id": sample_row.get("id", "unknown"),
                "image_bytes": None,
                "pending_masks": [],
                "prep_stats_json": json.dumps(dict(prep_stats)),
                "error": True,
                "reason": "load_sample returned None",
            }
        prep_stats[SC_SAMPLE_LOADED] += 1

        sample_id = sample.id
        main_image = sample.image
        image_bytes = None  # Only serialize if needed
        pending_masks = []  # List to hold serialized masks for later processing

        for segment in sample.segments:
            prep_stats[SC_SEGMENT_LOADED] += 1

            # Apply mapping and sampling (pass local stats dict)
            mapping_result = _get_sampled_mapping_info(
                segment, phrase_map, prep_stats, args.sample_ratio
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

            valid_masks = [
                mask for mask in masks_to_process if mask.is_valid and mask.binary_mask is not None
            ]

            if len(masks_to_process) - len(valid_masks) > 0:
                logger.warning(
                    f"Skipping {len(masks_to_process) - len(valid_masks)} invalid or empty masks"
                )

            if args.skip_zero and len(valid_masks) == 0:
                prep_stats[SC_SEGMENT_SKIPPED_ZERO_MASK] += 1
                continue

            # Prepare valid masks for later polygon generation
            masks_prepared_for_segment = 0
            for mask in valid_masks:
                try:
                    # Use SegmMask.to_dict() for serialization
                    mask_dict = mask.to_dict()
                    pending_masks.append((class_id_str, mask_dict))
                    masks_prepared_for_segment += 1
                except Exception as e:
                    logger.error(
                        f"Sample {sample_id}: Failed to serialize mask for '{matched_phrase}': {e}"
                    )

            prep_stats[SC_MASK_PREPARED_FOR_LABELING] += masks_prepared_for_segment
            if masks_prepared_for_segment > 0:
                prep_stats[SC_SEGMENT_PROCESSED_PREP] += 1  # Count segment if masks were prepared

        # If masks were prepared, serialize the image
        if len(pending_masks) > 0:
            try:
                buffer = BytesIO()
                img_to_save = main_image
                if img_to_save.mode != "RGB":
                    img_to_save = img_to_save.convert("RGB")
                img_to_save.save(buffer, format="JPEG", quality=95)
                image_bytes = buffer.getvalue()
                prep_stats[SC_IMAGE_WITH_PENDING_LABELS] = 1  # Use 1/0 for aggregation
            except Exception as e:
                logger.error(f"Sample {sample_id}: Failed to prepare image bytes: {e}")
                raise

        return {
            "sample_id": sample_id,
            "image_bytes": image_bytes,
            "pending_masks": pending_masks,  # List of (class_id_str, mask_dict) tuples
            "prep_stats_json": json.dumps(dict(prep_stats)),
            "error": False,
            "reason": None,
        }

    except Exception as e:
        logger.error(
            f"Error preparing sample {sample_row.get('id', 'unknown')}: {e}", exc_info=True
        )
        return {
            "sample_id": sample_row.get("id", "unknown"),
            "image_bytes": None,
            "pending_masks": [],
            "prep_stats_json": json.dumps(dict(prep_stats)),
            "error": True,
            "reason": str(e),
        }


# Define Features for the output of prepare_sample_for_conversion
map_output_features = Features(
    {
        "sample_id": Value("string"),
        "image_bytes": Value("large_binary"),
        # Store pending masks as a sequence of tuples (class_id_str, mask_dict_json)
        "pending_masks": Sequence(
            {
                "class_id": Value("string"),
                "mask_dict_json": Value("large_string"),  # JSON string of mask_dict
            }
        ),
        "prep_stats_json": Value("large_string"),
        "error": Value("bool"),
        "reason": Value("string"),
    }
)


# Helper function to serialize pending masks for datasets.map
def _serialize_pending_masks(
    pending_masks_list: List[Tuple[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Convert list of (class_id, mask_dict) to format suitable for Features schema."""
    serialized = []
    for class_id, mask_dict in pending_masks_list:
        try:
            # Convert the mask_dict to JSON string
            mask_dict_json = json.dumps(mask_dict)
            serialized.append({"class_id": class_id, "mask_dict_json": mask_dict_json})
        except Exception as e:
            logger.error(f"Failed to serialize mask_dict to JSON: {e}")
            # Skip this mask if serialization fails
    return serialized


# Helper function to deserialize pending masks after datasets.map
def _deserialize_pending_masks(
    serialized_masks: List[Dict[str, Any]],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Convert serialized masks back to list of (class_id, mask_dict) tuples."""
    deserialized = []
    for item in serialized_masks:
        try:
            class_id = item["class_id"]
            mask_dict = json.loads(item["mask_dict_json"])
            deserialized.append((class_id, mask_dict))
        except Exception as e:
            logger.error(f"Failed to deserialize mask data: {e}")
            # Skip this mask if deserialization fails
    return deserialized


def _prepare_dataset_slice(
    start: int,
    end: int,
    phrase_map: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
    global_stats: Dict[str, Any],
) -> Optional[Any]:
    """
    Load a dataset slice and prepare it for conversion using parallel processing.

    Returns:
        The prepared dataset or None if loading fails.
    """
    try:
        logger.info(f"Loading dataset slice {args.train_split}[{start}:{end}]...")
        dset_slice = load_dataset(
            args.hf_dataset_path,
            split=f"{args.train_split}[{start}:{end}]",
            streaming=False,
        )
        logger.info(f"Slice loaded with {len(dset_slice)} samples.")

        # Prepare map arguments for parallel preparation
        map_fn_kwargs = {
            "phrase_map": phrase_map,
            "args": argparse.Namespace(
                sample_ratio=args.sample_ratio,
                mask_tag=args.mask_tag,
                skip_zero=args.skip_zero,
            ),
        }

        # Run parallel preparation
        logger.info(f"Applying preparation map function with {args.num_proc} processes...")

        # Serialize the pending_masks lists
        def prepare_with_serialization(sample_row):
            result = prepare_sample_for_conversion(sample_row, **map_fn_kwargs)
            result["pending_masks"] = _serialize_pending_masks(result["pending_masks"])
            return result

        prepared_dataset = dset_slice.map(
            prepare_with_serialization,
            num_proc=args.num_proc if args.num_proc > 1 else None,
            load_from_cache_file=False,
            desc=f"Preparing slice {start}-{end}",
            features=map_output_features,
            remove_columns=dset_slice.column_names,
        )
        logger.info("Preparation map operation completed for slice.")

        return prepared_dataset

    except Exception:
        logger.exception(
            f"Error loading or preparing dataset slice {args.train_split}[{start}:{end}]"
        )
        global_stats[SC_ERROR_INCIDENT] += 1
        return None


def _process_mask_to_polygon(
    class_id_str: str,
    mask_dict: Dict[str, Any],
    sample_id: str,
    global_stats: Dict[str, Any],
) -> Optional[str]:
    """
    Process a single mask: deserialize it, convert to polygons, and return the YOLO label.

    Returns:
        The YOLO label string or None if conversion fails.
    """
    try:
        # Reconstruct the mask using SegmMask.from_dict
        segm_mask = SegmMask.from_dict(mask_dict)

        # Check if binary_mask was successfully deserialized
        if segm_mask.binary_mask is None or not segm_mask.is_valid:
            logger.debug(f"Sample {sample_id}: Invalid mask after deserialization")
            return None

        # Call mask_to_yolo_polygons in main process
        polygons = mask_to_yolo_polygons(
            binary_mask=segm_mask.binary_mask,
            img_shape=segm_mask.binary_mask.shape,
            connect_parts=True,
        )

        if not polygons:
            global_stats[SC_MASK_SKIPPED_NO_POLYGON] += 1
            return None

        if len(polygons) > 1:
            global_stats[SC_MASK_SKIPPED_MULTIPLE_POLYGONS] += 1
            logger.debug(f"Sample {sample_id}: Using first of {len(polygons)} polygons")

        # Use first polygon
        polygon = polygons[0]
        label = f"{class_id_str} {' '.join(map(str, polygon))}"
        global_stats[SC_MASK_LABEL_GENERATED] += 1
        return label

    except Exception as e:
        logger.error(f"Error converting mask to polygon for sample {sample_id}: {str(e)}")
        global_stats[SC_ERROR_INCIDENT] += 1
        return None


def _write_label_and_image(
    sample_id: str,
    image_labels: List[str],
    image_bytes: bytes,
    image_dir: Path,
    label_dir: Path,
    args: argparse.Namespace,
    global_stats: Dict[str, Any],
) -> bool:
    """
    Write label and image files for a sample.

    Returns:
        True if files were written successfully, False otherwise.
    """
    try:
        # Setup file paths
        image_filename = f"{sample_id}.jpg"
        label_filename = f"{sample_id}.txt"
        image_path = image_dir / image_filename
        label_path = label_dir / label_filename

        # Write label file
        write_label = not (args.no_overwrite and label_path.exists())
        if write_label:
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(image_labels) + "\n")
            global_stats[SC_FILE_LABEL_WRITTEN] += 1
        elif args.no_overwrite and label_path.exists():
            global_stats[SC_FILE_LABEL_SKIPPED_EXISTING] += 1

        # Save image file (only if label was written or skipped)
        write_image = (
            image_bytes
            and (write_label or (args.no_overwrite and label_path.exists()))
            and not (args.no_overwrite and image_path.exists())
        )

        if write_image:
            img = Image.open(BytesIO(image_bytes))
            img.save(image_path, "JPEG", quality=95)
            global_stats[SC_FILE_IMAGE_WRITTEN] += 1
        elif image_bytes and args.no_overwrite and image_path.exists():
            global_stats[SC_FILE_IMAGE_SKIPPED_EXISTING] += 1

        return True

    except Exception as e:
        logger.error(f"Error writing files for sample {sample_id}: {str(e)}")
        global_stats[SC_ERROR_INCIDENT] += 1
        return False


def process_slice(
    start: int,
    end: int,
    phrase_map: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
    global_stats: Dict[str, Any],  # Single global stats dictionary
    image_dir: Path,
    label_dir: Path,
) -> bool:
    """
    Loads a slice, prepares data in parallel, then converts masks and writes sequentially.

    Returns True if slice processed without fatal errors, False otherwise.
    """
    logger.info(f"--- Processing slice: {args.train_split}[{start}:{end}] ---")

    try:
        # 1. Load Dataset Slice and Prepare in Parallel
        prepared_dataset = _prepare_dataset_slice(start, end, phrase_map, args, global_stats)

        if prepared_dataset is None:
            return False  # Fatal error during slice loading or preparation

        # Ensure output directories exist
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        # Ensure class stats dictionary exists in global_stats
        if SC_CLASS_STATS_DICT not in global_stats:
            global_stats[SC_CLASS_STATS_DICT] = defaultdict(list)

        # 2. Sequential Processing Loop
        logger.info("Converting masks and writing outputs...")

        write_iterator = tqdm(
            prepared_dataset,
            desc=f"Converting/Writing slice {start}-{end}",
            total=len(prepared_dataset),
        )

        for result in write_iterator:
            # Decode preparation stats JSON and aggregate
            try:
                prep_stats = json.loads(result.get("prep_stats_json", "{}"))

                # Aggregate preparation stats directly into global_stats
                for key, value in prep_stats.items():
                    if isinstance(value, (int, float)):
                        global_stats[key] += value

            except json.JSONDecodeError:
                logger.error(f"Failed to decode stats JSON for sample: {result.get('sample_id')}")
                global_stats[SC_ERROR_INCIDENT] += 1
                continue

            # Handle preparation errors
            if result.get("error"):
                logger.error(
                    f"Error preparing sample {result.get('sample_id')}: {result.get('reason')}"
                )
                global_stats[SC_ERROR_INCIDENT] += 1
                continue

            # Extract sample data
            sample_id = result["sample_id"]
            image_bytes = result.get("image_bytes")
            serialized_pending_masks = result.get("pending_masks", [])

            # Skip if no masks were prepared or image failed serialization
            if not serialized_pending_masks or not image_bytes:
                logger.debug(f"Sample {sample_id}: Skipping due to missing data")
                continue

            # 3. Process Masks to Generate Labels
            image_labels = []  # Labels for the current image
            sample_class_counts = defaultdict(int)  # Per-sample class counts

            # Deserialize the pending masks
            pending_masks = _deserialize_pending_masks(serialized_pending_masks)

            # Process each mask to generate polygon/label
            for class_id_str, mask_dict in pending_masks:
                label = _process_mask_to_polygon(class_id_str, mask_dict, sample_id, global_stats)
                if label:
                    image_labels.append(label)
                    sample_class_counts[class_id_str] += 1

            # Skip if no valid labels were generated
            if not image_labels:
                logger.debug(f"Sample {sample_id}: No valid labels generated, skipping")
                continue

            # 4. Write Label and Image Files
            success = _write_label_and_image(
                sample_id, image_labels, image_bytes, image_dir, label_dir, args, global_stats
            )

            if not success:
                continue

            # 5. Add class counts for this sample to the global class stats
            class_stats_dict = global_stats[SC_CLASS_STATS_DICT]
            for cls_id, count in sample_class_counts.items():
                class_stats_dict[cls_id].append(count)

        logger.info(f"--- Finished processing slice: {args.train_split}[{start}:{end}] ---")
        return True  # Successful slice processing

    except Exception:
        logger.exception(f"FATAL error processing slice {args.train_split}[{start}:{end}]")
        global_stats[SC_ERROR_INCIDENT] += 1
        return False  # Fatal error in slice processing


def _log_summary(stats: Dict[str, Any], class_names: Dict[int, str]):
    """Logs the final conversion statistics."""
    logging.info("--- Conversion Summary ---")
    logging.info(f"Total samples loaded: {stats.get(SC_SAMPLE_LOADED, 0)}")
    logging.info(f"Total segments loaded: {stats.get(SC_SEGMENT_LOADED, 0)}")
    logging.info(f"Segments skipped (no mapping): {stats.get(SC_SEGMENT_SKIPPED_NO_MATCH, 0)}")
    logging.info(f"Segments skipped (sampling): {stats.get(SC_SEGMENT_SKIPPED_SAMPLING, 0)}")
    logging.info(f"Segments skipped (zero mask): {stats.get(SC_SEGMENT_SKIPPED_ZERO_MASK, 0)}")
    logging.info(f"Segments prepared for conversion: {stats.get(SC_SEGMENT_PROCESSED_PREP, 0)}")

    logging.info(f"Masks prepared for labeling: {stats.get(SC_MASK_PREPARED_FOR_LABELING, 0)}")
    logging.info(f"Masks skipped (no polygon): {stats.get(SC_MASK_SKIPPED_NO_POLYGON, 0)}")
    logging.info(
        f"Masks skipped (multiple polygons): {stats.get(SC_MASK_SKIPPED_MULTIPLE_POLYGONS, 0)}"
    )
    logging.info(f"Masks converted to labels: {stats.get(SC_MASK_LABEL_GENERATED, 0)}")

    logging.info(f"Images with pending labels: {stats.get(SC_IMAGE_WITH_PENDING_LABELS, 0)}")
    logging.info(f"Images file written: {stats.get(SC_FILE_IMAGE_WRITTEN, 0)}")
    logging.info(
        f"Images file skipped (already exist): {stats.get(SC_FILE_IMAGE_SKIPPED_EXISTING, 0)}"
    )
    logging.info(f"Labels file written: {stats.get(SC_FILE_LABEL_WRITTEN, 0)}")
    logging.info(
        f"Labels file skipped (already exist): {stats.get(SC_FILE_LABEL_SKIPPED_EXISTING, 0)}"
    )

    # Log error incidents
    error_count = stats.get(SC_ERROR_INCIDENT, 0)
    if error_count > 0:
        logging.warning(f"Total error incidents during processing: {error_count}")
        logging.warning("See log file for detailed error information")

    # Calculate and log per-class statistics (based on successfully generated labels)
    cls_stats = stats.get(SC_CLASS_STATS_DICT, {})
    if cls_stats:
        logging.info("--- Class Label Statistics (Masks resulting in labels per image) ---")

        stats_data = {}
        for class_id_str, counts in cls_stats.items():
            try:
                class_id = int(class_id_str)
                class_name = class_names.get(class_id, f"Unknown ClassID {class_id_str}")
                stats_data[class_name] = counts
            except ValueError:
                logger.warning(
                    f"Invalid class ID string encountered in class stats: {class_id_str}"
                )

        if stats_data:
            # Define format string for the table
            format_string = "{key:<25} {count:>8d} {sum:>8d} {mean:>8.1f} {p25:>8d} {p50:>8d} {p75:>8d} {max:>8d}"

            # Generate the formatted table using the utility
            try:
                table_lines = format_statistics_table(stats_data, format_string)
                # Log each line of the table
                for line in table_lines:
                    logging.info(line)
            except Exception as e:
                logging.error(f"Failed to format class statistics table: {e}")
                global_stats[SC_ERROR_INCIDENT] += 1
        else:
            logging.info("No valid class statistics were generated.")
    else:
        logging.info("No class statistics were generated.")


def _find_dataset_samples(hf_dataset_path: str, train_split: str) -> int:
    """Determine total samples in the specified dataset split."""
    logger.debug(f"Getting total number of samples for split '{train_split}'...")
    try:
        builder = load_dataset_builder(hf_dataset_path)
        if train_split not in builder.info.splits:
            raise ValueError(f"Split '{train_split}' not found in dataset '{hf_dataset_path}'.")
        return builder.info.splits[train_split].num_examples
    except Exception as e:
        logger.error(f"Failed to get total sample count for split '{train_split}': {e}")
        raise


# --- Main Function ---
def main():
    parser = _setup_argparse()
    args = parser.parse_args()

    # Initialize a single global_stats dictionary
    global_stats = defaultdict(int)
    # Initialize the nested class_stats dictionary
    global_stats[SC_CLASS_STATS_DICT] = defaultdict(list)

    try:
        try:
            image_dir, label_dir = _configure_environment(args)
        except ValueError as e:
            logging.error(f"Configuration error: {e}")
            return 1

        phrase_map, class_names = _load_mapping_config(args.mapping_config)
        num_total_samples = _find_dataset_samples(args.hf_dataset_path, args.train_split)
        logger.info(f"Found {num_total_samples} total samples in split '{args.train_split}'.")

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
        if slice_size is None or slice_size <= 0 or slice_size > samples_to_process:
            slice_size = samples_to_process
            logger.info("Processing all samples in a single pass (no slicing).")
        else:
            logger.info(
                f"Processing in slices of {slice_size} with {args.num_proc} processes for preparation."
            )

        # Loop through slices
        for i in range(0, samples_to_process, slice_size):
            start = i
            end = min(i + slice_size, samples_to_process)
            if start >= end:
                continue  # Skip if start >= end

            # Process the slice, updates global_stats directly
            success = process_slice(
                start,
                end,
                phrase_map,
                args,
                global_stats,  # Pass the global stats dictionary
                image_dir,
                label_dir,
            )

            # Break loop if slice had a fatal error
            if not success:
                logger.error(f"Stopping processing due to fatal error in slice {start}-{end}.")
                break

        # Log final summary from aggregated global_stats
        _log_summary(global_stats, class_names)

        # Determine exit code based on error incidents
        if global_stats.get(SC_ERROR_INCIDENT, 0) > 0:
            logger.warning("Conversion finished with some errors. Check logs for details.")
            return (
                1 if global_stats.get(SC_ERROR_INCIDENT, 0) > 10 else 0
            )  # Only fail if many errors
        else:
            logger.info("Conversion finished successfully with no errors.")
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
