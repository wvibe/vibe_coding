import argparse
import csv
import logging
import os
import random
import re  # Added import for regex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # Added for new functions
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

# Import the OOP data models
from vibelab.dataops.cov_segm.datamodel import ClsSegment, SegmMask, SegmSample

# Import loader using full path
from vibelab.dataops.cov_segm.loader import load_sample
from vibelab.utils.common.bbox import xyxy_to_xywh
from vibelab.utils.common.mask import (
    calculate_mask_iou,
    mask_to_yolo_polygons_verified,
)

# Import statistics formatting utility
from vibelab.utils.common.stats import format_statistics_table

ANNOTATION_OVERLAP_IOU_THRESH = 0.95

# Global stats counters (initialized in main)
stats_counters: Dict[str, Any] = {
    "total_samples": 0,
    "processed_samples": 0,  # Samples successfully loaded by load_sample
    "skipped_sample_load_error": 0,
    "skipped_sample_no_mapping": 0,
    "skipped_sample_sampling": 0,
    "segments_loaded": 0,  # Count of all segments encountered from loaded samples
    "skipped_segment_no_mapping": 0,
    "skipped_segment_zero_masks": 0,
    "instances_loaded": 0,  # Count of all instances encountered from loaded samples
    "skipped_instance_invalid": 0,
    # Annotation generation stats
    "skipped_mask_no_polygon": 0,  # Masks that failed polygon conversion
    "mask_annotations_generated": 0,  # Number of unique mask instances after aggregation
    "mask_annotations_ious": [],
    "mask_annotations_vertices": [],
    "mask_annotations_updated": 0,
    "mask_annotations_appended": 0,
    "mask_annotations_skipped": 0,
    "skipped_bbox_no_bbox": 0,  # Bboxes that failed bbox generation
    "bbox_annotations_generated": 0,  # Number of unique bbox instances after aggregation
    "bbox_annotations_updated": 0,
    "bbox_annotations_appended": 0,
    "bbox_annotations_skipped": 0,
    # File I/O stats
    "skipped_existing_mask_labels": 0,
    "skipped_existing_bbox_labels": 0,
    "skipped_existing_images": 0,
    "sample_with_annotations": set(),  # Set of sample_ids that had at least one annotation
    "output_mask_labels": 0,
    "output_bbox_labels": 0,
    "copied_images": 0,
    # Per class statistics
    "mask_instances_per_class": {},
    "bbox_instances_per_class": {},
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _parse_slice(slice_str: str) -> slice:
    """Parses a string slice like '[:100]' or '[1000:2000]' into a slice object."""
    pattern = r"^\[(.*?)(:(.*?))?\]$"
    match = re.match(pattern, slice_str)
    if not match:
        raise ValueError(
            f"Invalid slice format: '{slice_str}'. Expected format like '[start:stop]'."
        )
    start_str, _, stop_str = match.groups(default="")
    try:
        start = int(start_str) if start_str else None
        stop = int(stop_str) if stop_str else None
    except ValueError as e:
        raise ValueError(
            f"Invalid slice components in '{slice_str}'. Start and stop must be integers."
        ) from e
    if start is not None and start < 0:
        raise ValueError(f"Slice start cannot be negative: {start}")
    if stop is not None and stop < 0:
        raise ValueError(f"Slice stop cannot be negative: {stop}")
    return slice(start, stop)


def _setup_argparse() -> argparse.ArgumentParser:
    """Sets up the argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert lab42/cov-segm-v3 dataset to YOLO segmentation format."
    )
    parser.add_argument(
        "--mapping-config",
        type=str,
        default="configs/dataops/cov_segm_yolo_mapping_20cls.csv",
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
        "--label-type",
        type=str,
        choices=["bbox", "mask"],
        default="bbox",
        help="Type of labels to generate ('bbox' or 'mask').",
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
        "--sample-slice",
        type=str,
        default=None,
        help="Slice of the dataset to process (e.g., '[:100]', '[1000:2000]'). Format: [start:stop]",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.0,
        help=(
            "Global sampling ratio (0.0-1.0) applied to all classes in addition to individual "
            "ratios. 0.0 means no sampling."
        ),
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


def _configure_environment(args: argparse.Namespace) -> Tuple[Path, Path]:
    """Configures logging, env vars, paths, and random seed.
    Returns paths for images and labels directories.
    """
    if not (0.0 <= args.sample_ratio <= 1.0):
        raise ValueError(f"Invalid sample_ratio: {args.sample_ratio}. Must be between 0.0 and 1.0.")
    load_dotenv()
    output_dir_str = args.output_dir or os.getenv("COV_SEGM_ROOT")
    if not output_dir_str:
        raise ValueError(
            "Output directory must be specified via --output-dir or COV_SEGM_ROOT env var."
        )
    yolo_root = Path(output_dir_str) / (args.output_name or args.mask_tag)
    # Add a subfolder for label type
    yolo_root_typed = yolo_root / args.label_type
    logging.info(f"Output dataset root: {yolo_root_typed}")
    if args.seed is not None:
        random.seed(args.seed)
        logging.info(f"Using random seed: {args.seed}")
    image_dir = yolo_root_typed / "images" / args.train_split
    label_dir = yolo_root_typed / "labels" / args.train_split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output images will be saved to: {image_dir}")
    logging.info(f"Output labels will be saved to: {label_dir}")
    return image_dir, label_dir


def _load_data(
    args: argparse.Namespace,
) -> Tuple[Dataset, Dict[str, Dict[str, Any]], Dict[int, str]]:
    """Loads the mapping config and the Hugging Face dataset, applying slicing if specified."""
    phrase_map, class_names = load_mapping_config(args.mapping_config)
    logging.info(f"Loading dataset '{args.hf_dataset_path}' split '{args.train_split}'...")
    try:
        full_dataset = load_dataset(args.hf_dataset_path, split=args.train_split)
        total_rows = len(full_dataset)
        logging.info(f"Full dataset loaded with {total_rows} samples.")
        if args.sample_slice:
            data_slice = _parse_slice(args.sample_slice)
            indices = range(*data_slice.indices(total_rows))
            if not indices:
                raise ValueError(f"Slice '{args.sample_slice}' resulted in zero samples selected.")
            dataset = full_dataset.select(indices)
            logging.info(
                f"Applying slice '{args.sample_slice}'. Processing {len(dataset)} samples."
            )
        else:
            dataset = full_dataset
            logging.info("Processing the full dataset (no slice specified).")
        return dataset, phrase_map, class_names
    except Exception as e:
        logging.error(f"Failed to load or slice dataset: {e}", exc_info=True)
        raise


def _get_phrase_mapping_info(
    segment: ClsSegment,
    phrase_map: Dict[str, Dict[str, Any]],
) -> Optional[Tuple[Dict[str, Any], str]]:
    """Gets the mapping information for a phrase."""
    for phrase in segment.phrases:
        phrase_text = phrase.text.strip()
        if not phrase_text:
            continue
        current_mapping = phrase_map.get(phrase_text)
        if current_mapping:
            return current_mapping, phrase_text
    return None, None


def _skip_sample_by_mapping_and_sampling(
    sample: SegmSample,
    phrase_map: Dict[str, Dict[str, Any]],
    global_sample_ratio: float = 0.0,  # 0.0 means no sampling
) -> bool:
    """Checks if a sample should be skipped based on the mapping and sampling config."""
    max_segment_sampling_ratio = 0.0
    for segment in sample.segments:
        mapping_info, _ = _get_phrase_mapping_info(segment, phrase_map)
        if mapping_info is None:
            continue
        if mapping_info["sampling_ratio"] > max_segment_sampling_ratio:
            max_segment_sampling_ratio = mapping_info["sampling_ratio"]
    if max_segment_sampling_ratio == 0.0:
        stats_counters["skipped_sample_no_mapping"] += 1
        return True
    if global_sample_ratio > 0.0:
        effective_ratio = global_sample_ratio * max_segment_sampling_ratio
        if random.random() > effective_ratio:
            stats_counters["skipped_sample_sampling"] += 1
            return True
    return False


def load_mapping_config(
    config_path: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, str]]:
    """Loads the phrase-to-class mapping configuration from a CSV file."""
    phrase_map: Dict[str, Dict[str, Any]] = {}
    class_names: Dict[int, str] = {}
    essential_columns = {"yolo_class_id", "yolo_class_name", "hf_phrase", "sampling_ratio"}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Mapping config file not found: {config_path}")
    with open(config_path, mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        if not essential_columns.issubset(reader.fieldnames or []):
            missing = essential_columns - set(reader.fieldnames or [])
            raise ValueError(f"Missing required columns in {config_path}: {missing}")
        for row_idx, row in enumerate(reader):
            try:
                class_id = int(row["yolo_class_id"])
                class_name = row["yolo_class_name"].strip()
                sampling_ratio = float(row["sampling_ratio"])
                priority = int(row.get("yolo_class_pri", 0) or 0)
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
                    "priority": priority,
                }
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
    if not phrase_map:
        raise ValueError(f"No valid mappings found in {config_path}")
    logging.info(f"Loaded {len(phrase_map)} phrase mappings for {len(class_names)} classes.")
    return phrase_map, class_names


def process_single_sample(sample_row):
    """Process a single sample row and return a serialized dictionary."""
    try:
        sample = load_sample(sample_row)
        if sample is None:
            return {"error": True, "reason": "load_sample returned None"}
        serialized = sample.to_dict()
        serialized["error"] = False
        return serialized
    except Exception as e:
        logging.warning(f"Sample loading error: {e}", exc_info=True)
        return {"error": True, "reason": str(e)}


def _get_yolo_bbox_str_from_abs_xyxy(
    abs_xyxy_bbox: Tuple[int, int, int, int], img_width: int, img_height: int
) -> str:
    """Converts an absolute pixel bounding box (xmin, ymin, xmax, ymax)
    to a normalized YOLO format annotation string "cx cy w h".
    First converts to [cx, cy, w, h] format, then normalizes.
    """
    x1, y1, x2, y2 = abs_xyxy_bbox
    abs_coords = np.array([x1, y1, x2, y2])
    abs_xywh = xyxy_to_xywh(abs_coords)
    norm_xywh = np.array(
        [
            abs_xywh[0] / img_width,  # center x
            abs_xywh[1] / img_height,  # center y
            abs_xywh[2] / img_width,  # width
            abs_xywh[3] / img_height,  # height
        ]
    )
    return f"{norm_xywh[0]} {norm_xywh[1]} {norm_xywh[2]} {norm_xywh[3]}"


def _update_annotations_list(
    existing_annotations_list: List[Dict],
    candidate_annotation_data: Dict,
    iou_cutoff: float = 0.95,
) -> int:
    """Manages a list of annotations based on mask IoU and class priority.
    Returns:
        - 0 if the candidate annotation is handled (replaced or appended)
        - 1 if the candidate annotation is appended
        - -1 if the candidate annotation is skipped
    """
    idx_to_replace = -1
    candidate_is_inferior_or_handled = False

    for i, existing_ann in enumerate(existing_annotations_list):
        iou = calculate_mask_iou(
            candidate_annotation_data["source_binary_mask"].astype(bool),
            existing_ann["source_binary_mask"].astype(bool),
        )
        if iou >= iou_cutoff:
            if candidate_annotation_data["priority"] > existing_ann["priority"]:
                idx_to_replace = i
                break
            elif (
                candidate_annotation_data["priority"] == existing_ann["priority"]
                and candidate_annotation_data["class_id"] < existing_ann["class_id"]
            ):  # Tie-breaking
                idx_to_replace = i
                break
            else:
                candidate_is_inferior_or_handled = True
                break
    if idx_to_replace != -1:
        existing_annotations_list[idx_to_replace] = candidate_annotation_data
        return 0
    elif not candidate_is_inferior_or_handled:
        existing_annotations_list.append(candidate_annotation_data)
        return 1
    return -1


def _generate_single_sample_labels(
    sample: SegmSample, phrase_map: Dict[str, Dict[str, Any]], args: argparse.Namespace
) -> Tuple[List[Dict], Optional[Any]]:
    """Processes a single SegmSample, generates annotations for the specified label type."""
    global stats_counters
    image_height, image_width = sample.image.height, sample.image.width
    if _skip_sample_by_mapping_and_sampling(sample, phrase_map, args.sample_ratio):
        return [], None

    current_annotations_rich: List[Dict] = []

    for segment in sample.segments:
        stats_counters["segments_loaded"] += 1
        mapping_info, matched_phrase = _get_phrase_mapping_info(segment, phrase_map)
        if mapping_info is None:
            stats_counters["skipped_segment_no_mapping"] += 1
            continue

        current_class_id = mapping_info["class_id"]
        current_priority = mapping_info["priority"]
        masks_to_process: List[SegmMask] = (
            segment.visible_masks if args.mask_tag == "visible" else segment.full_masks
        )
        valid_masks = [m for m in masks_to_process if m.is_valid]
        if args.skip_zero and not valid_masks:
            stats_counters["skipped_segment_zero_masks"] += 1
            continue

        invalid_mask_count = len(masks_to_process) - len(valid_masks)
        if invalid_mask_count > 0:
            stats_counters["skipped_instance_invalid"] += invalid_mask_count

        for v_mask in valid_masks:
            stats_counters["instances_loaded"] += 1
            current_v_mask_binary = v_mask.binary_mask

            if args.label_type == "mask":
                polygons_yolo_lists, iou, err_msg = mask_to_yolo_polygons_verified(
                    binary_mask=current_v_mask_binary, img_shape=(image_height, image_width)
                )

                if err_msg or not polygons_yolo_lists:
                    stats_counters["skipped_mask_no_polygon"] += 1
                    logging.warning(
                        f"Sample {sample.id}, Phrase '{matched_phrase}': Failed to convert mask to "
                        f"polygon during verification. Error: {err_msg}"
                    )
                else:
                    stats_counters["mask_annotations_ious"].append(iou)

                    # Concatenate all polygon parts into a single annotation string
                    all_coords_flat = []
                    for poly_part in polygons_yolo_lists:
                        all_coords_flat.extend(poly_part)

                    num_vertices = len(all_coords_flat) // 2
                    stats_counters["mask_annotations_vertices"].append(num_vertices)

                    annotation_str = " ".join(map(str, all_coords_flat))
                    candidate_payload = {
                        "source_binary_mask": current_v_mask_binary,
                        "class_id": current_class_id,
                        "priority": current_priority,
                        "annotation_str": annotation_str,
                    }
                    update_stats = _update_annotations_list(
                        current_annotations_rich,
                        candidate_payload,
                        ANNOTATION_OVERLAP_IOU_THRESH,
                    )
                    if update_stats == 0:
                        stats_counters["mask_annotations_updated"] += 1
                    elif update_stats == 1:
                        stats_counters["mask_annotations_appended"] += 1
                    else:
                        stats_counters["mask_annotations_skipped"] += 1

            elif args.label_type == "bbox":
                abs_xyxy_bbox = v_mask.bbox
                if abs_xyxy_bbox is None:
                    stats_counters["skipped_bbox_no_bbox"] += 1
                    logging.warning(
                        f"Sample {sample.id}, Phrase '{matched_phrase}': SegmMask bbox is None, "
                        f"cannot generate bbox label."
                    )
                else:
                    bbox_yolo_annotation_str = _get_yolo_bbox_str_from_abs_xyxy(
                        abs_xyxy_bbox, image_width, image_height
                    )
                    candidate_payload = {
                        "source_binary_mask": current_v_mask_binary,
                        "abs_bbox_xyxy": abs_xyxy_bbox,
                        "class_id": current_class_id,
                        "priority": current_priority,
                        "annotation_str": bbox_yolo_annotation_str,
                    }
                    update_stats = _update_annotations_list(
                        current_annotations_rich,
                        candidate_payload,
                        ANNOTATION_OVERLAP_IOU_THRESH,
                    )
                    if update_stats == 0:
                        stats_counters["bbox_annotations_updated"] += 1
                    elif update_stats == 1:
                        stats_counters["bbox_annotations_appended"] += 1
                    else:
                        stats_counters["bbox_annotations_skipped"] += 1

    final_output_list = [
        {"class_id": ann["class_id"], "annotation_str": ann["annotation_str"]}
        for ann in current_annotations_rich
    ]
    annotations_made_for_sample = bool(final_output_list)

    if annotations_made_for_sample:
        stats_counters["sample_with_annotations"].add(sample.id)

        if args.label_type == "mask":
            stats_counters["mask_annotations_generated"] += len(final_output_list)
            class_counts = {}
            for ann_data in final_output_list:
                final_cid = ann_data["class_id"]
                class_counts.setdefault(final_cid, 0)
                class_counts[final_cid] += 1
            for cid, count in class_counts.items():
                stats_counters["mask_instances_per_class"].setdefault(cid, []).append(count)
        else:  # args.label_type == "bbox"
            stats_counters["bbox_annotations_generated"] += len(final_output_list)
            class_counts = {}
            for ann_data in final_output_list:
                final_cid = ann_data["class_id"]
                class_counts.setdefault(final_cid, 0)
                class_counts[final_cid] += 1
            for cid, count in class_counts.items():
                stats_counters["bbox_instances_per_class"].setdefault(cid, []).append(count)

    return final_output_list, sample.image if annotations_made_for_sample else None


def _save_sample_outputs(
    sample_id: str,
    image_to_save: Any,
    annotations: List[Dict],
    image_dir: Path,
    label_dir: Path,
    label_type: str,
    no_overwrite_flag: bool,
):
    """Handles all file I/O for a single sample's outputs."""
    global stats_counters
    base_filename = sample_id

    if annotations:
        image_path = image_dir / f"{sample_id}.jpg"
        if no_overwrite_flag and image_path.exists():
            stats_counters["skipped_existing_images"] += 1
        else:
            try:
                if image_to_save.mode != "RGB":
                    image_to_save = image_to_save.convert("RGB")
                image_to_save.save(image_path, "JPEG", quality=95)
                stats_counters["copied_images"] += 1
            except Exception as e:
                logging.error(f"Failed to save image {sample_id}.jpg to {image_path}: {e}")

        label_path = label_dir / f"{base_filename}.txt"
        if no_overwrite_flag and label_path.exists():
            if label_type == "mask":
                stats_counters["skipped_existing_mask_labels"] += 1
            else:  # label_type == "bbox"
                stats_counters["skipped_existing_bbox_labels"] += 1
        else:
            annotation_lines = [f"{ann['class_id']} {ann['annotation_str']}" for ann in annotations]
            try:
                with open(label_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(annotation_lines) + "\n")
                if label_type == "mask":
                    stats_counters["output_mask_labels"] += 1
                else:  # label_type == "bbox"
                    stats_counters["output_bbox_labels"] += 1
            except Exception as e:
                logging.error(f"Failed to write label file {label_path}: {e}")


def _process_samples(
    dataset: Dataset,
    phrase_map: Dict[str, Dict[str, Any]],
    image_dir: Path,
    label_dir: Path,
    args: argparse.Namespace,
):
    """Process samples using serialization, new label generation, and saving logic."""
    global stats_counters
    total_samples = len(dataset)
    stats_counters["total_samples"] = total_samples

    logging.info(f"Loading and processing samples using {args.num_proc} processes...")
    logging.info(
        "Using load_from_cache_file=False and writer_batch_size=200 for datasets.map(). "
        "Note: If the process crashes, manual cleanup of the Hugging Face cache "
        "(~/.cache/huggingface/datasets) might be necessary."
    )
    processed_dataset = dataset.map(
        process_single_sample,
        num_proc=args.num_proc if args.num_proc > 1 else None,
        load_from_cache_file=False,
        writer_batch_size=200,
        desc="Loading and Serializing Samples",
    )

    logging.info("Generating annotations and saving outputs...")
    processed_iterator = tqdm(
        processed_dataset, desc="Generating Labels & Saving Outputs", total=len(processed_dataset)
    )
    for i, sample_dict in enumerate(processed_iterator):
        if sample_dict.get("error", False):
            stats_counters["skipped_sample_load_error"] += 1
            continue
        try:
            sample = SegmSample.from_dict(sample_dict)
            stats_counters["processed_samples"] += 1
        except Exception as e:
            logging.error(f"Error deserializing sample at index {i}: {e}")
            stats_counters["skipped_sample_load_error"] += 1
            continue

        annotations, image_obj_or_none = _generate_single_sample_labels(sample, phrase_map, args)
        if image_obj_or_none:
            _save_sample_outputs(
                sample.id,
                image_obj_or_none,
                annotations,
                image_dir,
                label_dir,
                args.label_type,
                args.no_overwrite,
            )
    logging.info(
        f"Finished processing. Samples with any annotations: "
        f"{len(stats_counters['sample_with_annotations'])}/{stats_counters['processed_samples']}"
    )


def _log_summary(counters: Dict[str, Any], class_names: Dict[int, str], label_type: str = "bbox"):
    """Logs the final conversion statistics."""
    logging.info("-" * 80)
    logging.info(f"{'CONVERSION SUMMARY':^40}")
    logging.info("-" * 80)

    logging.info(f"{'DATASET PROCESSING:':^40}")
    logging.info(f"  Total samples in dataset split: {counters['total_samples']}")
    logging.info(f"  Successfully loaded and processed: {counters['processed_samples']}")
    logging.info(f"  Due to load/deserialization errors: {counters['skipped_sample_load_error']}")
    logging.info(f"  No class mapping for any segment: {counters['skipped_sample_no_mapping']}")
    logging.info(f"  Due to mapping and sampling ratios: {counters['skipped_sample_sampling']}")
    logging.info("-" * 80)

    logging.info(f"{'SEGMENT & INSTANCE PROCESSING:':^40}")
    logging.info(f"  Total segments encountered: {counters['segments_loaded']}")
    logging.info(f"  Segments skipped (no mapping): {counters['skipped_segment_no_mapping']}")
    logging.info(f"  Segments skipped (empty mask): {counters['skipped_segment_zero_masks']}")
    logging.info(f"  Segment Instances loaded: {counters['instances_loaded']}")
    logging.info(f"  Instances skipped (invalid masks): {counters['skipped_instance_invalid']}")
    logging.info("-" * 80)

    if label_type == "mask":
        logging.info(f"{'MASK ANNOTATION GENERATION:':^80}")
        logging.info(f"  Mask annotations generated: {counters['mask_annotations_generated']}")
        logging.info(
            f"  Instances with failed polygon conversion: {counters['skipped_mask_no_polygon']}"
        )
        logging.info(f"  Mask annotations updated: {counters['mask_annotations_updated']}")
        logging.info(f"  Mask annotations appended: {counters['mask_annotations_appended']}")
        logging.info(f"  Mask annotations skipped: {counters['mask_annotations_skipped']}")

        # MODIFIED: Added IoU statistics reporting (using updated key name)
        iou_values = counters.get("mask_annotations_ious", [])
        if iou_values:
            logging.info(f"{'MASK CONVERSION IoU STATISTICS:':^80}")
            iou_format_string = "{key:<18} {count:>5} {mean:>5.2f} {min:>5.2f} {p10:>5.2f} {p25:>5.2f} {p50:>5.2f} {p75:>5.2f} {p90:>5.2f}"
            iou_stats_data = {"Mask Convert IoU": iou_values}
            iou_table_lines = format_statistics_table(
                data_dict=iou_stats_data, format_string=iou_format_string
            )
            for line in iou_table_lines:
                logging.info(f"  {line}")
        else:
            logging.info("No mask conversion IoUs were recorded.")

        # MODIFIED: Added Mask Polygon Vertex Count Statistics table
        vertex_counts_list = counters.get("mask_annotations_vertices", [])
        if vertex_counts_list:
            logging.info(f"{'MASK POLYGON VERTEX COUNT STATISTICS:':^80}")
            vertex_format_string = "{key:<18} {count:>5} {mean:>5.1f} {min:>5d} {p10:>5d} {p25:>5d} {p50:>5d} {p75:>5d} {p90:>5d} {max:>5d}"
            vertex_stats_data = {"Vertex Counts": vertex_counts_list}
            vertex_table_lines = format_statistics_table(
                data_dict=vertex_stats_data, format_string=vertex_format_string
            )
            for line in vertex_table_lines:
                logging.info(f"  {line}")
        else:
            logging.info("No mask polygon vertex counts were recorded.")
    else:  # label_type == "bbox"
        logging.info(f"{'BBOX ANNOTATION GENERATION:':^80}")
        logging.info(f"  Bbox annotations generated: {counters['bbox_annotations_generated']}")
        logging.info(f"  Instances with failed bbox generation: {counters['skipped_bbox_no_bbox']}")
        logging.info(f"  Bbox annotations updated: {counters['bbox_annotations_updated']}")
        logging.info(f"  Bbox annotations appended: {counters['bbox_annotations_appended']}")
        logging.info(f"  Bbox annotations skipped: {counters['bbox_annotations_skipped']}")

    logging.info("-" * 80)

    logging.info(f"{'FILE OUTPUT:':^80}")
    logging.info(f"  Samples with annotations: {len(counters['sample_with_annotations'])}")

    if label_type == "mask":
        logging.info(f"  Mask labels written: {counters['output_mask_labels']}")
        logging.info(
            f"  Mask labels skipped (already exist): {counters['skipped_existing_mask_labels']}"
        )
    else:  # label_type == "bbox"
        logging.info(f"  Bbox labels written: {counters['output_bbox_labels']}")
        logging.info(
            f"  Bbox labels skipped (already exist): {counters['skipped_existing_bbox_labels']}"
        )

    logging.info(f"  Images copied: {counters['copied_images']}")
    logging.info(f"  Images skipped (already exist): {counters['skipped_existing_images']}")
    logging.info("-" * 80)

    if label_type == "mask" and counters["mask_instances_per_class"]:
        logging.info(f"{'MASK ANNOTATIONS BY CLASS (Unique Samples per Class):':^80}")
        format_string = "{key:<18} {count:>5} {mean:>5.1f} {min:>5d} {p10:>5d} {p25:>5d} {p50:>5d} {p75:>5d} {p90:>5d} {max:>5d}"
        mask_cls_data = {}
        for cid, sids in counters["mask_instances_per_class"].items():
            class_name = class_names.get(cid, f"Unknown-{cid}")
            mask_cls_data[class_name] = sids

        table_lines = format_statistics_table(
            mask_cls_data,
            format_string=format_string,
        )
        for line in table_lines:
            logging.info(f"  {line}")

    elif label_type == "bbox" and counters["bbox_instances_per_class"]:
        logging.info(f"{'BBOX ANNOTATIONS BY CLASS (Unique Samples per Class):':^80}")
        format_string = "{key:<18} {count:>5} {mean:>5.1f} {min:>5d} {p10:>5d} {p25:>5d} {p50:>5d} {p75:>5d} {p90:>5d} {max:>5d}"
        bbox_cls_data = {}
        for cid, sids in counters["bbox_instances_per_class"].items():
            class_name = class_names.get(cid, f"Unknown-{cid}")
            bbox_cls_data[class_name] = sids

        table_lines = format_statistics_table(
            bbox_cls_data,
            format_string=format_string,
        )
        for line in table_lines:
            logging.info(f"  {line}")

    logging.info("-" * 80)


def main():
    """Main function to orchestrate the dataset conversion."""
    global stats_counters

    parser = _setup_argparse()
    args = parser.parse_args()

    # Log all arguments
    logging.info("-" * 80)
    logging.info(f"{'PROGRAM ARGUMENTS':^40}")
    logging.info("-" * 80)
    for arg_name, arg_value in sorted(vars(args).items()):
        logging.info(f"  {arg_name:<20}: {arg_value}")
    logging.info("-" * 80)

    try:
        image_dir, label_dir = _configure_environment(args)
        dataset, phrase_map, class_names = _load_data(args)
        _process_samples(dataset, phrase_map, image_dir, label_dir, args)
        _log_summary(stats_counters, class_names, args.label_type)
        logging.info("Conversion completed successfully.")
        return 0
    except Exception as e:
        logging.error(f"Conversion failed with unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
