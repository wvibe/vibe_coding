"""CLI entry point for the conversion verification tool."""

import argparse
import logging
import os
import random
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import datasets
from tqdm import tqdm

# Local imports
from vibelab.dataops.cov_segm.converter import load_mapping_config
from vibelab.dataops.cov_segm.datamodel import SegmSample
from vibelab.dataops.cov_segm.loader import load_sample
from vibelab.dataops.cov_segm.convert_verifier import (
    VerificationResult,
    verify_sample_conversion,
)
from vibelab.utils.common.stats import format_statistics_table

logger = logging.getLogger(__name__)


def _setup_argparse() -> argparse.ArgumentParser:
    """Sets up the argument parser for the verifier CLI."""
    parser = argparse.ArgumentParser(
        description="Verify YOLO dataset conversion against original HF dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--train-split",
        type=str,
        required=True,
        choices=["train", "validation"],
        help="Dataset split to verify (must match HF split and YOLO folder name).",
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        required=True,
        choices=["visible", "full"],
        help="Mask type used during conversion (must match YOLO folder name).",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        required=True,
        help="Number of samples to randomly select and verify.",
    )

    # Optional arguments - Paths and Configs
    parser.add_argument(
        "--target-root",
        type=str,
        default=os.getenv("COV_SEGM_ROOT"),
        help="Root directory of the generated YOLO dataset. Defaults to $COV_SEGM_ROOT.",
    )
    parser.add_argument(
        "--mapping-config",
        type=str,
        default="configs/dataops/cov_segm_yolo_mapping.csv",
        help="Path to the mapping CSV file used during conversion.",
    )
    parser.add_argument(
        "--hf-dataset-path",
        type=str,
        default="lab42/cov-segm-v3",
        help="Name or path of the Hugging Face dataset.",
    )

    # Optional arguments - Verification Parameters
    parser.add_argument(
        "--bbox-min-iou",
        type=float,
        default=0.95,
        help="Minimum IoU threshold for bounding box matching.",
    )
    parser.add_argument(
        "--mask-min-iou",
        type=float,
        default=0.95,
        help="Minimum IoU threshold for mask matching.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for sample selection."
    )

    # Optional arguments - Output and Debugging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose per-sample debugging logs.",
    )

    # Performance arguments
    parser.add_argument(
        "--num-proc",
        type=int,
        default=8,
        help="Number of processes to use for parallel dataset operations.",
    )

    return parser


def validate_paths(target_root_path: Path, mask_type: str, train_split: str) -> Tuple[Path, Path]:
    """Validate YOLO dataset paths and return label and image directories."""
    yolo_label_dir = target_root_path / mask_type / "labels" / train_split
    yolo_image_dir = target_root_path / mask_type / "images" / train_split

    if not yolo_label_dir.is_dir():
        logger.error(f"YOLO label directory not found: {yolo_label_dir}")
        exit(1)
    if not yolo_image_dir.is_dir():
        logger.error(f"YOLO image directory not found: {yolo_image_dir}")
        exit(1)

    return yolo_label_dir, yolo_image_dir


def sample_yolo_ids(yolo_label_dir: Path, sample_count: int) -> List[str]:
    """Sample YOLO IDs for verification."""
    logger.info(f"Scanning YOLO label directory: {yolo_label_dir}")
    all_label_files = list(yolo_label_dir.glob("*.txt"))
    all_sample_ids = [f.stem for f in all_label_files]
    logger.info(f"Found {len(all_sample_ids)} total samples in YOLO dataset.")

    if not all_sample_ids:
        logger.error(f"No label files found in {yolo_label_dir}. Cannot verify.")
        exit(1)

    if sample_count >= len(all_sample_ids):
        sampled_ids = all_sample_ids
        logger.info(f"Sample count ({sample_count}) >= total samples. Verifying all {len(sampled_ids)} samples.")
    else:
        sampled_ids = random.sample(all_sample_ids, sample_count)
        logger.info(f"Randomly selected {len(sampled_ids)} samples for verification.")

    return sampled_ids


def load_hf_data(hf_dataset_path: str, train_split: str, sampled_ids_set: Set[str], num_proc: int = 1) -> Dict[str, Optional[SegmSample]]:
    """Load corresponding data from HuggingFace dataset."""
    logger.info(f"Loading Hugging Face dataset '{hf_dataset_path}' split '{train_split}'...")
    try:
        hf_dataset = datasets.load_dataset(hf_dataset_path, split=train_split)
        # Filter the dataset *before* loading samples
        logger.info(f"Filtering HF dataset for {len(sampled_ids_set)} selected sample IDs using {num_proc} processes...")
        filtered_hf_dataset = hf_dataset.filter(
            lambda example: example['id'] in sampled_ids_set,
            num_proc=num_proc
        )
        logger.info(f"Loading {len(filtered_hf_dataset)} corresponding samples from HF dataset...")

        # Store results in a dictionary keyed by sample_id for easy lookup
        original_samples_dict: Dict[str, Optional[SegmSample]] = {}
        processed_samples = filtered_hf_dataset.map(
            load_sample,
            num_proc=num_proc
        )

        for sample_data in tqdm(processed_samples, desc="Processing HF samples"):
            # Check if load_sample returned a dict (success) or None (error)
            if sample_data:
                try:
                    # Process sample data based on its type
                    if isinstance(sample_data, SegmSample):
                        original_samples_dict[sample_data.id] = sample_data
                    elif isinstance(sample_data, dict) and "id" in sample_data:
                        sample_obj = SegmSample.from_dict(sample_data)
                        original_samples_dict[sample_obj.id] = sample_obj
                    else:
                        logger.warning(f"Unexpected data type from map: {type(sample_data)}. Skipping.")
                except Exception as e:
                    sample_id_from_data = sample_data.get("id", "unknown") if isinstance(sample_data, dict) else "unknown"
                    logger.error(f"Error processing loaded sample {sample_id_from_data}: {e}", exc_info=True)
                    original_samples_dict[sample_id_from_data] = None

        return original_samples_dict

    except Exception as e:
        logger.error(f"Failed to load or process Hugging Face dataset: {e}", exc_info=True)
        exit(1)


def verify_samples(
    sampled_ids: List[str],
    yolo_label_dir: Path,
    yolo_image_dir: Path,
    original_samples_dict: Dict[str, Optional[SegmSample]],
    phrase_map: Dict[str, Dict[str, Any]],
    mask_type: str,
    mask_min_iou: float,
    bbox_min_iou: float
) -> List[VerificationResult]:
    """Verify each sample against the original data."""
    verification_results: List[VerificationResult] = []
    logger.info(f"Verifying {len(sampled_ids)} samples...")

    for sample_id in tqdm(sampled_ids, desc="Verifying samples"):
        yolo_label_path = yolo_label_dir / f"{sample_id}.txt"
        yolo_image_path = yolo_image_dir / f"{sample_id}.jpg"
        original_sample = original_samples_dict.get(sample_id)

        result = verify_sample_conversion(
            sample_id=sample_id,
            yolo_label_path=yolo_label_path,
            yolo_image_path=yolo_image_path,
            original_sample=original_sample,
            phrase_map=phrase_map,
            mask_type=mask_type,
            mask_min_iou=mask_min_iou,
            bbox_min_iou=bbox_min_iou,
            global_sample_ratio=1.0
        )
        verification_results.append(result)

    return verification_results


def aggregate_results(verification_results: List[VerificationResult]) -> Tuple[Dict[str, int], Dict[str, List[float]]]:
    """Aggregate verification results into statistics."""
    stats = {
        "total_samples_verified": 0,
        "total_processing_errors": 0,
        "total_matched": 0,
        "total_lost": 0,
        "total_extra": 0,
        "total_bbox_fails": 0,
    }

    iou_data = {
        "mask_ious": [],
        "bbox_ious": [],
    }

    for res in verification_results:
        if res.processing_error:
            stats["total_processing_errors"] += 1
        else:
            stats["total_samples_verified"] += 1
            stats["total_matched"] += len(res.matched_pairs)
            stats["total_lost"] += len(res.lost_instances)
            stats["total_extra"] += len(res.extra_instances)
            stats["total_bbox_fails"] += len(res.bbox_iou_failures)

            for pair in res.matched_pairs:
                iou_data["mask_ious"].append(pair["mask_iou"])
                iou_data["bbox_ious"].append(pair["bbox_iou"])

    return stats, iou_data


def report_results(
    stats: Dict[str, int],
    iou_data: Dict[str, List[float]],
    args: argparse.Namespace,
    all_sample_count: int
) -> int:
    """Report verification results and return exit code."""
    logger.info("--- Verification Summary ---")
    logger.info(f"Samples Requested: {args.sample_count}")
    logger.info(f"Samples Found in YOLO: {all_sample_count}")
    logger.info(f"Samples Selected for Verification: {stats['total_samples_verified'] + stats['total_processing_errors']}")
    logger.info(f"Samples Successfully Verified: {stats['total_samples_verified']}")
    logger.info(f"Samples with Processing Errors: {stats['total_processing_errors']}")
    logger.info("--- Instance Matching --- >")
    logger.info(f"Matched Instances: {stats['total_matched']}")
    logger.info(f"Lost Instances (Expected but not found in YOLO): {stats['total_lost']}")
    logger.info(f"Extra Instances (Found in YOLO but not expected): {stats['total_extra']}")
    logger.info(f"Matched Instances Failing BBox IoU (>= {args.bbox_min_iou}): {stats['total_bbox_fails']}")

    # Use stats table for IoU distributions if matches exist
    if stats["total_matched"] > 0:
        iou_stats_data = {
            "Mask IoU": iou_data["mask_ious"],
            "BBox IoU": iou_data["bbox_ious"],
        }
        # Example format string - adjust columns/width as needed
        format_string = "{key:<12} {count:>6d} {mean:>7.3f} {min:>7.3f} {p50:>7.3f} {max:>7.3f}"
        logger.info("--- IoU Statistics for Matched Instances --- >")
        table_lines = format_statistics_table(iou_stats_data, format_string)
        for line in table_lines:
            logger.info(line)
    else:
        logger.info("No matched instances to report IoU statistics for.")

    # Determine Overall Success and Exit Code
    is_success = (
        stats["total_processing_errors"] == 0 and
        stats["total_lost"] == 0 and
        stats["total_extra"] == 0 and
        stats["total_bbox_fails"] == 0
    )

    if is_success:
        logger.info("--- Verification PASSED --- >")
        return 0
    else:
        logger.warning("--- Verification FAILED --- >")
        return 1


def main():
    """Main execution function for the verification CLI."""
    parser = _setup_argparse()
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.setLevel(log_level)

    if not args.target_root:
        logger.error(
            "Target root directory not specified and COV_SEGM_ROOT env var not set. Exiting."
        )
        exit(1)

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"Using random seed: {args.seed}")

    logger.info("Starting YOLO dataset conversion verification...")
    logger.info(f"Split: {args.train_split}, Mask Type: {args.mask_type}")
    logger.info(f"Target Root: {args.target_root}")
    logger.info(f"Sample Count: {args.sample_count}")
    logger.info(f"Using {args.num_proc} processes for dataset operations")

    # 1. Validate paths and Load Mapping Config
    target_root_path = Path(args.target_root)
    yolo_label_dir, yolo_image_dir = validate_paths(
        target_root_path, args.mask_type, args.train_split
    )

    try:
        phrase_map, class_names = load_mapping_config(args.mapping_config)
        logger.info(f"Loaded mapping config from: {args.mapping_config}")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load mapping config: {e}")
        exit(1)

    # 2. Sample YOLO IDs
    sampled_ids = sample_yolo_ids(yolo_label_dir, args.sample_count)
    sampled_ids_set = set(sampled_ids)
    all_sample_count = len(sampled_ids)

    # 3. Load Corresponding HF Data
    original_samples_dict = load_hf_data(
        args.hf_dataset_path, args.train_split, sampled_ids_set, args.num_proc
    )

    # 4. Verify Each Sample
    verification_results = verify_samples(
        sampled_ids,
        yolo_label_dir,
        yolo_image_dir,
        original_samples_dict,
        phrase_map,
        args.mask_type,
        args.mask_min_iou,
        args.bbox_min_iou
    )

    # 5. Aggregate Results
    stats, iou_data = aggregate_results(verification_results)

    # 6. Print Summary Report and determine exit code
    exit_code = report_results(stats, iou_data, args, all_sample_count)

    logger.info("Verification process finished.")
    exit(exit_code)


if __name__ == "__main__":
    main()