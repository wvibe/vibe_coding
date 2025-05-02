"""CLI entry point for the conversion verification tool."""

import argparse
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import datasets
from tqdm import tqdm

from vibelab.dataops.cov_segm.convert_verifier import (
    VerificationResult,
    verify_sample_conversion,
)

# Local imports
from vibelab.dataops.cov_segm.converter import load_mapping_config
from vibelab.dataops.cov_segm.datamodel import SegmSample
from vibelab.dataops.cov_segm.loader import load_sample
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

    # Sample selection mutually exclusive group
    sample_group = parser.add_mutually_exclusive_group(required=True)
    sample_group.add_argument(
        "--sample-count",
        type=int,
        help="Number of samples to randomly select and verify.",
    )
    sample_group.add_argument(
        "--sample-id",
        type=str,
        help="Specific sample ID to verify (for debugging).",
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
        "--iou-cutoff",
        type=float,
        default=0.5,
        help=(
            "Minimum mask IoU threshold required to consider an original and YOLO "
            "instance as a potential match."
        ),
    )
    parser.add_argument(
        "--iou-top",
        type=float,
        default=0.95,
        help=(
            "High IoU threshold used for quality assessment. Matched pairs below this "
            "(for mask or bbox) are flagged."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sample selection.")

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
        default=1,
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


def sample_yolo_ids(yolo_label_dir: Path, args: argparse.Namespace) -> List[str]:
    """Sample YOLO IDs for verification or find specific ID."""
    logger.info(f"Scanning YOLO label directory: {yolo_label_dir}")
    all_label_files = list(yolo_label_dir.glob("*.txt"))
    all_sample_ids = [f.stem for f in all_label_files]
    logger.info(f"Found {len(all_sample_ids)} total samples in YOLO dataset.")

    if not all_sample_ids:
        logger.error(f"No label files found in {yolo_label_dir}. Cannot verify.")
        exit(1)

    # Check for specific sample ID mode
    if args.sample_id:
        # Check if the requested sample ID exists
        if args.sample_id in all_sample_ids:
            logger.info(f"Found specified sample ID '{args.sample_id}' in YOLO dataset.")
            return [args.sample_id]
        else:
            logger.error(f"Specified sample ID '{args.sample_id}' not found in YOLO dataset.")
            exit(1)

    # Random sampling mode
    if args.sample_count >= len(all_sample_ids):
        sampled_ids = all_sample_ids
        log_msg = (
            f"Sample count ({args.sample_count}) >= total samples. "
            f"Verifying all {len(sampled_ids)} samples."
        )
        logger.info(log_msg)
    else:
        sampled_ids = random.sample(all_sample_ids, args.sample_count)
        logger.info(f"Randomly selected {len(sampled_ids)} samples for verification.")

    return sampled_ids


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
        logging.warning(f"Sample loading error: {e}")
        # Return an error flag that can be checked later
        return {"error": True, "reason": str(e)}


def load_hf_data(
    hf_dataset_path: str, train_split: str, sampled_ids_set: Set[str], num_proc: int = 1
) -> Dict[str, Optional[SegmSample]]:
    """Load corresponding data from HuggingFace dataset."""
    logger.info(f"Loading Hugging Face dataset '{hf_dataset_path}' split '{train_split}'...")
    try:
        hf_dataset = datasets.load_dataset(hf_dataset_path, split=train_split)
        # Filter the dataset *before* loading samples
        log_msg = (
            f"Filtering HF dataset for {len(sampled_ids_set)} selected sample IDs "
            f"using {num_proc} processes..."
        )
        logger.info(log_msg)
        filtered_hf_dataset = hf_dataset.filter(
            lambda example: example["id"] in sampled_ids_set, num_proc=num_proc
        )
        logger.info(f"Loading {len(filtered_hf_dataset)} corresponding samples from HF dataset...")

        # Store results in a dictionary keyed by sample_id for easy lookup
        original_samples_dict: Dict[str, Optional[SegmSample]] = {}
        processed_samples = filtered_hf_dataset.map(process_single_sample, num_proc=num_proc)

        for sample_data in tqdm(processed_samples, desc="Processing HF samples"):
            # Check if load_sample returned a dict (success) or None (error)
            if sample_data.get("error", True):
                continue
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
                sample_id_from_data = (
                    sample_data.get("id", "unknown") if isinstance(sample_data, dict) else "unknown"
                )
                logger.error(
                    f"Error processing loaded sample {sample_id_from_data}: {e}", exc_info=True
                )
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
    iou_cutoff: float,
    iou_top: float,
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
            iou_cutoff=iou_cutoff,
            iou_top=iou_top,
            global_sample_ratio=1.0,
        )
        verification_results.append(result)

    return verification_results


def aggregate_results(
    verification_results: List[VerificationResult],
) -> Tuple[Dict[str, int], Dict[str, List[float]], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """Aggregate verification results into statistics, including per-class counts."""
    stats = {
        "total_samples_verified": 0,
        "total_processing_errors": 0,
        "total_matched": 0,
        "total_lost": 0,
        "total_extra": 0,
        "total_mask_top_iou_fails": 0,
        "total_bbox_top_iou_fails": 0,
    }

    iou_data = {
        "mask_ious": [],
        "bbox_ious": [],
    }

    # Per-class counters
    expected_by_class = defaultdict(int)
    matched_by_class = defaultdict(int)
    lost_by_class = defaultdict(int)
    extra_by_class = defaultdict(int)

    for res in verification_results:
        if res.processing_error:
            stats["total_processing_errors"] += 1
        else:
            stats["total_samples_verified"] += 1
            stats["total_matched"] += len(res.matched_pairs)
            stats["total_lost"] += len(res.lost_instances)
            stats["total_extra"] += len(res.extra_instances)

            for pair in res.matched_pairs:
                class_id = pair["class_id"]
                iou_data["mask_ious"].append(pair["mask_iou"])
                iou_data["bbox_ious"].append(pair["bbox_iou"])
                matched_by_class[class_id] += 1
                expected_by_class[class_id] += 1

                if not pair["mask_threshold_passed"]:
                    stats["total_mask_top_iou_fails"] += 1
                if not pair["bbox_threshold_passed"]:
                    stats["total_bbox_top_iou_fails"] += 1

            for inst in res.lost_instances:
                class_id = inst.class_id
                lost_by_class[class_id] += 1
                expected_by_class[class_id] += 1

            for inst in res.extra_instances:
                class_id = inst.class_id
                extra_by_class[class_id] += 1

    return stats, iou_data, dict(expected_by_class), dict(lost_by_class), dict(extra_by_class)


def report_results(
    stats: Dict[str, int],
    iou_data: Dict[str, List[float]],
    args: argparse.Namespace,
    all_sample_count: int,
    class_names: Dict[int, str],
    expected_by_class: Dict[int, int],
    lost_by_class: Dict[int, int],
    extra_by_class: Dict[int, int],
) -> int:
    """Report verification results, including per-class stats, and return exit code."""
    logger.info("--- Verification Summary ---")
    if args.sample_id:
        logger.info(f"Verified Specific Sample ID: {args.sample_id}")
    else:
        logger.info(f"Samples Requested: {args.sample_count}")
    logger.info(f"Samples Found in YOLO: {all_sample_count}")
    logger.info(
        f"Samples Selected for Verification: "
        f"{stats['total_samples_verified'] + stats['total_processing_errors']}"
    )
    logger.info(f"Samples Successfully Verified: {stats['total_samples_verified']}")
    logger.info(f"Samples with Processing Errors: {stats['total_processing_errors']}")
    logger.info(f"--- Instance Matching (Mask IoU >= {args.iou_cutoff}) --- >")
    logger.info(f"Matched Instances: {stats['total_matched']}")
    logger.info(f"Lost Instances (Expected but not matched): {stats['total_lost']}")
    logger.info(f"Extra Instances (Found in YOLO but not matched): {stats['total_extra']}")
    logger.info(f"--- Matched Instance Quality (Threshold = {args.iou_top}) --- >")
    logger.info(
        f"Matched Instances Failing Mask IoU (>= {args.iou_top}): "
        f"{stats['total_mask_top_iou_fails']}"
    )
    logger.info(
        f"Matched Instances Failing BBox IoU (>= {args.iou_top}): "
        f"{stats['total_bbox_top_iou_fails']}"
    )

    # Use stats table for IoU distributions if matches exist
    if stats["total_matched"] > 0:
        iou_stats_data = {
            "Mask IoU": iou_data["mask_ious"],
            "BBox IoU": iou_data["bbox_ious"],
        }
        # Example format string - adjust columns/width as needed
        format_string = "{key:<12} {count:>6d} {mean:>7.3f} {min:>7.3f} {p50:>7.3f} {max:>7.3f}"
        logger.info(
            "--- IoU Statistics for Matched Instances (Pairs with Mask IoU >= {}) --- >".format(
                args.iou_cutoff
            )
        )
        table_lines = format_statistics_table(iou_stats_data, format_string)
        for line in table_lines:
            logger.info(line)
    else:
        logger.info("No matched instances to report IoU statistics for.")

    # Report Per-Class Statistics
    logger.info("--- Per-Class Instance Statistics --- >")
    all_class_ids = (
        set(expected_by_class.keys()) | set(lost_by_class.keys()) | set(extra_by_class.keys())
    )
    if not all_class_ids:
        logger.info("  No instances found across any class.")
    else:
        # Header with corrected alignment
        header1 = (
            f"  {'Class':<20} {'Expected':>8} {'Matched':>8} {'Lost':>8} "
            f"{'Lost (%)':>9} {'Extra':>8}"
        )
        header2 = (
            f"  {'--------------------':<20} {'--------':>8} {'--------':>8} "
            f"{'--------':>8} {'---------':>9} {'--------':>8}"
        )
        logger.info(header1)
        logger.info(header2)
        for cid in sorted(list(all_class_ids)):
            class_name = class_names.get(cid, f"Unknown ({cid})")
            expected = expected_by_class.get(cid, 0)
            lost = lost_by_class.get(cid, 0)
            extra = extra_by_class.get(cid, 0)
            matched = expected_by_class.get(cid, 0) - lost_by_class.get(cid, 0)
            lost_percentage = (lost / expected * 100) if expected > 0 else 0.0
            log_line = (
                f"  {f'{cid} ({class_name})':<20} {expected:>8} {matched:>8} "
                f"{lost:>8} ({lost_percentage:>5.1f}%) {extra:>8}"
            )
            logger.info(log_line)

    # Determine Overall Success and Exit Code
    is_success = (
        stats["total_processing_errors"] == 0
        and stats["total_lost"] == 0
        and stats["total_extra"] == 0
        and stats["total_mask_top_iou_fails"] == 0
        and stats["total_bbox_top_iou_fails"] == 0
    )

    if is_success:
        logger.info("--- Verification PASSED --- >")
        return 0
    else:
        logger.warning("--- Verification FAILED --- >")
        return 1


def print_single_sample_verification(result: VerificationResult, iou_top: float) -> None:
    """Print detailed verification result for a single sample."""
    sample_id = result.sample_id
    logger.info(f"===== Verification Result for Sample ID: {sample_id} =====")

    if result.processing_error:
        logger.error(f"PROCESSING ERROR: {result.processing_error}")
        return

    # Print matched pairs
    logger.info(f"MATCHED INSTANCES: {len(result.matched_pairs)}")
    if result.matched_pairs:
        for i, match in enumerate(result.matched_pairs):
            logger.info(f"  Match {i + 1}:")
            logger.info(f"    Class ID: {match['class_id']}")
            log_line = (
                f"    Original segment/mask: {match['original_segment_idx']}/"
                f"{match['original_mask_idx']}"
            )
            logger.info(log_line)
            logger.info(f"    YOLO instance: {match['yolo_instance_index']}")
            log_line = (
                f"    Mask IoU: {match['mask_iou']:.4f} (Threshold Passed: "
                f"{match['mask_threshold_passed']})"
            )
            logger.info(log_line)
            log_line = (
                f"    BBox IoU: {match['bbox_iou']:.4f} (Threshold Passed: "
                f"{match['bbox_threshold_passed']})"
            )
            logger.info(log_line)

    # Print lost instances
    logger.info(f"LOST INSTANCES (expected but not found in YOLO): {len(result.lost_instances)}")
    if result.lost_instances:
        for i, inst in enumerate(result.lost_instances):
            logger.info(f"  Lost {i + 1}:")
            logger.info(f"    Class ID: {inst.class_id}")
            logger.info(f"    Original segment/mask: {inst.segment_idx}/{inst.mask_idx}")
            logger.info(f"    BBox: {inst.bbox}")

    # Print extra instances
    logger.info(f"EXTRA INSTANCES (found in YOLO but not expected): {len(result.extra_instances)}")
    if result.extra_instances:
        for i, inst in enumerate(result.extra_instances):
            logger.info(f"  Extra {i + 1}:")
            logger.info(f"    Class ID: {inst.class_id}")
            logger.info(f"    BBox: {inst.bbox}")

    # Summary
    mask_fails = sum(1 for m in result.matched_pairs if not m["mask_threshold_passed"])
    bbox_fails = sum(1 for m in result.matched_pairs if not m["bbox_threshold_passed"])
    is_success = (
        len(result.lost_instances) == 0
        and len(result.extra_instances) == 0
        and mask_fails == 0
        and bbox_fails == 0
    )
    logger.info(f"High Threshold ({iou_top:.2f}) Failures: Mask={mask_fails}, BBox={bbox_fails}")
    logger.info(f"VERIFICATION RESULT: {'SUCCESS' if is_success else 'FAILURE'}")


def main():
    """Main execution function for the verification CLI."""
    parser = _setup_argparse()
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
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
    if args.sample_id:
        logger.info(f"Verifying specific sample ID: {args.sample_id}")
    else:
        logger.info(f"Sample Count: {args.sample_count}")
    logger.info(f"Using {args.num_proc} processes for dataset operations")
    logger.info(f"IoU Cutoff for Matching: {args.iou_cutoff}")
    logger.info(f"IoU Top Threshold for Quality: {args.iou_top}")

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

    # 2. Sample YOLO IDs or find specific ID
    sampled_ids = sample_yolo_ids(yolo_label_dir, args)
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
        args.iou_cutoff,
        args.iou_top,
    )

    # 5. Handle results
    # For single sample mode, provide detailed output
    if args.sample_id:
        if verification_results:
            print_single_sample_verification(verification_results[0], args.iou_top)
            res = verification_results[0]
            mask_fails = sum(1 for m in res.matched_pairs if not m["mask_threshold_passed"])
            bbox_fails = sum(1 for m in res.matched_pairs if not m["bbox_threshold_passed"])
            is_success = (
                not res.processing_error
                and len(res.lost_instances) == 0
                and len(res.extra_instances) == 0
                and mask_fails == 0
                and bbox_fails == 0
            )
            exit_code = 0 if is_success else 1
        else:
            logger.error(f"No verification result generated for sample ID: {args.sample_id}")
            exit_code = 1
    else:
        # For multi-sample mode, aggregate results
        stats, iou_data, expected_by_class, lost_by_class, extra_by_class = aggregate_results(
            verification_results
        )
        # 6. Print Summary Report and determine exit code
        exit_code = report_results(
            stats,
            iou_data,
            args,
            all_sample_count,
            class_names,
            expected_by_class,
            lost_by_class,
            extra_by_class,
        )

    logger.info("Verification process finished.")
    exit(exit_code)


if __name__ == "__main__":
    main()
