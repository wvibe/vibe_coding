import argparse
import json
import logging
import math
from typing import Dict, List, Optional

import datasets

# Use full import paths from src
from src.dataops.cov_segm.analyzer import (
    aggregate_phrase_stats,
    calculate_summary_stats,
)

logger = logging.getLogger(__name__)


def _format_percentiles(perc_dict: Optional[Dict[float, float]]) -> str:
    """Formats percentile dictionary for printing."""
    if perc_dict is None:
        return "N/A"
    if not perc_dict:
        return "(empty)"
    return ", ".join([f"{p * 100:.0f}%: {v:.1f}" for p, v in sorted(perc_dict.items())])


def _setup_argparse() -> argparse.ArgumentParser:
    """Creates and configures the argument parser for the analyzer CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate and summarize phrase statistics from the "
            "lab42/cov-segm-v3 dataset metadata or full data."
        )
    )
    # --- Mode Argument ---
    parser.add_argument(
        "--mode",
        type=str,
        default="count_only",
        choices=["count_only", "deep_stats"],
        help=(
            "Analysis mode: 'count_only' (default, fast, uses metadata), "
            "'deep_stats' (slow, loads masks, calculates geometry)."
        ),
    )
    # --- Dataset Arguments ---
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to process (e.g., 'train', 'validation').",
    )
    parser.add_argument(
        "--sample_slice",
        type=str,
        default="[:20]",
        help=(
            "Slice string to select samples (e.g., '[:100]', '[50:150]', ''). '' means all samples."
        ),
    )
    # --- Output Arguments ---
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help=("Optional path base to save the results (will append _agg.json and _summary.json)."),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top phrases (by appearance count) to print if not saving to file.",
    )
    # --- Calculation Arguments ---
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="+",  # Accept one or more float values
        default=[0.25, 0.5, 0.75],
        help="List of percentiles (0.0 to 1.0) to calculate for mask counts and geometry.",
    )
    parser.add_argument(
        "--skip_zero",
        action="store_true",
        help=(
            "If set, phrases with zero processed masks (deep_stats) or "
            "zero metadata counts (count_only) in a sample are ignored for that sample."
        ),
    )
    # --- Debug Arguments ---
    parser.add_argument(
        "--debug_phrase",
        type=str,
        default=None,
        help=(
            "If specified, print detailed mask counts/geometry when this exact phrase "
            "is encountered (requires --verbose)."
        ),
    )
    # --- Logging Arguments ---
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (including debug phrase details).",
    )
    return parser


def _handle_output(
    args: argparse.Namespace,
    aggregated_stats: Dict,
    summary_stats: List[Dict],
    total_processed: int,
):
    """Handles printing results to console or saving to files."""
    if args.output_file:
        # Save aggregated data
        agg_file_path = args.output_file + "_agg.json"
        logger.info(f"Saving aggregated statistics to: {agg_file_path}")
        try:
            with open(agg_file_path, "w") as f:
                json.dump(aggregated_stats, f, indent=4)
            logger.info("Successfully saved aggregated statistics.")
        except Exception as e:
            logger.error(
                f"Failed to save aggregated statistics to {agg_file_path}: {e}",
                exc_info=True,
            )

        summary_file_path = args.output_file + "_summary.json"
        logger.info(f"Saving summary statistics to: {summary_file_path}")
        try:
            with open(summary_file_path, "w") as f:
                json.dump(summary_stats, f, indent=4)
            logger.info("Successfully saved summary statistics.")
        except Exception as e:
            logger.error(
                f"Failed to save summary statistics to {summary_file_path}: {e}",
                exc_info=True,
            )
    else:
        # Print summary statistics in a more readable format
        print(
            f"\n--- Summary Statistics (Mode: {args.mode}, "
            f"Total Processed Samples: {total_processed}) ---"
        )
        print(f"--- Showing Top {min(args.top, len(summary_stats))} Phrases by Appearance ---")
        for i, stats in enumerate(summary_stats):
            if i >= args.top:
                break

            print(f'\n{i + 1}. Phrase: "{stats["phrase"]}"')
            appear_perc = stats["appearance_percentage"]
            print(
                f"   - Appeared in        : "
                f"{stats['appearance_count']} samples ({appear_perc:.1f}%)"
            )

            # --- Print Count Stats based on mode ---
            if args.mode == "count_only":
                if stats.get("avg_masks_per_image") is not None:
                    print(f"   - Avg Masks Per Image: {stats['avg_masks_per_image']:.2f}")
                    print(
                        f"   - Mask Cnt Perc.     : "
                        f"({_format_percentiles(stats.get('mask_percentiles'))})"
                    )
                else:
                    print("   - Mask Counts        : N/A (No data or calculation error)")
            elif args.mode == "deep_stats":
                # Print Visible/Full counts separately for deep_stats
                if stats.get("avg_visible_masks_per_image") is not None:
                    print(f"   - Avg Visible Masks  : {stats['avg_visible_masks_per_image']:.2f}")
                    print(
                        f"   - Visible Cnt Perc.  : "
                        f"({_format_percentiles(stats.get('visible_mask_percentiles'))})"
                    )
                else:
                    print("   - Avg Visible Masks  : N/A")

            # --- Print Deep Stats (Geometry) ---
            if args.mode == "deep_stats":
                # Check if geometry stats are available before trying to print
                if stats.get("avg_visible_mask_pixels") is not None:
                    # Combined Visible Geometry
                    vis_sat = stats.get("visible_mask_saturation")
                    vis_sat_str = (
                        f"{vis_sat * 100:.1f}%"
                        if isinstance(vis_sat, float) and not math.isnan(vis_sat)
                        else "N/A"
                    )
                    print(
                        f"   - Avg Visible Geom (Px, W, H, Sat%): "
                        f"{stats['avg_visible_mask_pixels']:.1f}, "
                        f"{stats['avg_visible_mask_width']:.1f}, "
                        f"{stats['avg_visible_mask_height']:.1f}, {vis_sat_str}"
                    )
                    print(
                        f"     - Vis Area Perc. : "
                        f"({_format_percentiles(stats.get('visible_mask_pixel_percentiles'))})"
                    )
                    print(
                        f"     - Vis Width Perc.: "
                        f"({_format_percentiles(stats.get('visible_mask_width_percentiles'))})"
                    )
                    print(
                        f"     - Vis Hgt Perc.  : "
                        f"({_format_percentiles(stats.get('visible_mask_height_percentiles'))})"
                    )
                else:
                    print("   - Avg Visible Geom   : N/A")

                if stats.get("avg_full_mask_pixels") is not None:
                    # Combined Full Geometry
                    full_sat = stats.get("full_mask_saturation")
                    full_sat_str = (
                        f"{full_sat * 100:.1f}%"
                        if isinstance(full_sat, float) and not math.isnan(full_sat)
                        else "N/A"
                    )
                    print(
                        f"   - Avg Full Geom    (Px, W, H, Sat%): "
                        f"{stats['avg_full_mask_pixels']:.1f}, "
                        f"{stats['avg_full_mask_width']:.1f}, "
                        f"{stats['avg_full_mask_height']:.1f}, {full_sat_str}"
                    )
                    print(
                        f"     - Full Area Perc. : "
                        f"({_format_percentiles(stats.get('full_mask_pixel_percentiles'))})"
                    )
                    print(
                        f"     - Full Width Perc.: "
                        f"({_format_percentiles(stats.get('full_mask_width_percentiles'))})"
                    )
                    print(
                        f"     - Full Hgt Perc.  : "
                        f"({_format_percentiles(stats.get('full_mask_height_percentiles'))})"
                    )
                else:
                    print("   - Avg Full Geom      : N/A")


def main():
    parser = _setup_argparse()
    args = parser.parse_args()

    # Validate percentiles
    valid_percentiles = []
    for p in args.percentiles:
        if 0.0 <= p <= 1.0:
            valid_percentiles.append(p)
        else:
            logging.warning(f"Ignoring invalid percentile value: {p}. Must be between 0.0 and 1.0.")
    if not valid_percentiles:
        # Use default if none are valid, instead of exiting
        logging.warning("No valid percentiles provided. Using defaults: [0.25, 0.5, 0.75]")
        valid_percentiles = [0.25, 0.5, 0.75]
    args.percentiles = valid_percentiles  # Use only valid ones or defaults

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    DATASET_NAME = "lab42/cov-segm-v3"
    SPLIT = args.split
    NUM_SAMPLES_STR = args.sample_slice if args.sample_slice else None

    try:
        logger.info(
            f"Loading dataset: {DATASET_NAME}, split: {SPLIT}, samples: {NUM_SAMPLES_STR or 'all'}"
        )
        # Stream only if using count_only mode for efficiency
        should_stream = args.mode == "count_only" and not NUM_SAMPLES_STR
        if should_stream:
            logger.info("Streaming dataset as mode is 'count_only' and no slice is specified.")
        dset = datasets.load_dataset(
            DATASET_NAME,
            split=f"{SPLIT}{NUM_SAMPLES_STR}" if NUM_SAMPLES_STR else SPLIT,
            streaming=should_stream,
            # trust_remote_code=True # Might be needed
        )
        dset_iterable = dset
        logger.info(f"Prepared dataset iterable for split '{SPLIT}'.")

        # --- Aggregation Step --- (Using imported function)
        logger.info(f"Starting phrase statistics aggregation (mode: {args.mode})...")
        aggregated_stats, total_processed = aggregate_phrase_stats(
            dset_iterable,
            mode=args.mode,
            verbose=args.verbose,
            debug_phrase=args.debug_phrase,
            skip_zero_masks=args.skip_zero,
        )
        logger.info("Aggregation finished.")

        # --- Summary Calculation Step --- (Using imported function)
        logger.info(f"Calculating summary statistics using percentiles: {args.percentiles}...")
        summary_stats = calculate_summary_stats(
            aggregated_stats,
            total_processed_samples=total_processed,
            mode=args.mode,
            percentiles=args.percentiles,
        )
        logger.info("Summary statistics calculation finished.")

        # --- Output ---
        _handle_output(args, aggregated_stats, summary_stats, total_processed)

    except ImportError as ie:
        logger.error(f"Import error: {ie}. Make sure 'datasets' and 'numpy' are installed.")
    except Exception as e:
        logger.error(f"An error occurred during script execution: {e}", exc_info=True)


if __name__ == "__main__":
    main()
