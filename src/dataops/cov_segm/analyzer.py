"""Functions for analyzing and aggregating statistics from the cov-segm dataset."""

import argparse
import json
import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np  # Import numpy

# Import the parser function and Pydantic models
from .loader import ConversationItem, parse_conversations

logger = logging.getLogger(__name__)

# Type alias for the aggregated stats dictionary
AggregatedStatsDict = Dict[str, Dict[str, Union[int, List[int], List[str]]]]
# Type alias for the summary stats dictionary
SummaryStat = Dict[str, Union[str, int, float, Dict[float, float]]]


def aggregate_phrase_stats(
    dataset_iterable: Iterable[Dict[str, Any]],
    verbose: bool = False,  # Add verbose flag
    debug_phrase: Optional[str] = None,  # Add debug_phrase parameter
    skip_zero_masks: bool = False,  # Add skip_zero_masks parameter
) -> Tuple[AggregatedStatsDict, int]:
    """
    Aggregates statistics about phrases by parsing 'conversations' JSON metadata.

    Iterates through raw dataset rows, parses the 'conversations' JSON string,
    and aggregates counts based on unique phrases found, including counts for
    both visible (instance_masks) and full (instance_full_masks) masks.
    This avoids loading actual image/mask data for faster analysis.

    Args:
        dataset_iterable: An iterable yielding raw dataset rows (dictionaries).
        verbose: If True, logs skipped samples and progress.
        debug_phrase: If set, logs detailed mask counts for items matching this phrase (requires verbose=True).
        skip_zero_masks: If True, phrases associated with zero visible AND zero full masks
                         in a sample will not contribute to the statistics for that sample.

    Returns:
        A tuple containing:
        - The aggregated statistics dictionary.
        - The total number of samples successfully processed.
    """
    phrase_agg_stats = defaultdict(
        lambda: {
            "appearance_count": 0,
            "sample_ids": [],
            "total_visible_mask_count": 0,
            "visible_mask_counts_per_image": [],
            "total_full_mask_count": 0,
            "full_mask_counts_per_image": [],
        }
    )

    processed_count = 0
    skipped_count = 0
    # from tqdm import tqdm
    # for row in tqdm(dataset_iterable, desc="Aggregating phrase stats"):
    for row in dataset_iterable:
        processed_count += 1
        sample_id = row.get("id", f"unknown_index_{processed_count - 1}")
        conversations_json_str = row.get("conversations")

        if not conversations_json_str:
            if verbose:
                logger.debug(f"Skipping sample '{sample_id}': Missing 'conversations' key.")
            skipped_count += 1
            continue
        try:
            # Parse the JSON string into Pydantic models
            parsed_conversations: List[ConversationItem] = parse_conversations(
                conversations_json_str
            )

            if not parsed_conversations:
                if verbose:
                    logger.debug(
                        f"Skipping sample '{sample_id}': Parsed conversations list is empty."
                    )
                # Still count as processed, just yielded no data
                continue

            # Store mask counts for the *first* appearance of each unique phrase in this sample
            phrase_data_for_sample: Dict[str, Dict[str, int]] = {}

            for item in parsed_conversations:
                phrases = item.phrases
                if not phrases:
                    continue

                phrase_text = phrases[0].text
                if not phrase_text:
                    continue

                # Only store data for the first time we see a phrase in this sample
                if phrase_text not in phrase_data_for_sample:
                    num_visible = len(getattr(item, "instance_masks", []) or [])
                    num_full = len(getattr(item, "instance_full_masks", []) or [])
                    phrase_data_for_sample[phrase_text] = {
                        "visible": num_visible,
                        "full": num_full,
                    }

                    # --- Debug Logic (Applied on first encounter) ---
                    if verbose and debug_phrase and phrase_text == debug_phrase:
                        logger.info(
                            f"[DEBUG PHRASE] Sample: {sample_id}, "
                            f'Phrase: "{phrase_text}" (First Encounter), '
                            f"Visible Masks: {num_visible}, "
                            f"Full Masks: {num_full}"
                        )
                    # --- End Debug Logic ---

            # Now, aggregate based on the collected first-encounter data
            for phrase_text, counts in phrase_data_for_sample.items():
                num_visible = counts["visible"]
                num_full = counts["full"]

                # Apply skip logic
                if skip_zero_masks and num_visible == 0 and num_full == 0:
                    if verbose:
                        logger.debug(
                            f"Skipping aggregation for phrase '{phrase_text}' in sample {sample_id} due to zero masks."
                        )
                    continue

                # Aggregate statistics
                stats = phrase_agg_stats[phrase_text]
                stats["appearance_count"] += 1
                stats["sample_ids"].append(sample_id)
                stats["visible_mask_counts_per_image"].append(num_visible)
                stats["total_visible_mask_count"] += num_visible
                stats["full_mask_counts_per_image"].append(num_full)
                stats["total_full_mask_count"] += num_full

        except (ValueError, TypeError, json.JSONDecodeError) as e:
            # Catches errors from parse_conversations (invalid JSON, Pydantic validation)
            skipped_count += 1
            if verbose:
                logger.warning(
                    f"Skipping sample '{sample_id}' due to parsing error: {e}",
                    exc_info=False,
                )
            continue
        except Exception as e:
            # Catch any other unexpected errors during processing
            skipped_count += 1
            if verbose:
                logger.error(
                    f"Skipping sample '{sample_id}' due to unexpected error: {e}",
                    exc_info=True,
                )
            continue

    successfully_processed_count = processed_count - skipped_count
    if verbose:
        logger.info(
            f"Aggregation complete. Processed rows: {processed_count}, "
            f"Skipped rows: {skipped_count} => Successfully processed: {successfully_processed_count}"
        )
        logger.info(f"Found {len(phrase_agg_stats)} unique phrases.")

    return dict(phrase_agg_stats), successfully_processed_count


def calculate_summary_stats(
    aggregated_stats: AggregatedStatsDict,
    total_processed_samples: int,
    percentiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
) -> List[SummaryStat]:
    """
    Calculates summary statistics from the aggregated phrase data.

    Args:
        aggregated_stats: The dictionary output from aggregate_phrase_stats.
        total_processed_samples: The total number of samples successfully processed.
        percentiles: A list of percentiles to calculate (e.g., [0.1, 0.5, 0.9]).

    Returns:
        A list of dictionaries, each containing summary statistics for a phrase,
        sorted by appearance_count in descending order.
        Stats include:
        - phrase: The phrase text.
        - appearance_count: Number of samples the phrase appeared in.
        - appearance_percentage: Percentage of processed samples the phrase appeared in.
        - avg_visible_masks_per_image: Average number of visible masks per appearance.
        - avg_full_masks_per_image: Average number of full masks per appearance.
        - visible_mask_percentiles: Dict mapping percentiles to visible mask counts.
        - full_mask_percentiles: Dict mapping percentiles to full mask counts.
    """
    summary_stats_list: List[SummaryStat] = []  # Ensure list type

    for phrase, stats in aggregated_stats.items():
        visible_counts = stats.get("visible_mask_counts_per_image", [])
        full_counts = stats.get("full_mask_counts_per_image", [])

        # Calculate averages (handle empty lists)
        avg_visible = np.mean(visible_counts) if visible_counts else 0.0
        avg_full = np.mean(full_counts) if full_counts else 0.0

        # Calculate percentiles (handle empty lists)
        visible_perc = {}
        if visible_counts:
            try:
                # Ensure percentiles are in [0, 100]
                percentile_values = np.percentile(visible_counts, [p * 100 for p in percentiles])
                visible_perc = {p: float(v) for p, v in zip(percentiles, percentile_values)}
            except IndexError as e:
                logger.warning(
                    f"Could not calculate visible percentiles for phrase '{phrase}' (IndexError: {e})"
                )
            except Exception as e:
                logger.warning(
                    f"Could not calculate visible percentiles for phrase '{phrase}' (Error: {e})"
                )

        full_perc = {}
        if full_counts:
            try:
                # Ensure percentiles are in [0, 100]
                percentile_values = np.percentile(full_counts, [p * 100 for p in percentiles])
                full_perc = {p: float(v) for p, v in zip(percentiles, percentile_values)}
            except IndexError as e:
                logger.warning(
                    f"Could not calculate full percentiles for phrase '{phrase}' (IndexError: {e})"
                )
            except Exception as e:
                logger.warning(
                    f"Could not calculate full percentiles for phrase '{phrase}' (Error: {e})"
                )

        appearance_count = stats.get("appearance_count", 0)
        appearance_percentage = (
            (appearance_count / total_processed_samples) * 100
            if total_processed_samples > 0
            else 0.0
        )

        summary: SummaryStat = {
            "phrase": phrase,
            "appearance_count": appearance_count,
            "appearance_percentage": float(appearance_percentage),
            "avg_visible_masks_per_image": float(avg_visible),
            "avg_full_masks_per_image": float(avg_full),
            "visible_mask_percentiles": visible_perc,
            "full_mask_percentiles": full_perc,
        }
        summary_stats_list.append(summary)

    # Sort by appearance_count descending
    summary_stats_list.sort(key=lambda x: x["appearance_count"], reverse=True)

    return summary_stats_list


# Example Usage
if __name__ == "__main__":
    import datasets
    # import pprint # Already imported above

    parser = argparse.ArgumentParser(
        description="Aggregate and summarize phrase statistics from the lab42/cov-segm-v3 dataset metadata."
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
        help="Slice string to select samples (e.g., '[:100]', '[50:150]', ''). '' means all samples.",
    )
    # --- Output Arguments ---
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path base to save the results (will append _agg.json and _summary.json).",
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
        help="List of percentiles (0.0 to 1.0) to calculate for mask counts.",
    )
    parser.add_argument(
        "--skip_zero",
        action="store_true",
        help="If set, phrases with zero visible AND zero full masks in a sample are ignored for that sample.",
    )
    # --- Debug Arguments ---
    parser.add_argument(
        "--debug_phrase",
        type=str,
        default=None,
        help="If specified, print detailed mask counts when this exact phrase is encountered (requires --verbose).",
    )
    # --- Logging Arguments ---
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (including debug phrase details).",
    )

    args = parser.parse_args()

    # Validate percentiles
    valid_percentiles = []
    for p in args.percentiles:
        if 0.0 <= p <= 1.0:
            valid_percentiles.append(p)
        else:
            logging.warning(f"Ignoring invalid percentile value: {p}. Must be between 0.0 and 1.0.")
    if not valid_percentiles:
        logging.error("No valid percentiles provided. Exiting.")
        exit(1)
    args.percentiles = valid_percentiles  # Use only valid ones

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
        dset = datasets.load_dataset(
            DATASET_NAME,
            split=f"{SPLIT}{NUM_SAMPLES_STR}" if NUM_SAMPLES_STR else SPLIT,
            # trust_remote_code=True # Might be needed
        )
        dset_iterable = dset
        logger.info(f"Loaded dataset subset for split '{SPLIT}'.")

        # --- Aggregation Step ---
        logger.info("Starting phrase statistics aggregation (using metadata only)...")
        aggregated_stats, total_processed = aggregate_phrase_stats(
            dset_iterable,
            verbose=args.verbose,
            debug_phrase=args.debug_phrase,
            skip_zero_masks=args.skip_zero,  # Pass skip_zero argument
        )
        logger.info("Aggregation finished.")

        # --- Summary Calculation Step ---
        logger.info(f"Calculating summary statistics using percentiles: {args.percentiles}...")
        summary_stats = calculate_summary_stats(
            aggregated_stats, total_processed_samples=total_processed, percentiles=args.percentiles
        )
        logger.info("Summary statistics calculation finished.")

        # --- Output ---
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
                    f"Failed to save aggregated statistics to {agg_file_path}: {e}", exc_info=True
                )

            summary_file_path = args.output_file + "_summary.json"
            logger.info(f"Saving summary statistics to: {summary_file_path}")
            try:
                with open(summary_file_path, "w") as f:
                    json.dump(summary_stats, f, indent=4)
                logger.info("Successfully saved summary statistics.")
            except Exception as e:
                logger.error(
                    f"Failed to save summary statistics to {summary_file_path}: {e}", exc_info=True
                )
        else:
            # Print summary statistics in a more readable format
            print(f"\n--- Summary Statistics (Total Processed Samples: {total_processed}) ---")
            print(f"--- Showing Top {min(args.top, len(summary_stats))} Phrases by Appearance ---")
            for i, stats in enumerate(summary_stats):
                if i >= args.top:
                    break

                print(f'\n{i + 1}. Phrase: "{stats["phrase"]}"')
                # Format percentage nicely
                appear_perc = stats["appearance_percentage"]
                print(
                    f"   - Appeared in        : {stats['appearance_count']} samples ({appear_perc:.1f}%)"
                )
                print(f"   - Avg Visible Masks  : {stats['avg_visible_masks_per_image']:.2f}")
                print(f"   - Avg Full Masks     : {stats['avg_full_masks_per_image']:.2f}")

                vis_perc_str = ", ".join(
                    [
                        f"{p * 100:.0f}%: {v:.1f}"
                        for p, v in sorted(stats["visible_mask_percentiles"].items())
                    ]
                )
                full_perc_str = ", ".join(
                    [
                        f"{p * 100:.0f}%: {v:.1f}"
                        for p, v in sorted(stats["full_mask_percentiles"].items())
                    ]
                )

                print(f"   - Visible Percentiles: ({vis_perc_str})")
                print(f"   - Full Percentiles   : ({full_perc_str})")

    except ImportError as ie:
        logger.error(f"Import error: {ie}. Make sure 'datasets' and 'numpy' are installed.")
    except Exception as e:
        logger.error(f"An error occurred during script execution: {e}", exc_info=True)
