"""Functions for analyzing and aggregating phrase statistics from cov-segm dataset metadata."""

import argparse
import json
import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple, Union

# Third-party imports
import datasets
from tqdm.auto import tqdm

# Project imports
from vibelab.dataops.cov_segm.datamodel import ConversationItem
from vibelab.dataops.cov_segm.loader import parse_conversations
from vibelab.utils.common.stats import format_statistics_table

logger = logging.getLogger(__name__)

# Type alias for the aggregated stats dictionary structure per phrase
# Stores lists of raw counts per appearance
PhraseStatsDict = Dict[str, Dict[str, Union[int, List[int]]]]


# --- Helper to define the structure of per-phrase aggregation results ----
def _phrase_stats_factory() -> Dict[str, Union[int, List[int]]]:
    """Factory to create the default dictionary structure for phrase aggregation."""
    return {
        "appearance_count": 0,
        "visible_mask_counts": [],  # List of visible mask counts per appearance
        "full_mask_counts": [],  # List of full mask counts per appearance
        "alternative_phrase_counts": [],  # List of counts of alt phrases per appearance
    }


# --- Aggregation Function ---


def _aggregate_stats_from_metadata(
    dataset_iterable: Iterable[Dict[str, Any]],
    skip_zero_masks: bool,
) -> Tuple[Dict[str, PhraseStatsDict], int, int, int]:
    """
    Aggregates phrase and overall stats using only dataset metadata (fast).

    Args:
        dataset_iterable: An iterable yielding raw dataset rows (dictionaries).
        skip_zero_masks: If True, phrases associated with zero masks in a sample
                         will not contribute to the stats for that sample.

    Returns:
        A tuple containing:
        - phrase_agg_stats: Dictionary mapping primary phrase text to its aggregated stats.
        - total_samples_processed: Total samples iterated over.
        - total_conversations: Total conversation items encountered across processed samples.
        - total_valid_conversations: Total conversation items with > 0 visible or full masks.
    """
    phrase_agg_stats: Dict[str, PhraseStatsDict] = defaultdict(_phrase_stats_factory)
    total_samples_processed = 0
    total_skipped_samples = 0
    total_conversations = 0
    total_valid_conversations = 0
    total_samples = None
    try:
        total_samples = len(dataset_iterable)  # type: ignore
    except TypeError:
        pass  # Iterable might not have __len__ (like streaming dataset)

    progress_bar_desc = "Aggregating metadata stats"
    iterable_with_progress = tqdm(
        dataset_iterable, desc=progress_bar_desc, total=total_samples, unit="sample"
    )

    for i, row in enumerate(iterable_with_progress):
        total_samples_processed += 1
        sample_id = row.get("id", f"unknown_index_{i}")

        try:
            conversations_json_str = row.get("conversations")
            if not conversations_json_str or not isinstance(conversations_json_str, str):
                logger.debug(f"Skipping sample '{sample_id}': Missing/invalid 'conversations'.")
                total_skipped_samples += 1
                continue

            parsed_conversations: List[ConversationItem] = parse_conversations(
                conversations_json_str
            )
            if not parsed_conversations:
                logger.debug(f"Skipping sample '{sample_id}': Parsed conversations list is empty.")
                # Still counts as processed, but contributes 0 conversations
                continue

            total_conversations += len(parsed_conversations)
            phrases_processed_in_sample = set()

            for item in parsed_conversations:
                # Calculate validity based on masks *first*
                num_visible = len(getattr(item, "instance_masks", []) or [])
                num_full = len(getattr(item, "instance_full_masks", []) or [])
                is_valid_conversation = num_visible > 0 or num_full > 0
                if is_valid_conversation:
                    total_valid_conversations += 1

                # Now check if there are phrases to process for stats
                phrases = item.phrases
                if not phrases:
                    continue

                # Use first phrase as the primary key
                phrase_text = phrases[0].text
                if not phrase_text or phrase_text in phrases_processed_in_sample:
                    continue  # Skip if no text or already processed in this sample

                phrases_processed_in_sample.add(phrase_text)

                # Apply skip_zero_masks logic *before* updating stats
                if skip_zero_masks and not is_valid_conversation:
                    logger.debug(f"Skipping '{phrase_text}' in {sample_id} (zero masks).")
                    continue  # Skip this phrase for this sample

                # Update stats for this phrase (first appearance in sample)
                agg_entry = phrase_agg_stats[phrase_text]
                agg_entry["appearance_count"] += 1
                agg_entry["visible_mask_counts"].append(num_visible)
                agg_entry["full_mask_counts"].append(num_full)

                alt_phrase_count = max(0, len(phrases) - 1)
                agg_entry["alternative_phrase_counts"].append(alt_phrase_count)

        except (ValueError, TypeError, json.JSONDecodeError) as e:
            total_skipped_samples += 1
            logger.debug(f"Skipping sample '{sample_id}' due to data error: {e}", exc_info=False)
            continue
        except Exception as e:
            total_skipped_samples += 1
            logger.error(
                f"Skipping sample '{sample_id}' due to unexpected error: {e}", exc_info=True
            )
            continue

    successfully_processed_count = total_samples_processed - total_skipped_samples
    logger.info(
        f"Metadata Aggregation complete. Total Samples Iterated: {total_samples_processed}, "
        f"Skipped: {total_skipped_samples} => Successfully Parsed: {successfully_processed_count}"
    )
    logger.info(f"Found {len(phrase_agg_stats)} unique primary phrases.")

    return (
        dict(phrase_agg_stats),
        successfully_processed_count,
        total_conversations,
        total_valid_conversations,
    )


# --- Command Line Interface Setup ---


def _setup_argparse() -> argparse.ArgumentParser:
    """Creates and configures the argument parser for the analyzer CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate and summarize phrase statistics from the lab42/cov-segm-v3 dataset metadata."
        )
    )
    # --- Dataset Arguments ---
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process (e.g., 'train', 'validation').",
    )
    parser.add_argument(
        "--sample-slice",
        type=str,
        default="[:100]",
        help=(
            "Slice string to select samples (e.g., '[:100]', '[50:150]', ''). '' means all samples."
        ),
    )
    # --- Output Arguments ---
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top phrases (by appearance count) to include in the statistics tables.",
    )
    parser.add_argument(
        "--metrics-format",
        type=str,
        default="{key:<18} {count:>8} {mean:>7.1f} {p25:>6.0f} {p50:>6.0f} {p75:>6.0f} {max:>6}",
        help="Format string for the statistics table output (follows f-string formatting).",
    )
    # --- Calculation Arguments ---
    parser.add_argument(
        "--skip-zero",
        action="store_true",
        help=(
            "If set, phrases with zero metadata mask counts in a sample are ignored for that sample."
        ),
    )
    return parser


# --- Output Formatting Function ---


def _print_phrase_details(
    top_phrases_data_with_pct: List[Tuple[str, Dict[str, Any]]],
    metrics_format: str,
    num_phrases_to_show: int,
) -> None:
    """
    Prints the detailed statistics for the top N phrases in a combined format.

    Args:
        top_phrases_data_with_pct: List of (phrase, stats_dict) tuples, sorted by appearance.
                                   The stats_dict must include 'appearance_count' and
                                   'appearance_percentage'.
        metrics_format: The f-string format for the statistics table (e.g., mean, p50).
        num_phrases_to_show: The number of phrases being shown (for the header).
    """
    if not top_phrases_data_with_pct:
        print("\n--- No Phrase Details to Display ---")
        return

    print(f"\n--- Top {num_phrases_to_show} Phrase Details ---")

    for rank, (phrase, data) in enumerate(top_phrases_data_with_pct, start=1):
        count = data["appearance_count"]
        percentage = data["appearance_percentage"]

        print(f"\n{rank}. {phrase}")
        print(f"   Appearances: {count}, Pct. of Total Processed: {percentage:.2f}%")

        # Prepare data for the stats table for this specific phrase
        single_phrase_stats_dict = {
            "visible_mask": data["visible_mask_counts"],
            "full_mask": data["full_mask_counts"],
            "alternative_phrase": data["alternative_phrase_counts"],
        }

        # Generate and print the table
        try:
            table_lines = format_statistics_table(single_phrase_stats_dict, metrics_format)
            # Indent the table lines for better readability under the phrase header
            for line in table_lines:
                print(f"   {line}")
        except Exception as e:
            logger.error(f"Could not format statistics table for phrase '{phrase}': {e}")
            print("   Error generating statistics table for this phrase.")


# --- Main Execution Logic ---


def main():
    parser = _setup_argparse()
    args = parser.parse_args()

    # Configure logging - default to INFO, can be changed by external config
    logging.basicConfig(
        level=logging.INFO,  # Default level
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    DATASET_NAME = "lab42/cov-segm-v3"
    SPLIT = args.split
    NUM_SAMPLES_STR = args.sample_slice if args.sample_slice else None

    try:
        logger.info(
            f"Loading dataset: {DATASET_NAME}, split: {SPLIT}, samples: {NUM_SAMPLES_STR or 'all'}"
        )
        # Stream only if processing all samples for efficiency
        should_stream = not NUM_SAMPLES_STR
        if should_stream:
            logger.info("Streaming dataset as no sample slice is specified.")
        dset = datasets.load_dataset(
            DATASET_NAME,
            split=f"{SPLIT}{NUM_SAMPLES_STR}" if NUM_SAMPLES_STR else SPLIT,
            streaming=should_stream,
            # trust_remote_code=True # Might be needed
        )
        dset_iterable = dset
        logger.info(f"Prepared dataset iterable for split '{SPLIT}'.")

        # --- Aggregation Step ---
        logger.info("Starting phrase statistics aggregation from metadata...")
        (
            phrase_agg_stats,
            total_processed,
            total_conversations,
            total_valid_conversations,
        ) = _aggregate_stats_from_metadata(
            dset_iterable,
            skip_zero_masks=args.skip_zero,
        )
        logger.info("Aggregation finished.")

        if total_processed == 0:
            logger.warning("No samples were successfully processed. Exiting.")
            return

        # --- Prepare Data for Output ---

        # 1. Calculate Overall Image Metrics
        avg_conv_per_sample = total_conversations / total_processed if total_processed > 0 else 0
        avg_valid_conv_per_sample = (
            total_valid_conversations / total_processed if total_processed > 0 else 0
        )

        # 2. Sort phrases by appearance count
        sorted_phrases = sorted(
            phrase_agg_stats.items(), key=lambda item: item[1]["appearance_count"], reverse=True
        )

        # 3. Determine how many top phrases to show
        num_phrases_to_show = min(args.top, len(sorted_phrases))

        # 4. Prepare data for the top phrases, including calculated percentage
        top_phrases_data_with_pct: List[Tuple[str, Dict[str, Any]]] = []
        for phrase, data in sorted_phrases[:num_phrases_to_show]:
            count = data["appearance_count"]
            percentage = (count / total_processed) * 100 if total_processed > 0 else 0
            updated_data = data.copy()  # Avoid modifying the original dict
            updated_data["appearance_percentage"] = percentage
            top_phrases_data_with_pct.append((phrase, updated_data))

        # --- Print Output ---
        print("\n--- Overall Statistics ---")
        print(f"Total Samples Successfully Processed: {total_processed}")
        print(f"Total Conversation Items Found     : {total_conversations}")
        print(f"Total Valid Conversation Items (>0 masks): {total_valid_conversations}")
        print(f"Average Conversations per Sample   : {avg_conv_per_sample:.2f}")
        print(f"Average Valid Conversations per Sample: {avg_valid_conv_per_sample:.2f}")

        # Print detailed phrase statistics using the new function
        _print_phrase_details(
            top_phrases_data_with_pct=top_phrases_data_with_pct,
            metrics_format=args.metrics_format,
            num_phrases_to_show=num_phrases_to_show,
        )

    except ImportError as ie:
        logger.error(
            f"Import error: {ie}. Make sure 'datasets', 'tqdm', and 'numpy' are installed."
        )
    except Exception as e:
        logger.error(f"An error occurred during script execution: {e}", exc_info=True)


if __name__ == "__main__":
    main()
