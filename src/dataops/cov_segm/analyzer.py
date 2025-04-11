"""Functions for analyzing and aggregating statistics from the cov-segm dataset."""

import json
import logging
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

from src.dataops.cov_segm.datamodel import (
    ConversationItem,
    ProcessedCovSegmSample,
    ProcessedMask,  # Import ProcessedMask for type hints if needed later
)

# Use full import paths from src
from src.dataops.cov_segm.loader import load_sample, parse_conversations

logger = logging.getLogger(__name__)

# Type alias for the aggregated stats dictionary
AggregatedStatsDict = Dict[str, Dict[str, Union[int, List[int], List[str]]]]
# Type alias for the summary stats dictionary
SummaryStat = Dict[str, Union[str, int, float, Dict[float, float], None]]


# --- Helper Functions for Sample Processing ---


def _extract_metadata_counts_for_sample(
    parsed_conversations: List[ConversationItem],
    sample_id: str,
    verbose: bool,
    debug_phrase: Optional[str],
) -> Dict[str, Dict[str, int]]:
    """Extracts visible mask counts from metadata for unique phrases in a sample."""
    phrase_data = {}
    for item in parsed_conversations:
        phrases = item.phrases
        if not phrases:
            continue
        phrase_text = phrases[0].text
        if not phrase_text:
            continue

        if phrase_text not in phrase_data:
            num_visible = len(getattr(item, "instance_masks", []) or [])
            num_full = len(getattr(item, "instance_full_masks", []) or [])

            if num_visible != num_full:
                logger.warning(
                    f"Sample '{sample_id}', Phrase '{phrase_text}': "
                    f"Visible mask count ({num_visible}) differs from full mask count ({num_full}) "
                    "in metadata. Using visible count for 'count_only' mode."
                )

            phrase_data[phrase_text] = {"count": num_visible}

            if verbose and debug_phrase and phrase_text == debug_phrase:
                logger.info(
                    f"[DEBUG PHRASE - Metadata] Sample: {sample_id}, "
                    f'Phrase: "{phrase_text}" (First Encounter), '
                    f"Mask Count: {num_visible}"
                )
    return phrase_data


def _extract_deep_stats_for_sample(
    processed_sample: ProcessedCovSegmSample,
    sample_id: str,
    verbose: bool,
    debug_phrase: Optional[str],
) -> Dict[str, Dict[str, Union[int, List[int], List[ProcessedMask]]]]:
    """Extracts counts and geometry stats from a processed sample for unique phrases."""
    phrase_data = {}

    # Helper to safely get geometry data from ProcessedMask, defaulting to 0
    def get_geo(mask: ProcessedMask, key: str) -> int:
        return mask.get(key, 0) or 0

    for item in processed_sample["processed_conversations"]:
        phrases_list = item.get("phrases", [])
        if not phrases_list:
            continue
        phrase_text = phrases_list[0].get("text")
        if not phrase_text:
            continue

        if phrase_text not in phrase_data:
            vis_masks = item.get("processed_instance_masks", [])
            full_masks = item.get("processed_full_masks", [])

            phrase_data[phrase_text] = {
                "visible_count": len(vis_masks),
                "full_count": len(full_masks),
                "vis_areas": [get_geo(m, "pixel_area") for m in vis_masks],
                "vis_widths": [get_geo(m, "width") for m in vis_masks],
                "vis_heights": [get_geo(m, "height") for m in vis_masks],
                "full_areas": [get_geo(m, "pixel_area") for m in full_masks],
                "full_widths": [get_geo(m, "width") for m in full_masks],
                "full_heights": [get_geo(m, "height") for m in full_masks],
                # Store masks for debug logging if needed (use list comp for typing)
                "_vis_masks_debug": [m for m in vis_masks] if verbose and debug_phrase else None,
                "_full_masks_debug": [m for m in full_masks] if verbose and debug_phrase else None,
            }

            if verbose and debug_phrase and phrase_text == debug_phrase:
                logger.info(
                    f"[DEBUG PHRASE - Deep Stats] Sample: {sample_id}, "
                    f'Phrase: "{phrase_text}" (First Encounter)'
                )
                # Use stored masks for logging
                debug_vis = phrase_data[phrase_text]["_vis_masks_debug"]
                debug_full = phrase_data[phrase_text]["_full_masks_debug"]
                if debug_vis:
                    logger.info(f"  Visible Masks ({len(debug_vis)}):")
                    for i, m in enumerate(debug_vis):
                        logger.info(
                            f"    - Mask {i}: Area={m.get('pixel_area')}, "
                            f"W={m.get('width')}, H={m.get('height')}, Src={m.get('source')}"
                        )
                if debug_full:
                    logger.info(f"  Full Masks ({len(debug_full)}):")
                    for i, m in enumerate(debug_full):
                        logger.info(
                            f"    - Mask {i}: Area={m.get('pixel_area')}, "
                            f"W={m.get('width')}, H={m.get('height')}, Src={m.get('source')}"
                        )
    return phrase_data


# --- Main Aggregation Function ---


def aggregate_phrase_stats(
    dataset_iterable: Iterable[Dict[str, Any]],
    mode: str = "count_only",
    verbose: bool = False,
    debug_phrase: Optional[str] = None,
    skip_zero_masks: bool = False,
) -> Tuple[AggregatedStatsDict, int]:
    """
    Aggregates statistics about phrases from dataset rows.

    In 'count_only' mode, parses 'conversations' JSON metadata for speed, using visible mask counts.
    In 'deep_stats' mode, loads full samples using `load_sample` to include mask geometry.

    Args:
        dataset_iterable: An iterable yielding raw dataset rows (dictionaries).
        mode: Analysis mode ('count_only' or 'deep_stats').
        verbose: If True, logs skipped samples and progress.
        debug_phrase: If set, logs detailed info for items matching this phrase.
        skip_zero_masks: If True, phrases associated with zero masks in a sample
                         (visible count for 'count_only', both for 'deep_stats')
                         will not contribute to the stats for that sample.

    Returns:
        A tuple containing:
        - The aggregated statistics dictionary.
        - The total number of samples successfully processed.
    """
    if mode not in ["count_only", "deep_stats"]:
        raise ValueError("Invalid mode. Choose 'count_only' or 'deep_stats'.")

    # Factory defines the structure, supporting fields for both modes
    def stats_factory():
        return {
            "appearance_count": 0,
            "sample_ids": [],
            # Unified Counts (populated ONLY in count_only mode)
            "total_mask_count": 0,
            "mask_counts_per_image": [],
            # Counts (populated ONLY in deep_stats mode)
            "total_visible_mask_count": 0,
            "visible_mask_counts_per_image": [],
            "total_full_mask_count": 0,
            "full_mask_counts_per_image": [],
            # Deep Stats Geometry (populated ONLY in deep_stats mode)
            "visible_mask_pixel_areas": [],
            "visible_mask_widths": [],
            "visible_mask_heights": [],
            "full_mask_pixel_areas": [],
            "full_mask_widths": [],
            "full_mask_heights": [],
        }

    phrase_agg_stats = defaultdict(stats_factory)
    processed_count = 0
    skipped_count = 0

    logger.info(f"Starting aggregation in '{mode}' mode...")

    # Determine total for progress bar if possible
    total_samples = None
    if hasattr(dataset_iterable, "__len__"):
        try:
            total_samples = len(dataset_iterable)
            logger.info(f"Total samples to process: {total_samples}")
        except TypeError:
            logger.info("Processing stream, total samples unknown.")
            total_samples = None

    progress_bar_desc = f"Aggregating '{mode}' stats"
    iterable_with_progress = tqdm(
        dataset_iterable, desc=progress_bar_desc, total=total_samples, unit="sample"
    )

    for row in iterable_with_progress:
        processed_count += 1
        sample_id = row.get("id", f"unknown_index_{processed_count - 1}")
        phrase_data_this_sample: Optional[Dict] = None

        try:
            # --- Process Sample based on Mode ---
            if mode == "count_only":
                conversations_json_str = row.get("conversations")
                if not conversations_json_str:
                    if verbose:
                        logger.debug(f"Skipping sample '{sample_id}': Missing 'conversations' key.")
                    skipped_count += 1
                    continue

                parsed_conversations: List[ConversationItem] = parse_conversations(
                    conversations_json_str
                )
                if not parsed_conversations:
                    if verbose:
                        logger.debug(
                            f"Skipping sample '{sample_id}': Parsed conversations list is empty."
                        )
                    continue

                phrase_data_this_sample = _extract_metadata_counts_for_sample(
                    parsed_conversations, sample_id, verbose, debug_phrase
                )

            elif mode == "deep_stats":
                processed_sample: Optional[ProcessedCovSegmSample] = load_sample(row)
                if not processed_sample:
                    logger.warning(f"Skipping sample '{sample_id}': Failed to load.")
                    skipped_count += 1
                    continue

                phrase_data_this_sample = _extract_deep_stats_for_sample(
                    processed_sample, sample_id, verbose, debug_phrase
                )

            # --- Update Aggregated Statistics for this Sample ---
            if phrase_data_this_sample is None:  # Should not happen if mode is valid
                logger.warning(
                    f"Internal logic error: phrase_data_this_sample is None for sample {sample_id}"
                )
                continue

            for phrase_text, data in phrase_data_this_sample.items():
                # Perform skip check BEFORE accessing defaultdict
                should_skip = False
                if mode == "count_only":
                    num_masks = data["count"]
                    if skip_zero_masks and num_masks == 0:
                        if verbose:
                            logger.debug(
                                f"Skipping '{phrase_text}' in {sample_id} (zero masks, count_only)."
                            )
                        should_skip = True
                elif mode == "deep_stats":
                    vis_count = data["visible_count"]
                    full_count = data["full_count"]
                    if skip_zero_masks and vis_count == 0 and full_count == 0:
                        if verbose:
                            logger.debug(
                                f"Skipping '{phrase_text}' in {sample_id} (zero masks, deep_stats)."
                            )
                        should_skip = True

                if should_skip:
                    continue

                # --- Phrase is NOT skipped, proceed with updates ---
                agg_entry = phrase_agg_stats[phrase_text]  # Now safe to access/create entry

                if mode == "count_only":
                    # Already got num_masks above
                    # Increment counts and add sample ID
                    agg_entry["appearance_count"] += 1
                    agg_entry["sample_ids"].append(sample_id)
                    # Update unified count fields
                    agg_entry["mask_counts_per_image"].append(num_masks)
                    agg_entry["total_mask_count"] += num_masks
                elif mode == "deep_stats":
                    # Already got vis_count and full_count above
                    # Increment counts and add sample ID
                    agg_entry["appearance_count"] += 1
                    agg_entry["sample_ids"].append(sample_id)
                    # Update deep_stats fields
                    agg_entry["visible_mask_counts_per_image"].append(vis_count)
                    agg_entry["total_visible_mask_count"] += vis_count
                    agg_entry["full_mask_counts_per_image"].append(full_count)
                    agg_entry["total_full_mask_count"] += full_count
                    # Geometry Stats
                    agg_entry["visible_mask_pixel_areas"].extend(data["vis_areas"])
                    agg_entry["visible_mask_widths"].extend(data["vis_widths"])
                    agg_entry["visible_mask_heights"].extend(data["vis_heights"])
                    agg_entry["full_mask_pixel_areas"].extend(data["full_areas"])
                    agg_entry["full_mask_widths"].extend(data["full_widths"])
                    agg_entry["full_mask_heights"].extend(data["full_heights"])

        except (ValueError, TypeError, json.JSONDecodeError) as e:
            skipped_count += 1
            if verbose:
                logger.warning(
                    f"Skipping sample '{sample_id}' due to data error: {e}", exc_info=False
                )
            continue
        except Exception as e:
            skipped_count += 1
            if verbose:
                logger.error(
                    f"Skipping sample '{sample_id}' due to unexpected error: {e}", exc_info=True
                )
            continue

    successfully_processed_count = processed_count - skipped_count
    if verbose:
        logger.info(
            f"Aggregation complete. Processed rows: {processed_count}, "
            f"Skipped rows: {skipped_count} => "
            f"Successfully processed: {successfully_processed_count}"
        )
        logger.info(f"Found {len(phrase_agg_stats)} unique phrases.")

    return dict(phrase_agg_stats), successfully_processed_count


def _calculate_metric_stats(data: List[Union[int, float]], percentiles: List[float]) -> Dict:
    """Helper to calculate mean and percentiles for a list of numbers."""
    stats_results = {"avg": 0.0, "percentiles": {}}
    # Filter out potential None or NaN values if geometry calculation failed or was invalid
    valid_data = [d for d in data if d is not None and not (isinstance(d, float) and math.isnan(d))]

    if valid_data:
        try:
            # Ensure data is suitable for numpy operations
            np_data = np.array(valid_data, dtype=float)
            stats_results["avg"] = float(np.mean(np_data))
            percentile_values = np.percentile(np_data, [p * 100 for p in percentiles])
            stats_results["percentiles"] = {
                p: float(v) for p, v in zip(percentiles, percentile_values, strict=False)
            }
        except (IndexError, ValueError, Exception) as e:  # Added ValueError
            logger.warning(
                f"Could not calculate statistics for data (len={len(valid_data)}, Error: {e})"
            )
            # Reset on error, keeping keys but with default values
            stats_results = {"avg": 0.0, "percentiles": {}}  # Assign default float for avg
    return stats_results


def calculate_summary_stats(
    aggregated_stats: AggregatedStatsDict,
    total_processed_samples: int,
    mode: str,
    percentiles: Optional[List[float]] = None,
) -> List[SummaryStat]:
    """
    Calculates summary statistics from the aggregated phrase data based on the provided mode.

    Args:
        aggregated_stats: The dictionary output from aggregate_phrase_stats.
        total_processed_samples: The total number of samples successfully processed.
        mode: The analysis mode used ('count_only' or 'deep_stats').
        percentiles: A list of percentiles to calculate (e.g., [0.1, 0.5, 0.9]).

    Returns:
        A list of dictionaries, each containing summary statistics for a phrase,
        sorted by appearance_count in descending order.
    """
    if not aggregated_stats:
        return []

    if percentiles is None:
        percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    summary_stats_list: List[SummaryStat] = []

    logger.info(f"Calculating summary stats based on provided mode: '{mode}'")

    for phrase, phrase_agg_data in aggregated_stats.items():
        appearance_count = phrase_agg_data.get("appearance_count", 0)
        appearance_percentage = (
            (appearance_count / total_processed_samples) * 100
            if total_processed_samples > 0
            else 0.0
        )

        # Initialize base summary dictionary
        summary: SummaryStat = {
            "phrase": phrase,
            "appearance_count": appearance_count,
            "appearance_percentage": float(appearance_percentage),
        }

        # --- Calculate Count Statistics based on detected mode ---
        if mode == "count_only":
            counts = phrase_agg_data.get("mask_counts_per_image", [])
            # Always call helper, it handles empty lists returning 0.0 for avg
            count_stats = _calculate_metric_stats(counts, percentiles)
            summary["avg_masks_per_image"] = count_stats["avg"]
            summary["mask_percentiles"] = count_stats["percentiles"]
        elif mode == "deep_stats":
            vis_counts = phrase_agg_data.get("visible_mask_counts_per_image", [])
            full_counts = phrase_agg_data.get("full_mask_counts_per_image", [])

            # Always call helper for visible counts
            vis_count_stats = _calculate_metric_stats(vis_counts, percentiles)
            summary["avg_visible_masks_per_image"] = vis_count_stats["avg"]
            summary["visible_mask_percentiles"] = vis_count_stats["percentiles"]

            # Always call helper for full counts
            full_count_stats = _calculate_metric_stats(full_counts, percentiles)
            summary["avg_full_masks_per_image"] = full_count_stats["avg"]
            summary["full_mask_percentiles"] = full_count_stats["percentiles"]

        # --- Calculate Deep Statistics (Geometry) if mode is 'deep_stats' ---
        if mode == "deep_stats":
            vis_area_stats = _calculate_metric_stats(
                phrase_agg_data.get("visible_mask_pixel_areas", []), percentiles
            )
            vis_width_stats = _calculate_metric_stats(
                phrase_agg_data.get("visible_mask_widths", []), percentiles
            )
            vis_height_stats = _calculate_metric_stats(
                phrase_agg_data.get("visible_mask_heights", []), percentiles
            )
            full_area_stats = _calculate_metric_stats(
                phrase_agg_data.get("full_mask_pixel_areas", []), percentiles
            )
            full_width_stats = _calculate_metric_stats(
                phrase_agg_data.get("full_mask_widths", []), percentiles
            )
            full_height_stats = _calculate_metric_stats(
                phrase_agg_data.get("full_mask_heights", []), percentiles
            )

            # Update summary with geometry stats
            summary["avg_visible_mask_pixels"] = vis_area_stats["avg"]
            summary["visible_mask_pixel_percentiles"] = vis_area_stats["percentiles"]
            summary["avg_visible_mask_width"] = vis_width_stats["avg"]
            summary["visible_mask_width_percentiles"] = vis_width_stats["percentiles"]
            summary["avg_visible_mask_height"] = vis_height_stats["avg"]
            summary["visible_mask_height_percentiles"] = vis_height_stats["percentiles"]

            summary["avg_full_mask_pixels"] = full_area_stats["avg"]
            summary["full_mask_pixel_percentiles"] = full_area_stats["percentiles"]
            summary["avg_full_mask_width"] = full_width_stats["avg"]
            summary["full_mask_width_percentiles"] = full_width_stats["percentiles"]
            summary["avg_full_mask_height"] = full_height_stats["avg"]
            summary["full_mask_height_percentiles"] = full_height_stats["percentiles"]

            # Calculate Saturation
            avg_vis_pixels = summary["avg_visible_mask_pixels"]
            avg_vis_w = summary["avg_visible_mask_width"]
            avg_vis_h = summary["avg_visible_mask_height"]
            vis_saturation = float("nan")
            if (
                isinstance(avg_vis_w, (int, float))
                and avg_vis_w > 0
                and isinstance(avg_vis_h, (int, float))
                and avg_vis_h > 0
                and isinstance(avg_vis_pixels, (int, float))
            ):
                vis_saturation = avg_vis_pixels / (avg_vis_w * avg_vis_h)
            summary["visible_mask_saturation"] = vis_saturation

            avg_full_pixels = summary["avg_full_mask_pixels"]
            avg_full_w = summary["avg_full_mask_width"]
            avg_full_h = summary["avg_full_mask_height"]
            full_saturation = float("nan")
            if (
                isinstance(avg_full_w, (int, float))
                and avg_full_w > 0
                and isinstance(avg_full_h, (int, float))
                and avg_full_h > 0
                and isinstance(avg_full_pixels, (int, float))
            ):
                full_saturation = avg_full_pixels / (avg_full_w * avg_full_h)
            summary["full_mask_saturation"] = full_saturation

        summary_stats_list.append(summary)

    # Sort by appearance_count descending
    summary_stats_list.sort(key=lambda x: x["appearance_count"], reverse=True)

    return summary_stats_list
