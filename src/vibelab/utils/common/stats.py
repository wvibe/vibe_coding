import logging
import math
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Default value to display when a statistic cannot be computed or is missing
DEFAULT_NA_VALUE = "N/A"


def calculate_numeric_summary(
    values: List[Optional[Union[int, float]]], metrics: List[str]
) -> Tuple[Dict[str, Union[float, int, None]], bool]:
    """
    Calculates requested summary statistics for a list of numeric values.

    Handles None values by filtering them out before calculation.

    Args:
        values: A list of numbers (int or float), potentially containing None.
        metrics: A list of strings specifying the statistics to calculate.
                 Supported metrics (case-insensitive):
                 'count', 'sum', 'mean', 'min', 'max', 'std', 'pX' (where X is 0-100).

    Returns:
        A tuple containing:
        - A dictionary mapping each requested metric name (lowercase) to its
          calculated value. Returns None for a metric if calculation is not
          possible (e.g., mean of empty list, invalid metric). Returns 0 for
          'count' if the input list is empty or contains only None.
        - A boolean indicating whether all requested metrics were successfully
          calculated.
    """
    # Filter out None values
    valid_values = [v for v in values if v is not None and not math.isnan(v)]
    results: Dict[str, Union[float, int, None]] = {}
    np_values = np.array(valid_values, dtype=float)  # Use float for consistency

    # Handle case with no valid data early
    if np_values.size == 0:
        for metric in metrics:
            metric_lower = metric.lower()
            if metric_lower == "count":
                results[metric_lower] = 0
            else:
                # For stats other than count, return None if no valid data
                results[metric_lower] = None
        return results, True

    skipped_count = 0
    # Calculate requested metrics
    for metric in metrics:
        metric_lower = metric.lower()
        value: Union[float, int, None] = None  # Default to None

        try:
            if metric_lower == "count":
                # Count includes only valid numbers
                value = len(valid_values)
            elif metric_lower == "sum":
                value = float(np.sum(np_values))
            elif metric_lower == "mean":
                value = float(np.mean(np_values))
            elif metric_lower == "min":
                value = float(np.min(np_values))
            elif metric_lower == "max":
                value = float(np.max(np_values))
            elif metric_lower == "std":
                value = float(np.std(np_values))
            elif metric_lower.startswith("p") and metric_lower[1:].isdigit():
                percentile = int(metric_lower[1:])
                if 0 <= percentile <= 100:
                    # Use interpolation='nearest' for consistency with pandas/Excel?
                    # Or 'linear' which is numpy default? Let's stick with nearest for now.
                    value = float(np.percentile(np_values, percentile, method="nearest"))
                else:
                    logger.warning(
                        f"Invalid percentile requested: {metric}. Must be p0-p100. Returning None."
                    )
            else:
                logger.warning(f"Unsupported statistic requested: {metric}. Returning None.")
        except (ValueError, TypeError, RuntimeWarning) as e:
            # Catch numpy errors if calculation fails on valid (but perhaps unusual) data
            logger.warning(
                f"Could not calculate metric '{metric_lower}' for data (size={np_values.size}): "
                f"{e}."
            )
        if value is None:
            skipped_count += 1
        results[metric_lower] = value

    return results, skipped_count == 0


def _parse_format_string(
    format_string: str,
) -> Tuple[List[str], Dict[str, str], List[str]]:
    """Parses the f-string-like format string to extract metrics and formats."""
    # Regex to find {placeholder:format_spec}
    # Allows simple placeholders like {key} too (implicitly treated as string)
    pattern = re.compile(r"\{(\w+)(?::([^{}]+))?\}")
    matches = pattern.findall(format_string)

    if not matches:
        raise ValueError(
            "Invalid format_string: No valid placeholders found (e.g., {key}, {mean:.2f})."
        )

    ordered_placeholders: List[str] = []
    format_specs: Dict[str, str] = {}
    required_metrics: List[str] = []

    for name, spec in matches:
        name_lower = name.lower()
        ordered_placeholders.append(name_lower)
        format_specs[name_lower] = spec if spec else ""  # Store empty spec if none provided

        if name_lower != "key":
            required_metrics.append(name_lower)

    # Check if 'key' placeholder exists and is the first placeholder
    if "key" not in ordered_placeholders or ordered_placeholders[0] != "key":
        raise ValueError(
            "Invalid format_string: Must include a {key} placeholder as the first placeholder."
        )

    return ordered_placeholders, format_specs, list(set(required_metrics))


def _generate_header_and_divider(
    format_string: str, ordered_placeholders: List[str]
) -> Tuple[str, str]:
    """Generates the header and divider lines based on parsed format info."""
    # Create a dummy dict with placeholder names for formatting the header
    # Use strings for all header values to avoid format specifier issues
    header_dict = {}
    for name in ordered_placeholders:
        header_dict[name] = name.capitalize()

    # Create a new format string that keeps alignment and width but removes type specifiers
    # This regex finds format specifiers like {name:specs} and processes them
    def format_replacer(match):
        name = match.group(1)
        if match.group(2):  # If there's a format spec
            # Extract alignment and width (e.g. '<10', '>5')
            # but remove type specifiers (e.g. 'd', 'f', '.2f')
            format_spec = match.group(2)
            # Keep only alignment (<, >, ^) and width (numbers)
            # Remove precision (.X) and type (d, f, etc.)
            clean_spec = re.sub(r"\..*|[^0-9<>^]$", "", format_spec)
            return f"{{{name}:{clean_spec}}}"
        return f"{{{name}}}"

    header_format = re.sub(r"\{(\w+)(?::([^{}]+))?\}", format_replacer, format_string)

    try:
        header = header_format.format(**header_dict)
    except KeyError as e:
        raise ValueError(
            f"Placeholder '{{{e}}}' in format_string does not match parsed placeholders."
        ) from e

    # Create divider based on header length (simple approach)
    divider = "-" * len(header)
    return header, divider


def _create_row_format_string(
    format_string: str, calculated_stats: Dict[str, Union[float, int, None, str]]
) -> str:
    """Create a safe format string that handles non-numeric values properly."""
    # Find all placeholders with format specs in the original string
    pattern = re.compile(r"\{(\w+)(?::([^{}]+))?\}")
    matches = pattern.findall(format_string)

    # Create a new format string that handles None/NA values and type mismatches
    safe_format = format_string
    for name, spec in matches:
        name_lower = name.lower()
        # Skip key which is always a string
        if name_lower == "key":
            continue

        value = calculated_stats.get(name_lower)

        # Case 1: Value is None or N/A - remove format spec
        if value is None or value == DEFAULT_NA_VALUE:
            orig_pattern = rf"{{{name_lower}(?::{spec})?\}}"
            replacement = f"{{{name_lower}}}"
            safe_format = re.sub(orig_pattern, replacement, safe_format)
            continue

        # Case 2: Integer format ('d') used with float - convert to integer format
        if spec and "d" in spec and isinstance(value, float):
            # Replace the 'd' with equivalent float format (removing decimals)
            # e.g. '>5d' becomes '>5.0f'
            new_spec = spec.replace("d", ".0f")
            orig_pattern = rf"{{{name_lower}:{spec}}}"
            replacement = f"{{{name_lower}:{new_spec}}}"
            safe_format = re.sub(orig_pattern, replacement, safe_format)

    return safe_format


def _format_row(
    item_key: str,
    calculated_stats: Dict[str, Union[float, int, None]],
    format_string: str,
    ordered_placeholders: List[str],
) -> str:
    """Formats a single data row using calculated stats and the format string."""
    # Prepare the data to format
    row_data_to_format = {"key": item_key}  # Ensure key is used correctly

    # Ensure all placeholders have a value, defaulting to DEFAULT_NA_VALUE
    for placeholder in ordered_placeholders:
        if placeholder != "key" and placeholder in calculated_stats:
            row_data_to_format[placeholder] = calculated_stats[placeholder]
            if row_data_to_format[placeholder] is None:
                # Explicitly replace None with N/A for formatting
                row_data_to_format[placeholder] = DEFAULT_NA_VALUE
        elif placeholder != "key":
            row_data_to_format[placeholder] = DEFAULT_NA_VALUE

    # Create a safe format string that handles non-numeric values properly
    safe_format = _create_row_format_string(format_string, row_data_to_format)

    try:
        # Format the row with the safe format string
        formatted_row = safe_format.format(**row_data_to_format)
        return formatted_row
    except (ValueError, TypeError) as e:
        # If we still have formatting errors, use a simple fallback
        logger.warning(
            f"Error formatting row for key '{item_key}' with data {row_data_to_format}: {e}."
        )

        # Create a simple space-separated row as fallback
        parts = [f"{item_key}"]
        for placeholder in ordered_placeholders[1:]:  # Skip key which is already added
            value = row_data_to_format.get(placeholder, DEFAULT_NA_VALUE)
            parts.append(f"{value}")
        return "  ".join(parts)


def _trim_key_if_needed(item_key: str, format_string: str) -> str:
    """
    Trims the key if it exceeds the length specified in the format string.

    Args:
        item_key: The key to potentially trim
        format_string: The format string that may contain a length specification for the key

    Returns:
        The trimmed key if needed, otherwise the original key
    """
    # Extract the key format specification from the format string
    key_format_match = re.search(r"\{key:([<^>])(\d+)\}", format_string)
    if key_format_match:
        alignment, width = key_format_match.groups()
        width = int(width)
        if len(item_key) > width:
            return item_key[:width]
    return item_key


def format_statistics_table(
    data_dict: Dict[str, List[Optional[Union[int, float]]]], format_string: str
) -> List[str]:
    """
    Generates a formatted statistics table from raw numeric data lists.

    Calculates summary statistics for each value list in the input dictionary
    based on metrics specified within the format string, then formats the
    results into a multi-line table.

    Args:
        data_dict: A dictionary where keys are identifiers (e.g., category names)
                   and values are lists of numbers (int/float, possibly with None).
        format_string: A Python f-string-like template defining the table columns,
                       their order, and formatting. It should include placeholders
                       like {key:<...>} for the identifier and {metric:<...>}
                       (e.g., {count:<6d}, {mean:8.2f}, {p50:<5}) for the
                       desired statistics. The metric names within the curly
                       braces must match the keys supported by
                       `calculate_numeric_summary` (case-insensitive) or be 'key'.

    Returns:
        A list of strings representing the formatted table (header, divider, data rows).
        Returns an empty list if data_dict is empty.
    """
    if not data_dict:
        return []

    try:
        ordered_placeholders, format_specs, required_metrics = _parse_format_string(format_string)
        header, divider = _generate_header_and_divider(format_string, ordered_placeholders)
    except ValueError as e:
        logger.error(f"Failed to initialize statistics table: {e}")
        return [f"Error: {e}"]  # Return error message if parsing/header fails

    table_rows: List[str] = []
    for item_key, item_values in data_dict.items():
        # Trim the key if it exceeds the format length
        trimmed_key = _trim_key_if_needed(item_key, format_string)

        calculated_stats, all_metrics_succeeded = calculate_numeric_summary(
            values=item_values, metrics=required_metrics
        )

        # Continue with formatting even if some metrics failed
        formatted_row = _format_row(
            trimmed_key, calculated_stats, format_string, ordered_placeholders
        )
        table_rows.append(formatted_row)

    return [header, divider] + table_rows
