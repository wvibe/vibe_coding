from typing import Dict, List, Optional

import numpy as np
import pytest
import torch

from vibelab.utils.common.stats import (
    DEFAULT_NA_VALUE,
    calculate_numeric_summary,
    format_statistics_table
)


class TestCalculateNumericSummary:
    """Tests for the calculate_numeric_summary function."""

    def test_with_normal_data(self):
        """Test calculation with normal numeric data."""
        values = [10, 12, 11, 13, 10]
        metrics = ["count", "mean", "max", "min", "p50"]
        result, success = calculate_numeric_summary(values, metrics)

        assert success is True
        assert result["count"] == 5
        assert result["mean"] == 11.2
        assert result["max"] == 13
        assert result["min"] == 10
        assert result["p50"] == 11  # Median

    def test_with_none_and_nan(self):
        """Test calculation with None values and NaN values."""
        values = [10, None, 12, np.nan, 14]
        metrics = ["count", "mean", "max"]
        result, success = calculate_numeric_summary(values, metrics)

        assert success is True
        assert result["count"] == 3  # Only valid numbers are counted
        assert result["mean"] == 12.0
        assert result["max"] == 14

    def test_with_empty_list(self):
        """Test calculation with empty list."""
        values: List[Optional[float]] = []
        metrics = ["count", "mean", "p50", "max"]
        result, success = calculate_numeric_summary(values, metrics)

        assert success is True
        assert result["count"] == 0
        assert result["mean"] is None
        assert result["p50"] is None
        assert result["max"] is None

    def test_with_only_none(self):
        """Test calculation with list of only None values."""
        values = [None, None, None]
        metrics = ["count", "mean", "max"]
        result, success = calculate_numeric_summary(values, metrics)

        assert success is True
        assert result["count"] == 0
        assert result["mean"] is None
        assert result["max"] is None

    def test_with_invalid_metrics(self):
        """Test calculation with invalid metric names."""
        values = [1, 2, 3, 4, 5]
        metrics = ["count", "invalid_metric", "p200", "mean"]
        result, success = calculate_numeric_summary(values, metrics)

        assert success is False  # Should be False because not all metrics were valid
        assert result["count"] == 5
        assert result["mean"] == 3.0
        assert result["invalid_metric"] is None
        assert result["p200"] is None

    def test_all_supported_metrics(self):
        """Test all supported metrics."""
        values = [1, 2, 3, 4, 5]
        metrics = ["count", "sum", "mean", "min", "max", "std", "p25", "p50", "p75", "p99"]
        result, success = calculate_numeric_summary(values, metrics)

        assert success is True
        assert result["count"] == 5
        assert result["sum"] == 15.0
        assert result["mean"] == 3.0
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert result["std"] == pytest.approx(1.4142, abs=0.001)
        assert result["p25"] == 2.0
        assert result["p50"] == 3.0
        assert result["p75"] == 4.0
        assert result["p99"] == 5.0

    def test_case_insensitivity(self):
        """Test that metric names are case-insensitive."""
        values = [1, 2, 3, 4, 5]
        metrics = ["COUNT", "Mean", "mAx", "P50"]
        result, success = calculate_numeric_summary(values, metrics)

        assert success is True
        assert "count" in result
        assert "mean" in result
        assert "max" in result
        assert "p50" in result
        assert result["count"] == 5
        assert result["mean"] == 3.0


class TestFormatStatisticsTable:
    """Tests for the format_statistics_table function."""

    def test_with_valid_data(self):
        """Test formatting with valid data and format string."""
        data_dict = {
            "apples": [10, 12, 11, 13],
            "bananas": [5, 6, 5, 7, 6],
        }
        format_string = "{key:<10} {count:>5} {mean:>6.1f} {max:>4}"

        lines = format_statistics_table(data_dict, format_string)

        assert len(lines) == 4  # Header, divider, 2 data rows
        assert "Key" in lines[0]  # Header should capitalize placeholders
        assert "Count" in lines[0]
        assert "Mean" in lines[0]
        assert "Max" in lines[0]
        assert "-" * 5 in lines[1]  # Divider
        assert "apples" in lines[2]
        assert "bananas" in lines[3]

    def test_with_empty_data(self):
        """Test formatting with empty data dict."""
        data_dict: Dict[str, List[float]] = {}
        format_string = "{key:<10} {count:>5} {mean:>6.1f}"

        lines = format_statistics_table(data_dict, format_string)

        assert lines == []  # Should return empty list for empty data

    def test_with_missing_key(self):
        """Test formatting with format string missing 'key' placeholder."""
        data_dict = {"apples": [10, 12, 11]}
        format_string = "{count:>5} {mean:>6.1f}"  # Missing key

        lines = format_statistics_table(data_dict, format_string)

        assert len(lines) == 1
        assert "Error" in lines[0]  # Should return error message

    def test_with_key_not_first(self):
        """Test formatting with 'key' not being the first placeholder."""
        data_dict = {"apples": [10, 12, 11]}
        format_string = "{count:>5} {key:<10} {mean:>6.1f}"  # Key not first

        lines = format_statistics_table(data_dict, format_string)

        assert len(lines) == 1
        assert "Error" in lines[0]  # Should return error message

    def test_with_invalid_metric(self):
        """Test formatting with invalid metric in format string."""
        data_dict = {"apples": [10, 12, 11]}
        format_string = "{key:<10} {countt:>5}"  # Typo in count

        lines = format_statistics_table(data_dict, format_string)

        assert len(lines) == 3  # Header, divider, 1 data row
        assert "Key" in lines[0]
        assert "Countt" in lines[0]
        assert "apples" in lines[2]
        assert DEFAULT_NA_VALUE in lines[2]

    def test_skipping_rows_with_calculation_failures(self):
        """Test handling of rows with calculation failures."""
        data_dict = {
            "valid": [1, 2, 3, 4, 5],
            "invalid_for_p101": [],  # Empty list will cause p101 to fail
        }
        format_string = "{key:<20} {count:>5} {p101:>6}"  # p101 is invalid

        lines = format_statistics_table(data_dict, format_string)

        assert len(lines) == 4  # Header, divider, 2 data rows (both shown)
        assert "valid" in lines[2]
        assert "invalid_for_p101" in lines[3]
        assert "5" in lines[2]  # Count for valid
        assert "0" in lines[3]  # Count for invalid_for_p101
        assert DEFAULT_NA_VALUE in lines[2]
        assert DEFAULT_NA_VALUE in lines[3]

    def test_with_none_values(self):
        """Test formatting with None values in the data."""
        data_dict = {"mix": [10, None, 12, None, 14], "all_none": [None, None]}
        format_string = "{key:<10} {count:>5} {mean:>6.1f}"

        lines = format_statistics_table(data_dict, format_string)

        assert len(lines) == 4  # Header, divider, 2 data rows
        assert "mix" in lines[2]
        assert "all_none" in lines[3]
        assert "3" in lines[2]  # Count is 3 (excluding None values)
        assert "12" in lines[2]  # Mean is 12.0
        assert "0" in lines[3]  # Count is 0
        assert DEFAULT_NA_VALUE in lines[3]  # Mean is N/A

    def test_complex_format_with_multiple_metrics(self):
        """Test formatting with a complex format string and multiple metrics."""
        data_dict = {"sample": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        format_string = (
            "{key:<10} {count:>5} {mean:>6.1f} {min:>4} {max:>4} {p25:>4} {p50:>4} "
            "{p75:>4} {std:>6.2f}"
        )

        lines = format_statistics_table(data_dict, format_string)

        assert len(lines) == 3  # Header, divider, 1 data row
        assert "sample" in lines[2]
        assert "10" in lines[2]  # count
        assert "5.5" in lines[2]  # mean
        assert "1" in lines[2]  # min
        assert "10" in lines[2]  # max

    def test_expected_typical_format_case(self):
        """Test formatting with a typical format string."""
        data_dict = {"apples": [10, 12, 11, 13], "bananas": [5, 6, 5, 7, 6]}
        format_string = "{key:<8} {count:<5d} {mean:<6.1f} {p50:<4d} {max:>4d}"

        lines = format_statistics_table(data_dict, format_string)

        assert len(lines) == 4  # Header, divider, 2 data rows

        # Check header format - update to match actual implementation
        assert "Key      Count Mean   P50   Max" == lines[0]
        assert "-" * len(lines[0]) == lines[1]
        assert "apples   4     11.5   12     13" == lines[2]
        assert "bananas  5     5.8    6       7" == lines[3]

    def test_trimmed_long_keys(self):
        """Test formatting with long keys that are trimmed."""
        very_long_key = "very_long_key_that_should_be_trimmed"
        format_key_len = 10

        data_dict = {very_long_key: [1, 2, 3, 4, 5]}
        format_string = "{key:<" + str(format_key_len) + "} {count:>5} {mean:>5.1f}"

        lines = format_statistics_table(data_dict, format_string)

        assert len(lines) == 3  # Header, divider, 1 data row
        assert very_long_key[:format_key_len] + "     5   3.0" == lines[2]
