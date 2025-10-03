"""
Result formatters for LLM context.

Different serialization formats for SQL results to optimize token usage.
"""

import os
import csv
from io import StringIO
from typing import List, Any


def format_as_csv(result_text: str) -> str:
    """
    Format SQL result as CSV.

    CSV format is ~26% more token-efficient than tabular format.

    Note: This is a best-effort parser. It assumes result_text is a
    tabular format output (with headers and rows separated by lines).

    :param result_text: Tabular formatted result text
    :return: CSV formatted string
    """
    lines = result_text.strip().split('\n')

    if len(lines) < 3:
        # Not enough lines for a table (need header separator and at least one row)
        # Return as-is (might be an error message or special output)
        return result_text

    # Try to parse as table
    # Format is typically:
    # +------+-------+
    # | col1 | col2  |
    # |------+-------|
    # | val1 | val2  |
    # +------+-------+

    # Find header line and separator
    # Table format is:
    # +------+-------+
    # |   id | name  |  <-- header line
    # |------+-------|  <-- separator line (contains -, not +)
    # |    1 | Alice |  <-- data lines
    # +------+-------+

    header_line = None
    separator_line_idx = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or not '|' in stripped:
            continue

        # Check if this is a separator/border line (all + - | and spaces)
        if all(c in '+-| ' for c in stripped):
            # Separator line is after the header and contains mostly dashes
            # Border lines are at top/bottom and contain mostly pluses
            # Count dashes vs pluses to distinguish
            dash_count = stripped.count('-')
            plus_count = stripped.count('+')

            if dash_count > plus_count and header_line is not None:
                # This is the separator line (comes after header)
                separator_line_idx = idx
            continue

        # This line has | and is not a border/separator -> it's data
        if header_line is None:
            header_line = stripped
            continue

    if header_line is None or separator_line_idx is None:
        # Not a table format, return as-is
        return result_text

    # Parse header
    headers = [col.strip() for col in header_line.split('|')[1:-1]]

    # Parse data rows (everything after separator line)
    rows = []
    for idx in range(separator_line_idx + 1, len(lines)):
        line = lines[idx].strip()

        # Skip empty lines and border lines
        if not line or all(c in '+-| ' for c in line):
            continue

        if '|' in line:
            values = [val.strip() for val in line.split('|')[1:-1]]
            if values and len(values) == len(headers):  # Skip malformed rows
                rows.append(values)

    # Convert to CSV
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)

    return output.getvalue().strip()


def format_as_tabular(result_text: str) -> str:
    """
    Keep SQL result in tabular format (current behavior).

    This is the default format - returns input unchanged.

    :param result_text: Tabular formatted result text
    :return: Same tabular formatted string
    """
    return result_text


def get_result_formatter():
    """
    Get the configured result formatter function.

    Checks PGCLI_LLM_RESULT_FORMAT environment variable:
    - "csv": Use CSV format (~26% token savings)
    - "tabular" or unset: Use tabular format (default, current behavior)

    :return: Formatter function that takes result_text and returns formatted string
    """
    format_type = os.environ.get('PGCLI_LLM_RESULT_FORMAT', 'tabular').lower()

    if format_type == 'csv':
        return format_as_csv
    else:
        # Default to tabular (current behavior)
        return format_as_tabular
