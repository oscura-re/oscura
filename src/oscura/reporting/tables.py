"""Table generation and formatting for Oscura reports.

This module provides utilities for creating and formatting measurement
summary tables with professional appearance.


Example:
    >>> from oscura.reporting.tables import create_measurement_table
    >>> table = create_measurement_table(measurements, format="markdown")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from oscura.reporting.formatting import NumberFormatter

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _build_table_headers(show_spec: bool, show_margin: bool, show_status: bool) -> list[str]:
    """Build table headers based on display options.

    Args:
        show_spec: Include specification column.
        show_margin: Include margin column.
        show_status: Include status column.

    Returns:
        List of header strings.
    """
    headers = ["Parameter", "Value"]
    if show_spec:
        headers.append("Specification")
    if show_margin:
        headers.append("Margin")
    if show_status:
        headers.append("Status")
    return headers


def _format_value_cell(value: Any, unit: str, formatter: NumberFormatter) -> str:
    """Format measurement value cell.

    Args:
        value: Measurement value.
        unit: Unit string.
        formatter: Number formatter instance.

    Returns:
        Formatted value string.
    """
    if value is None:
        return "N/A"
    return formatter.format(value, unit)


def _format_spec_cell(spec: Any, spec_type: str, unit: str, formatter: NumberFormatter) -> str:
    """Format specification cell with comparison operator.

    Args:
        spec: Specification value.
        spec_type: Spec type ("max", "min", or "exact").
        unit: Unit string.
        formatter: Number formatter instance.

    Returns:
        Formatted specification string.
    """
    if spec is None:
        return "-"

    prefix = "<" if spec_type == "max" else ">" if spec_type == "min" else "="
    return f"{prefix}{formatter.format(spec, unit)}"


def _calculate_margin(value: float, spec: float, spec_type: str) -> str:
    """Calculate margin percentage between value and spec.

    Args:
        value: Measured value.
        spec: Specification limit.
        spec_type: Specification type ("max" or "min").

    Returns:
        Formatted margin percentage string.
    """
    if spec == 0:
        return "-"

    if spec_type == "max":
        margin = (spec - value) / spec * 100
    else:
        margin = (value - spec) / spec * 100

    return f"{margin:.1f}%"


def _build_measurement_row(
    name: str,
    meas: dict[str, Any],
    formatter: NumberFormatter,
    show_spec: bool,
    show_margin: bool,
    show_status: bool,
) -> list[str]:
    """Build single measurement table row.

    Args:
        name: Measurement parameter name.
        meas: Measurement data dictionary.
        formatter: Number formatter instance.
        show_spec: Include specification column.
        show_margin: Include margin column.
        show_status: Include status column.

    Returns:
        List of cell values for table row.
    """
    row = [name]
    value = meas.get("value")
    unit = meas.get("unit", "")

    # Value column
    row.append(_format_value_cell(value, unit, formatter))

    # Specification column
    if show_spec:
        spec = meas.get("spec")
        spec_type = meas.get("spec_type", "max")
        row.append(_format_spec_cell(spec, spec_type, unit, formatter))

    # Margin column
    if show_margin:
        spec = meas.get("spec")
        if value is not None and spec is not None:
            spec_type = meas.get("spec_type", "max")
            row.append(_calculate_margin(value, spec, spec_type))
        else:
            row.append("-")

    # Status column
    if show_status:
        passed = meas.get("passed", True)
        if value is None:
            row.append("N/A")
        else:
            row.append("✓ PASS" if passed else "✗ FAIL")

    return row


def create_measurement_table(
    measurements: dict[str, Any],
    *,
    format: Literal["dict", "markdown", "html", "csv"] = "dict",
    show_spec: bool = True,
    show_margin: bool = True,
    show_status: bool = True,
    sort_by: str | None = None,
) -> dict[str, Any] | str:
    """Create formatted measurement summary table.

    Args:
        measurements: Dictionary of measurement name -> measurement data.
        format: Output format (dict, markdown, html, csv).
        show_spec: Include specification column.
        show_margin: Include margin column.
        show_status: Include pass/fail status column.
        sort_by: Column to sort by (None for original order).

    Returns:
        Formatted table as dictionary or string depending on format.

    Raises:
        ValueError: If format is unknown.

    Example:
        >>> measurements = {
        ...     "rise_time": {"value": 2.3e-9, "spec": 5e-9, "unit": "s"},
        ...     "fall_time": {"value": 1.8e-9, "spec": 5e-9, "unit": "s"},
        ... }
        >>> table = create_measurement_table(measurements, format="markdown")

    References:
        REPORT-004, REPORT-006
    """
    headers = _build_table_headers(show_spec, show_margin, show_status)
    formatter = NumberFormatter(sig_figs=3)

    # Build rows
    rows = [
        _build_measurement_row(name, meas, formatter, show_spec, show_margin, show_status)
        for name, meas in measurements.items()
    ]

    # Sort if requested
    if sort_by and sort_by in headers:
        col_idx = headers.index(sort_by)
        rows.sort(key=lambda r: r[col_idx])

    # Format output
    if format == "dict":
        return {"type": "table", "headers": headers, "data": rows}
    elif format == "markdown":
        return _format_markdown_table(headers, rows)
    elif format == "html":
        return _format_html_table(headers, rows)
    elif format == "csv":
        return _format_csv_table(headers, rows)
    else:
        raise ValueError(f"Unknown format: {format}")


def create_comparison_table(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    format: Literal["dict", "markdown", "html"] = "dict",
    show_delta: bool = True,
    show_percent_change: bool = True,
) -> dict[str, Any] | str:
    """Create comparison table between baseline and current measurements.

    Args:
        baseline: Baseline measurements.
        current: Current measurements.
        format: Output format.
        show_delta: Show absolute difference.
        show_percent_change: Show percentage change.

    Returns:
        Formatted comparison table.

    Raises:
        ValueError: If format is unknown.

    References:
        REPORT-008 (Comparison Reports)
    """
    headers = ["Parameter", "Baseline", "Current"]
    if show_delta:
        headers.append("Delta")
    if show_percent_change:
        headers.append("% Change")

    rows = []
    formatter = NumberFormatter(sig_figs=3)

    # Get all parameter names
    all_params = set(baseline.keys()) | set(current.keys())

    for name in sorted(all_params):
        row = [name]

        base_meas = baseline.get(name, {})
        curr_meas = current.get(name, {})

        base_val = base_meas.get("value")
        curr_val = curr_meas.get("value")
        unit = base_meas.get("unit", curr_meas.get("unit", ""))

        # Baseline value
        if base_val is not None:
            row.append(formatter.format(base_val, unit))
        else:
            row.append("-")

        # Current value
        if curr_val is not None:
            row.append(formatter.format(curr_val, unit))
        else:
            row.append("-")

        # Delta
        if show_delta:
            if base_val is not None and curr_val is not None:
                delta = curr_val - base_val
                row.append(formatter.format(delta, unit))
            else:
                row.append("-")

        # Percent change
        if show_percent_change:
            if base_val is not None and curr_val is not None and base_val != 0:
                pct_change = (curr_val - base_val) / base_val * 100
                row.append(f"{pct_change:+.1f}%")
            else:
                row.append("-")

        rows.append(row)

    if format == "dict":
        return {"type": "table", "headers": headers, "data": rows}
    elif format == "markdown":
        return _format_markdown_table(headers, rows)
    elif format == "html":
        return _format_html_table(headers, rows)
    else:
        raise ValueError(f"Unknown format: {format}")


def create_statistics_table(
    data: dict[str, NDArray[np.float64]],
    *,
    format: Literal["dict", "markdown", "html"] = "dict",
    statistics: list[str] | None = None,
) -> dict[str, Any] | str:
    """Create statistics summary table.

    Args:
        data: Dictionary of parameter name -> data array.
        format: Output format.
        statistics: List of statistics to include (mean, std, min, max, median).

    Returns:
        Formatted statistics table.

    Raises:
        ValueError: If format is unknown.

    References:
        REPORT-004
    """
    if statistics is None:
        statistics = ["mean", "std", "min", "max", "median"]

    headers = ["Parameter"] + [stat.capitalize() for stat in statistics]
    rows = []

    for name, values in data.items():
        row = [name]

        for stat in statistics:
            if stat == "mean":
                row.append(f"{np.mean(values):.3g}")
            elif stat == "std":
                row.append(f"{np.std(values):.3g}")
            elif stat == "min":
                row.append(f"{np.min(values):.3g}")
            elif stat == "max":
                row.append(f"{np.max(values):.3g}")
            elif stat == "median":
                row.append(f"{np.median(values):.3g}")
            else:
                row.append("-")

        rows.append(row)

    if format == "dict":
        return {"type": "table", "headers": headers, "data": rows}
    elif format == "markdown":
        return _format_markdown_table(headers, rows)
    elif format == "html":
        return _format_html_table(headers, rows)
    else:
        raise ValueError(f"Unknown format: {format}")


def _format_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Format table as Markdown."""
    lines = []

    # Header row
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    # Data rows
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

    return "\n".join(lines)


def _format_html_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Format table as HTML."""
    lines = ['<table class="measurement-table">']

    # Header
    lines.append("<thead><tr>")
    for h in headers:
        lines.append(f"<th>{h}</th>")
    lines.append("</tr></thead>")

    # Body
    lines.append("<tbody>")
    for row in rows:
        lines.append("<tr>")
        for cell in row:
            cell_str = str(cell)
            # Apply CSS classes for status
            if "PASS" in cell_str:
                lines.append(f'<td class="pass">{cell}</td>')
            elif "FAIL" in cell_str:
                lines.append(f'<td class="fail">{cell}</td>')
            else:
                lines.append(f"<td>{cell}</td>")
        lines.append("</tr>")
    lines.append("</tbody>")

    lines.append("</table>")
    return "\n".join(lines)


def _format_csv_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Format table as CSV."""
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)

    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)

    return output.getvalue()


def format_batch_summary_table(
    batch_results: list[dict[str, Any]],
    *,
    format: Literal["dict", "markdown", "html"] = "dict",
) -> dict[str, Any] | str:
    """Create batch summary table for multi-DUT testing.

    Args:
        batch_results: List of result dictionaries, one per DUT.
        format: Output format.

    Returns:
        Formatted batch summary table.

    Raises:
        ValueError: If format is unknown.

    References:
        REPORT-009 (Batch Report Aggregation)
    """
    if not batch_results:
        return {"type": "table", "headers": [], "data": []}

    headers = ["DUT ID", "Total Tests", "Passed", "Failed", "Yield"]
    rows = []

    for i, result in enumerate(batch_results):
        dut_id = result.get("dut_id", f"DUT-{i + 1}")
        total = result.get("total_count", 0)
        passed = result.get("pass_count", 0)
        failed = total - passed
        yield_pct = (passed / total * 100) if total > 0 else 0

        rows.append([dut_id, total, passed, failed, f"{yield_pct:.1f}%"])

    # Add summary row
    total_tests = sum(r.get("total_count", 0) for r in batch_results)
    total_passed = sum(r.get("pass_count", 0) for r in batch_results)
    total_failed = total_tests - total_passed
    overall_yield = (total_passed / total_tests * 100) if total_tests > 0 else 0

    rows.append(
        [
            "TOTAL",
            total_tests,
            total_passed,
            total_failed,
            f"{overall_yield:.1f}%",
        ]
    )

    if format == "dict":
        return {"type": "table", "headers": headers, "data": rows}
    elif format == "markdown":
        return _format_markdown_table(headers, rows)
    elif format == "html":
        return _format_html_table(headers, rows)
    else:
        raise ValueError(f"Unknown format: {format}")
