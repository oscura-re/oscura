"""CSV export functionality.

This module provides trace and measurement export to CSV format.


Example:
    >>> from oscura.exporters.csv import export_csv
    >>> export_csv(trace, "output.csv")
    >>> export_csv(measurements, "results.csv")

References:
    RFC 4180 (CSV format)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.core.types import DigitalTrace, WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray


def export_csv(
    data: WaveformTrace | DigitalTrace | dict[str, Any] | NDArray[Any],
    path: str | Path,
    *,
    include_time: bool = True,
    time_unit: str = "s",
    precision: int = 9,
    delimiter: str = ",",
    header: bool = True,
) -> None:
    """Export data to CSV format.

    Args:
        data: Data to export. Can be:
            - WaveformTrace or DigitalTrace (with metadata as comments)
            - Dictionary of measurements
            - NumPy array
        path: Output file path.
        include_time: Include time column for traces.
        time_unit: Time unit ("s", "ms", "us", "ns").
        precision: Decimal precision for floating point values.
        delimiter: Column delimiter.
        header: Include header row and metadata comments.

    Raises:
        TypeError: If data type is not supported.

    Example:
        >>> export_csv(trace, "waveform.csv")
        >>> export_csv(trace, "data.csv", precision=6, delimiter="\t")
        >>> export_csv(measurements, "results.csv")

    Note:
        When exporting traces, metadata is included as comment lines
        starting with '#' when header=True.

    References:
        EXP-001
    """
    path = Path(path)

    if isinstance(data, WaveformTrace | DigitalTrace):
        _export_trace(data, path, include_time, time_unit, precision, delimiter, header)
    elif isinstance(data, dict):
        _export_dict(data, path, precision, delimiter, header)
    elif isinstance(data, np.ndarray):
        _export_array(data, path, precision, delimiter, header)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def _export_trace(
    trace: WaveformTrace | DigitalTrace,
    path: Path,
    include_time: bool,
    time_unit: str,
    precision: int,
    delimiter: str,
    header: bool,
) -> None:
    """Export trace to CSV.

    Args:
        trace: Trace to export.
        path: Output file path.
        include_time: Include time column.
        time_unit: Time unit for column.
        precision: Decimal precision.
        delimiter: Column delimiter.
        header: Include header row.
    """
    time_multipliers = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
    multiplier = time_multipliers.get(time_unit, 1.0)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)

        # Write metadata as comments if header is enabled
        if header:
            # Metadata comments
            meta = trace.metadata
            f.write("# Oscura CSV Export\n")
            f.write(f"# Sample Rate: {meta.sample_rate} Hz\n")
            f.write(f"# Time Base: {meta.time_base} s\n")
            f.write(f"# Samples: {len(trace.data)}\n")
            f.write(f"# Duration: {trace.duration} s\n")

            if meta.vertical_scale is not None:
                f.write(f"# Vertical Scale: {meta.vertical_scale} V/div\n")
            if meta.vertical_offset is not None:
                f.write(f"# Vertical Offset: {meta.vertical_offset} V\n")
            if meta.acquisition_time is not None:
                f.write(f"# Acquisition Time: {meta.acquisition_time.isoformat()}\n")
            if meta.source_file is not None:
                f.write(f"# Source File: {meta.source_file}\n")
            if meta.channel_name is not None:
                f.write(f"# Channel: {meta.channel_name}\n")

            f.write("#\n")

            # Column headers
            if include_time:
                if isinstance(trace, WaveformTrace):
                    writer.writerow([f"Time ({time_unit})", "Voltage"])
                else:
                    writer.writerow([f"Time ({time_unit})", "Digital"])
            elif isinstance(trace, WaveformTrace):
                writer.writerow(["Voltage"])
            else:
                writer.writerow(["Digital"])

        # Data
        n_samples = len(trace.data)
        time_base = trace.metadata.time_base

        for i in range(n_samples):
            if include_time:
                time_val = i * time_base * multiplier
                if isinstance(trace, WaveformTrace):
                    writer.writerow([f"{time_val:.{precision}g}", f"{trace.data[i]:.{precision}g}"])
                else:
                    writer.writerow([f"{time_val:.{precision}g}", int(trace.data[i])])
            elif isinstance(trace, WaveformTrace):
                writer.writerow([f"{trace.data[i]:.{precision}g}"])
            else:
                writer.writerow([int(trace.data[i])])


def _export_dict(
    data: dict[str, Any],
    path: Path,
    precision: int,
    delimiter: str,
    header: bool,
) -> None:
    """Export dictionary to CSV.

    Args:
        data: Dictionary to export.
        path: Output file path.
        precision: Decimal precision.
        delimiter: Column delimiter.
        header: Include header row.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)

        if header:
            writer.writerow(["Parameter", "Value", "Unit"])

        for key, value in data.items():
            if isinstance(value, dict):
                # Nested dict with value/unit
                val = value.get("value", value)
                unit = value.get("unit", "")
                if isinstance(val, float):
                    writer.writerow([key, f"{val:.{precision}g}", unit])
                else:
                    writer.writerow([key, val, unit])
            elif isinstance(value, float):
                writer.writerow([key, f"{value:.{precision}g}", ""])
            else:
                writer.writerow([key, value, ""])


def _export_array(
    data: NDArray[Any],
    path: Path,
    precision: int,
    delimiter: str,
    header: bool,
) -> None:
    """Export numpy array to CSV.

    Args:
        data: NumPy array to export.
        path: Output file path.
        precision: Decimal precision.
        delimiter: Column delimiter.
        header: Include header row.
    """
    # Handle different array dimensions
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)

        if header:
            cols = [f"Column_{i}" for i in range(data.shape[1])]
            writer.writerow(cols)

        for row in data:
            formatted = []
            for val in row:
                if isinstance(val, float | np.floating):
                    formatted.append(f"{val:.{precision}g}")
                else:
                    formatted.append(str(val))  # type: ignore[unreachable]
            writer.writerow(formatted)


def export_multi_trace_csv(
    traces: list[WaveformTrace | DigitalTrace],
    path: str | Path,
    *,
    names: list[str] | None = None,
    include_time: bool = True,
    time_unit: str = "s",
    precision: int = 9,
) -> None:
    """Export multiple traces to single CSV file.

    Args:
        traces: List of traces to export.
        path: Output file path.
        names: Column names for each trace.
        include_time: Include time column.
        time_unit: Time unit.
        precision: Decimal precision.

    Example:
        >>> export_multi_trace_csv([ch1, ch2, ch3], "channels.csv",
        ...                         names=["CH1", "CH2", "CH3"])
    """
    if len(traces) == 0:
        return

    path = Path(path)

    if names is None:
        names = [f"Trace_{i}" for i in range(len(traces))]

    # Use first trace for timing
    ref_trace = traces[0]
    n_samples = len(ref_trace.data)
    time_base = ref_trace.metadata.time_base

    time_multipliers = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
    multiplier = time_multipliers.get(time_unit, 1.0)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header_row = []
        if include_time:
            header_row.append(f"Time ({time_unit})")
        header_row.extend(names)
        writer.writerow(header_row)

        # Data
        for i in range(n_samples):
            row = []

            if include_time:
                time_val = i * time_base * multiplier
                row.append(f"{time_val:.{precision}g}")

            for trace in traces:
                if i < len(trace.data):
                    if isinstance(trace, WaveformTrace):
                        row.append(f"{trace.data[i]:.{precision}g}")
                    else:
                        row.append(str(int(trace.data[i])))
                else:
                    row.append("")

            writer.writerow(row)


__all__ = [
    "export_csv",
    "export_multi_trace_csv",
]
