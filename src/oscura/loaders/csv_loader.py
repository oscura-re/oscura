"""CSV file loader for waveform data.

This module provides loading of waveform data from CSV files with
automatic header detection and column mapping.


Example:
    >>> from oscura.loaders.csv_loader import load_csv
    >>> trace = load_csv("oscilloscope_export.csv")
    >>> print(f"Sample rate: {trace.metadata.sample_rate} Hz")
"""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.core.exceptions import FormatError, LoaderError
from oscura.core.types import TraceMetadata, WaveformTrace

if TYPE_CHECKING:
    from os import PathLike

# Try to import pandas for better CSV handling
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# Common column names for time data
TIME_COLUMN_NAMES = [
    "time",
    "t",
    "time_s",
    "time_sec",
    "seconds",
    "timestamp",
    "x",
    "Time",
    "TIME",
]

# Common column names for voltage data
VOLTAGE_COLUMN_NAMES = [
    "voltage",
    "v",
    "volt",
    "volts",
    "amplitude",
    "signal",
    "y",
    "value",
    "data",
    "ch1",
    "ch2",
    "ch3",
    "ch4",
    "channel1",
    "channel2",
    "Voltage",
    "VOLTAGE",
]


def load_csv(
    path: str | PathLike[str],
    *,
    time_column: str | int | None = None,
    voltage_column: str | int | None = None,
    sample_rate: float | None = None,
    delimiter: str | None = None,
    skip_rows: int = 0,
    encoding: str = "utf-8",
    mmap: bool = False,
) -> WaveformTrace | Any:
    """Load waveform data from a CSV file.

    Parses CSV files exported from oscilloscopes or other data sources.
    Automatically detects header rows and maps columns for time and
    voltage data.

    Args:
        path: Path to the CSV file.
        time_column: Name or index of time column. If None, auto-detects.
        voltage_column: Name or index of voltage column. If None, auto-detects.
        sample_rate: Override sample rate. If None, computed from time column.
        delimiter: Column delimiter. If None, auto-detects.
        skip_rows: Number of rows to skip before header.
        encoding: File encoding (default: utf-8).
        mmap: If True, return memory-mapped trace for large files.

    Returns:
        WaveformTrace containing the waveform data and metadata.
        If mmap=True, returns MmapWaveformTrace instead.

    Raises:
        LoaderError: If the file cannot be loaded.

    Example:
        >>> trace = load_csv("oscilloscope.csv")
        >>> print(f"Sample rate: {trace.metadata.sample_rate} Hz")

        >>> # Specify columns explicitly
        >>> trace = load_csv("data.csv", time_column="Time", voltage_column="CH1")

        >>> # Load as memory-mapped for large files
        >>> trace = load_csv("huge_capture.csv", mmap=True)
    """
    path = Path(path)

    if not path.exists():
        raise LoaderError(
            "File not found",
            file_path=str(path),
        )

    if PANDAS_AVAILABLE:
        trace = _load_with_pandas(
            path,
            time_column=time_column,
            voltage_column=voltage_column,
            sample_rate=sample_rate,
            delimiter=delimiter,
            skip_rows=skip_rows,
            encoding=encoding,
        )
    else:
        trace = _load_basic(
            path,
            time_column=time_column,
            voltage_column=voltage_column,
            sample_rate=sample_rate,
            delimiter=delimiter,
            skip_rows=skip_rows,
            encoding=encoding,
        )

    # Convert to memory-mapped if requested
    if mmap:
        import tempfile

        from oscura.loaders.mmap_loader import load_mmap

        # Save data to temporary .npy file for memory mapping
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        np.save(tmp_path, trace.data)

        # Load as memory-mapped trace
        return load_mmap(
            tmp_path,
            sample_rate=trace.metadata.sample_rate,
        )

    return trace


def _load_with_pandas(
    path: Path,
    *,
    time_column: str | int | None,
    voltage_column: str | int | None,
    sample_rate: float | None,
    delimiter: str | None,
    skip_rows: int,
    encoding: str,
) -> WaveformTrace:
    """Load CSV using pandas for better parsing."""
    try:
        # Auto-detect delimiter if not specified
        if delimiter is None:
            delimiter = _detect_delimiter(path, encoding)

        # Read CSV with pandas
        df = pd.read_csv(
            path,
            delimiter=delimiter,
            skiprows=skip_rows,
            encoding=encoding,
            engine="python",  # More flexible parsing
        )

        if df.empty:
            raise FormatError(
                "CSV file is empty",
                file_path=str(path),
            )

        # Find time column
        time_data = None
        time_col_name = None

        if time_column is not None:
            if isinstance(time_column, int):
                if time_column < len(df.columns):
                    time_col_name = df.columns[time_column]
                    time_data = df.iloc[:, time_column].values
            elif time_column in df.columns:
                time_col_name = time_column
                time_data = df[time_column].values
        else:
            # Auto-detect time column
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in [n.lower() for n in TIME_COLUMN_NAMES]:
                    time_col_name = col
                    time_data = df[col].values
                    break

        # Find voltage column
        voltage_data = None
        voltage_col_name = None

        if voltage_column is not None:
            if isinstance(voltage_column, int):
                if voltage_column < len(df.columns):
                    voltage_col_name = df.columns[voltage_column]
                    voltage_data = df.iloc[:, voltage_column].values
            elif voltage_column in df.columns:
                voltage_col_name = voltage_column
                voltage_data = df[voltage_column].values
        else:
            # Auto-detect voltage column (first non-time numeric column)
            for col in df.columns:
                if col == time_col_name:
                    continue
                col_lower = col.lower().strip()
                # Check if numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Prefer columns with voltage-like names
                    if col_lower in [n.lower() for n in VOLTAGE_COLUMN_NAMES]:
                        voltage_col_name = col
                        voltage_data = df[col].values
                        break
                    elif voltage_data is None:
                        voltage_col_name = col
                        voltage_data = df[col].values

        if voltage_data is None:
            raise FormatError(
                "No voltage data found in CSV",
                file_path=str(path),
                expected="Numeric column for voltage data",
                got=f"Columns: {', '.join(df.columns)}",
            )

        # Convert to float64
        data = np.asarray(voltage_data, dtype=np.float64)

        # Determine sample rate
        detected_sample_rate = sample_rate
        if detected_sample_rate is None and time_data is not None:
            time_data = np.asarray(time_data, dtype=np.float64)
            if len(time_data) > 1:
                # Calculate sample rate from time intervals
                dt = np.median(np.diff(time_data))
                if dt > 0:
                    detected_sample_rate = 1.0 / dt

        if detected_sample_rate is None:
            detected_sample_rate = 1e6  # Default to 1 MSa/s

        # Build metadata
        metadata = TraceMetadata(
            sample_rate=detected_sample_rate,
            source_file=str(path),
            channel_name=voltage_col_name or "CH1",
        )

        return WaveformTrace(data=data, metadata=metadata)

    except pd.errors.ParserError as e:
        raise FormatError(
            "Failed to parse CSV file",
            file_path=str(path),
            details=str(e),
        ) from e
    except Exception as e:
        if isinstance(e, LoaderError | FormatError):
            raise
        raise LoaderError(
            "Failed to load CSV file",
            file_path=str(path),
            details=str(e),
        ) from e


def _load_basic(
    path: Path,
    *,
    time_column: str | int | None,
    voltage_column: str | int | None,
    sample_rate: float | None,
    delimiter: str | None,
    skip_rows: int,
    encoding: str,
) -> WaveformTrace:
    """Basic CSV loader without pandas."""
    try:
        with open(path, encoding=encoding) as f:
            # Skip rows
            for _ in range(skip_rows):
                next(f)

            content = f.read()

        # Auto-detect delimiter
        if delimiter is None:
            delimiter = _detect_delimiter_from_content(content)

        # Parse CSV
        reader = csv.reader(StringIO(content), delimiter=delimiter)
        rows = list(reader)

        if not rows:
            raise FormatError("CSV file is empty", file_path=str(path))

        # Detect header
        header = None
        data_start = 0
        first_row = rows[0]

        # Check if first row is a header (contains non-numeric values)
        is_header = False
        for cell in first_row:
            try:
                float(cell)
            except ValueError:
                if cell.strip():  # Non-empty, non-numeric
                    is_header = True
                    break

        if is_header:
            header = [cell.strip() for cell in first_row]
            data_start = 1

        # Determine column indices
        time_idx = None
        voltage_idx = None

        if header:
            # Find columns by name
            if time_column is not None:
                if isinstance(time_column, int):
                    time_idx = time_column
                elif time_column in header:
                    time_idx = header.index(time_column)
            else:
                # Auto-detect
                for i, col in enumerate(header):
                    if col.lower() in [n.lower() for n in TIME_COLUMN_NAMES]:
                        time_idx = i
                        break

            if voltage_column is not None:
                if isinstance(voltage_column, int):
                    voltage_idx = voltage_column
                elif voltage_column in header:
                    voltage_idx = header.index(voltage_column)
            else:
                # Auto-detect (first column that's not time)
                for i, col in enumerate(header):
                    if i == time_idx:
                        continue
                    if col.lower() in [n.lower() for n in VOLTAGE_COLUMN_NAMES]:
                        voltage_idx = i
                        break
                if voltage_idx is None:
                    voltage_idx = 1 if time_idx == 0 else 0
        else:
            # No header - use indices
            if isinstance(time_column, int):
                time_idx = time_column
            else:
                time_idx = 0  # Assume first column is time

            if isinstance(voltage_column, int):
                voltage_idx = voltage_column
            else:
                voltage_idx = 1  # Assume second column is voltage

        # Extract data
        time_data = []
        voltage_data = []

        for row in rows[data_start:]:
            if not row:
                continue
            try:
                if voltage_idx is not None and voltage_idx < len(row):
                    voltage_data.append(float(row[voltage_idx]))
                    if time_idx is not None and time_idx < len(row):
                        time_data.append(float(row[time_idx]))
            except (ValueError, IndexError):
                continue  # Skip malformed rows

        if not voltage_data:
            raise FormatError(
                "No valid voltage data found in CSV",
                file_path=str(path),
            )

        data = np.array(voltage_data, dtype=np.float64)

        # Determine sample rate
        detected_sample_rate = sample_rate
        if detected_sample_rate is None and time_data:
            time_arr = np.array(time_data, dtype=np.float64)
            if len(time_arr) > 1:
                dt = np.median(np.diff(time_arr))
                if dt > 0:
                    detected_sample_rate = 1.0 / dt

        if detected_sample_rate is None:
            detected_sample_rate = 1e6

        # Channel name
        channel_name = "CH1"
        if header and voltage_idx is not None and voltage_idx < len(header):
            channel_name = header[voltage_idx]

        metadata = TraceMetadata(
            sample_rate=detected_sample_rate,
            source_file=str(path),
            channel_name=channel_name,
        )

        return WaveformTrace(data=data, metadata=metadata)

    except Exception as e:
        if isinstance(e, LoaderError | FormatError):
            raise
        raise LoaderError(
            "Failed to load CSV file",
            file_path=str(path),
            details=str(e),
        ) from e


def _detect_delimiter(path: Path, encoding: str) -> str:
    """Detect the delimiter used in a CSV file."""
    try:
        with open(path, encoding=encoding) as f:
            sample = f.read(4096)
        return _detect_delimiter_from_content(sample)
    except Exception:
        return ","


def _detect_delimiter_from_content(content: str) -> str:
    """Detect delimiter from CSV content."""
    # Try common delimiters and count occurrences
    delimiters = [",", "\t", ";", "|", " "]
    counts: dict[str, int] = {}

    for delim in delimiters:
        counts[delim] = content.count(delim)

    # Return the most common delimiter
    if counts:
        return max(counts, key=lambda d: counts[d])
    return ","


__all__ = ["load_csv"]
