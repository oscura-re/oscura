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
    """Load CSV using pandas for better parsing.

    Args:
        path: Path to CSV file.
        time_column: Name or index of time column (None for auto-detect).
        voltage_column: Name or index of voltage column (None for auto-detect).
        sample_rate: Override sample rate (None to compute from time column).
        delimiter: Column delimiter (None for auto-detect).
        skip_rows: Number of rows to skip before header.
        encoding: File encoding.

    Returns:
        WaveformTrace containing waveform data and metadata.

    Raises:
        FormatError: If CSV format is invalid or missing data.
        LoaderError: If file cannot be loaded.
    """
    try:
        # Read CSV with pandas
        delimiter = delimiter or _detect_delimiter(path, encoding)
        df = _read_csv_with_pandas(path, delimiter, skip_rows, encoding)

        # Find time and voltage columns
        time_data, time_col_name = _find_pandas_time_column(df, time_column)
        voltage_data, voltage_col_name = _find_pandas_voltage_column(
            df, voltage_column, time_col_name, path
        )

        # Compute sample rate from time data if not provided
        detected_sample_rate = _compute_sample_rate_from_array(sample_rate, time_data)

        # Create metadata and trace
        metadata = TraceMetadata(
            sample_rate=detected_sample_rate,
            source_file=str(path),
            channel_name=voltage_col_name or "CH1",
        )

        return WaveformTrace(data=np.asarray(voltage_data, dtype=np.float64), metadata=metadata)

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


def _read_csv_with_pandas(path: Path, delimiter: str, skip_rows: int, encoding: str) -> Any:
    """Read CSV file using pandas.

    Args:
        path: Path to CSV file.
        delimiter: Column delimiter.
        skip_rows: Number of rows to skip before header.
        encoding: File encoding.

    Returns:
        Pandas DataFrame.

    Raises:
        FormatError: If CSV is empty.
    """
    df = pd.read_csv(
        path,
        delimiter=delimiter,
        skiprows=skip_rows,
        encoding=encoding,
        engine="python",  # More flexible parsing
    )

    if df.empty:
        raise FormatError("CSV file is empty", file_path=str(path))

    return df


def _find_pandas_time_column(
    df: Any, time_column: str | int | None
) -> tuple[Any | None, str | None]:
    """Find time column in pandas DataFrame.

    Args:
        df: Pandas DataFrame.
        time_column: User-specified time column name or index.

    Returns:
        Tuple of (time_data, time_column_name).
    """
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

    return time_data, time_col_name


def _find_pandas_voltage_column(
    df: Any, voltage_column: str | int | None, time_col_name: str | None, path: Path
) -> tuple[Any, str | None]:
    """Find voltage column in pandas DataFrame.

    Args:
        df: Pandas DataFrame.
        voltage_column: User-specified voltage column name or index.
        time_col_name: Name of time column (to exclude from voltage search).
        path: Path to CSV file (for error reporting).

    Returns:
        Tuple of (voltage_data, voltage_column_name).

    Raises:
        FormatError: If no voltage data found.
    """
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

    return voltage_data, voltage_col_name


def _compute_sample_rate_from_array(sample_rate: float | None, time_data: Any | None) -> float:
    """Compute sample rate from numpy array or use override.

    Args:
        sample_rate: User-specified sample rate (None to compute).
        time_data: Numpy array of time values.

    Returns:
        Sample rate in Hz. Defaults to 1 MHz if cannot be computed.
    """
    if sample_rate is not None:
        return sample_rate

    if time_data is not None:
        time_arr = np.asarray(time_data, dtype=np.float64)
        if len(time_arr) > 1:
            dt = float(np.median(np.diff(time_arr)))
            if dt > 0:
                return 1.0 / dt

    return 1e6  # Default to 1 MSa/s


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
    """Basic CSV loader without pandas.

    Args:
        path: Path to CSV file.
        time_column: Name or index of time column (None for auto-detect).
        voltage_column: Name or index of voltage column (None for auto-detect).
        sample_rate: Override sample rate (None to compute from time column).
        delimiter: Column delimiter (None for auto-detect).
        skip_rows: Number of rows to skip before header.
        encoding: File encoding.

    Returns:
        WaveformTrace containing waveform data and metadata.

    Raises:
        FormatError: If CSV format is invalid or missing data.
        LoaderError: If file cannot be loaded.
    """
    try:
        # Read and parse CSV
        content = _read_file_content(path, skip_rows, encoding)
        delimiter = delimiter or _detect_delimiter_from_content(content)
        rows = _parse_csv_rows(content, delimiter, path)

        # Detect header and determine data start position
        header, data_start = _detect_header(rows)

        # Find column indices for time and voltage data
        time_idx, voltage_idx = _determine_column_indices(header, time_column, voltage_column)

        # Extract numeric data from rows
        time_data, voltage_data = _extract_data_from_rows(
            rows, data_start, time_idx, voltage_idx, path
        )

        # Calculate sample rate from time data if not provided
        detected_sample_rate = _compute_sample_rate(sample_rate, time_data)

        # Build channel name from header if available
        channel_name = _get_channel_name(header, voltage_idx)

        # Create metadata and trace
        metadata = TraceMetadata(
            sample_rate=detected_sample_rate,
            source_file=str(path),
            channel_name=channel_name,
        )

        return WaveformTrace(data=np.array(voltage_data, dtype=np.float64), metadata=metadata)

    except Exception as e:
        if isinstance(e, LoaderError | FormatError):
            raise
        raise LoaderError(
            "Failed to load CSV file",
            file_path=str(path),
            details=str(e),
        ) from e


def _read_file_content(path: Path, skip_rows: int, encoding: str) -> str:
    """Read file content after skipping specified rows.

    Args:
        path: Path to CSV file.
        skip_rows: Number of rows to skip.
        encoding: File encoding.

    Returns:
        File content as string.
    """
    with open(path, encoding=encoding, buffering=65536) as f:
        for _ in range(skip_rows):
            next(f)
        return f.read()


def _parse_csv_rows(content: str, delimiter: str, path: Path) -> list[list[str]]:
    """Parse CSV content into rows.

    Args:
        content: CSV file content.
        delimiter: Column delimiter.
        path: Path to CSV file (for error reporting).

    Returns:
        List of rows, where each row is a list of cell values.

    Raises:
        FormatError: If CSV is empty.
    """
    reader = csv.reader(StringIO(content), delimiter=delimiter)
    rows = list(reader)

    if not rows:
        raise FormatError("CSV file is empty", file_path=str(path))

    return rows


def _detect_header(rows: list[list[str]]) -> tuple[list[str] | None, int]:
    """Detect if first row is a header row.

    Args:
        rows: Parsed CSV rows.

    Returns:
        Tuple of (header row, data start index).
        If no header detected, returns (None, 0).
    """
    first_row = rows[0]

    # Check if first row contains non-numeric values (indicates header)
    for cell in first_row:
        try:
            float(cell)
        except ValueError:
            if cell.strip():  # Non-empty, non-numeric
                return [cell.strip() for cell in first_row], 1

    return None, 0


def _determine_column_indices(
    header: list[str] | None,
    time_column: str | int | None,
    voltage_column: str | int | None,
) -> tuple[int | None, int | None]:
    """Determine column indices for time and voltage data.

    Args:
        header: Header row if detected, None otherwise.
        time_column: User-specified time column name or index.
        voltage_column: User-specified voltage column name or index.

    Returns:
        Tuple of (time_index, voltage_index).
    """
    if header:
        time_idx = _find_time_column_index(header, time_column)
        voltage_idx = _find_voltage_column_index(header, voltage_column, time_idx)
    else:
        time_idx = _get_index_or_default(time_column, 0)
        voltage_idx = _get_index_or_default(voltage_column, 1)

    return time_idx, voltage_idx


def _find_time_column_index(header: list[str], time_column: str | int | None) -> int | None:
    """Find time column index from header.

    Args:
        header: Header row.
        time_column: User-specified time column name or index.

    Returns:
        Time column index, or None if not found.
    """
    if time_column is not None:
        if isinstance(time_column, int):
            return time_column
        if time_column in header:
            return header.index(time_column)
    else:
        # Auto-detect
        for i, col in enumerate(header):
            if col.lower() in [n.lower() for n in TIME_COLUMN_NAMES]:
                return i

    return None


def _find_voltage_column_index(
    header: list[str], voltage_column: str | int | None, time_idx: int | None
) -> int | None:
    """Find voltage column index from header.

    Args:
        header: Header row.
        voltage_column: User-specified voltage column name or index.
        time_idx: Time column index (to exclude from voltage search).

    Returns:
        Voltage column index.
    """
    if voltage_column is not None:
        if isinstance(voltage_column, int):
            return voltage_column
        if voltage_column in header:
            return header.index(voltage_column)
    else:
        # Auto-detect (first column that's not time)
        for i, col in enumerate(header):
            if i == time_idx:
                continue
            if col.lower() in [n.lower() for n in VOLTAGE_COLUMN_NAMES]:
                return i

        # Default: column 1 if time is 0, otherwise column 0
        return 1 if time_idx == 0 else 0

    return None


def _get_index_or_default(column: str | int | None, default: int) -> int:
    """Get column index or return default.

    Args:
        column: User-specified column (int or string).
        default: Default index if column is not an int.

    Returns:
        Column index.
    """
    if isinstance(column, int):
        return column
    return default


def _extract_data_from_rows(
    rows: list[list[str]],
    data_start: int,
    time_idx: int | None,
    voltage_idx: int | None,
    path: Path,
) -> tuple[list[float], list[float]]:
    """Extract numeric data from CSV rows.

    Args:
        rows: Parsed CSV rows.
        data_start: Index of first data row.
        time_idx: Time column index.
        voltage_idx: Voltage column index.
        path: Path to CSV file (for error reporting).

    Returns:
        Tuple of (time_data, voltage_data) lists.

    Raises:
        FormatError: If no valid voltage data found.
    """
    time_data: list[float] = []
    voltage_data: list[float] = []

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

    return time_data, voltage_data


def _compute_sample_rate(sample_rate: float | None, time_data: list[float]) -> float:
    """Compute sample rate from time data or use override.

    Args:
        sample_rate: User-specified sample rate (None to compute).
        time_data: List of time values.

    Returns:
        Sample rate in Hz. Defaults to 1 MHz if cannot be computed.
    """
    if sample_rate is not None:
        return sample_rate

    if time_data:
        time_arr = np.array(time_data, dtype=np.float64)
        if len(time_arr) > 1:
            dt = float(np.median(np.diff(time_arr)))
            if dt > 0:
                return 1.0 / dt

    return 1e6  # Default to 1 MSa/s


def _get_channel_name(header: list[str] | None, voltage_idx: int | None) -> str:
    """Get channel name from header or use default.

    Args:
        header: Header row if available.
        voltage_idx: Voltage column index.

    Returns:
        Channel name string.
    """
    if header and voltage_idx is not None and voltage_idx < len(header):
        return header[voltage_idx]
    return "CH1"


def _detect_delimiter(path: Path, encoding: str) -> str:
    """Detect the delimiter used in a CSV file."""
    try:
        with open(path, encoding=encoding, buffering=65536) as f:
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
