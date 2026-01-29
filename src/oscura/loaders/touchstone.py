"""Touchstone file loader for S-parameter data.

Supports .s1p through .s8p formats (Touchstone 1.0 and 2.0).

Example:
    >>> from oscura.loaders import load_touchstone
    >>> s_params = load_touchstone("cable.s2p")
    >>> print(f"Loaded {s_params.n_ports}-port, {len(s_params.frequencies)} points")

References:
    Touchstone 2.0 File Format Specification
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from oscura.core.exceptions import FormatError, LoaderError

# LAZY IMPORT: SParameterData is only needed at return time, not parse time
# This breaks the touchstone -> signal_integrity -> analyzer dependency chain
# See .claude/PERFORMANCE_PROFILE_2026-01-25.md for performance analysis
# Import time reduced from 879ms to <50ms by deferring this import
if TYPE_CHECKING:
    from oscura.analyzers.signal_integrity.sparams import SParameterData

# Module-level cache for lazy import
_SParameterData: type[SParameterData] | None = None


def _get_s_parameter_data_class() -> type[SParameterData]:
    """Get SParameterData class with lazy import.

    Returns:
        SParameterData class.
    """
    global _SParameterData
    if _SParameterData is None:
        from oscura.analyzers.signal_integrity.sparams import SParameterData as SPD

        _SParameterData = SPD
    # At this point _SParameterData cannot be None due to the if statement above
    return _SParameterData


def load_touchstone(path: str | Path) -> SParameterData:
    """Load S-parameter data from Touchstone file.

    Supports .s1p through .s8p formats and both Touchstone 1.0
    and 2.0 file formats.

    Args:
        path: Path to Touchstone file.

    Returns:
        SParameterData with loaded S-parameters.

    Raises:
        LoaderError: If file cannot be read.
        FormatError: If file format is invalid.

    Example:
        >>> s_params = load_touchstone("cable.s2p")
        >>> print(f"Loaded {s_params.n_ports}-port, {len(s_params.frequencies)} points")

    References:
        Touchstone 2.0 File Format Specification
    """
    path = Path(path)

    if not path.exists():
        raise LoaderError(f"File not found: {path}")

    # Determine number of ports from extension
    suffix = path.suffix.lower()
    match = re.match(r"\.s(\d+)p", suffix)
    if not match:
        raise FormatError(f"Unsupported file extension: {suffix}")

    n_ports = int(match.group(1))

    try:
        with open(path, encoding="utf-8", buffering=65536) as f:
            lines = f.readlines()
    except Exception as e:
        raise LoaderError(f"Failed to read file: {e}") from e

    return _parse_touchstone(lines, n_ports, str(path))


def _separate_touchstone_lines(lines: list[str]) -> tuple[list[str], str | None, list[str]]:
    """Separate file lines into comments, options, and data.

    Args:
        lines: All file lines.

    Returns:
        Tuple of (comments, option_line, data_lines).
    """
    comments = []
    option_line = None
    data_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("!"):
            comments.append(line[1:].strip())
        elif line.startswith("#"):
            option_line = line
        else:
            data_lines.append(line)

    return (comments, option_line, data_lines)


def _parse_touchstone_options(option_line: str | None) -> tuple[float, str, float]:
    """Parse option line to extract frequency unit, format, and impedance.

    Args:
        option_line: Option line starting with '#' or None.

    Returns:
        Tuple of (freq_unit multiplier, format_type, z0 impedance).
    """
    freq_unit = 1e9  # Default GHz
    format_type = "ma"  # Default MA (magnitude/angle)
    z0 = 50.0

    if not option_line:
        return (freq_unit, format_type, z0)

    option_line = option_line.lower()
    parts = option_line.split()

    for i, part in enumerate(parts):
        if part in ("hz", "khz", "mhz", "ghz"):
            freq_unit = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}[part]
        elif part in ("db", "ma", "ri"):
            format_type = part
        elif part == "r" and i + 1 < len(parts):
            # Reference impedance follows
            with contextlib.suppress(ValueError):
                z0 = float(parts[i + 1])

    return (freq_unit, format_type, z0)


def _is_new_frequency_line(parts: list[str]) -> bool:
    """Check if line represents a new frequency point.

    Args:
        parts: Split line parts.

    Returns:
        True if this is a new frequency line (odd number of numeric values).
    """
    try:
        float(parts[0])
        return len(parts) % 2 == 1
    except (ValueError, IndexError):
        return False


def _collect_s_parameter_values(
    data_lines: list[str], start_idx: int, n_s_params: int
) -> tuple[list[tuple[float, float]], int]:
    """Collect S-parameter value pairs for one frequency.

    Args:
        data_lines: All data lines.
        start_idx: Index of first line (contains frequency).
        n_s_params: Expected number of S-parameters.

    Returns:
        Tuple of (value pairs list, next line index).
    """
    s_values = []
    parts = data_lines[start_idx].split()

    # Add values from first line (skip frequency at parts[0])
    for j in range(1, len(parts), 2):
        if j + 1 < len(parts):
            val1 = float(parts[j])
            val2 = float(parts[j + 1])
            s_values.append((val1, val2))

    i = start_idx + 1

    # Continue collecting from subsequent lines if needed
    while len(s_values) < n_s_params and i < len(data_lines):
        parts = data_lines[i].split()

        # Stop if this is a new frequency line
        if _is_new_frequency_line(parts):
            break

        for j in range(0, len(parts), 2):
            if j + 1 < len(parts):
                val1 = float(parts[j])
                val2 = float(parts[j + 1])
                s_values.append((val1, val2))

        i += 1

    return (s_values, i)


def _convert_to_complex(val1: float, val2: float, format_type: str) -> complex:
    """Convert value pair to complex number based on format.

    Args:
        val1: First value (real/magnitude/dB).
        val2: Second value (imaginary/angle).
        format_type: Format type ("ri", "ma", or "db").

    Returns:
        Complex number representation.
    """
    if format_type == "ri":
        # Real/Imaginary
        return complex(val1, val2)
    elif format_type == "ma":
        # Magnitude/Angle (degrees)
        angle_rad = np.radians(val2)
        return complex(val1 * np.exp(1j * angle_rad))
    elif format_type == "db":
        # dB/Angle (degrees)
        mag = 10 ** (val1 / 20)
        angle_rad = np.radians(val2)
        return complex(mag * np.exp(1j * angle_rad))
    return complex(0, 0)


def _parse_touchstone(
    lines: list[str],
    n_ports: int,
    source_file: str,
) -> SParameterData:
    """Parse Touchstone file content.

    Args:
        lines: File lines.
        n_ports: Number of ports.
        source_file: Source file path.

    Returns:
        Parsed SParameterData.

    Raises:
        FormatError: If file format is invalid.
    """
    # Separate lines into components
    comments, option_line, data_lines = _separate_touchstone_lines(lines)

    # Parse options
    freq_unit, format_type, z0 = _parse_touchstone_options(option_line)

    # Parse data
    frequencies = []
    s_data = []
    n_s_params = n_ports * n_ports

    i = 0
    while i < len(data_lines):
        parts = data_lines[i].split()

        if len(parts) < 1:
            i += 1
            continue

        # Extract frequency
        freq = float(parts[0]) * freq_unit
        frequencies.append(freq)

        # Collect S-parameter values
        s_values, next_i = _collect_s_parameter_values(data_lines, i, n_s_params)

        # Convert to complex numbers
        s_complex = [_convert_to_complex(val1, val2, format_type) for val1, val2 in s_values]

        # Reshape into matrix
        if len(s_complex) == n_s_params:
            s_matrix = np.array(s_complex).reshape(n_ports, n_ports)
            s_data.append(s_matrix)

        i = next_i

    if len(frequencies) == 0:
        raise FormatError("No valid frequency points found")

    frequencies_arr = np.array(frequencies, dtype=np.float64)
    s_matrix_arr = np.array(s_data, dtype=np.complex128)

    # Lazy import of SParameterData class - only happens on first actual file load
    SParameterDataClass = _get_s_parameter_data_class()

    return SParameterDataClass(
        frequencies=frequencies_arr,
        s_matrix=s_matrix_arr,
        n_ports=n_ports,
        z0=z0,
        format=format_type,
        source_file=source_file,
        comments=comments,
    )


__all__ = ["load_touchstone"]
