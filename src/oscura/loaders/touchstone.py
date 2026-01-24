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

import numpy as np

from oscura.analyzers.signal_integrity.sparams import SParameterData
from oscura.core.exceptions import FormatError, LoaderError


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
        with open(path) as f:
            lines = f.readlines()
    except Exception as e:
        raise LoaderError(f"Failed to read file: {e}")  # noqa: B904

    return _parse_touchstone(lines, n_ports, str(path))


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

    # Parse option line
    freq_unit = 1e9  # Default GHz
    format_type = "ma"  # Default MA (magnitude/angle)
    z0 = 50.0

    if option_line:
        option_line = option_line.lower()
        parts = option_line.split()

        for i, part in enumerate(parts):
            if part in ("hz", "khz", "mhz", "ghz"):
                freq_unit = {
                    "hz": 1.0,
                    "khz": 1e3,
                    "mhz": 1e6,
                    "ghz": 1e9,
                }[part]
            elif part in ("db", "ma", "ri"):
                format_type = part
            elif part == "r":
                # Reference impedance follows
                if i + 1 < len(parts):
                    with contextlib.suppress(ValueError):
                        z0 = float(parts[i + 1])

    # Parse data
    frequencies = []
    s_data = []

    # Number of S-parameters per frequency
    n_s_params = n_ports * n_ports

    i = 0
    while i < len(data_lines):
        # First line has frequency and first S-parameters
        parts = data_lines[i].split()

        if len(parts) < 1:
            i += 1
            continue

        freq = float(parts[0]) * freq_unit
        frequencies.append(freq)

        # Collect all S-parameter values for this frequency
        s_values = []

        # Add values from first line
        for j in range(1, len(parts), 2):
            if j + 1 < len(parts):
                val1 = float(parts[j])
                val2 = float(parts[j + 1])
                s_values.append((val1, val2))

        i += 1

        # Continue collecting from subsequent lines if needed
        while len(s_values) < n_s_params and i < len(data_lines):
            parts = data_lines[i].split()

            # Check if this is a new frequency (has odd number of values)
            try:
                float(parts[0])
                if len(parts) % 2 == 1:
                    break  # New frequency line
            except (ValueError, IndexError):
                pass  # Skip lines that can't be parsed as numeric data

            for j in range(0, len(parts), 2):
                if j + 1 < len(parts):
                    val1 = float(parts[j])
                    val2 = float(parts[j + 1])
                    s_values.append((val1, val2))

            i += 1

        # Convert to complex based on format
        s_complex = []
        for val1, val2 in s_values:
            if format_type == "ri":
                # Real/Imaginary
                s_complex.append(complex(val1, val2))
            elif format_type == "ma":
                # Magnitude/Angle (degrees)
                mag = val1
                angle_rad = np.radians(val2)
                s_complex.append(mag * np.exp(1j * angle_rad))
            elif format_type == "db":
                # dB/Angle (degrees)
                mag = 10 ** (val1 / 20)
                angle_rad = np.radians(val2)
                s_complex.append(mag * np.exp(1j * angle_rad))

        # Reshape into matrix
        if len(s_complex) == n_s_params:
            s_matrix = np.array(s_complex).reshape(n_ports, n_ports)
            s_data.append(s_matrix)

    if len(frequencies) == 0:
        raise FormatError("No valid frequency points found")

    frequencies_arr = np.array(frequencies, dtype=np.float64)
    s_matrix_arr = np.array(s_data, dtype=np.complex128)

    return SParameterData(
        frequencies=frequencies_arr,
        s_matrix=s_matrix_arr,
        n_ports=n_ports,
        z0=z0,
        format=format_type,
        source_file=source_file,
        comments=comments,
    )


__all__ = ["load_touchstone"]
