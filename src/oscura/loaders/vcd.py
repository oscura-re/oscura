"""IEEE 1364 VCD (Value Change Dump) file loader.

This module provides loading of VCD files, which are commonly used
for digital waveform data from logic analyzers and simulators.


Example:
    >>> from oscura.loaders.vcd import load_vcd
    >>> trace = load_vcd("simulation.vcd")
    >>> print(f"Sample rate: {trace.metadata.sample_rate} Hz")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from oscura.core.exceptions import FormatError, LoaderError
from oscura.core.types import DigitalTrace, TraceMetadata

if TYPE_CHECKING:
    from os import PathLike


@dataclass
class VCDVariable:
    """VCD variable definition.

    Attributes:
        var_type: Variable type (wire, reg, etc.).
        size: Bit width of the variable.
        identifier: Single-character identifier code.
        name: Human-readable variable name.
        scope: Hierarchical scope path.
    """

    var_type: str
    size: int
    identifier: str
    name: str
    scope: str = ""


@dataclass
class VCDHeader:
    """Parsed VCD file header information.

    Attributes:
        timescale: Timescale in seconds (e.g., 1e-9 for 1ns).
        variables: Dictionary mapping identifier to VCDVariable.
        date: Date string from header.
        version: VCD version string.
        comment: Comment from header.
    """

    timescale: float = 1e-9  # Default 1ns
    variables: dict[str, VCDVariable] = field(default_factory=dict)
    date: str = ""
    version: str = ""
    comment: str = ""


def load_vcd(
    path: str | PathLike[str],
    *,
    signal: str | None = None,
    sample_rate: float | None = None,
) -> DigitalTrace:
    """Load an IEEE 1364 VCD (Value Change Dump) file.

    VCD files contain digital waveform data with value changes and
    timestamps. This loader converts the event-based format to a
    sampled digital trace.

    Args:
        path: Path to the VCD file.
        signal: Optional signal name to load. If None, loads the
            first signal found.
        sample_rate: Sample rate for conversion to sampled data.
            If None, automatically determined from timescale.

    Returns:
        DigitalTrace containing the digital signal data and metadata.

    Raises:
        LoaderError: If the file cannot be loaded.
        FormatError: If the file is not a valid VCD file.

    Example:
        >>> trace = load_vcd("simulation.vcd", signal="clk")
        >>> print(f"Duration: {trace.duration:.6f} seconds")
        >>> print(f"Edges: {len(trace.edges or [])}")

    References:
        IEEE 1364-2005: Verilog Hardware Description Language
    """
    path = Path(path)

    if not path.exists():
        raise LoaderError(
            "File not found",
            file_path=str(path),
        )

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Parse header
        header = _parse_vcd_header(content, path)

        if not header.variables:
            raise FormatError(
                "No variables found in VCD file",
                file_path=str(path),
                expected="At least one $var definition",
            )

        # Select signal to load
        if signal is not None:
            # Find by name
            target_var = None
            for var in header.variables.values():
                if signal in (var.name, var.identifier):
                    target_var = var
                    break
            if target_var is None:
                available = [v.name for v in header.variables.values()]
                raise LoaderError(
                    f"Signal '{signal}' not found",
                    file_path=str(path),
                    details=f"Available signals: {available}",
                )
        else:
            # Use first variable
            target_var = next(iter(header.variables.values()))

        # Parse value changes
        changes = _parse_value_changes(content, target_var.identifier)

        if not changes:
            raise FormatError(
                f"No value changes found for signal '{target_var.name}'",
                file_path=str(path),
            )

        # Determine sample rate and convert to sampled data
        if sample_rate is None:
            # Auto-determine from timescale and value changes
            sample_rate = _determine_sample_rate(changes, header.timescale)

        # Convert to sampled digital trace
        data, edges = _changes_to_samples(
            changes,
            header.timescale,
            sample_rate,
        )

        # Build metadata
        metadata = TraceMetadata(
            sample_rate=sample_rate,
            source_file=str(path),
            channel_name=target_var.name,
            trigger_info={
                "timescale": header.timescale,
                "var_type": target_var.var_type,
                "bit_width": target_var.size,
            },
        )

        return DigitalTrace(
            data=data.astype(np.bool_),  # type: ignore[arg-type]
            metadata=metadata,
            edges=edges,
        )

    except UnicodeDecodeError as e:
        raise FormatError(
            "VCD file contains invalid characters",
            file_path=str(path),
            expected="UTF-8 or ASCII text",
        ) from e
    except Exception as e:
        if isinstance(e, LoaderError | FormatError):
            raise
        raise LoaderError(
            "Failed to load VCD file",
            file_path=str(path),
            details=str(e),
            fix_hint="Ensure the file is a valid IEEE 1364 VCD format.",
        ) from e


def _parse_vcd_header(content: str, path: Path) -> VCDHeader:
    """Parse VCD file header section.

    Args:
        content: Full VCD file content.
        path: Path for error messages.

    Returns:
        Parsed VCDHeader object.

    Raises:
        FormatError: If VCD header is invalid.
    """
    header = VCDHeader()
    current_scope: list[str] = []

    # Find header section (before $enddefinitions)
    end_def_match = re.search(r"\$enddefinitions\s+\$end", content)
    if not end_def_match:
        raise FormatError(
            "Invalid VCD file: missing $enddefinitions",
            file_path=str(path),
        )

    header_content = content[: end_def_match.end()]

    # Parse timescale
    timescale_match = re.search(r"\$timescale\s+(\d+)\s*(s|ms|us|ns|ps|fs)\s+\$end", header_content)
    if timescale_match:
        value = int(timescale_match.group(1))
        unit = timescale_match.group(2)
        unit_multipliers = {
            "s": 1.0,
            "ms": 1e-3,
            "us": 1e-6,
            "ns": 1e-9,
            "ps": 1e-12,
            "fs": 1e-15,
        }
        header.timescale = value * unit_multipliers.get(unit, 1e-9)

    # Parse date
    date_match = re.search(r"\$date\s+(.*?)\s*\$end", header_content, re.DOTALL)
    if date_match:
        header.date = date_match.group(1).strip()

    # Parse version
    version_match = re.search(r"\$version\s+(.*?)\s*\$end", header_content, re.DOTALL)
    if version_match:
        header.version = version_match.group(1).strip()

    # Parse comment
    comment_match = re.search(r"\$comment\s+(.*?)\s*\$end", header_content, re.DOTALL)
    if comment_match:
        header.comment = comment_match.group(1).strip()

    # Parse scopes and variables
    scope_pattern = re.compile(r"\$scope\s+(\w+)\s+(\w+)\s+\$end")
    upscope_pattern = re.compile(r"\$upscope\s+\$end")
    var_pattern = re.compile(r"\$var\s+(\w+)\s+(\d+)\s+(\S+)\s+(\S+)(?:\s+\[.*?\])?\s+\$end")

    pos = 0
    while pos < len(header_content):
        # Check for scope
        scope_match = scope_pattern.match(header_content, pos)
        if scope_match:
            current_scope.append(scope_match.group(2))
            pos = scope_match.end()
            continue

        # Check for upscope
        upscope_match = upscope_pattern.match(header_content, pos)
        if upscope_match:
            if current_scope:
                current_scope.pop()
            pos = upscope_match.end()
            continue

        # Check for variable
        var_match = var_pattern.match(header_content, pos)
        if var_match:
            var = VCDVariable(
                var_type=var_match.group(1),
                size=int(var_match.group(2)),
                identifier=var_match.group(3),
                name=var_match.group(4),
                scope=".".join(current_scope),
            )
            header.variables[var.identifier] = var
            pos = var_match.end()
            continue

        pos += 1

    return header


def _parse_value_changes(
    content: str,
    identifier: str,
) -> list[tuple[int, str]]:
    """Parse value changes for a specific signal.

    Args:
        content: Full VCD file content.
        identifier: Signal identifier to track.

    Returns:
        List of (timestamp, value) tuples.
    """
    changes: list[tuple[int, str]] = []
    current_time = 0

    # Find data section (after $enddefinitions)
    end_def_match = re.search(r"\$enddefinitions\s+\$end", content)
    if not end_def_match:
        return changes

    data_content = content[end_def_match.end() :]

    # Parse line by line
    for line in data_content.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Timestamp
        if line.startswith("#"):
            try:
                current_time = int(line[1:])
            except ValueError:
                continue

        # Binary value change: 0x, 1x, xx, zx (single bit)
        elif line[0] in "01xXzZ" and len(line) >= 2:
            value = line[0]
            var_id = line[1:]
            if var_id == identifier:
                changes.append((current_time, value))

        # Multi-bit value: bVALUE IDENTIFIER or BVALUE IDENTIFIER
        elif line[0] in "bB" or line[0] in "rR":
            parts = line[1:].split()
            if len(parts) >= 2:
                value = parts[0]
                var_id = parts[1]
                if var_id == identifier:
                    changes.append((current_time, value))

    return changes


def _determine_sample_rate(
    changes: list[tuple[int, str]],
    timescale: float,
) -> float:
    """Determine appropriate sample rate from value changes.

    Args:
        changes: List of (timestamp, value) tuples.
        timescale: VCD timescale in seconds.

    Returns:
        Sample rate in Hz.
    """
    if len(changes) < 2:
        # Default to 1 MHz if not enough data
        return 1e6

    # Calculate minimum time interval between changes
    timestamps = sorted({t for t, _ in changes})
    if len(timestamps) < 2:
        return 1e6

    min_interval = min(timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1))

    if min_interval <= 0:
        return 1e6

    # Convert to seconds and set sample rate for ~10 samples per interval
    interval_seconds = min_interval * timescale
    sample_rate = 10.0 / interval_seconds

    # Clamp to reasonable range
    sample_rate = max(1e3, min(1e12, sample_rate))

    return sample_rate


def _changes_to_samples(
    changes: list[tuple[int, str]],
    timescale: float,
    sample_rate: float,
) -> tuple[NDArray[np.bool_], list[tuple[float, bool]]]:
    """Convert value changes to sampled data.

    Args:
        changes: List of (timestamp, value) tuples.
        timescale: VCD timescale in seconds.
        sample_rate: Target sample rate in Hz.

    Returns:
        Tuple of (data array, edges list).
    """
    if not changes:
        return np.array([], dtype=np.bool_), []

    # Sort changes by timestamp
    changes = sorted(changes, key=lambda x: x[0])

    # Get time range
    start_time = changes[0][0]
    end_time = changes[-1][0]

    # Calculate number of samples
    duration_seconds = (end_time - start_time) * timescale
    n_samples = max(1, int(duration_seconds * sample_rate) + 1)

    # Initialize data array
    data = np.zeros(n_samples, dtype=np.bool_)
    edges: list[tuple[float, bool]] = []

    # Convert values to boolean (for single-bit) or LSB (for multi-bit)
    def value_to_bool(val: str) -> bool:
        """Convert VCD value to boolean."""
        val = val.lower()
        if val in ("1", "h"):
            return True
        if val in ("0", "l"):
            return False
        # For multi-bit, check LSB
        return bool(val and val[-1] in ("1", "h"))

    # Fill samples based on value changes
    prev_value = False
    for i, (timestamp, value) in enumerate(changes):
        current_value = value_to_bool(value)

        # Calculate sample index
        time_seconds = (timestamp - start_time) * timescale
        sample_idx = int(time_seconds * sample_rate)

        # Calculate next change sample index
        if i + 1 < len(changes):
            next_time_seconds = (changes[i + 1][0] - start_time) * timescale
            next_sample_idx = int(next_time_seconds * sample_rate)
        else:
            next_sample_idx = n_samples

        # Fill samples
        sample_idx = max(0, min(sample_idx, n_samples - 1))
        next_sample_idx = max(0, min(next_sample_idx, n_samples))
        data[sample_idx:next_sample_idx] = current_value

        # Record edge
        if current_value != prev_value:
            edge_time = time_seconds
            is_rising = current_value
            edges.append((edge_time, is_rising))

        prev_value = current_value

    return data, edges


__all__ = ["load_vcd"]
