"""SPICE PWL export functionality for Oscura.

This module provides export to SPICE Piece-Wise Linear (PWL) format for
use in circuit simulation tools like LTspice, ngspice, Cadence, etc.


Example:
    >>> from oscura.exporters.spice_export import export_pwl
    >>> export_pwl(trace, "stimulus.pwl")
    >>> # Use in SPICE: V1 in 0 PWL file=stimulus.pwl
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.core.types import DigitalTrace, WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray


def export_pwl(
    data: WaveformTrace | DigitalTrace | NDArray[Any] | tuple[NDArray[Any], NDArray[Any]],
    path: str | Path,
    *,
    time_scale: float = 1.0,
    amplitude_scale: float = 1.0,
    time_offset: float = 0.0,
    amplitude_offset: float = 0.0,
    precision: int = 12,
    comment: str | None = None,
    downsample: int = 1,
    format_style: str = "ltspice",
) -> None:
    """Export data to SPICE PWL (Piece-Wise Linear) format.

    Creates a PWL file that can be used as a stimulus source in SPICE
    circuit simulators. The format consists of time-value pairs.

    Args:
        data: Data to export. Can be:
            - WaveformTrace or DigitalTrace
            - NumPy array (uses trace.time_vector or generates index-based time)
            - Tuple of (time_array, value_array)
        path: Output file path.
        time_scale: Scaling factor for time values (e.g., 1e-9 for ns).
        amplitude_scale: Scaling factor for amplitude values.
        time_offset: Offset to add to all time values.
        amplitude_offset: Offset to add to all amplitude values.
        precision: Decimal precision for output values.
        comment: Optional comment to include at top of file.
        downsample: Downsample factor to reduce file size (1 = no downsampling).
        format_style: Output format style:
            - "ltspice": LTspice compatible (time value pairs)
            - "ngspice": ngspice compatible (same as ltspice)
            - "hspice": HSPICE compatible (with header)

    Raises:
        TypeError: If data type is not supported.

    Example:
        >>> # Export as stimulus for simulation
        >>> export_pwl(trace, "input.pwl")
        >>> # In LTspice: V1 in 0 PWL file=input.pwl

        >>> # Scale time to nanoseconds for display
        >>> export_pwl(trace, "input.pwl", time_scale=1e9)

    References:
        EXP-005
    """
    path = Path(path)

    # Extract time and value arrays
    if isinstance(data, WaveformTrace | DigitalTrace):
        time = data.time_vector
        values = data.data
    elif isinstance(data, tuple) and len(data) == 2:
        time, values = data
    elif isinstance(data, np.ndarray):
        # Generate time based on array index (assume 1 unit per sample)
        time = np.arange(len(data), dtype=np.float64)
        values = data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    # Apply downsampling
    if downsample > 1:
        time = time[::downsample]
        values = values[::downsample]

    # Apply scaling and offset
    time = time * time_scale + time_offset
    values = values * amplitude_scale + amplitude_offset

    # Write to file
    with open(path, "w") as f:
        # Write comment/header
        if format_style == "hspice":
            f.write("* HSPICE PWL Data\n")
            if comment:
                f.write(f"* {comment}\n")
            f.write(f"* Points: {len(time)}\n")
            f.write("*\n")
        elif comment:
            f.write(f"; {comment}\n")

        # Write time-value pairs
        fmt = f"{{:.{precision}g}} {{:.{precision}g}}\n"
        f.writelines(fmt.format(t, v) for t, v in zip(time, values, strict=False))


def export_pwl_multi(
    traces: dict[str, WaveformTrace | DigitalTrace | NDArray[Any]],
    path: str | Path,
    *,
    time_scale: float = 1.0,
    amplitude_scale: float = 1.0,
    precision: int = 12,
    downsample: int = 1,
) -> None:
    """Export multiple traces to individual PWL files.

    Creates separate PWL files for each trace in the dictionary,
    with filenames based on the dictionary keys.

    Args:
        traces: Dictionary mapping signal names to trace data.
        path: Output directory path.
        time_scale: Scaling factor for time values.
        amplitude_scale: Scaling factor for amplitude values.
        precision: Decimal precision for output values.
        downsample: Downsample factor to reduce file size.

    Example:
        >>> traces = {
        ...     "clk": clock_trace,
        ...     "data": data_trace,
        ...     "reset": reset_trace,
        ... }
        >>> export_pwl_multi(traces, "stimuli/")
        >>> # Creates: stimuli/clk.pwl, stimuli/data.pwl, stimuli/reset.pwl

    References:
        EXP-005
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    for name, data in traces.items():
        # Sanitize filename
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        file_path = path / f"{safe_name}.pwl"

        export_pwl(
            data,
            file_path,
            time_scale=time_scale,
            amplitude_scale=amplitude_scale,
            precision=precision,
            downsample=downsample,
            comment=f"Signal: {name}",
        )


def generate_spice_source(
    pwl_path: str | Path,
    node_positive: str = "in",
    node_negative: str = "0",
    source_name: str = "V1",
    source_type: str = "voltage",
) -> str:
    """Generate SPICE source definition for a PWL file.

    Args:
        pwl_path: Path to PWL file.
        node_positive: Positive node name.
        node_negative: Negative node name (usually "0" for ground).
        source_name: Source instance name.
        source_type: Source type ("voltage" or "current").

    Returns:
        SPICE source definition string.

    Example:
        >>> line = generate_spice_source("input.pwl", "in", "0", "V1")
        >>> print(line)
        V1 in 0 PWL file=input.pwl

    References:
        EXP-005
    """
    prefix = "V" if source_type == "voltage" else "I"

    # Ensure source name starts with correct prefix
    if not source_name.upper().startswith(prefix):
        source_name = f"{prefix}{source_name}"

    return f"{source_name} {node_positive} {node_negative} PWL file={pwl_path}"


__all__ = [
    "export_pwl",
    "export_pwl_multi",
    "generate_spice_source",
]
