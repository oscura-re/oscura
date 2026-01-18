"""Oscura DSL Commands.

Built-in command implementations for DSL.
"""

import sys
from pathlib import Path
from typing import Any

from oscura.core.exceptions import OscuraError


def cmd_load(filename: str) -> Any:
    """Load a trace file.

    Args:
        filename: Path to trace file

    Returns:
        Loaded trace object

    Raises:
        OscuraError: If file cannot be loaded
    """
    path = Path(filename)

    if not path.exists():
        raise OscuraError(f"File not found: {filename}")

    # Determine loader based on extension
    ext = path.suffix.lower()

    try:
        if ext == ".csv":
            from oscura.loaders.csv import (  # type: ignore[import-not-found]
                load_csv,  # type: ignore[import-not-found]
            )

            return load_csv(str(path))
        elif ext == ".bin":
            from oscura.loaders.binary import (  # type: ignore[import-not-found]
                load_binary,  # type: ignore[import-not-found]
            )

            return load_binary(str(path))
        elif ext in (".h5", ".hdf5"):
            from oscura.loaders.hdf5 import (  # type: ignore[import-not-found]
                load_hdf5,  # type: ignore[import-not-found]
            )

            return load_hdf5(str(path))
        else:
            raise OscuraError(f"Unsupported file format: {ext}")

    except ImportError as e:
        raise OscuraError(f"Loader not available for {ext}: {e}")  # noqa: B904


def cmd_filter(trace: Any, filter_type: str, *args: Any, **kwargs: Any) -> Any:
    """Apply filter to trace.

    Args:
        trace: Input trace
        filter_type: Filter type (lowpass, highpass, bandpass, bandstop)
        *args: Filter parameters (cutoff frequency, etc.)
        **kwargs: Additional filter options

    Returns:
        Filtered trace

    Raises:
        OscuraError: If filter cannot be applied
    """
    try:
        from oscura.filtering import filters  # type: ignore[attr-defined]

        if filter_type.lower() == "lowpass":
            if len(args) < 1:
                raise OscuraError("lowpass filter requires cutoff frequency")
            return filters.low_pass(trace, cutoff=args[0], **kwargs)

        elif filter_type.lower() == "highpass":
            if len(args) < 1:
                raise OscuraError("highpass filter requires cutoff frequency")
            return filters.high_pass(trace, cutoff=args[0], **kwargs)

        elif filter_type.lower() == "bandpass":
            if len(args) < 2:
                raise OscuraError("bandpass filter requires low and high cutoff frequencies")
            return filters.band_pass(trace, low=args[0], high=args[1], **kwargs)

        elif filter_type.lower() == "bandstop":
            if len(args) < 2:
                raise OscuraError("bandstop filter requires low and high cutoff frequencies")
            return filters.band_stop(trace, low=args[0], high=args[1], **kwargs)

        else:
            raise OscuraError(f"Unknown filter type: {filter_type}")

    except ImportError:
        raise OscuraError("Filtering module not available")  # noqa: B904


def cmd_measure(trace: Any, *measurements: str) -> Any:
    """Measure properties of trace.

    Args:
        trace: Input trace
        *measurements: Measurement names (rise_time, fall_time, etc.)

    Returns:
        Measurement results (single value or dict)

    Raises:
        OscuraError: If measurement cannot be performed
    """
    try:
        from oscura.analyzers import (  # type: ignore[attr-defined]
            measurements as meas,  # type: ignore[attr-defined]
        )

        if len(measurements) == 0:
            raise OscuraError("measure command requires at least one measurement name")

        results = {}

        for measurement in measurements:
            meas_name = measurement.lower()

            if meas_name == "rise_time":
                results["rise_time"] = meas.rise_time(trace)
            elif meas_name == "fall_time":
                results["fall_time"] = meas.fall_time(trace)
            elif meas_name == "period":
                results["period"] = meas.period(trace)
            elif meas_name == "frequency":
                results["frequency"] = meas.frequency(trace)
            elif meas_name == "amplitude":
                results["amplitude"] = meas.amplitude(trace)
            elif meas_name == "mean":
                results["mean"] = meas.mean(trace)
            elif meas_name == "rms":
                results["rms"] = meas.rms(trace)
            elif meas_name == "all":
                # Measure all available measurements
                results = meas.measure_all(trace)
                break
            else:
                raise OscuraError(f"Unknown measurement: {measurement}")

        # Return single value if only one measurement
        if len(results) == 1:
            return next(iter(results.values()))

        return results

    except ImportError:
        raise OscuraError("Measurements module not available")  # noqa: B904


def cmd_plot(trace: Any, **options: Any) -> None:
    """Plot trace.

    Args:
        trace: Input trace
        **options: Plot options (title, annotate, etc.)

    Raises:
        OscuraError: If plotting fails
    """
    try:
        from oscura.visualization import (  # type: ignore[attr-defined]
            plot as plot_module,  # type: ignore[attr-defined]
        )

        title = options.get("title", "Trace Plot")
        annotate = options.get("annotate")

        plot_module.plot_trace(trace, title=title)

        if annotate:
            plot_module.add_annotation(annotate)

        plot_module.show()

    except ImportError:
        raise OscuraError("Visualization module not available")  # noqa: B904


def cmd_export(data: Any, format_type: str, filename: str | None = None) -> None:
    """Export data to file.

    Args:
        data: Data to export (trace, measurements, etc.)
        format_type: Export format (json, csv, hdf5)
        filename: Output filename (optional, auto-generated if None)

    Raises:
        OscuraError: If export fails
    """
    try:
        from oscura.exporters import exporters  # type: ignore[attr-defined]

        if filename is None:
            filename = f"export.{format_type}"

        fmt = format_type.lower()

        if fmt == "json":
            exporters.json(data, filename)
        elif fmt == "csv":
            exporters.csv(data, filename)
        elif fmt in ("h5", "hdf5"):
            exporters.hdf5(data, filename)
        else:
            raise OscuraError(f"Unknown export format: {format_type}")

        print(f"Exported to {filename}", file=sys.stderr)

    except ImportError:
        raise OscuraError("Export module not available")  # noqa: B904


def cmd_glob(pattern: str) -> list[str]:
    """Glob files matching pattern.

    Args:
        pattern: Glob pattern (*.csv, etc.)

    Returns:
        List of matching filenames
    """
    from glob import glob as glob_func

    return list(glob_func(pattern))  # noqa: PTH207


# Command registry
BUILTIN_COMMANDS = {
    "load": cmd_load,
    "filter": cmd_filter,
    "measure": cmd_measure,
    "plot": cmd_plot,
    "export": cmd_export,
    "glob": cmd_glob,
}
