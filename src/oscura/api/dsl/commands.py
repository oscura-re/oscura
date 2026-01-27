"""Oscura DSL Commands.

Built-in command implementations for DSL.
"""

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

    # Use the unified loader
    try:
        from oscura.loaders import load

        return load(str(path))

    except ImportError as e:
        raise OscuraError(f"Loader not available: {e}")


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
        from oscura.utils import filtering

        if filter_type.lower() == "lowpass":
            if len(args) < 1:
                raise OscuraError("lowpass filter requires cutoff frequency")
            return filtering.low_pass(trace, cutoff=args[0], **kwargs)

        elif filter_type.lower() == "highpass":
            if len(args) < 1:
                raise OscuraError("highpass filter requires cutoff frequency")
            return filtering.high_pass(trace, cutoff=args[0], **kwargs)

        elif filter_type.lower() == "bandpass":
            if len(args) < 2:
                raise OscuraError("bandpass filter requires low and high cutoff frequencies")
            return filtering.band_pass(trace, low=args[0], high=args[1], **kwargs)

        elif filter_type.lower() == "bandstop":
            if len(args) < 2:
                raise OscuraError("bandstop filter requires low and high cutoff frequencies")
            return filtering.band_stop(trace, low=args[0], high=args[1], **kwargs)

        else:
            raise OscuraError(f"Unknown filter type: {filter_type}")

    except ImportError:
        raise OscuraError("Filtering module not available")


def cmd_measure(trace: Any, *measurement_names: str) -> Any:
    """Measure properties of trace.

    Args:
        trace: Input trace
        *measurement_names: Measurement names (rise_time, fall_time, etc.)

    Returns:
        Measurement results (single value or dict)

    Raises:
        OscuraError: If measurement cannot be performed
    """
    try:
        from oscura.analyzers import measurements

        if len(measurement_names) == 0:
            raise OscuraError("measure command requires at least one measurement name")

        results = {}

        for measurement_name in measurement_names:
            meas_name = measurement_name.lower()

            if meas_name == "rise_time":
                results["rise_time"] = measurements.rise_time(trace)
            elif meas_name == "fall_time":
                results["fall_time"] = measurements.fall_time(trace)
            elif meas_name == "period":
                results["period"] = measurements.period(trace)
            elif meas_name == "frequency":
                results["frequency"] = measurements.frequency(trace)
            elif meas_name == "amplitude":
                results["amplitude"] = measurements.amplitude(trace)
            elif meas_name == "mean":
                results["mean"] = measurements.mean(trace)
            elif meas_name == "rms":
                results["rms"] = measurements.rms(trace)
            elif meas_name == "all":
                # Measure all available measurements
                results = measurements.measure(trace, parameters=None)
                break
            else:
                raise OscuraError(f"Unknown measurement: {measurement_name}")

        # Return single value if only one measurement
        if len(results) == 1:
            return next(iter(results.values()))

        return results

    except ImportError:
        raise OscuraError("Measurements module not available")


def cmd_plot(trace: Any, **options: Any) -> None:
    """Plot trace.

    Args:
        trace: Input trace
        **options: Plot options (title, annotate, etc.)

    Raises:
        OscuraError: If plotting fails
    """
    try:
        from oscura.visualization import plot

        title = options.get("title", "Trace Plot")
        annotate = options.get("annotate")

        plot.plot_trace(trace, title=title)

        if annotate:
            plot.add_annotation(annotate)

        # Import matplotlib.pyplot for show()
        import matplotlib.pyplot as plt

        plt.show()

    except ImportError:
        raise OscuraError("Visualization module not available")


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
        # Export functionality has been redesigned
        # Use oscura.export.* modules for protocol export
        raise NotImplementedError(
            "Data export has been redesigned. Use oscura.export.wireshark, "
            "oscura.export.kaitai_struct, or oscura.export.scapy_layer for protocol exports."
        )
    except ImportError:
        raise OscuraError("Export module not available")


def cmd_glob(pattern: str) -> list[str]:
    """Glob files matching pattern.

    Args:
        pattern: Glob pattern (*.csv, etc.)

    Returns:
        List of matching filenames
    """
    from glob import glob as glob_func

    return list(glob_func(pattern))


# Command registry
BUILTIN_COMMANDS = {
    "load": cmd_load,
    "filter": cmd_filter,
    "measure": cmd_measure,
    "plot": cmd_plot,
    "export": cmd_export,
    "glob": cmd_glob,
}
