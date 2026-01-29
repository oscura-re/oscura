"""Rich display integration for Jupyter notebooks.

This module provides rich HTML display for Oscura objects including
traces, measurements, and spectral data.

  - HTML tables for measurements
  - Inline plot rendering
  - Interactive result display

Example:
    In [1]: from oscura.jupyter.display import display_trace
    In [2]: display_trace(trace)  # Shows rich HTML summary
"""

from typing import Any

try:
    from IPython.display import HTML, SVG
    from IPython.display import display as ipython_display

    IPYTHON_AVAILABLE = True

    def display(*args: Any, **kwargs: Any) -> None:
        """Wrapper for IPython display."""
        ipython_display(*args, **kwargs)  # type: ignore[no-untyped-call]

except ImportError:
    IPYTHON_AVAILABLE = False

    class HTML:  # type: ignore[no-redef]
        """Fallback HTML class when IPython not available."""

        def __init__(self, data: str) -> None:
            self.data = data

    class SVG:  # type: ignore[no-redef]
        """Fallback SVG class when IPython not available."""

        def __init__(self, data: str) -> None:
            self.data = data

    def display(*args: Any, **kwargs: Any) -> None:
        """Fallback display when IPython not available."""
        for arg in args:
            print(arg)


class TraceDisplay:
    """Rich display wrapper for trace objects.

    Provides _repr_html_ for Jupyter notebook rendering.
    """

    def __init__(self, trace: Any, title: str = "Trace") -> None:
        """Initialize trace display.

        Args:
            trace: WaveformTrace or DigitalTrace object
            title: Display title
        """
        self.trace = trace
        self.title = title

    def _repr_html_(self) -> str:
        """Generate HTML representation for Jupyter."""
        trace = self.trace
        rows: list[tuple[str, str]] = []

        # Collect trace information
        self._add_basic_rows(trace, rows)
        self._add_metadata_rows(trace, rows)
        self._add_duration_row(trace, rows)
        self._add_statistics_rows(trace, rows)

        # Build HTML table
        return self._build_html_table(rows)

    def _add_basic_rows(self, trace: Any, rows: list[tuple[str, str]]) -> None:
        """Add basic trace information rows.

        Args:
            trace: Trace object.
            rows: List of (label, value) tuples.
        """
        if hasattr(trace, "data"):
            rows.append(("Samples", f"{len(trace.data):,}"))

    def _add_metadata_rows(self, trace: Any, rows: list[tuple[str, str]]) -> None:
        """Add metadata information rows.

        Args:
            trace: Trace object.
            rows: List of (label, value) tuples.
        """
        if not hasattr(trace, "metadata"):
            return

        meta = trace.metadata

        # Sample rate
        if hasattr(meta, "sample_rate") and meta.sample_rate:
            rate_str = self._format_sample_rate(meta.sample_rate)
            rows.append(("Sample Rate", rate_str))

        # Channel name
        if hasattr(meta, "channel_name") and meta.channel_name:
            rows.append(("Channel", meta.channel_name))

        # Source file
        if hasattr(meta, "source_file") and meta.source_file:
            rows.append(("Source", meta.source_file))

    def _format_sample_rate(self, rate: float) -> str:
        """Format sample rate with appropriate units.

        Args:
            rate: Sample rate in Hz.

        Returns:
            Formatted string with units.
        """
        if rate >= 1e9:
            return f"{rate / 1e9:.3f} GSa/s"
        elif rate >= 1e6:
            return f"{rate / 1e6:.3f} MSa/s"
        else:
            return f"{rate / 1e3:.3f} kSa/s"

    def _add_duration_row(self, trace: Any, rows: list[tuple[str, str]]) -> None:
        """Add duration information row.

        Args:
            trace: Trace object.
            rows: List of (label, value) tuples.
        """
        if not (hasattr(trace, "data") and hasattr(trace, "metadata")):
            return

        if not (hasattr(trace.metadata, "sample_rate") and trace.metadata.sample_rate):
            return

        duration = len(trace.data) / trace.metadata.sample_rate
        dur_str = self._format_duration(duration)
        rows.append(("Duration", dur_str))

    def _format_duration(self, duration: float) -> str:
        """Format duration with appropriate units.

        Args:
            duration: Duration in seconds.

        Returns:
            Formatted string with units.
        """
        if duration >= 1:
            return f"{duration:.3f} s"
        elif duration >= 1e-3:
            return f"{duration * 1e3:.3f} ms"
        elif duration >= 1e-6:
            return f"{duration * 1e6:.3f} us"
        else:
            return f"{duration * 1e9:.3f} ns"

    def _add_statistics_rows(self, trace: Any, rows: list[tuple[str, str]]) -> None:
        """Add data statistics rows.

        Args:
            trace: Trace object.
            rows: List of (label, value) tuples.
        """
        if not hasattr(trace, "data"):
            return

        import numpy as np

        data = trace.data
        rows.append(("Min", f"{np.min(data):.4g}"))
        rows.append(("Max", f"{np.max(data):.4g}"))
        rows.append(("Mean", f"{np.mean(data):.4g}"))
        rows.append(("Std Dev", f"{np.std(data):.4g}"))

    def _build_html_table(self, rows: list[tuple[str, str]]) -> str:
        """Build HTML table from rows.

        Args:
            rows: List of (label, value) tuples.

        Returns:
            HTML string.
        """
        html = f"""
<div style="border: 1px solid #ccc; border-radius: 4px; padding: 10px; max-width: 400px;">
    <h4 style="margin: 0 0 10px 0; color: #333;">{self.title}</h4>
    <table style="width: 100%; border-collapse: collapse;">
"""
        for label, value in rows:
            html += f"""
        <tr>
            <td style="padding: 4px 8px; border-bottom: 1px solid #eee; font-weight: bold; color: #666;">{label}</td>
            <td style="padding: 4px 8px; border-bottom: 1px solid #eee;">{value}</td>
        </tr>
"""
        html += """
    </table>
</div>
"""
        return html


class MeasurementDisplay:
    """Rich display wrapper for measurement results.

    Provides _repr_html_ for Jupyter notebook rendering.
    """

    def __init__(self, measurements: dict[str, Any], title: str = "Measurements") -> None:
        """Initialize measurement display.

        Args:
            measurements: Dictionary of measurement name -> value
            title: Display title
        """
        self.measurements = measurements
        self.title = title

    def _format_value(self, value: Any) -> str:
        """Format a measurement value with appropriate units."""
        if isinstance(value, float):
            # Determine scale and units for common measurements
            abs_val = abs(value)
            if abs_val == 0:
                return "0"
            elif abs_val >= 1e9:
                return f"{value / 1e9:.3f} G"
            elif abs_val >= 1e6:
                return f"{value / 1e6:.3f} M"
            elif abs_val >= 1e3:
                return f"{value / 1e3:.3f} k"
            elif abs_val >= 1:
                return f"{value:.4f}"
            elif abs_val >= 1e-3:
                return f"{value * 1e3:.3f} m"
            elif abs_val >= 1e-6:
                return f"{value * 1e6:.3f} u"
            elif abs_val >= 1e-9:
                return f"{value * 1e9:.3f} n"
            elif abs_val >= 1e-12:
                return f"{value * 1e12:.3f} p"
            else:
                return f"{value:.3e}"
        else:
            return str(value)

    def _repr_html_(self) -> str:
        """Generate HTML representation for Jupyter."""
        html = f"""
<div style="border: 1px solid #ccc; border-radius: 4px; padding: 10px; max-width: 500px;">
    <h4 style="margin: 0 0 10px 0; color: #333;">{self.title}</h4>
    <table style="width: 100%; border-collapse: collapse;">
        <tr style="background-color: #f5f5f5;">
            <th style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Measurement</th>
            <th style="padding: 8px; text-align: right; border-bottom: 2px solid #ddd;">Value</th>
        </tr>
"""
        for name, value in self.measurements.items():
            formatted = self._format_value(value)
            html += f"""
        <tr>
            <td style="padding: 6px 8px; border-bottom: 1px solid #eee;">{name}</td>
            <td style="padding: 6px 8px; border-bottom: 1px solid #eee; text-align: right; font-family: monospace;">{formatted}</td>
        </tr>
"""
        html += """
    </table>
</div>
"""
        return html


def display_trace(trace: Any, title: str = "Trace") -> None:
    """Display a trace with rich HTML formatting.

    Args:
        trace: WaveformTrace or DigitalTrace object
        title: Display title
    """
    wrapper = TraceDisplay(trace, title)
    if IPYTHON_AVAILABLE:
        display(HTML(wrapper._repr_html_()))  # type: ignore[no-untyped-call]
    else:
        print(wrapper._repr_html_())


def display_measurements(measurements: dict[str, Any], title: str = "Measurements") -> None:
    """Display measurements with rich HTML formatting.

    Args:
        measurements: Dictionary of measurement name -> value
        title: Display title
    """
    wrapper = MeasurementDisplay(measurements, title)
    if IPYTHON_AVAILABLE:
        display(HTML(wrapper._repr_html_()))  # type: ignore[no-untyped-call]
    else:
        for name, value in measurements.items():
            print(f"{name}: {value}")


def display_spectrum(
    frequencies: Any,
    magnitudes: Any,
    title: str = "Spectrum",
    log_scale: bool = True,
    figsize: tuple[int, int] = (10, 4),
) -> None:
    """Display a spectrum plot inline in Jupyter.

    Args:
        frequencies: Frequency array (Hz)
        magnitudes: Magnitude array (dB or linear)
        title: Plot title
        log_scale: Use log scale for x-axis
        figsize: Figure size tuple
    """
    import matplotlib.pyplot as plt
    import numpy as np

    _fig, ax = plt.subplots(figsize=figsize)

    if log_scale and np.min(frequencies[frequencies > 0]) > 0:
        ax.semilogx(frequencies, magnitudes)
    else:
        ax.plot(frequencies, magnitudes)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if IPYTHON_AVAILABLE:
        # Display inline
        plt.show()
    else:
        plt.show()
