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
    from IPython.display import HTML, SVG, display

    IPYTHON_AVAILABLE = True
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

    def display(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef,misc]
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

        # Build info rows
        rows = []

        if hasattr(trace, "data"):
            rows.append(("Samples", f"{len(trace.data):,}"))

        if hasattr(trace, "metadata"):
            meta = trace.metadata
            if hasattr(meta, "sample_rate") and meta.sample_rate:
                rate = meta.sample_rate
                if rate >= 1e9:
                    rate_str = f"{rate / 1e9:.3f} GSa/s"
                elif rate >= 1e6:
                    rate_str = f"{rate / 1e6:.3f} MSa/s"
                else:
                    rate_str = f"{rate / 1e3:.3f} kSa/s"
                rows.append(("Sample Rate", rate_str))

            if hasattr(meta, "channel_name") and meta.channel_name:
                rows.append(("Channel", meta.channel_name))

            if hasattr(meta, "source_file") and meta.source_file:
                rows.append(("Source", meta.source_file))

        # Calculate duration if possible
        if hasattr(trace, "data") and hasattr(trace, "metadata"):
            if hasattr(trace.metadata, "sample_rate") and trace.metadata.sample_rate:
                duration = len(trace.data) / trace.metadata.sample_rate
                if duration >= 1:
                    dur_str = f"{duration:.3f} s"
                elif duration >= 1e-3:
                    dur_str = f"{duration * 1e3:.3f} ms"
                elif duration >= 1e-6:
                    dur_str = f"{duration * 1e6:.3f} us"
                else:
                    dur_str = f"{duration * 1e9:.3f} ns"
                rows.append(("Duration", dur_str))

        # Data statistics
        if hasattr(trace, "data"):
            import numpy as np

            data = trace.data
            rows.append(("Min", f"{np.min(data):.4g}"))
            rows.append(("Max", f"{np.max(data):.4g}"))
            rows.append(("Mean", f"{np.mean(data):.4g}"))
            rows.append(("Std Dev", f"{np.std(data):.4g}"))

        # Build HTML table
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
