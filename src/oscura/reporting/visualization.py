"""Professional plot generation for reports with IEEE compliance.

This module extends PlotGenerator with comprehensive plot types for
signal analysis including waveforms, FFT, PSD, spectrograms, eye diagrams,
histograms, jitter analysis, and power plots.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# IEEE compliant color scheme
IEEE_COLORS = {
    "primary": "#003f87",  # IEEE blue
    "secondary": "#00629b",
    "accent": "#009fdf",
    "success": "#27ae60",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "grid": "#cccccc",
    "text": "#2c3e50",
}


class PlotStyler:
    """Apply consistent IEEE-compliant styling to plots.

    Example:
        >>> styler = PlotStyler()
        >>> fig, ax = plt.subplots()
        >>> styler.apply_ieee_style(ax, "Time (s)", "Voltage (V)", "Waveform")
    """

    @staticmethod
    def apply_ieee_style(
        ax: Any,
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        grid: bool = True,
    ) -> None:
        """Apply IEEE-compliant styling to matplotlib axes.

        Args:
            ax: Matplotlib axes object.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            title: Plot title.
            grid: Whether to show grid.

        Example:
            >>> fig, ax = plt.subplots()
            >>> PlotStyler.apply_ieee_style(ax, "Time", "Voltage", "Signal")
        """
        if not HAS_MATPLOTLIB:
            return

        # Labels and title
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10, fontweight="normal")
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10, fontweight="normal")
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

        # Grid
        if grid:
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, color=IEEE_COLORS["grid"])

        # Spines
        for spine in ax.spines.values():
            spine.set_color(IEEE_COLORS["text"])
            spine.set_linewidth(0.8)

        # Ticks
        ax.tick_params(labelsize=9, colors=IEEE_COLORS["text"])

        # Tight layout
        ax.figure.tight_layout()


class IEEEPlotGenerator:
    """Generate IEEE-compliant plots for signal analysis reports.

    Provides comprehensive plot types with consistent styling and
    proper axis labels, units, and annotations.

    Example:
        >>> generator = IEEEPlotGenerator()
        >>> fig = generator.plot_waveform(time, voltage, title="Input Signal")
        >>> base64_img = generator.figure_to_base64(fig)
    """

    def __init__(self, dpi: int = 150, figsize: tuple[int, int] = (10, 6)) -> None:
        """Initialize plot generator.

        Args:
            dpi: Resolution in dots per inch.
            figsize: Figure size in inches (width, height).

        Raises:
            ImportError: If matplotlib is not installed.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for plot generation")

        self.dpi = dpi
        self.figsize = figsize
        self.styler = PlotStyler()

    def plot_waveform(
        self,
        time: NDArray[np.floating[Any]],
        signal: NDArray[np.floating[Any]],
        title: str = "Waveform",
        xlabel: str = "Time (s)",
        ylabel: str = "Amplitude",
        markers: dict[str, float] | None = None,
    ) -> Figure:
        """Plot time-series waveform.

        Args:
            time: Time array in seconds.
            signal: Signal amplitude array.
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            markers: Optional dict of marker labels to time positions.

        Returns:
            Matplotlib Figure object.

        Example:
            >>> t = np.linspace(0, 1, 1000)
            >>> s = np.sin(2 * np.pi * 10 * t)
            >>> fig = generator.plot_waveform(t, s, "Sine Wave")
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot signal
        ax.plot(time, signal, color=IEEE_COLORS["primary"], linewidth=1.5, label="Signal")

        # Add markers if provided
        if markers:
            for label, pos in markers.items():
                ax.axvline(pos, color=IEEE_COLORS["accent"], linestyle="--", alpha=0.7)
                ax.text(
                    pos,
                    ax.get_ylim()[1] * 0.9,
                    label,
                    rotation=90,
                    verticalalignment="top",
                    fontsize=8,
                )

        self.styler.apply_ieee_style(ax, xlabel, ylabel, title)
        return fig

    def plot_fft(
        self,
        frequencies: NDArray[np.floating[Any]],
        magnitude_db: NDArray[np.floating[Any]],
        title: str = "FFT Magnitude Spectrum",
        peak_markers: int = 5,
    ) -> Figure:
        """Plot FFT magnitude spectrum.

        Args:
            frequencies: Frequency array in Hz.
            magnitude_db: Magnitude in dB.
            title: Plot title.
            peak_markers: Number of peak frequencies to mark (0 to disable).

        Returns:
            Matplotlib Figure object.

        Example:
            >>> freq = np.fft.rfftfreq(1000, 1/1000)
            >>> mag_db = 20 * np.log10(np.abs(np.fft.rfft(signal)))
            >>> fig = generator.plot_fft(freq, mag_db)
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot spectrum
        ax.plot(frequencies, magnitude_db, color=IEEE_COLORS["primary"], linewidth=1.5)

        # Mark peaks
        if peak_markers > 0:
            # Find peaks (ignore DC)
            valid_idx = frequencies > 0
            valid_freq = frequencies[valid_idx]
            valid_mag = magnitude_db[valid_idx]

            if len(valid_mag) > 0:
                peak_indices = np.argsort(valid_mag)[-peak_markers:]
                for idx in peak_indices:
                    freq_val = valid_freq[idx]
                    mag_val = valid_mag[idx]
                    ax.plot(freq_val, mag_val, "ro", markersize=6, alpha=0.7)
                    ax.annotate(
                        f"{freq_val:.1f} Hz",
                        (freq_val, mag_val),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        color=IEEE_COLORS["danger"],
                    )

        self.styler.apply_ieee_style(ax, "Frequency (Hz)", "Magnitude (dB)", title)

        # Log scale for frequency if range > 2 decades
        if len(frequencies) > 1 and frequencies[-1] / frequencies[1] > 100:
            ax.set_xscale("log")

        return fig

    def plot_psd(
        self,
        frequencies: NDArray[np.floating[Any]],
        psd: NDArray[np.floating[Any]],
        title: str = "Power Spectral Density",
        units: str = "V²/Hz",
    ) -> Figure:
        """Plot Power Spectral Density.

        Args:
            frequencies: Frequency array in Hz.
            psd: Power spectral density array.
            title: Plot title.
            units: PSD units.

        Returns:
            Matplotlib Figure object.

        Example:
            >>> from scipy import signal as sp_signal
            >>> freq, psd = sp_signal.welch(data, fs=sample_rate)
            >>> fig = generator.plot_psd(freq, psd)
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Convert to dB scale
        psd_db = 10 * np.log10(psd + 1e-12)  # Add epsilon to avoid log(0)

        ax.plot(frequencies, psd_db, color=IEEE_COLORS["primary"], linewidth=1.5)
        self.styler.apply_ieee_style(ax, "Frequency (Hz)", f"PSD (dB {units})", title)

        # Log scale for frequency
        if len(frequencies) > 1 and frequencies[-1] / frequencies[1] > 100:
            ax.set_xscale("log")

        return fig

    def plot_spectrogram(
        self,
        time: NDArray[np.floating[Any]],
        frequencies: NDArray[np.floating[Any]],
        spectrogram: NDArray[np.floating[Any]],
        title: str = "Spectrogram",
    ) -> Figure:
        """Plot time-frequency spectrogram.

        Args:
            time: Time array in seconds.
            frequencies: Frequency array in Hz.
            spectrogram: 2D spectrogram array (freq x time).
            title: Plot title.

        Returns:
            Matplotlib Figure object.

        Example:
            >>> from scipy import signal as sp_signal
            >>> f, t, Sxx = sp_signal.spectrogram(data, fs=sample_rate)
            >>> fig = generator.plot_spectrogram(t, f, Sxx)
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Convert to dB scale
        spec_db = 10 * np.log10(spectrogram + 1e-12)

        # Plot spectrogram
        im = ax.pcolormesh(
            time, frequencies, spec_db, shading="auto", cmap="viridis", rasterized=True
        )

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, label="Power (dB)")
        cbar.ax.tick_params(labelsize=9)

        self.styler.apply_ieee_style(ax, "Time (s)", "Frequency (Hz)", title, grid=False)

        return fig

    def plot_eye_diagram(
        self,
        signal: NDArray[np.floating[Any]],
        samples_per_symbol: int,
        title: str = "Eye Diagram",
        num_traces: int = 100,
    ) -> Figure:
        """Plot eye diagram for digital signals.

        Args:
            signal: Signal array.
            samples_per_symbol: Samples per symbol period.
            title: Plot title.
            num_traces: Number of eye traces to plot.

        Returns:
            Matplotlib Figure object.

        Example:
            >>> # For 1000 samples at 10 samples/symbol
            >>> fig = generator.plot_eye_diagram(signal, 10, num_traces=50)
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Extract symbol windows
        num_symbols = len(signal) // samples_per_symbol
        num_traces = min(num_traces, num_symbols - 1)

        for i in range(num_traces):
            start = i * samples_per_symbol
            end = start + 2 * samples_per_symbol  # Two symbol periods
            if end <= len(signal):
                trace = signal[start:end]
                ax.plot(
                    np.arange(len(trace)),
                    trace,
                    color=IEEE_COLORS["primary"],
                    alpha=0.3,
                    linewidth=0.5,
                )

        self.styler.apply_ieee_style(ax, "Sample", "Amplitude", title)
        return fig

    def plot_histogram(
        self,
        data: NDArray[np.floating[Any]],
        bins: int = 50,
        title: str = "Sample Distribution",
        xlabel: str = "Value",
    ) -> Figure:
        """Plot histogram of sample distribution.

        Args:
            data: Data array.
            bins: Number of histogram bins.
            title: Plot title.
            xlabel: X-axis label.

        Returns:
            Matplotlib Figure object.

        Example:
            >>> fig = generator.plot_histogram(signal, bins=100, title="Voltage Distribution")
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot histogram
        n, bins_edges, _ = ax.hist(
            data, bins=bins, color=IEEE_COLORS["primary"], alpha=0.7, edgecolor="black"
        )

        # Fit Gaussian
        mu = np.mean(data)
        sigma = np.std(data)
        x = np.linspace(bins_edges[0], bins_edges[-1], 200)
        gaussian = (
            len(data)
            * (bins_edges[1] - bins_edges[0])
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        )
        ax.plot(x, gaussian, color=IEEE_COLORS["danger"], linewidth=2, label="Gaussian fit")

        # Add statistics text
        stats_text = f"mean = {mu:.4f}\nstd = {sigma:.4f}"
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        ax.legend(fontsize=9)
        self.styler.apply_ieee_style(ax, xlabel, "Count", title)
        return fig

    def plot_jitter(
        self,
        time_intervals: NDArray[np.floating[Any]],
        title: str = "Jitter Analysis",
    ) -> Figure:
        """Plot jitter analysis with histogram and time series.

        Args:
            time_intervals: Array of time interval measurements.
            title: Plot title.

        Returns:
            Matplotlib Figure object with two subplots.

        Example:
            >>> # time_intervals in seconds
            >>> fig = generator.plot_jitter(time_intervals)
        """
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)

        # Time series plot
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(
            np.arange(len(time_intervals)),
            time_intervals * 1e9,  # Convert to nanoseconds
            color=IEEE_COLORS["primary"],
            linewidth=1,
            marker="o",
            markersize=2,
            alpha=0.6,
        )
        self.styler.apply_ieee_style(ax1, "Interval #", "Jitter (ns)", f"{title} - Time Series")

        # Histogram
        ax2 = fig.add_subplot(gs[1])
        jitter_ns = time_intervals * 1e9
        ax2.hist(jitter_ns, bins=50, color=IEEE_COLORS["secondary"], alpha=0.7, edgecolor="black")

        # Statistics
        mean_jitter = np.mean(jitter_ns)
        std_jitter = np.std(jitter_ns)
        pk_pk_jitter = np.max(jitter_ns) - np.min(jitter_ns)

        stats_text = (
            f"Mean: {mean_jitter:.3f} ns\nStd: {std_jitter:.3f} ns\nPk-Pk: {pk_pk_jitter:.3f} ns"
        )
        ax2.text(
            0.98,
            0.98,
            stats_text,
            transform=ax2.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        self.styler.apply_ieee_style(ax2, "Jitter (ns)", "Count", "Distribution")

        return fig

    def plot_power(
        self,
        time: NDArray[np.floating[Any]],
        voltage: NDArray[np.floating[Any]],
        current: NDArray[np.floating[Any]],
        title: str = "Power Waveform",
    ) -> Figure:
        """Plot power waveforms (voltage, current, power).

        Args:
            time: Time array in seconds.
            voltage: Voltage array.
            current: Current array.
            title: Plot title.

        Returns:
            Matplotlib Figure object with three subplots.

        Example:
            >>> fig = generator.plot_power(time, voltage, current)
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figsize, dpi=self.dpi, sharex=True)

        # Voltage
        ax1.plot(time, voltage, color=IEEE_COLORS["primary"], linewidth=1.5)
        self.styler.apply_ieee_style(ax1, "", "Voltage (V)", "Voltage", grid=True)

        # Current
        ax2.plot(time, current, color=IEEE_COLORS["secondary"], linewidth=1.5)
        self.styler.apply_ieee_style(ax2, "", "Current (A)", "Current", grid=True)

        # Power
        power = voltage * current
        ax3.plot(time, power, color=IEEE_COLORS["accent"], linewidth=1.5)
        ax3.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        self.styler.apply_ieee_style(ax3, "Time (s)", "Power (W)", "Instantaneous Power", grid=True)

        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
        fig.tight_layout()

        return fig

    @staticmethod
    def figure_to_base64(fig: Figure, format: str = "png") -> str:
        """Convert matplotlib figure to base64-encoded string for HTML embedding.

        Args:
            fig: Matplotlib Figure object.
            format: Image format (png, jpg, svg).

        Returns:
            Base64-encoded image string with data URI prefix.

        Example:
            >>> fig = plt.figure()
            >>> img_str = IEEEPlotGenerator.figure_to_base64(fig)
            >>> "data:image/png;base64," in img_str
            True
        """
        buffer = BytesIO()
        fig.savefig(buffer, format=format, bbox_inches="tight", dpi=150)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        buffer.close()
        plt.close(fig)

        return f"data:image/{format};base64,{img_base64}"
